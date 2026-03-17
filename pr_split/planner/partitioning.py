from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ..constants import AssignmentType, PartitionStrategy, Priority
from ..exceptions import PRSplitError
from ..schemas import Group, GroupAssignment
from .chunker import recompute_estimated_loc

if TYPE_CHECKING:
    from ..config import Settings
    from ..diff_ops import ParsedDiff


@dataclass(frozen=True)
class PartitionUnit:
    id: str
    file_path: str
    hunk_indices: tuple[int, ...]
    loc: int
    position: int


def build_partition_units(parsed_diff: ParsedDiff, max_loc: int) -> list[PartitionUnit]:
    units: list[PartitionUnit] = []
    position = 0
    for patch_file in parsed_diff.patch_set:
        current_indices: list[int] = []
        current_loc = 0

        for hunk_index, hunk in enumerate(patch_file):
            hunk_loc = hunk.added + hunk.removed
            if current_indices and current_loc + hunk_loc > max_loc:
                units.append(
                    PartitionUnit(
                        id=f"{patch_file.path}:{current_indices[0]}-{current_indices[-1]}",
                        file_path=patch_file.path,
                        hunk_indices=tuple(current_indices),
                        loc=current_loc,
                        position=position,
                    )
                )
                position += 1
                current_indices = []
                current_loc = 0

            current_indices.append(hunk_index)
            current_loc += hunk_loc

        if current_indices:
            units.append(
                PartitionUnit(
                    id=f"{patch_file.path}:{current_indices[0]}-{current_indices[-1]}",
                    file_path=patch_file.path,
                    hunk_indices=tuple(current_indices),
                    loc=current_loc,
                    position=position,
                )
            )
            position += 1

    return units


def _shared_dir_depth(path_a: str, path_b: str) -> int:
    parts_a = PurePosixPath(path_a).parts[:-1]
    parts_b = PurePosixPath(path_b).parts[:-1]
    depth = 0
    for left, right in zip(parts_a, parts_b, strict=False):
        if left != right:
            break
        depth += 1
    return depth


def _test_pair_bonus(path_a: str, path_b: str) -> int:
    stem_a = PurePosixPath(path_a).stem.removesuffix("_test")
    stem_b = PurePosixPath(path_b).stem.removesuffix("_test")
    names_match = stem_a == stem_b
    has_test_name = "test" in PurePosixPath(path_a).stem or "test" in PurePosixPath(path_b).stem
    return 40 if names_match and has_test_name else 0


def _affinity_score(unit_a: PartitionUnit, unit_b: PartitionUnit, priority: Priority) -> int:
    if unit_a.file_path == unit_b.file_path:
        return 500

    score = 0
    shared_depth = _shared_dir_depth(unit_a.file_path, unit_b.file_path)
    score += shared_depth * 25
    if PurePosixPath(unit_a.file_path).suffix == PurePosixPath(unit_b.file_path).suffix:
        score += 10
    score += _test_pair_bonus(unit_a.file_path, unit_b.file_path)

    if priority == Priority.ORTHOGONAL:
        score = max(0, score - 20)
    else:
        score += 10 if shared_depth else 0

    return score


def _group_units_graph(
    units: list[PartitionUnit], *, settings: Settings
) -> list[list[PartitionUnit]]:
    remaining = set(range(len(units)))
    grouped_units: list[list[PartitionUnit]] = []

    while remaining:
        seed = max(remaining, key=lambda idx: (units[idx].loc, -units[idx].position))
        current_group = [seed]
        remaining.remove(seed)
        current_load = units[seed].loc

        while remaining:
            current_files = {units[idx].file_path for idx in current_group}
            best_idx: int | None = None
            best_score = 0.0
            for candidate in remaining:
                candidate_unit = units[candidate]
                if current_load and current_load + candidate_unit.loc > settings.max_loc:
                    continue

                affinity = sum(
                    _affinity_score(candidate_unit, units[group_idx], settings.priority)
                    for group_idx in current_group
                )
                fill_bonus = candidate_unit.loc / max(1, settings.max_loc)
                file_penalty = (
                    25.0
                    if settings.priority == Priority.ORTHOGONAL
                    and candidate_unit.file_path not in current_files
                    else 0.0
                )
                score = affinity + fill_bonus - file_penalty
                if score > best_score:
                    best_idx = candidate
                    best_score = score

            if best_idx is None or best_score <= 0:
                break

            current_group.append(best_idx)
            current_load += units[best_idx].loc
            remaining.remove(best_idx)

        grouped_units.append(
            sorted((units[idx] for idx in current_group), key=lambda unit: unit.position)
        )

    return grouped_units


def _build_affinity_edges(
    units: list[PartitionUnit], priority: Priority
) -> list[tuple[int, int, int]]:
    edges: list[tuple[int, int, int]] = []
    for left in range(len(units)):
        for right in range(left + 1, len(units)):
            weight = _affinity_score(units[left], units[right], priority)
            if weight > 0:
                edges.append((left, right, weight))
    return edges


def _group_units_cp_sat(
    units: list[PartitionUnit], *, settings: Settings
) -> list[list[PartitionUnit]]:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise PRSplitError(
            "CP-SAT partitioning requires the optional 'ortools' package to be installed"
        ) from exc

    if not units:
        return []

    group_slots = len(units)
    max_total_loc = max(settings.max_loc, sum(unit.loc for unit in units), 1)

    model = cp_model.CpModel()
    x = {
        (unit_idx, group_idx): model.NewBoolVar(f"x_{unit_idx}_{group_idx}")
        for unit_idx in range(len(units))
        for group_idx in range(group_slots)
    }
    y = [model.NewBoolVar(f"y_{group_idx}") for group_idx in range(group_slots)]
    overflow = [
        model.NewIntVar(0, max_total_loc, f"overflow_{group_idx}")
        for group_idx in range(group_slots)
    ]

    for unit_idx in range(len(units)):
        model.Add(sum(x[(unit_idx, group_idx)] for group_idx in range(group_slots)) == 1)

    for group_idx in range(group_slots):
        load = sum(
            units[unit_idx].loc * x[(unit_idx, group_idx)]
            for unit_idx in range(len(units))
        )
        model.Add(load <= max_total_loc * y[group_idx])
        model.Add(load - settings.max_loc <= overflow[group_idx])
        for unit_idx in range(len(units)):
            model.Add(x[(unit_idx, group_idx)] <= y[group_idx])

    for group_idx in range(group_slots - 1):
        model.Add(y[group_idx] >= y[group_idx + 1])

    pair_terms: list[tuple[int, int, object]] = []
    for left in range(len(units)):
        for right in range(left + 1, len(units)):
            affinity = _affinity_score(units[left], units[right], settings.priority)
            same_group_terms = []
            for group_idx in range(group_slots):
                pair_var = model.NewBoolVar(f"pair_{left}_{right}_{group_idx}")
                model.Add(pair_var <= x[(left, group_idx)])
                model.Add(pair_var <= x[(right, group_idx)])
                model.Add(pair_var >= x[(left, group_idx)] + x[(right, group_idx)] - 1)
                same_group_terms.append(pair_var)
            same_group = model.NewBoolVar(f"same_{left}_{right}")
            model.Add(sum(same_group_terms) == same_group)
            is_cross_file = int(units[left].file_path != units[right].file_path)
            pair_terms.append((affinity, is_cross_file, same_group))

    overflow_weight = 1_000
    group_weight = 200 if settings.priority == Priority.LOGICAL else 250
    cross_file_penalty = 0 if settings.priority == Priority.LOGICAL else 400
    affinity_divisor = 5 if settings.priority == Priority.LOGICAL else 10

    model.Minimize(
        group_weight * sum(y)
        + overflow_weight * sum(overflow)
        + cross_file_penalty
        * sum(same_group for _, is_cross_file, same_group in pair_terms if is_cross_file)
        - sum(
            (affinity // affinity_divisor) * same_group
            for affinity, _, same_group in pair_terms
            if affinity > 0
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise PRSplitError("CP-SAT partitioning failed to find a feasible hunk assignment")

    assignments: dict[int, list[PartitionUnit]] = defaultdict(list)
    for unit_idx, unit in enumerate(units):
        for group_idx in range(group_slots):
            if solver.BooleanValue(x[(unit_idx, group_idx)]):
                assignments[group_idx].append(unit)
                break

    return [
        sorted(group_units, key=lambda unit: unit.position)
        for _, group_units in sorted(assignments.items())
        if group_units
    ]


def _build_group_title(group_index: int, units: list[PartitionUnit]) -> str:
    file_paths = sorted({unit.file_path for unit in units})
    if len(file_paths) == 1:
        stem = PurePosixPath(file_paths[0]).stem.replace("_", "-")
        return f"chore(split): review {stem}-{group_index}"
    return f"chore(split): review-slice-{group_index}"


def _build_group_description(backend: PartitionStrategy, units: list[PartitionUnit]) -> str:
    file_paths = sorted({unit.file_path for unit in units})
    files = ", ".join(file_paths)
    return f"{backend.value} partition over {len(file_paths)} file(s): {files}"


def _build_groups_from_units(
    grouped_units: list[list[PartitionUnit]],
    parsed_diff: ParsedDiff,
    *,
    backend: PartitionStrategy,
) -> list[Group]:
    file_hunk_counts = {patch_file.path: len(patch_file) for patch_file in parsed_diff.patch_set}
    groups: list[Group] = []

    for group_index, units in enumerate(grouped_units, start=1):
        assignments: list[GroupAssignment] = []
        grouped_by_file: dict[str, list[int]] = defaultdict(list)
        for unit in units:
            grouped_by_file[unit.file_path].extend(unit.hunk_indices)

        for file_path, hunk_indices in grouped_by_file.items():
            sorted_indices = sorted(set(hunk_indices))
            file_hunk_count = file_hunk_counts[file_path]
            if sorted_indices == list(range(file_hunk_count)):
                assignment_type = AssignmentType.WHOLE_FILE
            else:
                assignment_type = AssignmentType.PARTIAL_HUNKS
            assignments.append(
                GroupAssignment(
                    file_path=file_path,
                    assignment_type=assignment_type,
                    hunk_indices=sorted_indices,
                )
            )

        groups.append(
            Group(
                id=f"pr-{group_index}",
                title=_build_group_title(group_index, units),
                description=_build_group_description(backend, units),
                assignments=sorted(assignments, key=lambda assignment: assignment.file_path),
                estimated_loc=sum(unit.loc for unit in units),
            )
        )

    recompute_estimated_loc(groups, parsed_diff)
    _derive_merge_order_dependencies(groups)
    return groups


def _has_alternative_path(dep_map: dict[str, set[str]], source: str, target: str) -> bool:
    stack = [source]
    seen = {source}
    while stack:
        current = stack.pop()
        for parent in dep_map[current]:
            if current == source and parent == target:
                continue
            if parent == target:
                return True
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return False


def _derive_merge_order_dependencies(groups: list[Group]) -> None:
    file_occurrences: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for group_index, group in enumerate(groups):
        for assignment in group.assignments:
            min_hunk = min(assignment.hunk_indices, default=0)
            ordering_key = group_index * 10_000 + min_hunk
            file_occurrences[assignment.file_path].append((ordering_key, group.id))

    dep_map = {group.id: set(group.depends_on) for group in groups}
    for occurrences in file_occurrences.values():
        ordered_group_ids: list[str] = []
        for _, group_id in sorted(occurrences):
            if group_id not in ordered_group_ids:
                ordered_group_ids.append(group_id)
        for parent_id, child_id in pairwise(ordered_group_ids):
            dep_map[child_id].add(parent_id)

    for group in groups:
        reduced_deps = set(dep_map[group.id])
        for dep in list(reduced_deps):
            if _has_alternative_path(dep_map, group.id, dep):
                reduced_deps.remove(dep)
        group.depends_on = sorted(reduced_deps)


def partition_diff(parsed_diff: ParsedDiff, settings: Settings) -> list[Group]:
    units = build_partition_units(parsed_diff, settings.max_loc)
    if not units:
        return []

    match settings.partition_strategy:
        case PartitionStrategy.GRAPH:
            grouped_units = _group_units_graph(units, settings=settings)
        case PartitionStrategy.CP_SAT:
            grouped_units = _group_units_cp_sat(units, settings=settings)
        case _:
            raise PRSplitError(f"Unsupported partition strategy '{settings.partition_strategy}'")

    return _build_groups_from_units(
        grouped_units,
        parsed_diff,
        backend=settings.partition_strategy,
    )
