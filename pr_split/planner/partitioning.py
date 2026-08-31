from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from itertools import pairwise
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ..constants import AssignmentType, PartitionStrategy, Priority
from ..exceptions import PRSplitError
from ..graph import PlanDAG
from ..schemas import Group, GroupAssignment
from .chunker import recompute_estimated_loc

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ortools.sat.python.cp_model import IntVar

    from ..config import Settings
    from ..diff_ops import ParsedDiff

_TEST_PAIR_BONUS = 40
_AFFINITY_SAME_FILE = 500
_AFFINITY_SHARED_DIR_MULTIPLIER = 25
_AFFINITY_SAME_SUFFIX = 10
_AFFINITY_ORTHOGONAL_PENALTY = 20
_AFFINITY_LOGICAL_SHARED_DIR_BONUS = 10
_GRAPH_ORTHOGONAL_FILE_PENALTY = 25.0
_CP_SAT_OVERFLOW_WEIGHT = 1_000
_CP_SAT_UNDERFLOW_WEIGHT = 1_000
_CP_SAT_GROUP_WEIGHT_LOGICAL = 200
_CP_SAT_GROUP_WEIGHT_ORTHOGONAL = 250
_CP_SAT_CROSS_FILE_PENALTY_LOGICAL = 0
_CP_SAT_CROSS_FILE_PENALTY_ORTHOGONAL = 400
_CP_SAT_AFFINITY_DIVISOR_LOGICAL = 5
_CP_SAT_AFFINITY_DIVISOR_ORTHOGONAL = 10


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


def _normalize_test_stem(path: str) -> tuple[str, bool]:
    stem = PurePosixPath(path).stem
    normalized = stem.removeprefix("test_").removesuffix("_test")
    return normalized, normalized != stem


def _test_pair_bonus(path_a: str, path_b: str) -> int:
    stem_a, is_test_a = _normalize_test_stem(path_a)
    stem_b, is_test_b = _normalize_test_stem(path_b)
    return _TEST_PAIR_BONUS if (is_test_a ^ is_test_b) and stem_a == stem_b else 0


def _affinity_score(unit_a: PartitionUnit, unit_b: PartitionUnit, priority: Priority) -> int:
    if unit_a.file_path == unit_b.file_path:
        return _AFFINITY_SAME_FILE

    score = 0
    shared_depth = _shared_dir_depth(unit_a.file_path, unit_b.file_path)
    score += shared_depth * _AFFINITY_SHARED_DIR_MULTIPLIER
    if PurePosixPath(unit_a.file_path).suffix == PurePosixPath(unit_b.file_path).suffix:
        score += _AFFINITY_SAME_SUFFIX
    score += _test_pair_bonus(unit_a.file_path, unit_b.file_path)

    if priority == Priority.ORTHOGONAL:
        score = max(0, score - _AFFINITY_ORTHOGONAL_PENALTY)
    else:
        score += _AFFINITY_LOGICAL_SHARED_DIR_BONUS if shared_depth else 0

    return score


def _merge_order_is_acyclic(grouped_units: Iterable[Sequence[PartitionUnit]]) -> bool:
    """Check that the merge-order dependencies implied by ``grouped_units`` form a DAG.

    Mirrors ``_derive_merge_order_dependencies``: within each file, groups are ordered by
    their earliest unit and each group depends on the one before it.
    """
    file_occurrences: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for group_idx, group_units in enumerate(grouped_units):
        first_positions: dict[str, int] = {}
        for unit in group_units:
            previous = first_positions.get(unit.file_path)
            if previous is None or unit.position < previous:
                first_positions[unit.file_path] = unit.position
        for file_path, position in first_positions.items():
            file_occurrences[file_path].append((position, group_idx))

    sorter: TopologicalSorter[int] = TopologicalSorter()
    for occurrences in file_occurrences.values():
        ordered_group_indices = [group_idx for _, group_idx in sorted(occurrences)]
        for parent_idx, child_idx in pairwise(ordered_group_indices):
            sorter.add(child_idx, parent_idx)
    try:
        sorter.prepare()
    except CycleError:
        return False
    return True


def _group_units_graph(
    units: list[PartitionUnit], *, settings: Settings
) -> list[list[PartitionUnit]]:
    remaining = set(range(len(units)))
    grouped_units: list[list[PartitionUnit]] = []
    grouped_files: set[str] = set()

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
                if candidate_unit.file_path in grouped_files and not _merge_order_is_acyclic(
                    [*grouped_units, [*(units[idx] for idx in current_group), candidate_unit]]
                ):
                    continue

                affinity = sum(
                    _affinity_score(candidate_unit, units[group_idx], settings.priority)
                    for group_idx in current_group
                )
                fill_bonus = candidate_unit.loc / max(1, settings.max_loc)
                file_penalty = (
                    _GRAPH_ORTHOGONAL_FILE_PENALTY
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
        grouped_files.update(units[idx].file_path for idx in current_group)

    return grouped_units


def _group_load(group_units: list[PartitionUnit]) -> int:
    return sum(unit.loc for unit in group_units)


def _group_anchor_position(group_units: list[PartitionUnit]) -> int:
    return min(unit.position for unit in group_units)


def _group_affinity(
    left_group: list[PartitionUnit], right_group: list[PartitionUnit], priority: Priority
) -> int:
    return sum(
        _affinity_score(left_unit, right_unit, priority)
        for left_unit in left_group
        for right_unit in right_group
    )


def _shared_file_merge_is_contiguous(
    grouped_units: list[list[PartitionUnit]], source_idx: int, target_idx: int
) -> bool:
    source_files = {unit.file_path for unit in grouped_units[source_idx]}
    target_files = {unit.file_path for unit in grouped_units[target_idx]}
    shared_files = source_files & target_files

    if not shared_files:
        return True

    for file_path in shared_files:
        ordered_occurrences: list[tuple[int, int]] = []
        for group_idx, group_units in enumerate(grouped_units):
            positions = [unit.position for unit in group_units if unit.file_path == file_path]
            if positions:
                ordered_occurrences.append((min(positions), group_idx))

        occurrence_order = [group_idx for _, group_idx in sorted(ordered_occurrences)]
        if abs(occurrence_order.index(source_idx) - occurrence_order.index(target_idx)) != 1:
            return False

    return True


def _best_graph_merge_target(
    grouped_units: list[list[PartitionUnit]], source_idx: int, *, settings: Settings
) -> int | None:
    if settings.min_loc is None:
        return None

    source_group = grouped_units[source_idx]
    source_load = _group_load(source_group)
    source_underflow = max(0, settings.min_loc - source_load)
    if source_underflow == 0:
        return None

    best_idx: int | None = None
    best_key: tuple[int, int, int, int, int, int] | None = None
    source_anchor = _group_anchor_position(source_group)

    for target_idx, target_group in enumerate(grouped_units):
        if target_idx == source_idx:
            continue

        merged_load = source_load + _group_load(target_group)
        if merged_load > settings.max_loc:
            continue
        if not _shared_file_merge_is_contiguous(grouped_units, source_idx, target_idx):
            continue
        if not _merge_order_is_acyclic(_merge_group_units(grouped_units, source_idx, target_idx)):
            continue

        current_underflow = source_underflow + max(0, settings.min_loc - _group_load(target_group))
        merged_underflow = max(0, settings.min_loc - merged_load)
        if merged_underflow >= current_underflow:
            continue

        target_anchor = _group_anchor_position(target_group)
        merge_key = (
            int(merged_load >= settings.min_loc),
            _group_affinity(source_group, target_group, settings.priority),
            -abs(source_anchor - target_anchor),
            -abs(settings.max_loc - merged_load),
            -min(source_idx, target_idx),
            -max(source_idx, target_idx),
        )
        if best_key is None or merge_key > best_key:
            best_idx = target_idx
            best_key = merge_key

    return best_idx


def _merge_group_units(
    grouped_units: list[list[PartitionUnit]], source_idx: int, target_idx: int
) -> list[list[PartitionUnit]]:
    merged_group = sorted(
        grouped_units[target_idx] + grouped_units[source_idx],
        key=lambda unit: unit.position,
    )
    repaired_groups = [list(group_units) for group_units in grouped_units]
    repaired_groups[target_idx] = merged_group
    del repaired_groups[source_idx]
    return repaired_groups


def _repair_graph_min_loc(
    grouped_units: list[list[PartitionUnit]], *, settings: Settings
) -> list[list[PartitionUnit]]:
    if settings.min_loc is None or len(grouped_units) < 2:
        return grouped_units

    repaired_groups = [list(group_units) for group_units in grouped_units]

    while True:
        undersized_group_indices = sorted(
            (
                group_idx
                for group_idx, group_units in enumerate(repaired_groups)
                if _group_load(group_units) < settings.min_loc
            ),
            key=lambda group_idx: (
                _group_load(repaired_groups[group_idx]),
                _group_anchor_position(repaired_groups[group_idx]),
                group_idx,
            ),
        )

        merged = False
        for source_idx in undersized_group_indices:
            target_idx = _best_graph_merge_target(
                repaired_groups,
                source_idx,
                settings=settings,
            )
            if target_idx is None:
                continue
            repaired_groups = _merge_group_units(repaired_groups, source_idx, target_idx)
            merged = True
            break

        if not merged:
            return repaired_groups


def _group_units_cp_sat(
    units: list[PartitionUnit], *, settings: Settings
) -> list[list[PartitionUnit]]:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise PRSplitError(
            "CP-SAT partitioning requires the optional 'ortools' package; "
            "install it with `pip install 'pr-split[cp-sat]'`"
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
    underflow = (
        [
            model.NewIntVar(0, max_total_loc, f"underflow_{group_idx}")
            for group_idx in range(group_slots)
        ]
        if settings.min_loc is not None
        else []
    )

    for unit_idx in range(len(units)):
        model.Add(sum(x[(unit_idx, group_idx)] for group_idx in range(group_slots)) == 1)

    for group_idx in range(group_slots):
        load = sum(
            units[unit_idx].loc * x[(unit_idx, group_idx)] for unit_idx in range(len(units))
        )
        model.Add(load <= max_total_loc * y[group_idx])
        model.Add(load - settings.max_loc <= overflow[group_idx])
        if underflow:
            # Underflow only applies to active groups: inactive groups (y==0)
            # have zero load and should not be penalised for being undersized.
            model.Add(settings.min_loc * y[group_idx] - load <= underflow[group_idx])
        for unit_idx in range(len(units)):
            model.Add(x[(unit_idx, group_idx)] <= y[group_idx])

    for group_idx in range(group_slots - 1):
        model.Add(y[group_idx] >= y[group_idx + 1])

    pair_terms: list[tuple[int, int, IntVar]] = []
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

    overflow_weight = _CP_SAT_OVERFLOW_WEIGHT
    group_weight = (
        _CP_SAT_GROUP_WEIGHT_LOGICAL
        if settings.priority == Priority.LOGICAL
        else _CP_SAT_GROUP_WEIGHT_ORTHOGONAL
    )
    cross_file_penalty = (
        _CP_SAT_CROSS_FILE_PENALTY_LOGICAL
        if settings.priority == Priority.LOGICAL
        else _CP_SAT_CROSS_FILE_PENALTY_ORTHOGONAL
    )
    affinity_divisor = (
        _CP_SAT_AFFINITY_DIVISOR_LOGICAL
        if settings.priority == Priority.LOGICAL
        else _CP_SAT_AFFINITY_DIVISOR_ORTHOGONAL
    )

    underflow_cost = _CP_SAT_UNDERFLOW_WEIGHT * sum(underflow) if underflow else 0
    model.Minimize(
        group_weight * sum(y)
        + overflow_weight * sum(overflow)
        + underflow_cost
        + cross_file_penalty
        * sum(same_group for _, is_cross_file, same_group in pair_terms if is_cross_file)
        - sum(
            (affinity // affinity_divisor) * same_group
            for affinity, _, same_group in pair_terms
            if affinity > 0
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = settings.cp_sat_timeout
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
    for group in groups:
        for assignment in group.assignments:
            min_hunk = min(assignment.hunk_indices, default=0)
            file_occurrences[assignment.file_path].append((min_hunk, group.id))

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
            grouped_units = _repair_graph_min_loc(grouped_units, settings=settings)
        case PartitionStrategy.CP_SAT:
            grouped_units = _group_units_cp_sat(units, settings=settings)
        case _:
            raise PRSplitError(f"Unsupported partition strategy '{settings.partition_strategy}'")

    groups = _build_groups_from_units(
        grouped_units,
        parsed_diff,
        backend=settings.partition_strategy,
    )
    PlanDAG(groups).validate_acyclic()
    return groups
