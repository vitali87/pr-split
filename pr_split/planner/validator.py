from __future__ import annotations

from loguru import logger

from .. import logs
from ..diff_ops import ParsedDiff
from ..exceptions import ErrorMsg, PlanValidationError
from ..graph import PlanDAG
from ..schemas import Group


def validate_coverage(groups: list[Group], parsed_diff: ParsedDiff) -> None:
    assigned: dict[tuple[str, int], list[str]] = {}
    for group in groups:
        for assignment in group.assignments:
            for idx in assignment.hunk_indices:
                key = (assignment.file_path, idx)
                assigned.setdefault(key, []).append(group.id)

    all_hunks: set[tuple[str, int]] = set()
    for pf in parsed_diff.patch_set:
        for i in range(len(pf)):
            all_hunks.add((pf.path, i))

    for key in all_hunks:
        if key not in assigned:
            raise PlanValidationError(ErrorMsg.COVERAGE_GAP(file=key[0], index=key[1]))

    for key, group_ids in assigned.items():
        if len(group_ids) > 1:
            raise PlanValidationError(
                ErrorMsg.COVERAGE_OVERLAP(file=key[0], index=key[1], groups=", ".join(group_ids))
            )


def validate_loc(groups: list[Group], parsed_diff: ParsedDiff) -> None:
    total_estimated = sum(g.estimated_loc for g in groups)
    total_actual = parsed_diff.stats["total_loc"]
    if total_estimated != total_actual:
        raise PlanValidationError(
            ErrorMsg.LOC_MISMATCH(actual=total_estimated, expected=total_actual)
        )


def validate_no_conflicts(groups: list[Group], dag: PlanDAG) -> None:
    group_files: dict[str, dict[str, set[int]]] = {}
    for group in groups:
        file_hunks: dict[str, set[int]] = {}
        for assignment in group.assignments:
            file_hunks.setdefault(assignment.file_path, set()).update(assignment.hunk_indices)
        group_files[group.id] = file_hunks

    group_ids = [g.id for g in groups]
    for i, gid_a in enumerate(group_ids):
        ancestors_a = dag.ancestors(gid_a)
        descendants_a = dag.descendants(gid_a)
        for gid_b in group_ids[i + 1 :]:
            if gid_b in ancestors_a or gid_b in descendants_a:
                continue
            files_a = group_files[gid_a]
            files_b = group_files[gid_b]
            shared_files = set(files_a.keys()) & set(files_b.keys())
            for file_path in shared_files:
                overlap = files_a[file_path] & files_b[file_path]
                if overlap:
                    raise PlanValidationError(
                        ErrorMsg.MERGE_CONFLICT(a=gid_a, b=gid_b, file=file_path)
                    )


def validate_loc_bounds(groups: list[Group], max_loc: int) -> list[str]:
    warnings: list[str] = []
    for group in groups:
        if group.estimated_loc > max_loc:
            msg = logs.LOC_SOFT_WARN.format(group=group.id, loc=group.estimated_loc, limit=max_loc)
            logger.warning(msg)
            warnings.append(msg)
    return warnings


def validate_plan(
    groups: list[Group],
    parsed_diff: ParsedDiff,
    dag: PlanDAG,
    max_loc: int,
) -> list[str]:
    logger.info(logs.VALIDATING_PLAN)
    dag.validate_acyclic()
    validate_coverage(groups, parsed_diff)
    validate_loc(groups, parsed_diff)
    validate_no_conflicts(groups, dag)
    warnings = validate_loc_bounds(groups, max_loc)
    logger.info(logs.VALIDATION_PASSED)
    return warnings
