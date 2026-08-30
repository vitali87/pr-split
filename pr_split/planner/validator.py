from __future__ import annotations

from .. import logs
from ..constants import AssignmentType, LocViolationType
from ..diff_ops import ParsedDiff
from ..exceptions import ErrorMsg, PlanValidationError
from ..graph import PlanDAG
from ..schemas import Group
from ..types_defs import LocBoundViolation


def validate_coverage(groups: list[Group], parsed_diff: ParsedDiff) -> None:
    hunk_counts = {pf.path: len(pf) for pf in parsed_diff.patch_set}
    assigned: dict[tuple[str, int], list[str]] = {}
    for group in groups:
        for assignment in group.assignments:
            if assignment.file_path not in hunk_counts:
                raise PlanValidationError(
                    ErrorMsg.UNKNOWN_FILE(file=assignment.file_path, group=group.id)
                )
            # A WHOLE_FILE assignment claims every hunk of the file even when
            # its hunk_indices list was left empty.
            if assignment.assignment_type is AssignmentType.WHOLE_FILE:
                indices = range(hunk_counts.get(assignment.file_path, 0))
            else:
                indices = assignment.hunk_indices
            for idx in indices:
                key = (assignment.file_path, idx)
                assigned.setdefault(key, []).append(group.id)

    all_hunks = {(pf.path, i) for pf in parsed_diff.patch_set for i in range(len(pf))}

    for key, group_ids in assigned.items():
        if key not in all_hunks:
            raise PlanValidationError(
                ErrorMsg.UNKNOWN_HUNK(file=key[0], index=key[1], group=group_ids[0])
            )

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


def detect_loc_bound_violations(
    groups: list[Group],
    max_loc: int,
    min_loc: int | None = None,
) -> list[LocBoundViolation]:
    violations: list[LocBoundViolation] = []
    for group in groups:
        if min_loc is not None and group.estimated_loc < min_loc:
            violations.append(
                LocBoundViolation(
                    group_id=group.id,
                    violation_type=LocViolationType.BELOW_MIN,
                    limit=min_loc,
                    estimated_loc=group.estimated_loc,
                    estimated_added=group.estimated_added,
                    estimated_removed=group.estimated_removed,
                )
            )
        if group.estimated_loc > max_loc:
            violations.append(
                LocBoundViolation(
                    group_id=group.id,
                    violation_type=LocViolationType.ABOVE_MAX,
                    limit=max_loc,
                    estimated_loc=group.estimated_loc,
                    estimated_added=group.estimated_added,
                    estimated_removed=group.estimated_removed,
                )
            )
    return violations


def _format_loc_bound_violation(violation: LocBoundViolation) -> str:
    template = (
        logs.LOC_MIN_WARN
        if violation.violation_type == LocViolationType.BELOW_MIN
        else logs.LOC_MAX_WARN
    )
    return template.format(
        group=violation.group_id,
        loc=violation.estimated_loc,
        added=violation.estimated_added,
        removed=violation.estimated_removed,
        limit=violation.limit,
    )


def validate_loc_bounds(
    groups: list[Group],
    max_loc: int,
    min_loc: int | None = None,
) -> list[str]:
    return [
        _format_loc_bound_violation(violation)
        for violation in detect_loc_bound_violations(groups, max_loc, min_loc)
    ]


def validate_plan(
    groups: list[Group],
    parsed_diff: ParsedDiff,
    dag: PlanDAG,
    max_loc: int,
    min_loc: int | None = None,
) -> list[str]:
    dag.validate_acyclic()
    validate_coverage(groups, parsed_diff)
    validate_loc(groups, parsed_diff)
    validate_no_conflicts(groups, dag)
    return validate_loc_bounds(groups, max_loc, min_loc)
