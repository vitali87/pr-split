"""Mechanically repair an LLM plan before it is validated.

Models, small local ones most of all, sometimes skip a hunk, claim one hunk
in two groups, invent a file or hunk index, or depend on a group that does
not exist. Each has one obvious fix, so it is applied with a warning rather
than failing a whole planning call. A cycle or a duplicate group id has no
single right fix and is left for validation to report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .. import logs
from ..constants import AssignmentType
from .chunker import assign_uncovered_hunks, recompute_estimated_loc

if TYPE_CHECKING:
    from ..diff_ops import ParsedDiff
    from ..schemas import Group


def _drop_unknown_hunks(groups: list[Group], hunk_counts: dict[str, int]) -> None:
    for group in groups:
        kept = []
        for assignment in group.assignments:
            count = hunk_counts.get(assignment.file_path)
            if count is None:
                logger.warning(
                    logs.REPAIR_UNKNOWN_FILE.format(group=group.id, file=assignment.file_path)
                )
                continue
            if assignment.assignment_type is AssignmentType.PARTIAL_HUNKS:
                bad = [i for i in assignment.hunk_indices if not 0 <= i < count]
                if bad:
                    logger.warning(
                        logs.REPAIR_UNKNOWN_HUNKS.format(
                            group=group.id, file=assignment.file_path, indices=bad
                        )
                    )
                    assignment.hunk_indices = [i for i in assignment.hunk_indices if i not in bad]
                if not assignment.hunk_indices:
                    continue
            kept.append(assignment)
        group.assignments = kept


def _drop_duplicate_claims(groups: list[Group], hunk_counts: dict[str, int]) -> None:
    owner: dict[tuple[str, int], str] = {}
    for group in groups:
        kept = []
        for assignment in group.assignments:
            count = hunk_counts[assignment.file_path]
            covered = assignment.covered_indices(count)
            taken = [i for i in covered if (assignment.file_path, i) in owner]
            if taken:
                logger.warning(
                    logs.REPAIR_DUPLICATE_HUNKS.format(
                        group=group.id,
                        file=assignment.file_path,
                        indices=taken,
                        owner=owner[(assignment.file_path, taken[0])],
                    )
                )
                assignment.assignment_type = AssignmentType.PARTIAL_HUNKS
                assignment.hunk_indices = [i for i in covered if i not in taken]
            if not assignment.hunk_indices and assignment.assignment_type is not (
                AssignmentType.WHOLE_FILE
            ):
                continue
            for idx in assignment.covered_indices(count):
                owner[(assignment.file_path, idx)] = group.id
            kept.append(assignment)
        group.assignments = kept


def _drop_bad_dependencies(groups: list[Group]) -> None:
    ids = {g.id for g in groups}
    for group in groups:
        bad = [d for d in group.depends_on if d not in ids or d == group.id]
        if bad:
            logger.warning(logs.REPAIR_UNKNOWN_DEPENDENCY.format(group=group.id, deps=bad))
            group.depends_on = [d for d in group.depends_on if d not in bad]
        # A repeated id is harmless but would show up twice in every PR body.
        group.depends_on = list(dict.fromkeys(group.depends_on))


def _drop_empty_groups(groups: list[Group]) -> list[Group]:
    empty = {g.id for g in groups if not g.assignments}
    if not empty:
        return groups
    by_id = {g.id: g for g in groups}
    for gid in sorted(empty):
        logger.warning(logs.REPAIR_EMPTY_GROUP.format(group=gid))
    kept = [g for g in groups if g.id not in empty]
    for group in kept:
        # Keep the order an emptied group imposed by inheriting its parents.
        deps: list[str] = []
        pending = list(group.depends_on)
        seen: set[str] = set()
        while pending:
            dep = pending.pop(0)
            if dep in seen:
                continue
            seen.add(dep)
            if dep in empty:
                pending.extend(by_id[dep].depends_on)
            elif dep != group.id:
                deps.append(dep)
        group.depends_on = deps
    return kept


def repair_plan(groups: list[Group], parsed_diff: ParsedDiff) -> list[Group]:
    if not groups:
        return groups  # nothing to place hunks into; validation reports the gap
    hunk_counts = {pf.path: len(pf) for pf in parsed_diff.patch_set}
    _drop_unknown_hunks(groups, hunk_counts)
    _drop_duplicate_claims(groups, hunk_counts)
    _drop_bad_dependencies(groups)
    if any(g.assignments for g in groups):
        groups = _drop_empty_groups(groups)
    assign_uncovered_hunks(groups, parsed_diff)
    recompute_estimated_loc(groups, parsed_diff)
    return groups
