from __future__ import annotations

import subprocess

from loguru import logger
from unidiff import Hunk, PatchedFile

from .. import logs
from ..constants import AssignmentType
from ..exceptions import GitOperationError
from ..schemas import Group, GroupAssignment
from .parser import ParsedDiff


def merge_chain_assignments(
    group: Group,
    ancestors: list[Group],
    hunk_counts: dict[str, int] | None = None,
    *,
    carry_ancestor_files: bool = False,
) -> Group:
    counts = hunk_counts or {}
    ancestor_hunks: dict[str, set[int]] = {}
    for ancestor in ancestors:
        for assignment in ancestor.assignments:
            # A WHOLE_FILE assignment covers every hunk even when its
            # hunk_indices list was left empty, so expand from the diff.
            if assignment.assignment_type is AssignmentType.WHOLE_FILE:
                covered = set(range(counts.get(assignment.file_path, 0)))
                covered.update(assignment.hunk_indices)
            else:
                covered = set(assignment.hunk_indices)
            ancestor_hunks.setdefault(assignment.file_path, set()).update(covered)

    merged = []
    own_files = set[str]()
    for assignment in group.assignments:
        own_files.add(assignment.file_path)
        extra = ancestor_hunks.get(assignment.file_path)
        if assignment.assignment_type is AssignmentType.PARTIAL_HUNKS and extra:
            merged.append(
                assignment.model_copy(
                    update={"hunk_indices": sorted(set(assignment.hunk_indices) | extra)}
                )
            )
        else:
            merged.append(assignment)

    if carry_ancestor_files:
        for file_path, covered in ancestor_hunks.items():
            if file_path not in own_files:
                merged.append(
                    GroupAssignment(
                        file_path=file_path,
                        assignment_type=AssignmentType.PARTIAL_HUNKS,
                        hunk_indices=sorted(covered),
                    )
                )
    return group.model_copy(update={"assignments": merged})


def _get_base_file_content(file_path: str, ref: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{file_path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitOperationError(result.stderr.strip())
    return result.stdout


NO_NEWLINE_MARKER = "\\"


def _hunk_target_lines(hunk: Hunk) -> list[str]:
    """Return the post-image lines of a hunk with their exact line endings.

    unidiff always attaches a newline to a line's value, and reports a
    missing trailing newline as a separate ``\\ No newline at end of file``
    marker following that line. Honour the marker so a file that ends
    without a newline is reconstructed byte-for-byte.
    """
    target: list[str] = []
    last_was_target = False
    for line in hunk:
        if line.is_added or line.is_context:
            value = line.value
            target.append(value if value.endswith("\n") else value + "\n")
            last_was_target = True
        elif line.line_type == NO_NEWLINE_MARKER:
            if last_was_target:
                target[-1] = target[-1].rstrip("\n")
            last_was_target = False
        else:
            last_was_target = False
    return target


def split_git_lines(content: str) -> list[str]:
    """Split file content into lines the way git counts them: on ``\\n`` only.

    ``str.splitlines`` also breaks on form feed, vertical tab, ``\\x1c``-``\\x1e``,
    ``\\x85``, ``\\u2028`` and ``\\u2029``, none of which git treats as a line
    break, so every hunk after such a character would land at the wrong offset.
    """
    if not content:
        return []
    parts = content.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def apply_hunks(base_content: str, patch_file: PatchedFile, assigned_indices: list[int]) -> str:
    lines = split_git_lines(base_content)
    sorted_indices = sorted(assigned_indices, reverse=True)
    for idx in sorted_indices:
        hunk = patch_file[idx]
        start = hunk.source_start - 1
        end = start + hunk.source_length
        lines[start:end] = _hunk_target_lines(hunk)
    return "".join(lines)


def _assigned_hunk_indices(
    patch_file: PatchedFile, assignments: list[GroupAssignment]
) -> list[int]:
    """Union of the hunks every assignment for one file claims."""
    covered: set[int] = set()
    for assignment in assignments:
        if assignment.assignment_type is AssignmentType.WHOLE_FILE:
            covered.update(range(len(patch_file)))
        else:
            covered.update(assignment.hunk_indices)
    return sorted(covered)


def materialize_group_files(
    parsed_diff: ParsedDiff, group: Group, ref: str
) -> dict[str, str | None]:
    pf_map = {pf.path: pf for pf in parsed_diff.patch_set}
    # Several assignments may name the same file (e.g. merged across diff
    # chunks); each file is written once from the union of their hunks.
    assignments_by_path: dict[str, list[GroupAssignment]] = {}
    for assignment in group.assignments:
        assignments_by_path.setdefault(assignment.file_path, []).append(assignment)
    logger.info(logs.MATERIALIZING_FILES.format(count=len(assignments_by_path), group=group.id))
    result: dict[str, str | None] = {}
    for file_path, assignments in assignments_by_path.items():
        patch_file = pf_map.get(file_path)
        if patch_file is None:
            continue
        if patch_file.is_removed_file:
            result[file_path] = None
            continue
        indices = _assigned_hunk_indices(patch_file, assignments)
        if patch_file.is_added_file:
            result[file_path] = "".join(
                "".join(_hunk_target_lines(patch_file[idx])) for idx in indices
            )
            continue
        base_content = _get_base_file_content(file_path, ref)
        result[file_path] = apply_hunks(base_content, patch_file, indices)
    return result
