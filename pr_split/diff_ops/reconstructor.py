from __future__ import annotations

import re
import subprocess

from loguru import logger
from unidiff import PatchedFile

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


def apply_hunks(base_content: str, patch_file: PatchedFile, assigned_indices: list[int]) -> str:
    lines = base_content.splitlines(keepends=True)
    sorted_indices = sorted(assigned_indices, reverse=True)
    for idx in sorted_indices:
        hunk = patch_file[idx]
        start = hunk.source_start - 1
        end = start + hunk.source_length
        target_lines = [str(line)[1:] for line in hunk if line.is_added or line.is_context]
        target_with_endings = [ln if ln.endswith("\n") else ln + "\n" for ln in target_lines]
        lines[start:end] = target_with_endings
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
        # unidiff's is_removed_file is a heuristic that is also true for a
        # file truncated to empty (single hunk with target length 0); only a
        # /dev/null target is a real deletion.
        if patch_file.target_file == "/dev/null":
            result[file_path] = None
            continue
        indices = _assigned_hunk_indices(patch_file, assignments)
        if patch_file.is_added_file:
            target_lines = []
            for idx in indices:
                hunk = patch_file[idx]
                for line in hunk:
                    if line.is_added or line.is_context:
                        target_lines.append(str(line)[1:])
            # Lines from unidiff keep their trailing newline, so they are
            # concatenated as-is; joining on "\n" double-spaces the file.
            result[file_path] = "".join(
                ln if ln.endswith("\n") else ln + "\n" for ln in target_lines
            )
            continue
        base_content = _get_base_file_content(file_path, ref)
        result[file_path] = apply_hunks(base_content, patch_file, indices)
    return result


_TARGET_MODE_RE = re.compile(r"^(?:new file mode|new mode) (\d{6})$", re.MULTILINE)


def target_file_modes(parsed_diff: ParsedDiff, group: Group) -> dict[str, int]:
    """Map each file the group materializes to the mode the diff gives it.

    unidiff parses ``old mode``/``new mode``/``new file mode`` headers into
    ``patch_info`` only; nothing else applies them, so without this a
    ``chmod +x`` that comes with a content change silently loses the bit.
    """
    wanted = {assignment.file_path for assignment in group.assignments}
    modes: dict[str, int] = {}
    for patch_file in parsed_diff.patch_set:
        if patch_file.path not in wanted or patch_file.is_removed_file:
            continue
        header = "".join(str(line) for line in patch_file.patch_info or [])
        match = _TARGET_MODE_RE.search(header)
        if not match:
            continue
        mode = int(match.group(1), 8)
        # Only regular files (100644 / 100755): a symlink's 120000 would mask
        # to 000 and make the written file unreadable. Symlinks are written as
        # plain files by the reconstructor, unchanged from before.
        if mode & 0o170000 == 0o100000:
            modes[patch_file.path] = mode
    return modes
