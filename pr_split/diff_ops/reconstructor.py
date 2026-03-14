from __future__ import annotations

import subprocess

from loguru import logger
from unidiff import PatchedFile

from .. import logs
from ..constants import AssignmentType
from ..exceptions import GitOperationError
from ..schemas import Group
from .parser import ParsedDiff


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


def materialize_group_files(parsed_diff: ParsedDiff, group: Group, ref: str) -> dict[str, str]:
    logger.info(logs.MATERIALIZING_FILES.format(count=len(group.assignments), group=group.id))
    pf_map = {pf.path: pf for pf in parsed_diff.patch_set}
    result: dict[str, str] = {}
    for assignment in group.assignments:
        patch_file = pf_map.get(assignment.file_path)
        if patch_file is None:
            continue
        if patch_file.is_added_file:
            all_indices = list(range(len(patch_file)))
            match assignment.assignment_type:
                case AssignmentType.WHOLE_FILE:
                    indices = all_indices
                case AssignmentType.PARTIAL_HUNKS:
                    indices = assignment.hunk_indices
            target_lines = []
            for idx in indices:
                hunk = patch_file[idx]
                for line in hunk:
                    if line.is_added or line.is_context:
                        target_lines.append(str(line)[1:])
            result[assignment.file_path] = "\n".join(target_lines)
            if target_lines:
                result[assignment.file_path] += "\n"
            continue
        base_content = _get_base_file_content(assignment.file_path, ref)
        match assignment.assignment_type:
            case AssignmentType.WHOLE_FILE:
                indices = list(range(len(patch_file)))
            case AssignmentType.PARTIAL_HUNKS:
                indices = assignment.hunk_indices
        result[assignment.file_path] = apply_hunks(base_content, patch_file, indices)
    return result
