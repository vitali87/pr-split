from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import cached_property

from loguru import logger
from unidiff import PatchSet

from .. import logs
from ..exceptions import DiffParseError, GitOperationError
from ..types_defs import DiffStats, FileSummary, HunkInfo


def extract_diff(dev_branch: str, base_branch: str) -> str:
    logger.info(logs.EXTRACTING_DIFF.format(base=base_branch, dev=dev_branch))
    result = subprocess.run(
        ["git", "diff", f"{base_branch}...{dev_branch}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitOperationError(result.stderr.strip())
    return result.stdout


def parse_diff(raw_diff: str) -> ParsedDiff:
    try:
        patch_set = PatchSet(raw_diff)
    except Exception as exc:
        raise DiffParseError(str(exc)) from exc
    return ParsedDiff(patch_set=patch_set, raw_diff=raw_diff)


@dataclass
class ParsedDiff:
    patch_set: PatchSet
    raw_diff: str

    @property
    def file_paths(self) -> list[str]:
        return [pf.path for pf in self.patch_set]

    @cached_property
    def stats(self) -> DiffStats:
        file_summaries: list[FileSummary] = []
        total_added = 0
        total_removed = 0
        for pf in self.patch_set:
            added = pf.added
            removed = pf.removed
            total_added += added
            total_removed += removed
            file_summaries.append(
                FileSummary(
                    path=pf.path,
                    added=added,
                    removed=removed,
                    is_new=pf.is_added_file,
                    is_deleted=pf.is_removed_file,
                    is_renamed=pf.is_rename,
                    hunk_count=len(pf),
                )
            )
        return DiffStats(
            total_files=len(self.patch_set),
            total_added=total_added,
            total_removed=total_removed,
            total_loc=total_added + total_removed,
            file_summaries=file_summaries,
        )

    def hunks_for_file(self, path: str) -> list[HunkInfo]:
        for pf in self.patch_set:
            if pf.path == path:
                return [
                    HunkInfo(
                        index=i,
                        source_start=hunk.source_start,
                        source_length=hunk.source_length,
                        target_start=hunk.target_start,
                        target_length=hunk.target_length,
                        added_lines=hunk.added,
                        removed_lines=hunk.removed,
                    )
                    for i, hunk in enumerate(pf)
                ]
        return []

    @property
    def labeled_diff(self) -> str:
        parts: list[str] = []
        for pf in self.patch_set:
            header = f"--- {pf.source_file}\n+++ {pf.target_file}\n"
            labeled_hunks = [f"[hunk_index={i}]\n{hunk}" for i, hunk in enumerate(pf)]
            parts.append(header + "".join(labeled_hunks))
        return "\n".join(parts)

    def hunk_content(self, path: str, index: int) -> str:
        for pf in self.patch_set:
            if pf.path == path:
                return str(pf[index])
        return ""
