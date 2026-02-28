from __future__ import annotations

from typing import NamedTuple, TypedDict


class HunkInfo(NamedTuple):
    index: int
    source_start: int
    source_length: int
    target_start: int
    target_length: int
    added_lines: int
    removed_lines: int


class FileSummary(TypedDict):
    path: str
    added: int
    removed: int
    is_new: bool
    is_deleted: bool
    is_renamed: bool
    hunk_count: int


class ForkPRInfo(TypedDict):
    pr_number: int | None
    local_ref: str
    base_branch: str
    author: str
    fork_full_name: str


class HunkRef(NamedTuple):
    file_path: str
    hunk_index: int
    token_estimate: int


class DiffStats(TypedDict):
    total_files: int
    total_added: int
    total_removed: int
    total_loc: int
    file_summaries: list[FileSummary]
