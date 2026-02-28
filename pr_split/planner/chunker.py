from __future__ import annotations

from collections import defaultdict

from ..diff_ops import ParsedDiff
from ..exceptions import ErrorMsg
from ..schemas import Group
from ..types_defs import DiffStats, FileSummary, HunkRef

DEFAULT_TOKEN_RATIO = 0.25


def build_hunk_sequence(
    parsed_diff: ParsedDiff, token_ratio: float = DEFAULT_TOKEN_RATIO
) -> list[HunkRef]:
    sequence: list[HunkRef] = []
    for pf in parsed_diff.patch_set:
        for i, hunk in enumerate(pf):
            token_estimate = max(1, int(len(str(hunk)) * token_ratio))
            sequence.append(
                HunkRef(file_path=pf.path, hunk_index=i, token_estimate=token_estimate)
            )
    return sequence


def chunk_hunks(hunk_sequence: list[HunkRef], token_budget: int) -> list[list[HunkRef]]:
    chunks: list[list[HunkRef]] = []
    current: list[HunkRef] = []
    current_tokens = 0

    for href in hunk_sequence:
        if href.token_estimate > token_budget:
            raise ValueError(
                ErrorMsg.HUNK_TOO_LARGE(
                    file=href.file_path,
                    index=href.hunk_index,
                    tokens=href.token_estimate,
                    budget=token_budget,
                )
            )
        if current and current_tokens + href.token_estimate > token_budget:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(href)
        current_tokens += href.token_estimate

    if current:
        chunks.append(current)
    return chunks


def build_chunk_diff_from_hunks(parsed_diff: ParsedDiff, hunk_refs: list[HunkRef]) -> str:
    file_hunks: dict[str, list[int]] = defaultdict(list)
    for href in hunk_refs:
        file_hunks[href.file_path].append(href.hunk_index)

    parts: list[str] = []
    for pf in parsed_diff.patch_set:
        if pf.path not in file_hunks:
            continue
        indices = file_hunks[pf.path]
        header = f"--- {pf.source_file}\n+++ {pf.target_file}\n"
        hunk_texts = [str(pf[i]) for i in sorted(indices)]
        parts.append(header + "".join(hunk_texts))
    return "\n".join(parts)


def build_chunk_stats_from_hunks(parsed_diff: ParsedDiff, hunk_refs: list[HunkRef]) -> DiffStats:
    file_hunks: dict[str, list[int]] = defaultdict(list)
    for href in hunk_refs:
        file_hunks[href.file_path].append(href.hunk_index)

    file_summaries: list[FileSummary] = []
    total_added = 0
    total_removed = 0
    for pf in parsed_diff.patch_set:
        if pf.path not in file_hunks:
            continue
        indices = file_hunks[pf.path]
        added = sum(pf[i].added for i in indices)
        removed = sum(pf[i].removed for i in indices)
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
                hunk_count=len(indices),
            )
        )
    return DiffStats(
        total_files=len(file_summaries),
        total_added=total_added,
        total_removed=total_removed,
        total_loc=total_added + total_removed,
        file_summaries=file_summaries,
    )


def recompute_estimated_loc(groups: list[Group], parsed_diff: ParsedDiff) -> list[Group]:
    for group in groups:
        loc = 0
        for assignment in group.assignments:
            for idx in assignment.hunk_indices:
                for pf in parsed_diff.patch_set:
                    if pf.path == assignment.file_path:
                        hunk = pf[idx]
                        loc += hunk.added + hunk.removed
                        break
        group.estimated_loc = loc
    return groups


def format_group_catalog(groups: list[Group]) -> str:
    lines: list[str] = []
    for group in groups:
        file_paths = sorted({a.file_path for a in group.assignments})
        deps = f" (depends on: {', '.join(group.depends_on)})" if group.depends_on else ""
        lines.append(f"Group {group.id}: {group.title}{deps}")
        lines.append(f"  Description: {group.description}")
        lines.append(f"  Files: {', '.join(file_paths)}")
        lines.append("")
    return "\n".join(lines)
