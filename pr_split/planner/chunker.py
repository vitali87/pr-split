from __future__ import annotations

from collections import defaultdict

from loguru import logger

from .. import logs
from ..constants import AssignmentType
from ..diff_ops import ParsedDiff
from ..exceptions import ErrorMsg
from ..schemas import Group, GroupAssignment
from ..types_defs import DiffStats, FileSummary, HunkRef

_DEFAULT_TOKEN_RATIO = 0.25


def build_hunk_sequence(
    parsed_diff: ParsedDiff, token_ratio: float = _DEFAULT_TOKEN_RATIO
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
        indices = sorted(file_hunks[pf.path])
        header = f"--- {pf.source_file}\n+++ {pf.target_file}\n"
        labeled_hunks = [f"[hunk_index={i}]\n{pf[i]}" for i in indices]
        parts.append(header + "".join(labeled_hunks))
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


def recompute_estimated_loc(groups: list[Group], parsed_diff: ParsedDiff) -> None:
    pf_map = {pf.path: pf for pf in parsed_diff.patch_set}
    file_hunk_counts = {path: len(pf) for path, pf in pf_map.items()}

    for group in groups:
        added = 0
        removed = 0
        for assignment in group.assignments:
            max_idx = file_hunk_counts.get(assignment.file_path, 0)
            for idx in assignment.hunk_indices:
                if idx >= max_idx:
                    logger.warning(
                        logs.INVALID_HUNK_INDEX.format(
                            group=group.id,
                            file=assignment.file_path,
                            index=idx,
                            max=max_idx - 1,
                        )
                    )
                    continue
                if pf := pf_map.get(assignment.file_path):
                    added += pf[idx].added
                    removed += pf[idx].removed
        group.estimated_added = added
        group.estimated_removed = removed
        group.estimated_loc = added + removed


def assign_uncovered_hunks(groups: list[Group], parsed_diff: ParsedDiff) -> int:
    assigned = {
        (a.file_path, idx) for g in groups for a in g.assignments for idx in a.hunk_indices
    }

    all_hunks = [(pf.path, i) for pf in parsed_diff.patch_set for i in range(len(pf))]

    unassigned = [h for h in all_hunks if h not in assigned]
    if not unassigned:
        return 0

    file_groups: dict[str, Group] = {}
    for group in groups:
        for assignment in group.assignments:
            if assignment.file_path not in file_groups:
                file_groups[assignment.file_path] = group

    largest = max(groups, key=lambda g: len(g.assignments))

    for file_path, hunk_idx in unassigned:
        target = file_groups.get(file_path, largest)
        existing_assignment = next(
            (a for a in target.assignments if a.file_path == file_path), None
        )
        if existing_assignment:
            existing_assignment.hunk_indices.append(hunk_idx)
        else:
            target.assignments.append(
                GroupAssignment(
                    file_path=file_path,
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[hunk_idx],
                )
            )
        logger.warning(
            logs.HUNK_AUTO_ASSIGNED.format(file=file_path, index=hunk_idx, group=target.id)
        )

    return len(unassigned)


def format_group_catalog(groups: list[Group]) -> str:
    lines: list[str] = []
    for group in groups:
        file_paths = sorted({a.file_path for a in group.assignments})
        deps = f" (depends on: {', '.join(group.depends_on)})" if group.depends_on else ""
        lines.append(f"Group {group.id}: {group.title}{deps}")
        lines.append(f"  Description: {group.description}")
        lines.append(f"  Files ({len(file_paths)}): {', '.join(file_paths)}")
        lines.append(f"  Assignments: {len(group.assignments)}, ~{group.estimated_loc} LOC")
        lines.append("")
    return "\n".join(lines)
