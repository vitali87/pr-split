from __future__ import annotations

from ..constants import Priority
from ..types_defs import DiffStats

SPLIT_TOOL_NAME = "propose_split_plan"

SPLIT_TOOL_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["groups"],
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "title",
                    "description",
                    "depends_on",
                    "assignments",
                    "estimated_loc",
                ],
                "properties": {
                    "id": {"type": "string", "description": "Unique group ID, e.g. pr-1, pr-2"},
                    "title": {
                        "type": "string",
                        "description": "PR title in conventional commits format",
                    },
                    "description": {
                        "type": "string",
                        "description": "What this group accomplishes",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of groups this depends on",
                    },
                    "assignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["file_path", "assignment_type", "hunk_indices"],
                            "properties": {
                                "file_path": {"type": "string"},
                                "assignment_type": {
                                    "type": "string",
                                    "enum": ["whole_file", "partial_hunks"],
                                },
                                "hunk_indices": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                            },
                        },
                    },
                    "estimated_loc": {
                        "type": "integer",
                        "description": "Estimated lines of code (added + removed)",
                    },
                },
            },
        },
    },
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior software engineer specializing in pull request decomposition.

Your task: given a unified diff, split it into a set of small, reviewable groups \
that can be submitted as stacked pull requests.

Rules:
1. Every hunk in the diff MUST be assigned to exactly one group. No hunk may be \
left unassigned and no hunk may appear in multiple groups.
2. Groups MUST form a directed acyclic graph (DAG) via their depends_on fields. \
No cycles are allowed.
3. Each group should stay within approximately {max_loc} lines of code \
(added + removed). Exceeding this is acceptable only when a logical unit cannot \
be split further.
4. Use the propose_split_plan tool to return your plan.
5. PR titles MUST follow conventional commits format: \
type(optional-scope): description. Allowed types: feat, fix, refactor, test, \
docs, chore, style, perf, ci, build, revert.
6. For whole_file assignments, set assignment_type to "whole_file" and \
hunk_indices to a list of ALL hunk indices for that file.
7. For partial file assignments, set assignment_type to "partial_hunks" and \
list only the specific hunk indices.
8. estimated_loc should reflect the sum of added + removed lines for the \
assigned hunks.

{priority_instructions}
"""

_PRIORITY_ORTHOGONAL = """\
Priority mode: ORTHOGONAL
Maximize independence between groups. Prefer groups that touch disjoint sets of \
files so they can be reviewed and merged in parallel. Only add dependencies when \
hunks within the same file force an ordering.\
"""

_PRIORITY_LOGICAL = """\
Priority mode: LOGICAL
Group changes by feature or logical concern. Hunks that implement the same \
feature, fix the same bug, or refactor the same component should be in the same \
group, even if they touch multiple files. Dependencies should reflect the natural \
build order of the feature.\
"""

_USER_PROMPT_TEMPLATE = """\
Below is the diff to split.

File summary:
{file_summary}

Full diff:
{full_diff}\
"""

_CHUNK_FIRST_USER_PROMPT_TEMPLATE = """\
Below is chunk 1 of {total_chunks} from a large diff. This is the first chunk; \
create initial groups and assign all hunks in this chunk.

File summary (this chunk only):
{file_summary}

Diff (this chunk only):
{chunk_diff}\
"""

_CHUNK_CONTINUATION_USER_PROMPT_TEMPLATE = """\
Below is chunk {chunk_index} of {total_chunks} from a large diff. \
Previous chunks have already been assigned to groups.

Existing groups from previous chunks:
{group_catalog}

Assign the hunks below to existing groups or create new groups as needed. \
When assigning to an existing group, reuse its exact ID. When creating new \
groups, use new IDs that do not conflict with existing ones. Only return groups \
that received assignments from THIS chunk (do not repeat groups with no new \
assignments).

File summary (this chunk only):
{file_summary}

Diff (this chunk only):
{chunk_diff}\
"""


def _format_file_summary(diff_stats: DiffStats) -> str:
    lines: list[str] = []
    for fs in diff_stats["file_summaries"]:
        flags: list[str] = []
        if fs["is_new"]:
            flags.append("new")
        if fs["is_deleted"]:
            flags.append("deleted")
        if fs["is_renamed"]:
            flags.append("renamed")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"  {fs['path']}: +{fs['added']}/-{fs['removed']} ({fs['hunk_count']} hunks){flag_str}"
        )
    header = (
        f"Total: {diff_stats['total_files']} files, "
        f"+{diff_stats['total_added']}/-{diff_stats['total_removed']} "
        f"({diff_stats['total_loc']} LOC)"
    )
    return header + "\n" + "\n".join(lines)


def build_system_prompt(priority: Priority, max_loc: int) -> str:
    match priority:
        case Priority.ORTHOGONAL:
            priority_instructions = _PRIORITY_ORTHOGONAL
        case Priority.LOGICAL:
            priority_instructions = _PRIORITY_LOGICAL
    return _SYSTEM_PROMPT_TEMPLATE.format(
        max_loc=max_loc,
        priority_instructions=priority_instructions,
    )


def build_user_prompt(diff_stats: DiffStats, full_diff: str) -> str:
    return _USER_PROMPT_TEMPLATE.format(
        file_summary=_format_file_summary(diff_stats),
        full_diff=full_diff,
    )


def build_chunk_first_prompt(chunk_stats: DiffStats, chunk_diff: str, total_chunks: int) -> str:
    return _CHUNK_FIRST_USER_PROMPT_TEMPLATE.format(
        total_chunks=total_chunks,
        file_summary=_format_file_summary(chunk_stats),
        chunk_diff=chunk_diff,
    )


def build_chunk_continuation_prompt(
    chunk_stats: DiffStats,
    chunk_diff: str,
    chunk_index: int,
    total_chunks: int,
    group_catalog: str,
) -> str:
    return _CHUNK_CONTINUATION_USER_PROMPT_TEMPLATE.format(
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        group_catalog=group_catalog,
        file_summary=_format_file_summary(chunk_stats),
        chunk_diff=chunk_diff,
    )
