from __future__ import annotations

from ..constants import Priority
from ..types_defs import DiffStats, LocBoundViolation

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
that can be submitted as dependency-ordered pull requests.

Rules:
1. Every hunk in the diff MUST be assigned to exactly one group. No hunk may be \
left unassigned and no hunk may appear in multiple groups.
2. Groups MUST form a directed acyclic graph (DAG) via their depends_on fields. \
No cycles are allowed.
3. Each group should stay within approximately {max_loc} lines of code \
(added + removed). Exceeding this is acceptable only when a logical unit cannot \
be split further.{min_loc_rule}
4. Use the propose_split_plan tool to return your plan.
5. PR titles MUST follow conventional commits format: \
type(optional-scope): description. Allowed types: feat, fix, refactor, test, \
docs, chore, style, perf, ci, build, revert.
6. Hunk indices are PER-FILE and 0-based. Each hunk in the diff is labeled \
with [hunk_index=N]. Use exactly those values. A file with 3 hunks has \
indices 0, 1, 2. Do NOT use global sequential numbers across files.
7. For whole_file assignments, set assignment_type to "whole_file" and \
hunk_indices to a list of ALL hunk indices for that file.
8. For partial file assignments, set assignment_type to "partial_hunks" and \
list only the specific hunk indices.
9. estimated_loc should reflect the sum of added + removed lines for the \
assigned hunks.
10. Only assign hunks that appear in the diff provided. Do NOT assign hunks \
for files not present in the diff.

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
Below is chunk 1 of {total_chunks} from a large diff. You will receive the \
remaining {remaining_chunks} chunk(s) one at a time after this. Create broad, \
coarse groups that future hunks can be assigned to. Prefer fewer, larger groups \
over many small ones since more hunks from the same features/modules will arrive \
in later chunks. Only assign hunks you can see in the diff below.

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

IMPORTANT: Strongly prefer assigning hunks to existing groups listed above. \
A hunk belongs to an existing group if it touches the same feature, module, \
or concern. Only create a new group when a hunk clearly does not fit ANY \
existing group. When assigning to an existing group, reuse its exact ID. \
When creating new groups, use new IDs that do not conflict with existing ones. \
Only return groups that received assignments from THIS chunk (do not repeat \
groups with no new assignments).

File summary (this chunk only):
{file_summary}

Diff (this chunk only):
{chunk_diff}\
"""


_REFINEMENT_USER_PROMPT_TEMPLATE = """\
Your previous split plan has LOC bound violations that need to be fixed.

Violations:
{violations}

Current plan:
{current_plan}

Full diff:
{full_diff}

Revise the plan to fix these violations by merging undersized groups or \
redistributing hunks from oversized groups. Return a complete revised plan \
covering ALL hunks. Keep groups that already satisfy the bounds unchanged \
where possible.\
"""


def _format_violations(violations: list[LocBoundViolation]) -> str:
    lines: list[str] = []
    for v in violations:
        direction = "below minimum" if v.violation_type == "below_min" else "above maximum"
        lines.append(
            f"  - {v.group_id}: {v.estimated_loc} LOC "
            f"(+{v.estimated_added}/-{v.estimated_removed}), {direction} {v.limit}"
        )
    return "\n".join(lines)


def _format_current_plan(groups: list[dict[str, object]]) -> str:
    import json

    return json.dumps(groups, indent=2)


def _format_file_summary(diff_stats: DiffStats) -> str:
    lines: list[str] = []
    for fs in diff_stats["file_summaries"]:
        flags = [
            f
            for f, c in [
                ("new", fs["is_new"]),
                ("deleted", fs["is_deleted"]),
                ("renamed", fs["is_renamed"]),
            ]
            if c
        ]
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        hunk_count = fs["hunk_count"]
        idx_range = f"indices 0..{hunk_count - 1}" if hunk_count > 0 else "no hunks"
        lines.append(
            f"  {fs['path']}: +{fs['added']}/-{fs['removed']}"
            f" ({hunk_count} hunks, {idx_range}){flag_str}"
        )
    header = (
        f"Total: {diff_stats['total_files']} files, "
        f"+{diff_stats['total_added']}/-{diff_stats['total_removed']} "
        f"({diff_stats['total_loc']} LOC)"
    )
    return header + "\n" + "\n".join(lines)


_PRIORITY_MAP = {
    Priority.ORTHOGONAL: _PRIORITY_ORTHOGONAL,
    Priority.LOGICAL: _PRIORITY_LOGICAL,
}


_MIN_LOC_RULE = (
    " Groups should also not be smaller than about {min_loc} lines: merge "
    "closely related small changes rather than creating tiny groups, unless "
    "a change genuinely stands alone."
)


def build_system_prompt(priority: Priority, max_loc: int, min_loc: int | None = None) -> str:
    # --min-loc was documented as applying to every backend, but the LLM only
    # ever heard about it via the (off by default) refinement prompt.
    min_loc_rule = _MIN_LOC_RULE.format(min_loc=min_loc) if min_loc else ""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        max_loc=max_loc,
        min_loc_rule=min_loc_rule,
        priority_instructions=_PRIORITY_MAP[priority],
    )


def build_user_prompt(diff_stats: DiffStats, full_diff: str) -> str:
    return _USER_PROMPT_TEMPLATE.format(
        file_summary=_format_file_summary(diff_stats),
        full_diff=full_diff,
    )


def build_chunk_first_prompt(
    chunk_stats: DiffStats,
    chunk_diff: str,
    total_chunks: int,
) -> str:
    return _CHUNK_FIRST_USER_PROMPT_TEMPLATE.format(
        total_chunks=total_chunks,
        remaining_chunks=total_chunks - 1,
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


def build_refinement_prompt(
    violations: list[LocBoundViolation],
    current_groups: list[dict[str, object]],
    full_diff: str,
) -> str:
    return _REFINEMENT_USER_PROMPT_TEMPLATE.format(
        violations=_format_violations(violations),
        current_plan=_format_current_plan(current_groups),
        full_diff=full_diff,
    )
