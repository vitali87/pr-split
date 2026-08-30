from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import PRRecord


class ErrorMsg(StrEnum):
    BRANCH_NOT_FOUND = "Branch '{branch}' does not exist"
    DIRTY_WORKTREE = "Working tree has uncommitted changes; commit or stash first"
    GH_AUTH_FAILED = "GitHub CLI authentication failed; run 'gh auth login'"
    CYCLE_DETECTED = "Dependency cycle detected in split plan"
    COVERAGE_GAP = "Hunk {file}[{index}] not assigned to any group"
    COVERAGE_OVERLAP = "Hunk {file}[{index}] assigned to multiple groups: {groups}"
    LOC_MISMATCH = "Total LOC {actual} does not match diff LOC {expected}"
    MERGE_CONFLICT = "Groups '{a}' and '{b}' modify overlapping regions in '{file}'"
    NO_PLAN = "No split plan found; run 'pr-split split' first"
    LLM_PARSE_ERROR = "Failed to parse LLM response: {detail}"
    LLM_OUTPUT_TRUNCATED = (
        "LLM response was cut off before the plan was complete ({detail});"
        " the partial plan cannot be trusted"
    )
    BRANCH_CREATE_FAILED = "Failed to create branch '{branch}': {detail}"
    PR_CREATE_FAILED = "Failed to create PR for group '{group}': {detail}"
    MERGE_FAILED = "Merge of '{source}' into '{target}' failed: {detail}"
    PR_NOT_FOUND = "PR #{number} not found or is not from a fork"
    PR_FETCH_FAILED = "Failed to fetch fork branch for PR #{number}: {detail}"
    FORK_FETCH_FAILED = "Failed to fetch {user}:{branch}: {detail}"
    HUNK_TOO_LARGE = "Hunk {file}[{index}] has ~{tokens} estimated tokens, exceeds budget {budget}"
    MIN_LOC_GE_MAX_LOC = "min_loc {min_loc} must be less than max_loc {max_loc}"
    LOC_BOUNDS_STRICT_FAILED = "Plan violates configured LOC bounds"

    def __call__(self, **kwargs: object) -> str:
        return self.value.format(**kwargs) if kwargs else self.value


class PRSplitError(Exception):
    pass


class DiffParseError(PRSplitError):
    pass


class PlanValidationError(PRSplitError):
    pass


class GitOperationError(PRSplitError):
    pass


class LLMError(PRSplitError):
    pass


class PRCreationError(PRSplitError):
    def __init__(self, message: str, pr_records: list[PRRecord]) -> None:
        super().__init__(message)
        self.pr_records = pr_records
