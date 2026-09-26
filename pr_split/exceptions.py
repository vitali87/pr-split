from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import PRRecord


class ErrorMsg(StrEnum):
    BRANCH_NOT_FOUND = "Branch '{branch}' does not exist"
    BASE_NOT_A_LOCAL_BRANCH = (
        "Base '{base}' is not a local branch; sub-PRs are opened against it on GitHub, "
        "so pass the branch name{suggestion}"
    )
    DIRTY_WORKTREE = "Working tree has uncommitted changes; commit or stash first"
    GH_AUTH_FAILED = "GitHub CLI authentication failed; run 'gh auth login'"
    CYCLE_DETECTED = "Dependency cycle detected in split plan"
    COVERAGE_GAP = "Hunk {file}[{index}] not assigned to any group"
    COVERAGE_OVERLAP = "Hunk {file}[{index}] assigned to multiple groups: {groups}"
    UNKNOWN_HUNK = "Hunk {file}[{index}] assigned to group '{group}' does not exist in the diff"
    UNKNOWN_FILE = "File '{file}' assigned to group '{group}' does not exist in the diff"
    UNKNOWN_DEPENDENCY = "Group '{group}' depends on unknown group '{dep}'"
    DUPLICATE_GROUP_ID = "Group id '{group}' is used more than once"
    NEW_FILE_PIECE_ORDER = (
        "Group '{group}' holds piece {piece} of new file '{file}' but does not build on"
        " group '{parent}', which holds piece {previous}; its sub-PR would create the file"
        " without its beginning"
    )
    LOC_MISMATCH = "Total LOC {actual} does not match diff LOC {expected}"
    MERGE_CONFLICT = "Groups '{a}' and '{b}' modify overlapping regions in '{file}'"
    NO_PLAN = "No split plan found; run 'pr-split split' first"
    PLAN_LOAD_FAILED = (
        "Cannot load split plan from '{path}': {detail}; delete it and run 'pr-split split' again"
    )
    LLM_PARSE_ERROR = "Failed to parse LLM response: {detail}"
    LOCAL_MODEL_REQUIRED = (
        "PR_SPLIT_MODEL must be set when provider is 'local'"
        " (the model name your local server serves, e.g. 'qwen2.5-coder:14b')"
    )
    LOCAL_OUTPUT_EXCEEDS_CONTEXT = (
        "PR_SPLIT_LOCAL_MAX_OUTPUT_TOKENS ({output}) must be less than"
        " PR_SPLIT_LOCAL_CONTEXT_TOKENS ({context})"
    )
    NO_DIFF_BUDGET = (
        "No room for the diff in a chunk: {budget} tokens left after the output budget"
        " ({output}) and prompt overhead ({overhead}) in a {context}-token window; raise"
        " PR_SPLIT_LOCAL_CONTEXT_TOKENS or lower PR_SPLIT_LOCAL_MAX_OUTPUT_TOKENS"
    )
    LOCAL_SERVER_UNREACHABLE = (
        "Cannot reach the local LLM server at {url}: {detail};"
        " start it (e.g. 'ollama serve') or set PR_SPLIT_LOCAL_BASE_URL"
    )
    LLM_OUTPUT_TRUNCATED = (
        "LLM response was cut off before the plan was complete ({detail});"
        " the partial plan cannot be trusted"
    )
    RESTACK_NOT_STACKED = "restack only applies to a plan split with --stack"
    RESTACK_NO_BRANCHES = "No branches recorded for this plan; run 'pr-split execute' first"
    RESTACK_CHECKED_OUT = (
        "Cannot restack branches checked out in a worktree: {branches};"
        " switch to another branch first"
    )
    RESTACK_DIVERGED = (
        "Branch '{branch}' and '{remote}/{branch}' both have commits the other lacks;"
        " reconcile them, then run 'pr-split restack' again"
    )
    RESTACK_BASE_FETCH_FAILED = "Cannot fetch base branch '{base}' to restack onto it ({detail})"
    RESTACK_CONFLICT = (
        "Rebasing '{branch}' onto '{parent}' conflicts; every local branch was left as it was."
        " Rebase it by hand, push it, then run 'pr-split restack' again ({detail})"
    )
    PER_GROUP_STEP_FAILED = "Per-group step failed for group '{group}' (exit {code}): {output}"
    CONFIG_INVALID = "Cannot read '{path}': {detail}"
    MOVE_NO_DIFF = "The saved plan has no diff; re-run 'pr-split split'"
    MOVE_UNKNOWN_LAYER = "Group '{group}' is not an executed layer of this plan"
    MOVE_NOT_UPWARD = (
        "Group '{target}' does not build on '{source}'; a hunk can only move up the stack"
    )
    MOVE_NOT_A_CHAIN = (
        "Group '{group}' has several parents, so the stack between the layers is not a chain"
    )
    MOVE_UNKNOWN_HUNK = "The plan's diff has no hunk {file}:{index}"
    MOVE_UNSUPPORTED_FILE = (
        "Moving a hunk of '{file}' is not supported: deleted files and pieces of a split"
        " new file move with their whole file only"
    )
    MOVE_HUNK_NOT_IN_LAYER = "Group '{group}' does not hold hunk {file}:{index}"
    MOVE_HUNK_DOES_NOT_APPLY = (
        "The hunk no longer applies to '{branch}' ({detail}); nothing on that branch was changed"
    )
    RECOVER_PLAN_EXISTS = "A split plan already exists at '{path}'; pass --force to replace it"
    RECOVER_NOTHING_FOUND = (
        "No branch or PR under '{prefix}' was found; pass the dev branch the stack was split from"
    )
    RECOVER_BASE_UNKNOWN = (
        "Cannot tell which base branch the stack under '{prefix}' targets ({bases}); pass --base"
    )
    RECOVER_PR_LIST_FAILED = "Cannot list PRs to rebuild the plan ({detail})"
    BRANCH_CREATE_FAILED = "Failed to create branch '{branch}': {detail}"
    PR_CREATE_FAILED = "Failed to create PR for group '{group}': {detail}"
    MERGE_FAILED = "Merge of '{source}' into '{target}' failed: {detail}"
    PR_NOT_FOUND = "PR #{number} not found"
    PR_RESPONSE_INVALID = "Unexpected response from GitHub for PR #{number}: {detail}"
    PR_NOT_FROM_FORK = (
        "PR #{number} is not from a fork; pass its head branch name instead of the PR number"
    )
    PR_FETCH_FAILED = "Failed to fetch fork branch for PR #{number}: {detail}"
    FORK_FETCH_FAILED = "Failed to fetch {user}:{branch}: {detail}"
    HUNK_TOO_LARGE = "Hunk {file}[{index}] has ~{tokens} estimated tokens, exceeds budget {budget}"
    MIN_LOC_GE_MAX_LOC = "min_loc {min_loc} must be less than max_loc {max_loc}"
    LOC_BOUNDS_STRICT_FAILED = "Plan violates configured LOC bounds"
    BINARY_FILES_UNSUPPORTED = (
        "Diff contains binary files, which cannot be split into hunks: {files}."
        " Commit them separately and re-run"
    )
    GH_STACK_MISSING = (
        "The gh-stack extension is required for stacked PRs;"
        " run 'gh extension install github/gh-stack'"
    )
    STACK_LINK_FAILED = "Failed to link stack for PRs {prs}: {detail}"

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
