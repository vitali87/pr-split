from enum import StrEnum


class AssignmentType(StrEnum):
    WHOLE_FILE = "whole_file"
    PARTIAL_HUNKS = "partial_hunks"


class Priority(StrEnum):
    ORTHOGONAL = "orthogonal"
    LOGICAL = "logical"


class PRState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


BRANCH_PREFIX = "pr-split/"
MERGE_BASE_PREFIX = "pr-split/base-"
PLAN_DIR = ".pr-split"
PLAN_FILE = ".pr-split/plan.json"
DEFAULT_MAX_LOC = 400
PR_REF_PREFIX = "refs/pr-split/pr-"
FORK_REF_PREFIX = "refs/pr-split/fork-"
CLAUDE_MODEL = "claude-opus-4-6"
CONTEXT_1M_BETA = "context-1m-2025-08-07"
MAX_CONTEXT_TOKENS = 1_000_000
MAX_OUTPUT_TOKENS = 128_000
LLM_TIMEOUT_SECONDS = 600
CHUNK_TARGET_RATIO = 2 / 3
CHUNK_RETRY_LIMIT = 2
