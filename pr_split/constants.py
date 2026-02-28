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
CLAUDE_MODEL = "claude-sonnet-4-6"
CONTEXT_1M_BETA = "context-1m-2025-08-07"
