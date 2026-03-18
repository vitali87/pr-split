from enum import StrEnum


class AssignmentType(StrEnum):
    WHOLE_FILE = "whole_file"
    PARTIAL_HUNKS = "partial_hunks"


class Priority(StrEnum):
    ORTHOGONAL = "orthogonal"
    LOGICAL = "logical"


class ChunkStrategy(StrEnum):
    GREEDY = "greedy"
    DYNAMIC_PROGRAMMING = "dynamic_programming"


class PartitionStrategy(StrEnum):
    LLM = "llm"
    GRAPH = "graph"
    CP_SAT = "cp_sat"


class PRState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


BRANCH_PREFIX = "pr-split/"
PLAN_DIR = ".pr-split"
PLAN_FILE = ".pr-split/plan.json"
DEFAULT_MAX_LOC = 400
DEFAULT_CP_SAT_TIMEOUT_SECONDS = 15.0
DEFAULT_CHUNK_STRATEGY = ChunkStrategy.DYNAMIC_PROGRAMMING
DEFAULT_PARTITION_STRATEGY = PartitionStrategy.LLM
PR_REF_PREFIX = "refs/pr-split/pr-"
FORK_REF_PREFIX = "refs/pr-split/fork-"
DEFAULT_MODEL = "claude-opus-4-6"
OPENAI_MODEL = "gpt-5.2"
ANTHROPIC_MAX_CONTEXT_TOKENS = 1_000_000
OPENAI_MAX_CONTEXT_TOKENS = 400_000
MAX_OUTPUT_TOKENS = 128_000
CHUNK_TARGET_RATIO = 2 / 3
CHUNK_RETRY_LIMIT = 2
