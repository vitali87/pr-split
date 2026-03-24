from __future__ import annotations

from pr_split.constants import (
    BRANCH_PREFIX,
    CHUNK_RETRY_LIMIT,
    CHUNK_TARGET_RATIO,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_MAX_LOC,
    DEFAULT_MIN_LOC,
    DEFAULT_PARTITION_STRATEGY,
    DEFAULT_STRICT_LOC_BOUNDS,
    FORK_REF_PREFIX,
    MAX_OUTPUT_TOKENS,
    PLAN_DIR,
    PLAN_FILE,
    PR_REF_PREFIX,
    AssignmentType,
    ChunkStrategy,
    LocViolationType,
    PartitionStrategy,
    Priority,
    Provider,
    PRState,
)


class TestAssignmentType:
    def test_values(self) -> None:
        assert AssignmentType.WHOLE_FILE == "whole_file"
        assert AssignmentType.PARTIAL_HUNKS == "partial_hunks"

    def test_is_str(self) -> None:
        assert isinstance(AssignmentType.WHOLE_FILE, str)


class TestPriority:
    def test_values(self) -> None:
        assert Priority.ORTHOGONAL == "orthogonal"
        assert Priority.LOGICAL == "logical"


class TestChunkStrategy:
    def test_values(self) -> None:
        assert ChunkStrategy.GREEDY == "greedy"
        assert ChunkStrategy.DYNAMIC_PROGRAMMING == "dynamic_programming"


class TestPartitionStrategy:
    def test_values(self) -> None:
        assert PartitionStrategy.LLM == "llm"
        assert PartitionStrategy.GRAPH == "graph"
        assert PartitionStrategy.CP_SAT == "cp_sat"


class TestLocViolationType:
    def test_values(self) -> None:
        assert LocViolationType.BELOW_MIN == "below_min"
        assert LocViolationType.ABOVE_MAX == "above_max"


class TestPRState:
    def test_values(self) -> None:
        assert PRState.OPEN == "open"
        assert PRState.CLOSED == "closed"
        assert PRState.MERGED == "merged"


class TestProvider:
    def test_values(self) -> None:
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OPENAI == "openai"


class TestConstants:
    def test_branch_prefix(self) -> None:
        assert BRANCH_PREFIX == "pr-split/"

    def test_plan_dir_and_file(self) -> None:
        assert PLAN_DIR == ".pr-split"
        assert PLAN_FILE == ".pr-split/plan.json"

    def test_default_max_loc(self) -> None:
        assert DEFAULT_MAX_LOC == 400

    def test_default_min_loc(self) -> None:
        assert DEFAULT_MIN_LOC is None

    def test_default_strict_loc_bounds(self) -> None:
        assert DEFAULT_STRICT_LOC_BOUNDS is False

    def test_default_strategies(self) -> None:
        assert DEFAULT_CHUNK_STRATEGY == ChunkStrategy.DYNAMIC_PROGRAMMING
        assert DEFAULT_PARTITION_STRATEGY == PartitionStrategy.LLM

    def test_chunk_constants(self) -> None:
        assert CHUNK_TARGET_RATIO > 0
        assert CHUNK_TARGET_RATIO < 1
        assert CHUNK_RETRY_LIMIT >= 1


class TestConstantsExtended:
    def test_ref_prefixes(self) -> None:
        assert PR_REF_PREFIX.startswith("refs/")
        assert FORK_REF_PREFIX.startswith("refs/")

    def test_max_output_tokens_positive(self) -> None:
        assert MAX_OUTPUT_TOKENS > 0
