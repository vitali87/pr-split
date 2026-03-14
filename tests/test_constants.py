from __future__ import annotations

from pr_split.constants import (
    BRANCH_PREFIX,
    CHUNK_RETRY_LIMIT,
    CHUNK_TARGET_RATIO,
    DEFAULT_MAX_LOC,
    PLAN_DIR,
    PLAN_FILE,
    AssignmentType,
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

    def test_chunk_constants(self) -> None:
        assert CHUNK_TARGET_RATIO > 0
        assert CHUNK_TARGET_RATIO < 1
        assert CHUNK_RETRY_LIMIT >= 1
