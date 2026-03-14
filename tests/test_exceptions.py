from __future__ import annotations

from pr_split.exceptions import (
    DiffParseError,
    ErrorMsg,
    GitOperationError,
    LLMError,
    PlanValidationError,
    PRSplitError,
)


class TestErrorMsg:
    def test_call_without_kwargs(self) -> None:
        assert ErrorMsg.DIRTY_WORKTREE() == "Working tree has uncommitted changes; commit or stash first"

    def test_call_with_kwargs(self) -> None:
        result = ErrorMsg.BRANCH_NOT_FOUND(branch="feature/xyz")
        assert "feature/xyz" in result

    def test_cycle_detected(self) -> None:
        assert "cycle" in ErrorMsg.CYCLE_DETECTED().lower()

    def test_coverage_gap(self) -> None:
        result = ErrorMsg.COVERAGE_GAP(file="foo.py", index=2)
        assert "foo.py" in result
        assert "2" in result

    def test_coverage_overlap(self) -> None:
        result = ErrorMsg.COVERAGE_OVERLAP(file="bar.py", index=1, groups="g1, g2")
        assert "bar.py" in result
        assert "g1, g2" in result

    def test_loc_mismatch(self) -> None:
        result = ErrorMsg.LOC_MISMATCH(actual=10, expected=20)
        assert "10" in result
        assert "20" in result

    def test_merge_conflict(self) -> None:
        result = ErrorMsg.MERGE_CONFLICT(a="g1", b="g2", file="x.py")
        assert "g1" in result
        assert "g2" in result
        assert "x.py" in result

    def test_no_plan(self) -> None:
        assert "No split plan" in ErrorMsg.NO_PLAN()

    def test_llm_parse_error(self) -> None:
        assert "bad json" in ErrorMsg.LLM_PARSE_ERROR(detail="bad json")

    def test_pr_not_found(self) -> None:
        assert "42" in ErrorMsg.PR_NOT_FOUND(number=42)

    def test_hunk_too_large(self) -> None:
        result = ErrorMsg.HUNK_TOO_LARGE(file="big.py", index=0, tokens=5000, budget=1000)
        assert "big.py" in result
        assert "5000" in result


class TestExceptionHierarchy:
    def test_diff_parse_error_is_prsplit(self) -> None:
        assert issubclass(DiffParseError, PRSplitError)

    def test_plan_validation_error_is_prsplit(self) -> None:
        assert issubclass(PlanValidationError, PRSplitError)

    def test_git_operation_error_is_prsplit(self) -> None:
        assert issubclass(GitOperationError, PRSplitError)

    def test_llm_error_is_prsplit(self) -> None:
        assert issubclass(LLMError, PRSplitError)

    def test_prsplit_error_is_exception(self) -> None:
        assert issubclass(PRSplitError, Exception)
