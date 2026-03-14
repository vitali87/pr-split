from __future__ import annotations

from pr_split.constants import Priority, PRState
from pr_split.schemas import (
    BranchRecord,
    GitState,
    Group,
    PlanFile,
    PRRecord,
    SplitPlan,
)


class TestSplitPlan:
    def test_defaults(self) -> None:
        plan = SplitPlan(
            dev_branch="feat/x", base_branch="main",
            max_loc=400, priority=Priority.ORTHOGONAL,
        )
        assert plan.groups == []
        assert plan.author is None

    def test_with_author(self) -> None:
        plan = SplitPlan(
            dev_branch="feat/x", base_branch="main",
            max_loc=400, priority=Priority.LOGICAL,
            author="Jane <jane@x.com>",
        )
        assert plan.author == "Jane <jane@x.com>"


class TestBranchRecord:
    def test_defaults(self) -> None:
        record = BranchRecord(
            group_id="pr-1", branch_name="pr-split/pr-1", base_branch="main",
        )
        assert record.commit_sha == ""

    def test_full_record(self) -> None:
        record = BranchRecord(
            group_id="pr-2", branch_name="pr-split/pr-2",
            base_branch="main", commit_sha="abc123",
        )
        assert record.commit_sha == "abc123"
        assert record.group_id == "pr-2"


class TestPRRecord:
    def test_default_state_is_open(self) -> None:
        record = PRRecord(
            group_id="pr-1", pr_number=42,
            pr_url="https://github.com/org/repo/pull/42",
        )
        assert record.state == PRState.OPEN

    def test_explicit_state(self) -> None:
        record = PRRecord(
            group_id="pr-1", pr_number=42,
            pr_url="https://github.com/org/repo/pull/42",
            state=PRState.MERGED,
        )
        assert record.state == PRState.MERGED


class TestGitState:
    def test_empty_defaults(self) -> None:
        gs = GitState()
        assert gs.branches == []
        assert gs.prs == []


class TestPlanFile:
    def test_roundtrip_json(self) -> None:
        plan_file = PlanFile(
            plan=SplitPlan(
                dev_branch="feat/big", base_branch="main",
                max_loc=400, priority=Priority.ORTHOGONAL,
                groups=[Group(id="pr-1", title="t", description="d")],
            ),
            git_state=GitState(
                branches=[BranchRecord(
                    group_id="pr-1", branch_name="pr-split/pr-1", base_branch="main",
                )],
                prs=[PRRecord(
                    group_id="pr-1", pr_number=1, pr_url="https://example.com/1",
                )],
            ),
        )
        json_str = plan_file.model_dump_json()
        restored = PlanFile.model_validate_json(json_str)
        assert restored.plan.dev_branch == "feat/big"
        assert len(restored.git_state.branches) == 1
        assert len(restored.git_state.prs) == 1

    def test_default_git_state(self) -> None:
        plan_file = PlanFile(
            plan=SplitPlan(
                dev_branch="feat/x", base_branch="main",
                max_loc=200, priority=Priority.LOGICAL,
            )
        )
        assert plan_file.git_state.branches == []
        assert plan_file.git_state.prs == []


class TestGroupPatchHashEdge:
    def test_explicit_hash_not_overwritten(self) -> None:
        g = Group(
            id="pr-1", title="t", description="d",
            expected_patch="content", expected_patch_sha256="custom_hash",
        )
        assert g.expected_patch_sha256 == "custom_hash"
