from __future__ import annotations

from pr_split.constants import AssignmentType, Priority
from pr_split.schemas import Group, GroupAssignment, SplitPlan


class TestGroup:
    def test_patch_hash_auto_computed(self) -> None:
        group = Group(
            id="pr-1",
            title="test",
            description="test",
            expected_patch="some diff content",
        )
        assert group.expected_patch_sha256 != ""
        assert len(group.expected_patch_sha256) == 64

    def test_empty_patch_no_hash(self) -> None:
        group = Group(
            id="pr-1",
            title="test",
            description="test",
        )
        assert group.expected_patch_sha256 == ""

    def test_compute_patch_hash_consistent(self) -> None:
        group = Group(
            id="pr-1",
            title="test",
            description="test",
            expected_patch="some diff content",
        )
        assert group.compute_patch_hash() == group.expected_patch_sha256


class TestGroupAssignment:
    def test_whole_file_assignment(self) -> None:
        assignment = GroupAssignment(
            file_path="hello.py",
            assignment_type=AssignmentType.WHOLE_FILE,
            hunk_indices=[0, 1, 2],
        )
        assert assignment.assignment_type == AssignmentType.WHOLE_FILE

    def test_whole_file_covers_exactly_the_parsed_range(self) -> None:
        for stored in ([], [1], [0, 1, 2], [99]):
            assignment = GroupAssignment(
                file_path="hello.py",
                assignment_type=AssignmentType.WHOLE_FILE,
                hunk_indices=stored,
            )
            assert assignment.covered_indices(3) == [0, 1, 2]
            assert assignment.hunk_indices == stored

    def test_partial_hunks_cover_exactly_what_they_list(self) -> None:
        assignment = GroupAssignment(
            file_path="hello.py",
            assignment_type=AssignmentType.PARTIAL_HUNKS,
            hunk_indices=[2, 0],
        )
        assert assignment.covered_indices(3) == [2, 0]

    def test_partial_hunks_assignment(self) -> None:
        assignment = GroupAssignment(
            file_path="hello.py",
            assignment_type=AssignmentType.PARTIAL_HUNKS,
            hunk_indices=[1],
        )
        assert assignment.assignment_type == AssignmentType.PARTIAL_HUNKS
        assert assignment.hunk_indices == [1]


class TestSplitPlanStacked:
    def test_defaults_to_flat(self) -> None:
        plan = SplitPlan(
            dev_branch="dev", base_branch="main", max_loc=400, priority=Priority.ORTHOGONAL
        )
        assert plan.stacked is False

    def test_stacked_round_trips_through_plan_file(self) -> None:
        plan = SplitPlan(
            dev_branch="dev",
            base_branch="main",
            max_loc=400,
            priority=Priority.ORTHOGONAL,
            stacked=True,
        )
        assert SplitPlan.model_validate(plan.model_dump()).stacked is True


class TestSplitPlanDraft:
    def test_defaults_to_ready(self) -> None:
        plan = SplitPlan(
            dev_branch="dev", base_branch="main", max_loc=400, priority=Priority.ORTHOGONAL
        )
        assert plan.draft is False

    def test_draft_round_trips_through_plan_file(self) -> None:
        plan = SplitPlan(
            dev_branch="dev",
            base_branch="main",
            max_loc=400,
            priority=Priority.ORTHOGONAL,
            draft=True,
        )
        assert SplitPlan.model_validate(plan.model_dump()).draft is True
