from __future__ import annotations

from pr_split.constants import AssignmentType
from pr_split.schemas import Group, GroupAssignment


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

    def test_partial_hunks_assignment(self) -> None:
        assignment = GroupAssignment(
            file_path="hello.py",
            assignment_type=AssignmentType.PARTIAL_HUNKS,
            hunk_indices=[1],
        )
        assert assignment.assignment_type == AssignmentType.PARTIAL_HUNKS
        assert assignment.hunk_indices == [1]
