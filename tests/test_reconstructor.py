from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from unidiff import PatchSet

from pr_split.constants import AssignmentType
from pr_split.diff_ops.parser import parse_diff
from pr_split.diff_ops.reconstructor import (
    _get_base_file_content,
    apply_hunks,
    materialize_group_files,
    merge_chain_assignments,
    target_file_modes,
)
from pr_split.exceptions import GitOperationError
from pr_split.schemas import Group, GroupAssignment

PATCH_TEXT = """\
--- a/example.py
+++ b/example.py
@@ -1,5 +1,6 @@
 line1
+inserted_after_1
 line2
 line3
 line4
 line5
@@ -10,4 +11,5 @@
 line10
 line11
+inserted_after_11
 line12
 line13
"""

NEW_FILE_DIFF = """\
diff --git a/new_file.py b/new_file.py
new file mode 100644
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+def hello():
+    return "world"
+
"""

MODIFY_DIFF = """\
diff --git a/existing.py b/existing.py
--- a/existing.py
+++ b/existing.py
@@ -1,3 +1,4 @@
 line1
+inserted
 line2
 line3
"""


def _base_content() -> str:
    return "\n".join(f"line{i}" for i in range(1, 16)) + "\n"


class TestApplyHunks:
    def test_apply_all_hunks(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [0, 1])
        lines = result.splitlines()
        assert "inserted_after_1" in lines
        assert "inserted_after_11" in lines

    def test_apply_first_hunk_only(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [0])
        lines = result.splitlines()
        assert "inserted_after_1" in lines
        assert "inserted_after_11" not in lines

    def test_apply_second_hunk_only(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [1])
        lines = result.splitlines()
        assert "inserted_after_1" not in lines
        assert "inserted_after_11" in lines

    def test_apply_no_hunks(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [])
        assert result == _base_content()

    def test_line_count_after_one_hunk(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        base = _base_content()
        base_lines = base.splitlines()
        result = apply_hunks(base, pf, [0])
        result_lines = result.splitlines()
        assert len(result_lines) == len(base_lines) + 1

    def test_preserves_untouched_lines(self) -> None:
        patch_set = PatchSet(PATCH_TEXT)
        pf = patch_set[0]
        result = apply_hunks(_base_content(), pf, [0])
        lines = result.splitlines()
        assert "line7" in lines
        assert "line8" in lines
        assert "line15" in lines


class TestGetBaseFileContent:
    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="file content\n", stderr="")
        result = _get_base_file_content("foo.py", "abc123")
        assert result == "file content\n"

    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
        with pytest.raises(GitOperationError):
            _get_base_file_content("missing.py", "abc123")


class TestMaterializeGroupFilesNewFile:
    def test_whole_file_new(self) -> None:
        parsed = parse_diff(NEW_FILE_DIFF)
        group = Group(
            id="g1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="new_file.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                ),
            ],
        )
        result = materialize_group_files(parsed, group, "abc123")
        assert "new_file.py" in result
        assert "hello" in result["new_file.py"]

    def test_partial_hunks_new(self) -> None:
        parsed = parse_diff(NEW_FILE_DIFF)
        group = Group(
            id="g1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="new_file.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0],
                ),
            ],
        )
        result = materialize_group_files(parsed, group, "abc123")
        assert "new_file.py" in result

    def test_file_not_in_diff_skipped(self) -> None:
        parsed = parse_diff(NEW_FILE_DIFF)
        group = Group(
            id="g1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="not_in_diff.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                ),
            ],
        )
        result = materialize_group_files(parsed, group, "abc123")
        assert result == {}


class TestGetBaseFileContentExtended:
    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_empty_file_returns_empty(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = _get_base_file_content("empty.py", "abc123")
        assert result == ""


class TestMaterializeGroupFilesExisting:
    @patch("pr_split.diff_ops.reconstructor._get_base_file_content")
    def test_whole_file_existing(self, mock_base: MagicMock) -> None:
        mock_base.return_value = "line1\nline2\nline3\n"
        parsed = parse_diff(MODIFY_DIFF)
        group = Group(
            id="g1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="existing.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                ),
            ],
        )
        result = materialize_group_files(parsed, group, "abc123")
        assert "existing.py" in result
        assert "inserted" in result["existing.py"]

    @patch("pr_split.diff_ops.reconstructor._get_base_file_content")
    def test_partial_hunks_existing(self, mock_base: MagicMock) -> None:
        mock_base.return_value = "line1\nline2\nline3\n"
        parsed = parse_diff(MODIFY_DIFF)
        group = Group(
            id="g1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="existing.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0],
                ),
            ],
        )
        result = materialize_group_files(parsed, group, "abc123")
        assert "existing.py" in result
        assert "inserted" in result["existing.py"]


class TestMergeChainAssignments:
    def _child(self) -> Group:
        return Group(
            id="pr-2",
            title="child",
            description="child",
            depends_on=["pr-1"],
            assignments=[
                GroupAssignment(
                    file_path="shared.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[1],
                ),
                GroupAssignment(
                    file_path="own.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                ),
            ],
        )

    def _parent(self) -> Group:
        return Group(
            id="pr-1",
            title="parent",
            description="parent",
            assignments=[
                GroupAssignment(
                    file_path="shared.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0],
                ),
                GroupAssignment(
                    file_path="parent_only.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                ),
            ],
        )

    def test_ancestor_hunks_join_shared_file(self) -> None:
        merged = merge_chain_assignments(self._child(), [self._parent()])
        by_path = {a.file_path: a for a in merged.assignments}
        assert by_path["shared.py"].hunk_indices == [0, 1]

    def test_ancestor_only_files_stay_out(self) -> None:
        merged = merge_chain_assignments(self._child(), [self._parent()])
        assert "parent_only.py" not in {a.file_path for a in merged.assignments}

    def test_own_whole_file_assignment_preserved(self) -> None:
        merged = merge_chain_assignments(self._child(), [self._parent()])
        by_path = {a.file_path: a for a in merged.assignments}
        assert by_path["own.py"].assignment_type is AssignmentType.WHOLE_FILE

    def test_no_ancestors_is_identity(self) -> None:
        child = self._child()
        assert merge_chain_assignments(child, []) == child


class TestMergeChainAssignmentsWholeFileAncestor:
    def _parent(self) -> Group:
        return Group(
            id="pr-1",
            title="parent",
            description="parent",
            assignments=[
                GroupAssignment(
                    file_path="shared.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                ),
            ],
        )

    def _child(self) -> Group:
        return Group(
            id="pr-2",
            title="child",
            description="child",
            depends_on=["pr-1"],
            assignments=[
                GroupAssignment(
                    file_path="shared.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[1],
                ),
            ],
        )

    def test_whole_file_ancestor_covers_every_hunk(self) -> None:
        merged = merge_chain_assignments(
            self._child(), [self._parent()], hunk_counts={"shared.py": 2}
        )
        by_path = {a.file_path: a for a in merged.assignments}
        assert by_path["shared.py"].hunk_indices == [0, 1]


class TestMergeChainAssignmentsCarryAncestorFiles:
    def _parent(self) -> Group:
        return Group(
            id="pr-1",
            title="parent",
            description="parent",
            assignments=[
                GroupAssignment(
                    file_path="parent_only.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                ),
            ],
        )

    def _child(self) -> Group:
        return Group(
            id="pr-3",
            title="child",
            description="child",
            depends_on=["pr-1"],
            assignments=[
                GroupAssignment(
                    file_path="child.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0],
                ),
            ],
        )

    def test_ancestor_only_file_is_carried(self) -> None:
        merged = merge_chain_assignments(
            self._child(),
            [self._parent()],
            hunk_counts={"parent_only.py": 1, "child.py": 1},
            carry_ancestor_files=True,
        )
        by_path = {a.file_path: a for a in merged.assignments}
        assert by_path["parent_only.py"].hunk_indices == [0]


class TestAddedFileLineEndings:
    def test_added_file_content_is_not_double_spaced(self) -> None:
        parsed = parse_diff(NEW_FILE_DIFF)
        group = Group(
            id="pr-1",
            title="t",
            description="t",
            assignments=[
                GroupAssignment(
                    file_path="new_file.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                )
            ],
        )
        result = materialize_group_files(parsed, group, "abc123")
        assert result["new_file.py"] == 'def hello():\n    return "world"\n\n'


class TestMaterializeDuplicateAssignments:
    @patch("pr_split.diff_ops.reconstructor._get_base_file_content")
    def test_two_assignments_for_one_file_apply_both_hunks(self, mock_base: MagicMock) -> None:
        mock_base.return_value = _base_content()
        parsed = parse_diff(PATCH_TEXT)
        group = Group(
            id="pr-1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="example.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0],
                ),
                GroupAssignment(
                    file_path="example.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[1],
                ),
            ],
        )
        with patch("pr_split.diff_ops.reconstructor.logger.info") as mock_log:
            result = materialize_group_files(parsed, group, "main")
        content = result["example.py"]
        assert content is not None
        assert "inserted_after_1" in content
        assert "inserted_after_11" in content
        assert mock_base.call_count == 1
        assert "Materializing 1 file" in mock_log.call_args[0][0]

    def test_duplicate_assignments_on_new_file_apply_both_hunks(self) -> None:
        # Two hunks in a new file (unidiff splits them when the context gap
        # is large enough), each claimed by a separate PARTIAL assignment.
        parsed = parse_diff(
            "diff --git a/n.py b/n.py\nnew file mode 100644\n--- /dev/null\n+++ b/n.py\n"
            "@@ -0,0 +1,2 @@\n+one\n+two\n"
            "@@ -0,0 +10,1 @@\n+ten\n"
        )
        assert len(parsed.patch_set[0]) == 2
        group = Group(
            id="pr-1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="n.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0],
                ),
                GroupAssignment(
                    file_path="n.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[1],
                ),
            ],
        )
        assert materialize_group_files(parsed, group, "main")["n.py"] == "one\ntwo\nten\n"


class TestTruncatedFileIsNotDeleted:
    @patch("pr_split.diff_ops.reconstructor._get_base_file_content", return_value="x\ny\n")
    def test_file_emptied_on_dev_is_written_empty(self, mock_base: MagicMock) -> None:
        parsed = parse_diff(
            "diff --git a/a.txt b/a.txt\nindex 1111111..e69de29 100644\n"
            "--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +0,0 @@\n-x\n-y\n"
        )
        assert parsed.patch_set[0].is_removed_file  # the heuristic that misled us
        group = Group(
            id="pr-1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="a.txt",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
        )
        assert materialize_group_files(parsed, group, "main") == {"a.txt": ""}

    def test_real_deletion_still_removes(self) -> None:
        parsed = parse_diff(
            "diff --git a/a.txt b/a.txt\ndeleted file mode 100644\nindex 1111111..0000000\n"
            "--- a/a.txt\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n"
        )
        group = Group(
            id="pr-1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="a.txt",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
        )
        assert materialize_group_files(parsed, group, "main") == {"a.txt": None}


class TestTargetFileModes:
    def _group(self, *paths: str) -> Group:
        return Group(
            id="pr-1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path=p, assignment_type=AssignmentType.WHOLE_FILE, hunk_indices=[0]
                )
                for p in paths
            ],
        )

    def test_mode_change_with_content_is_reported(self) -> None:
        parsed = parse_diff(
            "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n"
            "index 422c2b7..55dce13\n--- a/run.sh\n+++ b/run.sh\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n"
        )
        assert target_file_modes(parsed, self._group("run.sh")) == {"run.sh": 0o100755}

    def test_new_executable_file_is_reported(self) -> None:
        parsed = parse_diff(
            "diff --git a/n.sh b/n.sh\nnew file mode 100755\n--- /dev/null\n+++ b/n.sh\n"
            "@@ -0,0 +1 @@\n+x\n"
        )
        assert target_file_modes(parsed, self._group("n.sh")) == {"n.sh": 0o100755}

    def test_plain_change_and_unassigned_files_are_ignored(self) -> None:
        parsed = parse_diff(
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
            "diff --git a/b.sh b/b.sh\nold mode 100644\nnew mode 100755\n"
            "--- a/b.sh\n+++ b/b.sh\n@@ -1 +1 @@\n-x\n+y\n"
        )
        assert target_file_modes(parsed, self._group("a.py")) == {}

    def test_removed_file_is_ignored(self) -> None:
        parsed = parse_diff(
            "diff --git a/d.sh b/d.sh\ndeleted file mode 100755\n--- a/d.sh\n+++ /dev/null\n"
            "@@ -1 +0,0 @@\n-x\n"
        )
        assert target_file_modes(parsed, self._group("d.sh")) == {}

    def test_symlink_mode_is_not_reported(self) -> None:
        parsed = parse_diff(
            "diff --git a/link b/link\nnew file mode 120000\n--- /dev/null\n+++ b/link\n"
            "@@ -0,0 +1 @@\n+other.py\n\\ No newline at end of file\n"
        )
        assert target_file_modes(parsed, self._group("link")) == {}

    def test_mode_change_back_to_non_executable_is_reported(self) -> None:
        parsed = parse_diff(
            "diff --git a/run.sh b/run.sh\nold mode 100755\nnew mode 100644\n"
            "--- a/run.sh\n+++ b/run.sh\n@@ -1 +1 @@\n-x\n+y\n"
        )
        assert target_file_modes(parsed, self._group("run.sh")) == {"run.sh": 0o100644}
