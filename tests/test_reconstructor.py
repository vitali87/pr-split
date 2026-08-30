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
    split_git_lines,
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
        mock_run.return_value = MagicMock(returncode=0, stdout=b"file content\n", stderr=b"")
        result = _get_base_file_content("foo.py", "abc123")
        assert result == "file content\n"

    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout=b"", stderr=b"not found")
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
        mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
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


class TestMissingTrailingNewline:
    NO_EOL_PATCH = """\
--- a/f.txt
+++ b/f.txt
@@ -1,2 +1,2 @@
 a
-b
\\ No newline at end of file
+c
\\ No newline at end of file
"""

    def test_existing_file_keeps_missing_trailing_newline(self) -> None:
        patch_file = PatchSet(self.NO_EOL_PATCH)[0]
        assert apply_hunks("a\nb", patch_file, [0]) == "a\nc"

    def test_newline_added_when_dev_branch_adds_one(self) -> None:
        patch = """\
--- a/f.txt
+++ b/f.txt
@@ -1,2 +1,2 @@
 a
-b
\\ No newline at end of file
+b
"""
        patch_file = PatchSet(patch)[0]
        assert apply_hunks("a\nb", patch_file, [0]) == "a\nb\n"

    def test_marker_after_removed_line_does_not_strip_target(self) -> None:
        patch = """\
--- a/f.txt
+++ b/f.txt
@@ -1,2 +1,2 @@
 a
+c
-b
\\ No newline at end of file
"""
        patch_file = PatchSet(patch)[0]
        assert apply_hunks("a\nb", patch_file, [0]) == "a\nc\n"

    def test_new_file_without_trailing_newline(self) -> None:
        diff = """\
diff --git a/n.txt b/n.txt
new file mode 100644
--- /dev/null
+++ b/n.txt
@@ -0,0 +1,2 @@
+x
+y
\\ No newline at end of file
"""
        parsed = parse_diff(diff)
        group = Group(
            id="g",
            title="g",
            description="g",
            depends_on=[],
            assignments=[
                GroupAssignment(
                    file_path="n.txt",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
            estimated_loc=2,
        )
        assert materialize_group_files(parsed, group, "base")["n.txt"] == "x\ny"


class TestSplitGitLines:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            pytest.param("", [], id="empty"),
            pytest.param("a\n", ["a\n"], id="one-line"),
            pytest.param("a\nb", ["a\n", "b"], id="no-trailing-newline"),
            pytest.param("a\x0cb\nc\n", ["a\x0cb\n", "c\n"], id="form-feed"),
            pytest.param("a\x0bb\nc\n", ["a\x0bb\n", "c\n"], id="vertical-tab"),
            pytest.param("a\u2028b\nc\n", ["a\u2028b\n", "c\n"], id="line-separator"),
            pytest.param("a\x85b\n", ["a\x85b\n"], id="nel"),
            pytest.param("a\r\nb\r\n", ["a\r\n", "b\r\n"], id="crlf"),
            pytest.param("\n\n", ["\n", "\n"], id="blank-lines"),
        ],
    )
    def test_splits_on_newline_only(self, content: str, expected: list[str]) -> None:
        assert split_git_lines(content) == expected


class TestApplyHunksWithSplitlinesSeparators:
    def test_form_feed_in_earlier_line_does_not_shift_hunk(self) -> None:
        base = "a\x0cb\n" + "".join(f"{c}\n" for c in "cdefghijkl")
        dev = base.replace("k\n", "K\n")
        diff = "--- a/f.txt\n+++ b/f.txt\n@@ -7,5 +7,5 @@\n h\n i\n j\n-k\n+K\n l\n"
        pf = PatchSet(diff)[0]
        assert apply_hunks(base, pf, [0]) == dev


class TestCrlfPreserved:
    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_base_content_keeps_crlf(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout=b"a\r\nb\r\n", stderr=b"")
        assert _get_base_file_content("c.txt", "main") == "a\r\nb\r\n"

    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_partial_change_to_crlf_file_keeps_endings(self, mock_run: MagicMock) -> None:
        base = "".join(f"line{i}\r\n" for i in range(1, 8))
        mock_run.return_value = MagicMock(returncode=0, stdout=base.encode(), stderr=b"")
        diff = (
            "diff --git a/c.txt b/c.txt\n--- a/c.txt\n+++ b/c.txt\n"
            "@@ -1,6 +1,6 @@\n line1\r\n line2\r\n-line3\r\n+LINE3\r\n"
            " line4\r\n line5\r\n line6\r\n"
        )
        parsed = parse_diff(diff)
        group = Group(
            id="pr-1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="c.txt",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
        )
        result = materialize_group_files(parsed, group, "main")
        assert result["c.txt"] == base.replace("line3\r\n", "LINE3\r\n")

    @patch("pr_split.diff_ops.reconstructor.subprocess.run")
    def test_bare_carriage_return_does_not_shift_hunks(self, mock_run: MagicMock) -> None:
        # A lone CR is not a line break for git; with base content no longer
        # newline-translated, splitting on it would misplace every later hunk.
        mock_run.return_value = MagicMock(returncode=0, stdout=b"x\ry\nz\n", stderr=b"")
        diff = (
            "diff --git a/c.txt b/c.txt\n--- a/c.txt\n+++ b/c.txt\n"
            "@@ -1,2 +1,2 @@\n x\ry\n-z\n+Z\n"
        )
        parsed = parse_diff(diff)
        group = Group(
            id="pr-1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="c.txt",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                )
            ],
        )
        assert materialize_group_files(parsed, group, "main")["c.txt"] == "x\ry\nZ\n"
