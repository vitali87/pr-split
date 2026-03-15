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
            id="g1", title="t", description="d",
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
            id="g1", title="t", description="d",
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
            id="g1", title="t", description="d",
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
            id="g1", title="t", description="d",
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
            id="g1", title="t", description="d",
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
