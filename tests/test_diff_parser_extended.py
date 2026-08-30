from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pr_split.diff_ops.parser import extract_diff, parse_diff
from pr_split.exceptions import GitOperationError

SAMPLE_DIFF = (
    "diff --git a/hello.py b/hello.py\n"
    "new file mode 100644\n"
    "index 0000000..e69de29\n"
    "--- /dev/null\n"
    "+++ b/hello.py\n"
    "@@ -0,0 +1,5 @@\n"
    "+def hello():\n"
    '+    return "hello"\n'
    "+\n"
    "+def world():\n"
    '+    return "world"\n'
    "diff --git a/utils.py b/utils.py\n"
    "--- a/utils.py\n"
    "+++ b/utils.py\n"
    "@@ -1,3 +1,4 @@\n"
    " import os\n"
    "+import sys\n"
    " \n"
    " def helper():\n"
    "@@ -10,4 +11,7 @@ def helper():\n"
    "     pass\n"
    "     return True\n"
    "+\n"
    "+def new_func():\n"
    "+    pass\n"
    "     x = 1\n"
    "     y = 2\n"
)


class TestParseDiffInvalid:
    def test_empty_diff_produces_empty_patch_set(self) -> None:
        parsed = parse_diff("")
        assert parsed.file_paths == []
        assert parsed.stats["total_files"] == 0


class TestLabeledDiff:
    def test_labeled_diff_contains_hunk_index_markers(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        labeled = parsed.labeled_diff
        assert "[hunk_index=0]" in labeled
        assert "[hunk_index=1]" in labeled

    def test_labeled_diff_contains_file_headers(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        labeled = parsed.labeled_diff
        assert "+++ b/hello.py" in labeled
        assert "+++ b/utils.py" in labeled


class TestHunkContentEdge:
    def test_hunk_content_nonexistent_file_returns_empty(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        assert parsed.hunk_content("nope.py", 0) == ""

    def test_hunk_content_for_first_hunk(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        content = parsed.hunk_content("hello.py", 0)
        assert "hello" in content


class TestFileSummaryFlags:
    def test_non_new_file(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        stats = parsed.stats
        utils_summary = next(fs for fs in stats["file_summaries"] if fs["path"] == "utils.py")
        assert utils_summary["is_new"] is False
        assert utils_summary["is_deleted"] is False
        assert utils_summary["is_renamed"] is False


EXTRACT_DIFF_SAMPLE = """\
diff --git a/hello.py b/hello.py
new file mode 100644
--- /dev/null
+++ b/hello.py
@@ -0,0 +1,3 @@
+def hello():
+    return "hello"
+
"""


class TestExtractDiffSubprocess:
    @patch("pr_split.diff_ops.parser.subprocess.run")
    def test_extract_diff_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "diff"], returncode=0, stdout=EXTRACT_DIFF_SAMPLE.encode(), stderr=b""
        )
        result = extract_diff("feature", "main")
        assert result == EXTRACT_DIFF_SAMPLE
        mock_run.assert_called_once_with(
            [
                "git",
                "-c",
                "core.quotePath=false",
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "-U3",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                "main...feature",
            ],
            capture_output=True,
        )

    @patch("pr_split.diff_ops.parser.subprocess.run")
    def test_extract_diff_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "diff"],
            returncode=1,
            stdout=b"",
            stderr=b"fatal: bad revision",
        )
        with pytest.raises(GitOperationError, match="bad revision"):
            extract_diff("bad-branch", "main")


class TestRawDiffPreserved:
    def test_raw_diff_preserved(self) -> None:
        parsed = parse_diff(EXTRACT_DIFF_SAMPLE)
        assert parsed.raw_diff == EXTRACT_DIFF_SAMPLE


class TestExtractDiffPreservesCarriageReturns:
    @patch("pr_split.diff_ops.parser.subprocess.run")
    def test_crlf_diff_is_returned_verbatim(self, mock_run: MagicMock) -> None:
        raw = b"--- a/c.txt\n+++ b/c.txt\n@@ -1 +1 @@\n-old\r\n+new\r\n"
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "diff"], returncode=0, stdout=raw, stderr=b""
        )
        result = extract_diff("feature", "main")
        assert "\r\n" in result
        assert result == raw.decode()
        assert mock_run.call_args.kwargs.get("text") is not True


class TestNonAsciiPaths:
    def test_unicode_paths_survive_extraction_and_parsing(self, tmp_path: Path) -> None:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@x", *args],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout

        git("init", "-q", "-b", "main")
        (tmp_path / "ünï.txt").write_text("one\n", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "base")
        git("checkout", "-qb", "dev")
        (tmp_path / "ünï.txt").write_text("two\n", encoding="utf-8")
        (tmp_path / "new file ü.txt").write_text("x\n", encoding="utf-8")
        git("add", "-A")
        git("commit", "-qm", "dev")

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            parsed = parse_diff(extract_diff("dev", "main"))
        finally:
            os.chdir(cwd)

        assert sorted(pf.path for pf in parsed.patch_set) == ["new file ü.txt", "ünï.txt"]
