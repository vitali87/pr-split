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
            args=["git", "diff"], returncode=0, stdout=EXTRACT_DIFF_SAMPLE, stderr=""
        )
        result = extract_diff("feature", "main")
        assert result == EXTRACT_DIFF_SAMPLE
        mock_run.assert_called_once_with(
            ["git", "diff", "--submodule=short", "main...feature"],
            capture_output=True,
            text=True,
        )

    @patch("pr_split.diff_ops.parser.subprocess.run")
    def test_extract_diff_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "diff"],
            returncode=1,
            stdout="",
            stderr="fatal: bad revision",
        )
        with pytest.raises(GitOperationError, match="bad revision"):
            extract_diff("bad-branch", "main")


class TestRawDiffPreserved:
    def test_raw_diff_preserved(self) -> None:
        parsed = parse_diff(EXTRACT_DIFF_SAMPLE)
        assert parsed.raw_diff == EXTRACT_DIFF_SAMPLE


SUBMODULE_DIFF = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-x
+y
diff --git a/vendor b/vendor
index 0ebfaa8..2f687ea 160000
--- a/vendor
+++ b/vendor
@@ -1 +1 @@
-Subproject commit 0ebfaa8000000000000000000000000000000000
+Subproject commit 2f687ea000000000000000000000000000000000
"""

NEW_SUBMODULE_DIFF = """\
diff --git a/lib b/lib
new file mode 160000
index 0000000..2f687ea
--- /dev/null
+++ b/lib
@@ -0,0 +1 @@
+Subproject commit 2f687ea000000000000000000000000000000000
"""


DELETED_SUBMODULE_DIFF = """\
diff --git a/vendor b/vendor
deleted file mode 160000
index d075b20..0000000
--- a/vendor
+++ /dev/null
@@ -1 +0,0 @@
-Subproject commit d075b20000000000000000000000000000000000
"""


class TestSubmoduleChangesAreRejected:
    def test_deleted_submodule_is_rejected(self) -> None:
        from pr_split.exceptions import DiffParseError

        with pytest.raises(DiffParseError, match="pointer bump at vendor"):
            parse_diff(DELETED_SUBMODULE_DIFF)

    def test_pointer_bump_is_rejected_by_name(self) -> None:
        from pr_split.exceptions import DiffParseError

        with pytest.raises(DiffParseError, match="pointer bump at vendor"):
            parse_diff(SUBMODULE_DIFF)

    def test_added_submodule_is_rejected(self) -> None:
        from pr_split.exceptions import DiffParseError

        with pytest.raises(DiffParseError, match="Submodule changes are not supported"):
            parse_diff(NEW_SUBMODULE_DIFF)

    def test_regular_files_with_similar_content_are_fine(self) -> None:
        raw = (
            "diff --git a/notes.txt b/notes.txt\n"
            "index 0ebfaa8..2f687ea 100644\n"
            "--- a/notes.txt\n"
            "+++ b/notes.txt\n"
            "@@ -1 +1 @@\n"
            "-Subproject commit 0ebfaa8000000000000000000000000000000000\n"
            "+Subproject commit 2f687ea000000000000000000000000000000000\n"
        )
        assert [pf.path for pf in parse_diff(raw).patch_set] == ["notes.txt"]

    def test_real_submodule_bump_is_rejected_before_planning(self, tmp_path: Path) -> None:
        from pr_split.exceptions import DiffParseError

        def git(cwd: Path, *args: str) -> str:
            return subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@x",
                    "-c",
                    "protocol.file.allow=always",
                    *args,
                ],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout

        sub = tmp_path / "sub"
        sub.mkdir()
        git(sub, "init", "-q", "-b", "main")
        (sub / "f").write_text("1\n")
        git(sub, "add", "-A")
        git(sub, "commit", "-qm", "one")
        repo = tmp_path / "repo"
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        (repo / "app.py").write_text("x\n")
        git(repo, "add", "-A")
        git(repo, "submodule", "add", "-q", str(sub), "vendor")
        git(repo, "commit", "-qm", "base")
        git(repo, "checkout", "-qb", "dev")
        (sub / "f").write_text("2\n")
        git(sub, "commit", "-qam", "two")
        git(repo / "vendor", "pull", "-q", "origin", "main")
        (repo / "app.py").write_text("y\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "bump")

        cwd = os.getcwd()
        os.chdir(repo)
        try:
            for mode in ("short", "log", "diff"):
                # A user-level diff.submodule setting must not hide the bump.
                git(repo, "config", "diff.submodule", mode)
                with pytest.raises(DiffParseError, match="vendor"):
                    parse_diff(extract_diff("dev", "main"))
        finally:
            os.chdir(cwd)
