from __future__ import annotations

from pr_split.diff_ops.parser import parse_diff

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


class TestParseDiff:
    def test_parse_file_paths(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        assert set(parsed.file_paths) == {"hello.py", "utils.py"}

    def test_stats_total_files(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        stats = parsed.stats
        assert stats["total_files"] == 2

    def test_stats_added_removed(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        stats = parsed.stats
        assert stats["total_added"] == 9
        assert stats["total_removed"] == 0
        assert stats["total_loc"] == 9

    def test_file_summaries(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        stats = parsed.stats
        paths = {fs["path"] for fs in stats["file_summaries"]}
        assert paths == {"hello.py", "utils.py"}

    def test_new_file_flag(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        stats = parsed.stats
        hello_summary = next(fs for fs in stats["file_summaries"] if fs["path"] == "hello.py")
        assert hello_summary["is_new"] is True

    def test_hunks_for_file(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        hunks = parsed.hunks_for_file("utils.py")
        assert len(hunks) == 2
        assert hunks[0].index == 0
        assert hunks[1].index == 1

    def test_hunks_for_nonexistent_file(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        hunks = parsed.hunks_for_file("nonexistent.py")
        assert hunks == []

    def test_hunk_content(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        content = parsed.hunk_content("utils.py", 0)
        assert "+import sys" in content

    def test_hunk_count(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        stats = parsed.stats
        utils_summary = next(fs for fs in stats["file_summaries"] if fs["path"] == "utils.py")
        assert utils_summary["hunk_count"] == 2
