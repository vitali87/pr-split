from __future__ import annotations

from pr_split.types_defs import DiffStats, FileSummary, ForkPRInfo, HunkInfo, HunkRef


class TestHunkInfo:
    def test_fields(self) -> None:
        h = HunkInfo(index=0, source_start=1, source_length=5,
                     target_start=1, target_length=7, added_lines=3, removed_lines=1)
        assert h.index == 0
        assert h.source_start == 1
        assert h.added_lines == 3

    def test_tuple_behavior(self) -> None:
        h = HunkInfo(0, 1, 5, 1, 7, 3, 1)
        assert h[0] == 0
        assert len(h) == 7


class TestHunkRef:
    def test_fields(self) -> None:
        hr = HunkRef(file_path="foo.py", hunk_index=2, token_estimate=100)
        assert hr.file_path == "foo.py"
        assert hr.hunk_index == 2

    def test_tuple_unpacking(self) -> None:
        path, idx, tokens = HunkRef("bar.py", 0, 50)
        assert path == "bar.py"
        assert idx == 0
        assert tokens == 50


class TestFileSummary:
    def test_dict_access(self) -> None:
        fs: FileSummary = {
            "path": "test.py", "added": 10, "removed": 5,
            "is_new": True, "is_deleted": False, "is_renamed": False, "hunk_count": 2,
        }
        assert fs["path"] == "test.py"
        assert fs["added"] == 10


class TestDiffStats:
    def test_dict_access(self) -> None:
        ds: DiffStats = {
            "total_files": 3, "total_added": 20, "total_removed": 5,
            "total_loc": 25, "file_summaries": [],
        }
        assert ds["total_files"] == 3
        assert ds["total_loc"] == 25


class TestForkPRInfo:
    def test_dict_access(self) -> None:
        info: ForkPRInfo = {
            "pr_number": 42, "local_ref": "refs/pr-split/pr-42",
            "base_branch": "main", "author": "Jane <jane@x.com>",
            "fork_full_name": "jane/repo",
        }
        assert info["pr_number"] == 42

    def test_pr_number_none(self) -> None:
        info: ForkPRInfo = {
            "pr_number": None, "local_ref": "refs/pr-split/fork-user-branch",
            "base_branch": "main", "author": "User <u@x.com>",
            "fork_full_name": "user/repo",
        }
        assert info["pr_number"] is None
