from __future__ import annotations

from pr_split.constants import Priority
from pr_split.planner.prompts import (
    SPLIT_TOOL_NAME,
    SPLIT_TOOL_SCHEMA,
    build_chunk_continuation_prompt,
    build_chunk_first_prompt,
    build_system_prompt,
    build_user_prompt,
)
from pr_split.types_defs import DiffStats, FileSummary


def _make_diff_stats() -> DiffStats:
    return DiffStats(
        total_files=2,
        total_added=10,
        total_removed=3,
        total_loc=13,
        file_summaries=[
            FileSummary(
                path="foo.py",
                added=7,
                removed=2,
                is_new=True,
                is_deleted=False,
                is_renamed=False,
                hunk_count=2,
            ),
            FileSummary(
                path="bar.py",
                added=3,
                removed=1,
                is_new=False,
                is_deleted=False,
                is_renamed=False,
                hunk_count=1,
            ),
        ],
    )


class TestSplitToolSchema:
    def test_schema_has_groups(self) -> None:
        assert "groups" in SPLIT_TOOL_SCHEMA["properties"]

    def test_tool_name(self) -> None:
        assert SPLIT_TOOL_NAME == "propose_split_plan"


class TestBuildSystemPrompt:
    def test_orthogonal_includes_orthogonal(self) -> None:
        result = build_system_prompt(Priority.ORTHOGONAL, 400)
        assert "ORTHOGONAL" in result
        assert "400" in result

    def test_logical_includes_logical(self) -> None:
        result = build_system_prompt(Priority.LOGICAL, 200)
        assert "LOGICAL" in result
        assert "200" in result


class TestBuildUserPrompt:
    def test_contains_file_summary(self) -> None:
        stats = _make_diff_stats()
        result = build_user_prompt(stats, "the diff text")
        assert "foo.py" in result
        assert "bar.py" in result
        assert "the diff text" in result

    def test_contains_total_stats(self) -> None:
        stats = _make_diff_stats()
        result = build_user_prompt(stats, "diff")
        assert "2 files" in result
        assert "+10/-3" in result

    def test_new_file_flag_in_summary(self) -> None:
        stats = _make_diff_stats()
        result = build_user_prompt(stats, "diff")
        assert "new" in result

    def test_hunk_index_range_in_summary(self) -> None:
        stats = _make_diff_stats()
        result = build_user_prompt(stats, "diff")
        assert "indices 0..1" in result
        assert "indices 0..0" in result

    def test_zero_hunk_file(self) -> None:
        stats = DiffStats(
            total_files=1,
            total_added=0,
            total_removed=0,
            total_loc=0,
            file_summaries=[
                FileSummary(
                    path="empty.py",
                    added=0,
                    removed=0,
                    is_new=False,
                    is_deleted=False,
                    is_renamed=False,
                    hunk_count=0,
                ),
            ],
        )
        result = build_user_prompt(stats, "diff")
        assert "no hunks" in result


class TestBuildChunkFirstPrompt:
    def test_contains_chunk_info(self) -> None:
        stats = _make_diff_stats()
        result = build_chunk_first_prompt(stats, "chunk diff", 5)
        assert "chunk 1 of 5" in result
        assert "4 chunk(s)" in result
        assert "chunk diff" in result


class TestBuildChunkContinuationPrompt:
    def test_contains_chunk_index_and_catalog(self) -> None:
        stats = _make_diff_stats()
        result = build_chunk_continuation_prompt(
            stats, "chunk diff", 3, 5, "group catalog text"
        )
        assert "chunk 3 of 5" in result
        assert "group catalog text" in result
        assert "chunk diff" in result
