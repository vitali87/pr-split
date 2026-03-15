from __future__ import annotations

import pytest

from pr_split.constants import AssignmentType
from pr_split.diff_ops.parser import parse_diff
from pr_split.planner.chunker import (
    assign_uncovered_hunks,
    build_chunk_diff_from_hunks,
    build_chunk_stats_from_hunks,
    build_hunk_sequence,
    chunk_hunks,
    format_group_catalog,
    recompute_estimated_loc,
)
from pr_split.schemas import Group, GroupAssignment
from pr_split.types_defs import HunkRef

SAMPLE_DIFF = """\
diff --git a/a.py b/a.py
new file mode 100644
--- /dev/null
+++ b/a.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
diff --git a/b.py b/b.py
new file mode 100644
--- /dev/null
+++ b/b.py
@@ -0,0 +1,4 @@
+lineA
+lineB
+lineC
+lineD
"""

TWO_HUNK_DIFF = """\
diff --git a/c.py b/c.py
--- a/c.py
+++ b/c.py
@@ -1,3 +1,4 @@
 old1
+new1
 old2
 old3
@@ -10,3 +11,4 @@
 old10
+new10
 old11
 old12
"""


def _ga(path: str, indices: list[int]) -> GroupAssignment:
    return GroupAssignment(
        file_path=path,
        assignment_type=AssignmentType.PARTIAL_HUNKS,
        hunk_indices=indices,
    )


def _group(
    gid: str,
    assignments: list[GroupAssignment],
    depends_on: list[str] | None = None,
) -> Group:
    return Group(
        id=gid,
        title=f"Group {gid}",
        description=f"Description for {gid}",
        depends_on=depends_on or [],
        assignments=assignments,
        estimated_loc=0,
    )


class TestBuildHunkSequence:
    def test_returns_all_hunks(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        seq = build_hunk_sequence(parsed)
        assert len(seq) == 2
        assert seq[0].file_path == "a.py"
        assert seq[1].file_path == "b.py"

    def test_token_estimates_positive(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        seq = build_hunk_sequence(parsed)
        for href in seq:
            assert href.token_estimate >= 1

    def test_custom_token_ratio(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        seq_default = build_hunk_sequence(parsed, 0.25)
        seq_high = build_hunk_sequence(parsed, 1.0)
        assert seq_high[0].token_estimate >= seq_default[0].token_estimate

    def test_multi_hunk_file(self) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        seq = build_hunk_sequence(parsed)
        assert len(seq) == 2
        assert seq[0].hunk_index == 0
        assert seq[1].hunk_index == 1


class TestChunkHunks:
    def test_single_chunk_when_budget_large(self) -> None:
        refs = [HunkRef("a.py", 0, 100), HunkRef("b.py", 0, 100)]
        chunks = chunk_hunks(refs, 1000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 2

    def test_splits_when_budget_exceeded(self) -> None:
        refs = [HunkRef("a.py", 0, 100), HunkRef("b.py", 0, 100)]
        chunks = chunk_hunks(refs, 150)
        assert len(chunks) == 2

    def test_hunk_exceeding_budget_raises(self) -> None:
        refs = [HunkRef("a.py", 0, 500)]
        with pytest.raises(ValueError, match="exceeds budget"):
            chunk_hunks(refs, 100)

    def test_empty_input_returns_empty(self) -> None:
        chunks = chunk_hunks([], 100)
        assert chunks == []

    def test_exact_budget_no_split(self) -> None:
        refs = [HunkRef("a.py", 0, 50), HunkRef("b.py", 0, 50)]
        chunks = chunk_hunks(refs, 100)
        assert len(chunks) == 1


class TestBuildChunkDiffFromHunks:
    def test_filters_to_relevant_files(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        refs = [HunkRef("a.py", 0, 10)]
        result = build_chunk_diff_from_hunks(parsed, refs)
        assert "a.py" in result
        assert "b.py" not in result

    def test_includes_hunk_index_label(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        refs = [HunkRef("a.py", 0, 10)]
        result = build_chunk_diff_from_hunks(parsed, refs)
        assert "[hunk_index=0]" in result

    def test_empty_refs_returns_empty(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        result = build_chunk_diff_from_hunks(parsed, [])
        assert result == ""


class TestBuildChunkStatsFromHunks:
    def test_stats_for_single_file(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        refs = [HunkRef("a.py", 0, 10)]
        stats = build_chunk_stats_from_hunks(parsed, refs)
        assert stats["total_files"] == 1
        assert stats["total_added"] > 0

    def test_stats_for_multiple_files(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        refs = [HunkRef("a.py", 0, 10), HunkRef("b.py", 0, 10)]
        stats = build_chunk_stats_from_hunks(parsed, refs)
        assert stats["total_files"] == 2

    def test_empty_refs_returns_zero_stats(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        stats = build_chunk_stats_from_hunks(parsed, [])
        assert stats["total_files"] == 0
        assert stats["total_added"] == 0
        assert stats["total_removed"] == 0


class TestRecomputeEstimatedLoc:
    def test_recomputes_from_diff(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [_group("g1", [_ga("a.py", [0])])]
        recompute_estimated_loc(groups, parsed)
        assert groups[0].estimated_loc == 3
        assert groups[0].estimated_added == 3
        assert groups[0].estimated_removed == 0

    def test_invalid_hunk_index_skipped(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [_group("g1", [_ga("a.py", [0, 99])])]
        recompute_estimated_loc(groups, parsed)
        assert groups[0].estimated_loc == 3

    def test_multiple_groups(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [
            _group("g1", [_ga("a.py", [0])]),
            _group("g2", [_ga("b.py", [0])]),
        ]
        recompute_estimated_loc(groups, parsed)
        assert groups[0].estimated_loc == 3
        assert groups[1].estimated_loc == 4


class TestAssignUncoveredHunks:
    def test_all_covered_returns_zero(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [
            _group("g1", [_ga("a.py", [0])]),
            _group("g2", [_ga("b.py", [0])]),
        ]
        count = assign_uncovered_hunks(groups, parsed)
        assert count == 0

    def test_uncovered_assigned_to_file_group(self) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        groups = [_group("g1", [_ga("a.py", [0])])]
        count = assign_uncovered_hunks(groups, parsed)
        assert count == 1
        all_paths = [a.file_path for g in groups for a in g.assignments]
        assert "b.py" in all_paths

    def test_uncovered_added_to_existing_assignment(self) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        groups = [_group("g1", [_ga("c.py", [0])])]
        count = assign_uncovered_hunks(groups, parsed)
        assert count == 1
        c_assignment = next(a for a in groups[0].assignments if a.file_path == "c.py")
        assert 1 in c_assignment.hunk_indices


class TestFormatGroupCatalog:
    def test_basic_format(self) -> None:
        groups = [
            _group("g1", [_ga("a.py", [0])]),
            _group("g2", [_ga("b.py", [0])], depends_on=["g1"]),
        ]
        result = format_group_catalog(groups)
        assert "g1" in result
        assert "g2" in result
        assert "depends on" in result

    def test_no_deps(self) -> None:
        groups = [_group("g1", [_ga("a.py", [0])])]
        result = format_group_catalog(groups)
        assert "depends on" not in result


class TestChunkHunksExtended:
    def test_single_hunk_per_chunk_when_tight(self) -> None:
        refs = [
            HunkRef(file_path=f"f{i}.py", hunk_index=0, token_estimate=90)
            for i in range(3)
        ]
        result = chunk_hunks(refs, 100)
        assert len(result) == 3
        assert all(len(c) == 1 for c in result)

    def test_many_small_hunks_pack_efficiently(self) -> None:
        refs = [
            HunkRef(file_path=f"f{i}.py", hunk_index=0, token_estimate=10)
            for i in range(10)
        ]
        result = chunk_hunks(refs, 100)
        assert len(result) == 1
        assert len(result[0]) == 10


class TestFormatGroupCatalogExtended:
    def test_includes_group_id_and_title(self) -> None:
        g = Group(id="pr-1", title="feat: auth", description="Auth module")
        result = format_group_catalog([g])
        assert "pr-1" in result
        assert "feat: auth" in result

    def test_includes_file_paths(self) -> None:
        g = Group(
            id="pr-1",
            title="t",
            description="d",
            assignments=[
                GroupAssignment(
                    file_path="auth.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                ),
                GroupAssignment(
                    file_path="models.py",
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=[0],
                ),
            ],
        )
        result = format_group_catalog([g])
        assert "auth.py" in result
        assert "models.py" in result

    def test_empty_groups(self) -> None:
        result = format_group_catalog([])
        assert result == ""
