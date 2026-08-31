from __future__ import annotations

import pytest

from pr_split.config import Settings
from pr_split.constants import PartitionStrategy, Priority
from pr_split.diff_ops.parser import parse_diff
from pr_split.exceptions import PlanValidationError
from pr_split.planner.partitioning import (
    PartitionUnit,
    _group_units_graph,
    _merge_order_is_acyclic,
    _repair_graph_min_loc,
    build_partition_units,
    partition_diff,
)

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


def _settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_loc: int,
    partition_strategy: PartitionStrategy,
    priority: Priority = Priority.ORTHOGONAL,
    **overrides: object,
) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return Settings(
        max_loc=max_loc,
        partition_strategy=partition_strategy,
        priority=priority,
        **overrides,
    )


class TestBuildPartitionUnits:
    def test_splits_large_file_by_hunks(self) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        units = build_partition_units(parsed, 1)
        assert len(units) == 2
        assert units[0].file_path == "c.py"
        assert units[0].hunk_indices == (0,)
        assert units[1].hunk_indices == (1,)


class TestPartitionDiffGraph:
    def test_orthogonal_mode_keeps_unrelated_files_separate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parsed = parse_diff(SAMPLE_DIFF)
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.GRAPH,
        )
        groups = partition_diff(parsed, settings)
        assert len(groups) == 2
        assert all(group.depends_on == [] for group in groups)

    def test_split_file_groups_get_merge_order_dependency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        settings = _settings(
            monkeypatch,
            max_loc=1,
            partition_strategy=PartitionStrategy.GRAPH,
        )
        groups = partition_diff(parsed, settings)
        assert len(groups) == 2
        assert groups[1].depends_on == ["pr-1"]


class TestPartitionDiffCpSat:
    def test_cp_sat_backend_returns_groups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("ortools.sat.python.cp_model")
        parsed = parse_diff(SAMPLE_DIFF)
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.CP_SAT,
        )
        groups = partition_diff(parsed, settings)
        assert len(groups) == 2
        assert all(group.depends_on == [] for group in groups)

    def test_feasible_but_not_optimal_is_warned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cp_model = pytest.importorskip("ortools.sat.python.cp_model")
        from unittest.mock import patch

        parsed = parse_diff(SAMPLE_DIFF)
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.CP_SAT,
            cp_sat_timeout=0.5,
        )
        real_solve = cp_model.CpSolver.Solve

        def solve_feasible(self: object, model: object) -> int:
            real_solve(self, model)
            return cp_model.FEASIBLE

        with (
            patch.object(cp_model.CpSolver, "Solve", solve_feasible),
            patch("pr_split.planner.partitioning.logger") as mock_logger,
        ):
            groups = partition_diff(parsed, settings)

        assert len(groups) == 2
        warning = mock_logger.warning.call_args[0][0]
        assert "feasible but unproven-optimal plan" in warning
        assert "0.5s limit" in warning
        assert "raise --cp-sat-timeout" in warning

    def test_sub_decisecond_timeout_is_reported_exactly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cp_model = pytest.importorskip("ortools.sat.python.cp_model")
        from unittest.mock import patch

        parsed = parse_diff(SAMPLE_DIFF)
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.CP_SAT,
            cp_sat_timeout=0.04,
        )
        real_solve = cp_model.CpSolver.Solve

        def solve_feasible(self: object, model: object) -> int:
            real_solve(self, model)
            return cp_model.FEASIBLE

        with (
            patch.object(cp_model.CpSolver, "Solve", solve_feasible),
            patch("pr_split.planner.partitioning.logger") as mock_logger,
        ):
            partition_diff(parsed, settings)

        warning = mock_logger.warning.call_args[0][0]
        # A timeout below 0.1s must not be rounded away to "0.0s"; the warning
        # tells the user which limit to raise, so it has to name the real value.
        assert "0.04s limit" in warning

    def test_optimal_solution_does_not_warn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("ortools.sat.python.cp_model")
        from unittest.mock import patch

        parsed = parse_diff(SAMPLE_DIFF)
        settings = _settings(monkeypatch, max_loc=10, partition_strategy=PartitionStrategy.CP_SAT)
        with patch("pr_split.planner.partitioning.logger") as mock_logger:
            partition_diff(parsed, settings)
        mock_logger.warning.assert_not_called()


def _unit(file_path: str, hunks: tuple[int, ...], loc: int, position: int) -> PartitionUnit:
    return PartitionUnit(
        id=f"{file_path}:{hunks[0]}-{hunks[-1]}",
        file_path=file_path,
        hunk_indices=hunks,
        loc=loc,
        position=position,
    )


# Greedy grouping used to produce pr-1 -> pr-5 -> pr-2 -> pr-1 for these units:
# {f1[1,2]}, {f1[3], f2[0..2]}, {f1[0], f2[3]} invert the f1/f2 order between groups.
CYCLE_PRONE_UNITS = [
    _unit("lib/f0.py", (0,), 73, 0),
    _unit("lib/f1.py", (0,), 84, 1),
    _unit("lib/f1.py", (1, 2), 126, 2),
    _unit("lib/f1.py", (3,), 28, 3),
    _unit("lib/f2.py", (0, 1, 2), 121, 4),
    _unit("lib/f2.py", (3,), 60, 5),
    _unit("src/f3.py", (0, 1), 113, 6),
    _unit("src/f3.py", (2, 3), 104, 7),
]


class TestMergeOrderIsAcyclic:
    def test_consistent_file_order_is_acyclic(self) -> None:
        grouped = [
            [_unit("a.py", (0,), 10, 0), _unit("b.py", (0,), 10, 2)],
            [_unit("a.py", (1,), 10, 1), _unit("b.py", (1,), 10, 3)],
        ]
        assert _merge_order_is_acyclic(grouped) is True

    def test_inverted_file_order_is_cyclic(self) -> None:
        grouped = [
            [_unit("a.py", (0,), 10, 0), _unit("b.py", (1,), 10, 3)],
            [_unit("a.py", (1,), 10, 1), _unit("b.py", (0,), 10, 2)],
        ]
        assert _merge_order_is_acyclic(grouped) is False

    def test_three_group_cycle_is_detected(self) -> None:
        grouped = [
            [_unit("a.py", (0,), 10, 0), _unit("c.py", (1,), 10, 5)],
            [_unit("a.py", (1,), 10, 1), _unit("b.py", (0,), 10, 2)],
            [_unit("b.py", (1,), 10, 3), _unit("c.py", (0,), 10, 4)],
        ]
        assert _merge_order_is_acyclic(grouped) is False


class TestGraphGroupingStaysAcyclic:
    def test_greedy_grouping_never_inverts_file_order(self) -> None:
        settings = Settings(
            max_loc=150,
            priority=Priority.LOGICAL,
            partition_strategy=PartitionStrategy.GRAPH,
            max_refinement_iterations=0,
        )
        grouped = _group_units_graph(CYCLE_PRONE_UNITS, settings=settings)
        assert _merge_order_is_acyclic(grouped)
        assert sorted(unit.id for group in grouped for unit in group) == sorted(
            unit.id for unit in CYCLE_PRONE_UNITS
        )

    def test_min_loc_repair_refuses_cycle_creating_merge(self) -> None:
        # g0 is undersized. Merging it into g1 exceeds max_loc; merging it into g2 would
        # give {a[0], b[1]}, which a.py orders before g1 and b.py orders after g1 -> cycle.
        grouped = [
            [_unit("a.py", (0,), 10, 0)],
            [_unit("a.py", (1,), 95, 1), _unit("b.py", (0,), 0, 2)],
            [_unit("b.py", (1,), 80, 3)],
        ]
        settings = Settings(
            min_loc=30,
            max_loc=100,
            priority=Priority.LOGICAL,
            partition_strategy=PartitionStrategy.GRAPH,
            max_refinement_iterations=0,
        )
        repaired = _repair_graph_min_loc(grouped, settings=settings)
        assert repaired == grouped
        assert _merge_order_is_acyclic(repaired)

    def test_partition_diff_rejects_cyclic_backend_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cyclic = [
            [_unit("a.py", (0,), 10, 0), _unit("b.py", (1,), 10, 3)],
            [_unit("a.py", (1,), 10, 1), _unit("b.py", (0,), 10, 2)],
        ]
        monkeypatch.setattr(
            "pr_split.planner.partitioning._group_units_graph", lambda units, settings: cyclic
        )
        parsed_diff = parse_diff(
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
            "@@ -1,1 +1,2 @@\n x\n+y\n@@ -10,1 +11,2 @@\n x\n+z\n"
            "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n"
            "@@ -1,1 +1,2 @@\n x\n+y\n@@ -10,1 +11,2 @@\n x\n+z\n"
        )
        settings = Settings(
            max_loc=400, partition_strategy=PartitionStrategy.GRAPH, max_refinement_iterations=0
        )
        with pytest.raises(PlanValidationError, match="cycle"):
            partition_diff(parsed_diff, settings)
