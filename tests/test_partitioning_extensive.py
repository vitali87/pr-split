from __future__ import annotations

import builtins
import sys

import pytest

from pr_split.config import Settings
from pr_split.constants import AssignmentType, PartitionStrategy, Priority
from pr_split.diff_ops.parser import parse_diff
from pr_split.exceptions import PRSplitError
from pr_split.graph import PlanDAG
from pr_split.planner.partitioning import (
    _derive_merge_order_dependencies,
    _test_pair_bonus,
    partition_diff,
)
from pr_split.planner.scoring import score_plan
from pr_split.planner.validator import validate_plan
from pr_split.schemas import Group, GroupAssignment

UNRELATED_DIFF = """\
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

RELATED_DIFF = """\
diff --git a/src/auth.py b/src/auth.py
new file mode 100644
--- /dev/null
+++ b/src/auth.py
@@ -0,0 +1,3 @@
+def login():
+    return True
+
diff --git a/src/auth_test.py b/src/auth_test.py
new file mode 100644
--- /dev/null
+++ b/src/auth_test.py
@@ -0,0 +1,3 @@
+def test_login():
+    assert login()
+
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
) -> Settings:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return Settings(
        max_loc=max_loc,
        partition_strategy=partition_strategy,
        priority=priority,
    )


def _group_signature(groups: list[Group]) -> tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]:
    normalized: list[tuple[tuple[str, tuple[int, ...]], ...]] = []
    for group in groups:
        assignments = tuple(
            sorted(
                (assignment.file_path, tuple(assignment.hunk_indices))
                for assignment in group.assignments
            )
        )
        normalized.append(assignments)
    return tuple(normalized)


def _assert_valid_plan(groups: list[Group], diff_text: str, max_loc: int) -> None:
    parsed = parse_diff(diff_text)
    warnings = validate_plan(groups, parsed, PlanDAG(groups), max_loc)
    assert warnings == []


class TestPartitionDiffGraphExtensive:
    def test_orthogonal_keeps_unrelated_files_separate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.GRAPH,
            priority=Priority.ORTHOGONAL,
        )
        groups = partition_diff(parse_diff(UNRELATED_DIFF), settings)
        assert len(groups) == 2
        assert all(len(group.assignments) == 1 for group in groups)
        _assert_valid_plan(groups, UNRELATED_DIFF, 10)

    def test_logical_groups_related_files_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.GRAPH,
            priority=Priority.LOGICAL,
        )
        groups = partition_diff(parse_diff(RELATED_DIFF), settings)
        assert len(groups) == 1
        assert len(groups[0].assignments) == 2
        _assert_valid_plan(groups, RELATED_DIFF, 10)

    def test_same_file_hunks_split_with_dependency_when_budget_tight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings(
            monkeypatch,
            max_loc=1,
            partition_strategy=PartitionStrategy.GRAPH,
        )
        groups = partition_diff(parse_diff(TWO_HUNK_DIFF), settings)
        assert len(groups) == 2
        assert groups[1].depends_on == ["pr-1"]
        _assert_valid_plan(groups, TWO_HUNK_DIFF, 1)

    def test_is_deterministic_across_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.GRAPH,
            priority=Priority.LOGICAL,
        )
        parsed = parse_diff(RELATED_DIFF)
        signatures = {_group_signature(partition_diff(parsed, settings)) for _ in range(3)}
        assert len(signatures) == 1


class TestPartitionDiffCpSatExtensive:
    def test_missing_ortools_raises_clean_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parsed = parse_diff(UNRELATED_DIFF)
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.CP_SAT,
        )

        for module_name in list(sys.modules):
            if module_name == "ortools" or module_name.startswith("ortools."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)

        real_import = builtins.__import__

        def _fake_import(name: str, *args, **kwargs):
            if name == "ortools.sat.python" or name.startswith("ortools"):
                raise ImportError("ortools intentionally unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(PRSplitError, match="ortools"):
            partition_diff(parsed, settings)

    def test_orthogonal_keeps_unrelated_files_separate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("ortools.sat.python.cp_model")
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.CP_SAT,
            priority=Priority.ORTHOGONAL,
        )
        groups = partition_diff(parse_diff(UNRELATED_DIFF), settings)
        assert len(groups) == 2
        assert all(len(group.assignments) == 1 for group in groups)
        _assert_valid_plan(groups, UNRELATED_DIFF, 10)

    def test_logical_groups_related_files_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("ortools.sat.python.cp_model")
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.CP_SAT,
            priority=Priority.LOGICAL,
        )
        groups = partition_diff(parse_diff(RELATED_DIFF), settings)
        assert len(groups) == 1
        assert len(groups[0].assignments) == 2
        _assert_valid_plan(groups, RELATED_DIFF, 10)

    def test_same_file_hunks_split_when_budget_is_tight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("ortools.sat.python.cp_model")
        settings = _settings(
            monkeypatch,
            max_loc=1,
            partition_strategy=PartitionStrategy.CP_SAT,
            priority=Priority.ORTHOGONAL,
        )
        groups = partition_diff(parse_diff(TWO_HUNK_DIFF), settings)
        assert len(groups) == 2
        assert groups[1].depends_on == ["pr-1"]
        _assert_valid_plan(groups, TWO_HUNK_DIFF, 1)

    def test_is_deterministic_across_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("ortools.sat.python.cp_model")
        settings = _settings(
            monkeypatch,
            max_loc=10,
            partition_strategy=PartitionStrategy.CP_SAT,
            priority=Priority.LOGICAL,
        )
        parsed = parse_diff(RELATED_DIFF)
        signatures = {_group_signature(partition_diff(parsed, settings)) for _ in range(3)}
        assert len(signatures) == 1


@pytest.mark.parametrize(
    ("partition_strategy", "priority", "diff_text", "max_loc", "expected_groups"),
    [
        (PartitionStrategy.GRAPH, Priority.ORTHOGONAL, UNRELATED_DIFF, 10, 2),
        (PartitionStrategy.GRAPH, Priority.LOGICAL, RELATED_DIFF, 10, 1),
        (PartitionStrategy.GRAPH, Priority.ORTHOGONAL, TWO_HUNK_DIFF, 1, 2),
        (PartitionStrategy.CP_SAT, Priority.ORTHOGONAL, UNRELATED_DIFF, 10, 2),
        (PartitionStrategy.CP_SAT, Priority.LOGICAL, RELATED_DIFF, 10, 1),
        (PartitionStrategy.CP_SAT, Priority.ORTHOGONAL, TWO_HUNK_DIFF, 1, 2),
    ],
)
def test_partitioning_matrix(
    monkeypatch: pytest.MonkeyPatch,
    partition_strategy: PartitionStrategy,
    priority: Priority,
    diff_text: str,
    max_loc: int,
    expected_groups: int,
) -> None:
    if partition_strategy is PartitionStrategy.CP_SAT:
        pytest.importorskip("ortools.sat.python.cp_model")

    settings = _settings(
        monkeypatch,
        max_loc=max_loc,
        partition_strategy=partition_strategy,
        priority=priority,
    )
    parsed = parse_diff(diff_text)
    groups = partition_diff(parsed, settings)
    assert len(groups) == expected_groups
    _assert_valid_plan(groups, diff_text, max_loc)

    metrics = score_plan(groups, max_loc)
    assert metrics.total_groups == expected_groups
    assert metrics.loc_overflow == 0


class TestPartitioningHeuristics:
    def test_test_pair_bonus_matches_test_prefix_and_suffix(self) -> None:
        assert _test_pair_bonus("tests/test_auth.py", "src/auth.py") == 40
        assert _test_pair_bonus("src/auth_test.py", "src/auth.py") == 40

    def test_test_pair_bonus_avoids_false_positive_substrings(self) -> None:
        assert _test_pair_bonus("src/latest.py", "src/late.py") == 0
        assert _test_pair_bonus("src/contest.py", "src/con.py") == 0

    def test_merge_order_dependencies_ignore_group_creation_order(self) -> None:
        groups = [
            Group(
                id="pr-2",
                title="later",
                description="later hunk",
                assignments=[
                    GroupAssignment(
                        file_path="src/app.py",
                        assignment_type=AssignmentType.PARTIAL_HUNKS,
                        hunk_indices=[1],
                    )
                ],
            ),
            Group(
                id="pr-1",
                title="earlier",
                description="earlier hunk",
                assignments=[
                    GroupAssignment(
                        file_path="src/app.py",
                        assignment_type=AssignmentType.PARTIAL_HUNKS,
                        hunk_indices=[0],
                    )
                ],
            ),
        ]

        _derive_merge_order_dependencies(groups)

        assert groups[0].depends_on == ["pr-1"]
        assert groups[1].depends_on == []
