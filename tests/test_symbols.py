from __future__ import annotations

from unittest.mock import patch

from pr_split.constants import AssignmentType
from pr_split.diff_ops.parser import parse_diff
from pr_split.graph import PlanDAG
from pr_split.planner.partitioning import _add_symbol_dependencies
from pr_split.planner.symbols import symbol_dependencies
from pr_split.planner.validator import detect_symbol_order_violations
from pr_split.schemas import Group, GroupAssignment

# A new module and a new test file that imports it: no file in common.
DIFF = """\
diff --git a/pkg/context_pruning.py b/pkg/context_pruning.py
new file mode 100644
--- /dev/null
+++ b/pkg/context_pruning.py
@@ -0,0 +1,6 @@
+PRUNE_LIMIT = 3
+
+
+class PruneReport:
+    def describe_prune(self) -> str:
+        return "x"
diff --git a/tests/test_pruning.py b/tests/test_pruning.py
new file mode 100644
--- /dev/null
+++ b/tests/test_pruning.py
@@ -0,0 +1,4 @@
+from pkg.context_pruning import PRUNE_LIMIT, PruneReport
+
+def test_limit():
+    assert PruneReport().describe_prune() and PRUNE_LIMIT
"""


def _whole(gid: str, path: str, depends_on: list[str] | None = None) -> Group:
    return Group(
        id=gid,
        title=gid,
        description="",
        depends_on=depends_on or [],
        assignments=[
            GroupAssignment(
                file_path=path, assignment_type=AssignmentType.WHOLE_FILE, hunk_indices=[0]
            )
        ],
    )


def test_test_file_depends_on_the_module_it_imports() -> None:
    parsed = parse_diff(DIFF)
    groups = [_whole("pr-1", "tests/test_pruning.py"), _whole("pr-2", "pkg/context_pruning.py")]

    deps = symbol_dependencies(groups, parsed)

    assert deps == {"pr-1": {"pr-2": {"PRUNE_LIMIT", "PruneReport", "describe_prune"}}}


def test_graph_edges_make_the_test_build_on_the_module() -> None:
    parsed = parse_diff(DIFF)
    groups = [_whole("pr-1", "tests/test_pruning.py"), _whole("pr-2", "pkg/context_pruning.py")]

    _add_symbol_dependencies(groups, parsed)

    assert groups[0].depends_on == ["pr-2"]
    assert groups[1].depends_on == []


def test_edge_that_would_close_a_cycle_is_skipped() -> None:
    parsed = parse_diff(DIFF)
    groups = [
        _whole("pr-1", "tests/test_pruning.py"),
        _whole("pr-2", "pkg/context_pruning.py", depends_on=["pr-1"]),
    ]

    with patch("pr_split.planner.partitioning.logger") as mock_logger:
        _add_symbol_dependencies(groups, parsed)

    assert groups[0].depends_on == []
    PlanDAG(groups).validate_acyclic()
    assert "would create a cycle" in mock_logger.warning.call_args.args[0]


def test_definition_in_a_child_of_its_user_is_reported() -> None:
    # The #187 shape: the module is a child of the group that uses it.
    parsed = parse_diff(DIFF)
    groups = [
        _whole("pr-1", "tests/test_pruning.py"),
        _whole("pr-2", "pkg/context_pruning.py", depends_on=["pr-1"]),
    ]

    warnings = detect_symbol_order_violations(groups, parsed, PlanDAG(groups))

    assert len(warnings) == 1
    assert "Group 'pr-1' uses PRUNE_LIMIT, PruneReport, describe_prune" in warnings[0]
    assert "defined in group 'pr-2'" in warnings[0]


def test_correct_order_is_not_reported() -> None:
    parsed = parse_diff(DIFF)
    groups = [
        _whole("pr-1", "tests/test_pruning.py", depends_on=["pr-2"]),
        _whole("pr-2", "pkg/context_pruning.py"),
    ]
    assert detect_symbol_order_violations(groups, parsed, PlanDAG(groups)) == []


def test_name_defined_by_two_groups_is_ignored() -> None:
    diff = """\
diff --git a/a.py b/a.py
new file mode 100644
--- /dev/null
+++ b/a.py
@@ -0,0 +1,2 @@
+def helper_fn():
+    return 1
diff --git a/b.py b/b.py
new file mode 100644
--- /dev/null
+++ b/b.py
@@ -0,0 +1,2 @@
+def helper_fn():
+    return helper_fn
"""
    groups = [_whole("pr-1", "a.py"), _whole("pr-2", "b.py")]
    assert symbol_dependencies(groups, parse_diff(diff)) == {}


def test_graph_partition_links_the_test_group_to_the_module_group() -> None:
    from pr_split.config import Settings
    from pr_split.constants import PartitionStrategy
    from pr_split.planner.partitioning import partition_diff

    parsed = parse_diff(DIFF)
    # 10 LOC in total; a limit of 8 forces the two files into separate groups.
    settings = Settings(partition_strategy=PartitionStrategy.GRAPH, max_loc=8, min_loc=1)

    groups = partition_diff(parsed, settings)

    owner = {a.file_path: g for g in groups for a in g.assignments}
    test_group = owner["tests/test_pruning.py"]
    module_group = owner["pkg/context_pruning.py"]
    assert test_group.id != module_group.id
    assert module_group.id in PlanDAG(groups).ancestors(test_group.id)
