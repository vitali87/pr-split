from __future__ import annotations

from pr_split.constants import AssignmentType
from pr_split.diff_ops.parser import parse_diff
from pr_split.graph import PlanDAG
from pr_split.planner.coherence import make_groups_standalone
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

    # 10 LOC together; a limit of 8 keeps them apart, the test after the module.
    groups = make_groups_standalone(groups, parsed, max_loc=8)

    assert [g.depends_on for g in groups] == [["pr-2"], []]


def test_a_test_group_rejoins_the_module_it_tests_when_that_fits() -> None:
    parsed = parse_diff(DIFF)
    groups = [_whole("pr-1", "tests/test_pruning.py"), _whole("pr-2", "pkg/context_pruning.py")]

    groups = make_groups_standalone(groups, parsed, max_loc=400)

    assert [g.id for g in groups] == ["pr-2"]
    assert {a.file_path for a in groups[0].assignments} == {
        "pkg/context_pruning.py",
        "tests/test_pruning.py",
    }


def test_groups_that_need_each_other_are_combined() -> None:
    # The module already builds on the test group, and the test uses the
    # module: no order works, so the two become one group.
    parsed = parse_diff(DIFF)
    groups = [
        _whole("pr-1", "tests/test_pruning.py"),
        _whole("pr-2", "pkg/context_pruning.py", depends_on=["pr-1"]),
    ]

    groups = make_groups_standalone(groups, parsed)

    assert len(groups) == 1
    assert {a.file_path for a in groups[0].assignments} == {
        "tests/test_pruning.py",
        "pkg/context_pruning.py",
    }
    assert groups[0].depends_on == []
    PlanDAG(groups).validate_acyclic()


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
    from pr_split.planner.client import plan_split

    parsed = parse_diff(DIFF)
    # 10 LOC in total; a limit of 8 forces the two files into separate groups.
    settings = Settings(partition_strategy=PartitionStrategy.GRAPH, max_loc=8, min_loc=1)

    groups = plan_split(parsed, settings)

    owner = {a.file_path: g for g in groups for a in g.assignments}
    test_group = owner["tests/test_pruning.py"]
    module_group = owner["pkg/context_pruning.py"]
    assert test_group.id != module_group.id
    assert module_group.id in PlanDAG(groups).ancestors(test_group.id)


def test_graph_groups_a_test_file_with_the_module_it_imports() -> None:
    from pr_split.config import Settings
    from pr_split.constants import PartitionStrategy
    from pr_split.planner.partitioning import partition_diff

    # Different directories and file names: only the shared names relate them.
    settings = Settings(partition_strategy=PartitionStrategy.GRAPH, max_loc=400)

    groups = partition_diff(parse_diff(DIFF), settings)

    assert len(groups) == 1
    assert {a.file_path for a in groups[0].assignments} == {
        "pkg/context_pruning.py",
        "tests/test_pruning.py",
    }
    assert "review-slice" not in groups[0].title


def _title_of(diff: str, files: list[list[str]]) -> list[str]:
    from pr_split.planner.partitioning import retitle_groups

    parsed = parse_diff(diff)
    counts = {pf.path: len(pf) for pf in parsed.patch_set}
    groups = [
        Group(
            id=f"pr-{i}",
            title="",
            description="",
            assignments=[
                GroupAssignment(
                    file_path=path,
                    assignment_type=AssignmentType.WHOLE_FILE,
                    hunk_indices=list(range(counts[path])),
                )
                for path in paths
            ],
        )
        for i, paths in enumerate(files, start=1)
    ]
    retitle_groups(groups, parsed)
    return [g.title for g in groups]


def test_title_names_the_main_file_and_counts_the_rest() -> None:
    titles = _title_of(DIFF, [["pkg/context_pruning.py", "tests/test_pruning.py"]])
    assert titles == ["Add pkg/context_pruning.py and 1 test file"]


def test_title_of_a_single_file_group_has_no_group_number() -> None:
    titles = _title_of(DIFF, [["tests/test_pruning.py"], ["pkg/context_pruning.py"]])
    assert titles == ["Add tests/test_pruning.py", "Add pkg/context_pruning.py"]


def test_title_says_which_part_of_a_shared_file_a_group_holds() -> None:
    diff = """\
diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,2 +1,3 @@
 a = 1
+b = 2
 c = 3
@@ -20,2 +21,3 @@
 x = 1
+y = 2
 z = 3
"""
    from pr_split.planner.partitioning import retitle_groups

    parsed = parse_diff(diff)
    groups = [
        Group(
            id=gid,
            title="",
            description="",
            assignments=[
                GroupAssignment(
                    file_path="pkg/core.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[idx],
                )
            ],
        )
        for gid, idx in (("pr-1", 1), ("pr-2", 0))
    ]
    retitle_groups(groups, parsed)
    assert [g.title for g in groups] == [
        "Update pkg/core.py (part 2 of 2)",
        "Update pkg/core.py (part 1 of 2)",
    ]
