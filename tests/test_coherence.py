"""Each group of a plan must work on its own: see pr_split/planner/coherence.py."""

from __future__ import annotations

from pr_split.constants import AssignmentType
from pr_split.diff_ops.parser import parse_diff
from pr_split.graph import PlanDAG
from pr_split.planner.coherence import find_needs, hunk_ties, make_groups_standalone
from pr_split.planner.symbols import symbol_dependencies
from pr_split.schemas import Group, GroupAssignment


def _new_file(path: str, lines: list[str]) -> str:
    body = "".join(f"+{line}\n" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n{body}"
    )


def _group(gid: str, *files: str | tuple[str, list[int]], deps: list[str] | None = None) -> Group:
    assignments = []
    for item in files:
        if isinstance(item, tuple):
            path, hunks = item
            kind = AssignmentType.PARTIAL_HUNKS
        else:
            path, hunks, kind = item, [0], AssignmentType.WHOLE_FILE
        assignments.append(
            GroupAssignment(file_path=path, assignment_type=kind, hunk_indices=hunks)
        )
    return Group(
        id=gid, title=gid, description=gid, assignments=assignments, depends_on=deps or []
    )


def _reader(files: dict[str, str]):
    return files.get


def _files_of(group: Group) -> set[str]:
    return {a.file_path for a in group.assignments}


class TestNeeds:
    def test_patch_target_needs_the_group_that_binds_the_name(self) -> None:
        diff = _new_file("pkg/cli.py", ["from .tools import require_tools_now"]) + _new_file(
            "tests/conftest.py", ['patch("pkg.cli.require_tools_now")']
        )
        groups = [_group("pr-1", "pkg/cli.py"), _group("pr-2", "tests/conftest.py")]
        needs = {(n.user, n.provider) for n in find_needs(groups, parse_diff(diff))}
        assert ("pr-2", "pr-1") in needs

    def test_patch_target_on_a_name_the_diff_only_mentions_needs_nothing(self) -> None:
        # ``logger`` existed before the diff; the new line only calls it.
        diff = _new_file("pkg/cli.py", ["logger.info('hello there')"]) + _new_file(
            "tests/test_x.py", ['patch("pkg.cli.logger")']
        )
        groups = [_group("pr-1", "pkg/cli.py"), _group("pr-2", "tests/test_x.py")]
        assert find_needs(groups, parse_diff(diff)) == []

    def test_a_test_group_needs_every_changed_module_it_imports(self) -> None:
        diff = _new_file("pkg/engine.py", ["VALUE = 1"]) + _new_file(
            "tests/test_engine.py", ["def test_it():", "    assert run_it()"]
        )
        # The import sits outside the diff; only the file at head shows it.
        head = {
            "tests/test_engine.py": "from pkg.engine import run_it\n\ndef test_it():\n    pass\n",
            "pkg/engine.py": "VALUE = 1\n",
        }
        groups = [_group("pr-1", "pkg/engine.py"), _group("pr-2", "tests/test_engine.py")]
        needs = find_needs(groups, parse_diff(diff), _reader(head))
        assert [(n.user, n.provider) for n in needs] == [("pr-2", "pr-1")]

    def test_imports_are_followed_through_unchanged_modules(self) -> None:
        diff = _new_file("pkg/core.py", ["VALUE = 1"]) + _new_file(
            "tests/test_api.py", ["def test_it():", "    assert True"]
        )
        head = {
            "tests/test_api.py": "from pkg import api\n",
            "pkg/__init__.py": "",
            "pkg/api.py": "from .core import VALUE\n",
            "pkg/core.py": "VALUE = 1\n",
        }
        groups = [_group("pr-1", "pkg/core.py"), _group("pr-2", "tests/test_api.py")]
        needs = find_needs(groups, parse_diff(diff), _reader(head))
        assert [(n.user, n.provider) for n in needs] == [("pr-2", "pr-1")]

    def test_tests_need_the_conftest_above_them(self) -> None:
        diff = _new_file("tests/conftest.py", ["import pytest"]) + _new_file(
            "tests/unit/test_a.py", ["def test_a():", "    pass"]
        )
        groups = [_group("pr-1", "tests/conftest.py"), _group("pr-2", "tests/unit/test_a.py")]
        needs = {(n.user, n.provider) for n in find_needs(groups, parse_diff(diff))}
        assert ("pr-2", "pr-1") in needs

    def test_a_script_a_test_loads_by_path_is_a_need(self) -> None:
        diff = _new_file("scripts/score.py", ["LIMIT = 3"]) + _new_file(
            "tests/test_score.py", ['SCRIPT = ROOT / "scripts" / "score.py"']
        )
        groups = [_group("pr-1", "scripts/score.py"), _group("pr-2", "tests/test_score.py")]
        needs = {(n.user, n.provider) for n in find_needs(groups, parse_diff(diff))}
        assert ("pr-2", "pr-1") in needs


class TestSymbols:
    def test_an_import_line_binds_the_name_for_its_own_file(self) -> None:
        # The import and the test using it landed in different groups.
        diff = """\
diff --git a/tests/test_x.py b/tests/test_x.py
--- a/tests/test_x.py
+++ b/tests/test_x.py
@@ -1,2 +1,3 @@
 import os
+from pkg.mod import make_widget

@@ -30,1 +31,3 @@ def test_old():
     pass
+def test_new():
+    assert make_widget()
"""
        groups = [
            _group("pr-1", ("tests/test_x.py", [0])),
            _group("pr-2", ("tests/test_x.py", [1])),
        ]
        assert symbol_dependencies(groups, parse_diff(diff)) == {"pr-2": {"pr-1": {"make_widget"}}}

    def test_a_helper_local_to_one_test_file_is_not_shared(self) -> None:
        diff = _new_file("tests/test_a.py", ["def _make_group():", "    return 1"]) + _new_file(
            "tests/test_b.py", ["def _make_group():", "    return 2", "x = _make_group()"]
        )
        groups = [_group("pr-1", "tests/test_a.py"), _group("pr-2", "tests/test_b.py")]
        assert symbol_dependencies(groups, parse_diff(diff)) == {}

    def test_code_never_needs_a_name_a_test_defines(self) -> None:
        diff = _new_file("tests/test_a.py", ["def capture_output():", "    pass"]) + _new_file(
            "pkg/cli.py", ["console.capture_output()"]
        )
        groups = [_group("pr-1", "tests/test_a.py"), _group("pr-2", "pkg/cli.py")]
        assert symbol_dependencies(groups, parse_diff(diff)) == {}

    def test_an_edited_signature_is_not_a_new_definition(self) -> None:
        diff = """\
diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1,2 +1,2 @@
-def compute_total(x):
+def compute_total(x, y=0):
     return x
diff --git a/pkg/b.py b/pkg/b.py
new file mode 100644
--- /dev/null
+++ b/pkg/b.py
@@ -0,0 +1 @@
+VALUE = compute_total(1)
"""
        groups = [_group("pr-1", "pkg/a.py"), _group("pr-2", "pkg/b.py")]
        assert symbol_dependencies(groups, parse_diff(diff)) == {}

    def test_reading_a_new_field_needs_the_code_that_fills_it(self) -> None:
        diff = (
            _new_file("pkg/types.py", ["class Summary(TypedDict):", "    hunk_indices: list[int]"])
            + _new_file("pkg/parser.py", ["summary = Summary(hunk_indices=[0])"])
            + _new_file("pkg/prompts.py", ['rows = summary["hunk_indices"]'])
        )
        groups = [
            _group("pr-1", "pkg/types.py"),
            _group("pr-2", "pkg/parser.py"),
            _group("pr-3", "pkg/prompts.py"),
        ]
        deps = symbol_dependencies(groups, parse_diff(diff))
        assert set(deps["pr-3"]) == {"pr-1", "pr-2"}


class TestHunkTies:
    DIFF = """\
diff --git a/tests/test_x.py b/tests/test_x.py
--- a/tests/test_x.py
+++ b/tests/test_x.py
@@ -1,3 +1,4 @@
 class TestThing:
+    @patch("pkg.forget")
     @patch("pkg.one")
     @patch("pkg.two")
@@ -9,3 +10,4 @@ class TestThing:
         mock_two,
         mock_one,
+        mock_forget,
     ) -> None:
"""
    HEAD = (
        "class TestThing:\n"
        '    @patch("pkg.forget")\n'
        '    @patch("pkg.one")\n'
        '    @patch("pkg.two")\n'
        '    @patch("pkg.three")\n'
        '    @patch("pkg.four")\n'
        "    def test_it(\n"
        "        self,\n"
        "        mock_four,\n"
        "        mock_three,\n"
        "        mock_two,\n"
        "        mock_one,\n"
        "        mock_forget,\n"
        "    ) -> None:\n"
    )

    def test_a_decorator_and_the_parameter_it_injects_are_tied(self) -> None:
        ties = hunk_ties(parse_diff(self.DIFF), _reader({"tests/test_x.py": self.HEAD}))
        assert ties == {"tests/test_x.py": [[0, 1]]}

    def test_hunks_in_different_functions_are_not_tied(self) -> None:
        diff = """\
diff --git a/pkg/a.py b/pkg/a.py
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1,2 +1,3 @@
 def first():
+    x = 1
     return 1
@@ -9,2 +10,3 @@ def second():
 def second():
+    y = 2
     return 2
"""
        head = (
            "def first():\n    x = 1\n    return 1\n"
            + "\n" * 6
            + ("def second():\n    y = 2\n    return 2\n")
        )
        assert hunk_ties(parse_diff(diff), _reader({"pkg/a.py": head})) == {}

    def test_groups_holding_tied_hunks_become_one(self) -> None:
        groups = [
            _group("pr-1", ("tests/test_x.py", [0])),
            _group("pr-2", ("tests/test_x.py", [1])),
        ]
        kept = make_groups_standalone(
            groups, parse_diff(self.DIFF), _reader({"tests/test_x.py": self.HEAD})
        )
        assert len(kept) == 1
        assert kept[0].assignments[0].hunk_indices == [0, 1]


class TestMakeGroupsStandalone:
    def test_a_lockfile_ships_with_its_manifest(self) -> None:
        diff = _new_file("uv.lock", ["version = 1"]) + _new_file("pyproject.toml", ["[project]"])
        groups = [_group("pr-1", "uv.lock"), _group("pr-2", "pyproject.toml")]
        kept = make_groups_standalone(groups, parse_diff(diff), max_loc=1)
        assert len(kept) == 1
        assert _files_of(kept[0]) == {"uv.lock", "pyproject.toml"}

    def test_tests_mixed_into_unrelated_code_are_separated(self) -> None:
        # pr-1 packs a test for pr-2's module next to its own code; kept
        # together, pr-1 would need pr-2 and pr-2 (file order) would need pr-1.
        diff = (
            _new_file("pkg/alpha.py", ["ALPHA_VALUE = 1"])
            + _new_file("pkg/beta.py", ["def beta_value():", "    return ALPHA_VALUE"])
            + _new_file("tests/test_beta.py", ["from pkg.beta import beta_value"])
        )
        groups = [
            _group("pr-1", "pkg/alpha.py", "tests/test_beta.py"),
            _group("pr-2", "pkg/beta.py"),
        ]
        kept = make_groups_standalone(
            groups, parse_diff(diff), max_loc=3, keep_declared_deps=False
        )
        owner = {path: g.id for g in kept for path in _files_of(g)}
        dag = PlanDAG(kept)
        dag.validate_acyclic()
        # The test goes back to the module it tests, which builds on alpha.
        assert owner["tests/test_beta.py"] == owner["pkg/beta.py"]
        assert owner["pkg/alpha.py"] in dag.ancestors(owner["pkg/beta.py"])

    def test_an_updated_existing_test_moves_to_the_code_it_follows(self) -> None:
        diff = """\
diff --git a/pkg/engine.py b/pkg/engine.py
--- a/pkg/engine.py
+++ b/pkg/engine.py
@@ -1,2 +1,2 @@
 def compute_score():
-    return 1
+    return 2
diff --git a/pkg/other.py b/pkg/other.py
new file mode 100644
--- /dev/null
+++ b/pkg/other.py
@@ -0,0 +1 @@
+OTHER_VALUE = 3
diff --git a/tests/test_engine.py b/tests/test_engine.py
--- a/tests/test_engine.py
+++ b/tests/test_engine.py
@@ -1,2 +1,2 @@
 def test_compute_score():
-    assert compute_score() == 1
+    assert compute_score() == 2
"""
        head = {
            "tests/test_engine.py": (
                "from pkg.engine import compute_score\n"
                "from pkg.other import OTHER_VALUE\n"
                "def test_compute_score():\n    assert compute_score() == 2\n"
            ),
            "pkg/engine.py": "def compute_score():\n    return 2\n",
            "pkg/other.py": "OTHER_VALUE = 3\n",
        }
        groups = [
            _group("pr-1", "pkg/engine.py"),
            _group("pr-2", "pkg/other.py"),
            _group("pr-3", "tests/test_engine.py"),
        ]
        kept = make_groups_standalone(
            groups, parse_diff(diff), _reader(head), max_loc=3, keep_declared_deps=False
        )
        owner = {path: g.id for g in kept for path in _files_of(g)}
        # Without it, pr-1's own run of the old test would fail.
        assert owner["tests/test_engine.py"] == owner["pkg/engine.py"]

    def test_combined_groups_keep_the_larger_ones_id_and_both_descriptions(self) -> None:
        diff = _new_file("pkg/a.py", ["def alpha_one():", "    return beta_one()"]) + _new_file(
            "pkg/b.py", ["def beta_one():", "    return alpha_one()", "", ""]
        )
        groups = [_group("pr-1", "pkg/a.py"), _group("pr-2", "pkg/b.py")]
        kept = make_groups_standalone(groups, parse_diff(diff))
        assert [g.id for g in kept] == ["pr-2"]
        assert kept[0].description == "pr-2\n\npr-1"
        assert kept[0].depends_on == []

    def test_a_planner_s_own_dependencies_are_kept(self) -> None:
        diff = _new_file("pkg/a.py", ["A = 1"]) + _new_file("pkg/b.py", ["B = 2"])
        groups = [_group("pr-1", "pkg/a.py"), _group("pr-2", "pkg/b.py", deps=["pr-1"])]
        kept = make_groups_standalone(groups, parse_diff(diff))
        assert [g.depends_on for g in kept] == [[], ["pr-1"]]

    def test_a_single_group_is_returned_as_is(self) -> None:
        diff = _new_file("pkg/a.py", ["A = 1"])
        groups = [_group("pr-1", "pkg/a.py")]
        assert make_groups_standalone(groups, parse_diff(diff)) is groups


def test_plan_split_names_a_combined_group_after_what_it_holds() -> None:
    from pr_split.config import Settings
    from pr_split.constants import PartitionStrategy
    from pr_split.planner.client import plan_split

    diff = _new_file("uv.lock", [f"line {i}" for i in range(6)]) + _new_file(
        "pyproject.toml", ["[project]", 'name = "x"']
    )
    # A limit of 5 puts the two files in separate groups; they ship together.
    settings = Settings(partition_strategy=PartitionStrategy.GRAPH, max_loc=5, min_loc=1)

    groups = plan_split(parse_diff(diff), settings)

    assert len(groups) == 1
    assert groups[0].title == "Add pyproject.toml and 1 more file"
    assert groups[0].description == "graph partition over 2 file(s): pyproject.toml, uv.lock"
