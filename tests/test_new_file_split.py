from __future__ import annotations

import pytest

from pr_split.config import Settings
from pr_split.constants import AssignmentType, PartitionStrategy
from pr_split.diff_ops import materialize_group_files, merge_chain_assignments
from pr_split.diff_ops.new_file_split import block_starts, cut_points
from pr_split.diff_ops.parser import parse_diff
from pr_split.exceptions import PlanValidationError
from pr_split.graph import PlanDAG
from pr_split.planner.client import plan_split
from pr_split.planner.new_file_pieces import link_new_file_pieces
from pr_split.planner.validator import validate_new_file_pieces, validate_plan
from pr_split.schemas import Group, GroupAssignment, SplitPlan


def _new_file_diff(path: str, lines: list[str], *, trailing_newline: bool = True) -> str:
    body = "".join(f"+{line}\n" for line in lines)
    if not trailing_newline:
        body += "\\ No newline at end of file\n"
    return (
        f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n{body}"
    )


def _python_module(functions: int, body_lines: int) -> list[str]:
    lines = ['"""A generated module."""', "", "import os", ""]
    for n in range(functions):
        lines += ["", f"def func_{n}(value):", '    """Docstring."""']
        lines += [f"    value = value + {i}" for i in range(body_lines)]
        lines += ["", "    return os.fspath(str(value))"]
    return lines


_MODIFIED = """\
diff --git a/old.py b/old.py
--- a/old.py
+++ b/old.py
@@ -1,2 +1,3 @@
 a = 1
+b = 2
 c = 3
"""


class TestCutPoints:
    def test_packs_whole_blocks_up_to_the_limit(self) -> None:
        values = ["a = 1", "", "def f():", "    x", "", "def g():", "    y", "", "def h():"]
        # Blocks start at 0, 2, 5 and 8.
        assert cut_points(values, 5) == [5]
        assert cut_points(values, 4) == [2, 5]

    def test_block_larger_than_the_limit_stays_whole(self) -> None:
        values = ["class Big:", *[f"    x{i} = {i}" for i in range(20)], "", "y = 1"]
        assert cut_points(values, 5) == [22]

    def test_never_cuts_inside_indented_code(self) -> None:
        values = ["def f():", "    a = 1", "", "    b = 2", "", "    return a + b"]
        assert cut_points(values, 2) == []

    @pytest.mark.parametrize("line", ["}", "} else {", ")", "else:", "except ValueError:"])
    def test_continuation_lines_do_not_start_a_block(self, line: str) -> None:
        values = ["x", "y", "", line, "z"]
        assert cut_points(values, 2) == []

    def test_comment_above_a_definition_stays_with_it(self) -> None:
        values = ["a = 1", "b = 2", "", "# explains f", "@decorator", "def f():", "    pass"]
        assert cut_points(values, 4) == [3]


class TestBlockStarts:
    def test_python_docstring_paragraphs_are_not_cut(self) -> None:
        values = [
            '"""Module doc.\n',
            "\n",
            "Second paragraph at column 0.\n",
            '"""\n',
            "\n",
            "X = 1\n",
        ]
        assert block_starts(values, "m.py") == [5]

    def test_python_decorators_and_comments_stay_with_their_definition(self) -> None:
        values = ["import os\n", "\n", "# why\n", "@cache\n", "def f():\n", "    pass\n"]
        assert block_starts(values, "m.py") == [2]

    def test_python_cuts_between_statements_without_blank_lines(self) -> None:
        assert block_starts(["a = 1\n", "b = 2\n", "c = 3\n"], "m.py") == [1, 2]

    def test_invalid_python_falls_back_to_blank_lines(self) -> None:
        assert block_starts(["def f()\n", "\n", "x = 1\n"], "m.py") == [2]

    def test_blank_line_inside_brackets_is_not_a_boundary(self) -> None:
        values = ["const X = [", "  1,", "", "2,", "];", "", "const Y = 2;"]
        assert block_starts(values, "m.js") == [6]

    @pytest.mark.parametrize(
        ("opener", "closer"),
        [("const S = `", "`;"), ("/*", "*/"), ('s = """', '"""'), ("x = '''", "'''")],
    )
    def test_multi_line_strings_and_comments_are_not_boundaries(
        self, opener: str, closer: str
    ) -> None:
        values = [opener, "", "inside", closer, "", "after = 1"]
        assert block_starts(values, "m.txt") == [5]

    def test_brackets_in_strings_and_comments_are_ignored(self) -> None:
        values = ['let s = "(";', "// see (a", "", "fn g() {}"]
        assert block_starts(values, "m.rs") == [3]

    def test_rust_lifetimes_do_not_hide_brackets(self) -> None:
        values = ["fn f<'a>(x: &'a str) -> &'a str {", "    x", "}", "", "fn g() {}"]
        assert block_starts(values, "m.rs") == [4]


class TestParseDiffSplitsNewFiles:
    def test_pieces_reassemble_the_file(self) -> None:
        lines = _python_module(functions=12, body_lines=20)
        parsed = parse_diff(_new_file_diff("pkg/big.py", lines), split_new_files_over=100)

        patch_file = parsed.patch_set[0]
        assert len(patch_file) > 1
        assert all(hunk.added <= 100 for hunk in patch_file)
        assert parsed.new_file_pieces == {"pkg/big.py": len(patch_file)}
        assert sum(hunk.added for hunk in patch_file) == len(lines)
        assert parsed.stats["total_loc"] == len(lines)

        group = _whole("pr-1", "pkg/big.py")
        content = materialize_group_files(parsed, group, "unused")["pkg/big.py"]
        assert content == "".join(f"{line}\n" for line in lines)

    def test_missing_final_newline_is_kept(self) -> None:
        lines = _python_module(functions=6, body_lines=20)
        diff = _new_file_diff("big.py", lines, trailing_newline=False)
        parsed = parse_diff(diff, split_new_files_over=50)

        assert len(parsed.patch_set[0]) > 1
        content = materialize_group_files(parsed, _whole("pr-1", "big.py"), "unused")["big.py"]
        assert content == "\n".join(lines)

    def test_labeled_diff_shows_every_piece(self) -> None:
        lines = _python_module(functions=6, body_lines=20)
        parsed = parse_diff(_new_file_diff("big.py", lines), split_new_files_over=50)
        pieces = len(parsed.patch_set[0])
        for index in range(pieces):
            assert f"[hunk_index={index}]" in parsed.labeled_diff

    def test_small_new_files_and_modified_files_are_untouched(self) -> None:
        diff = _MODIFIED + _new_file_diff("small.py", ["x = 1", "", "y = 2"])
        parsed = parse_diff(diff, split_new_files_over=3)
        assert [len(pf) for pf in parsed.patch_set] == [1, 1]
        assert parsed.new_file_pieces == {}

    def test_off_by_default(self) -> None:
        lines = _python_module(functions=12, body_lines=20)
        parsed = parse_diff(_new_file_diff("big.py", lines))
        assert len(parsed.patch_set[0]) == 1


def _whole(gid: str, path: str) -> Group:
    return Group(
        id=gid,
        title=gid,
        description="",
        assignments=[
            GroupAssignment(
                file_path=path, assignment_type=AssignmentType.WHOLE_FILE, hunk_indices=[0]
            )
        ],
    )


def _pieces(gid: str, path: str, indices: list[int], depends_on: list[str] | None = None) -> Group:
    return Group(
        id=gid,
        title=gid,
        description="",
        depends_on=depends_on or [],
        assignments=[
            GroupAssignment(
                file_path=path,
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=indices,
            )
        ],
    )


class TestStackedGraphSplit:
    def test_every_layer_holds_a_compiling_prefix_under_max_loc(self) -> None:
        lines = _python_module(functions=16, body_lines=25)
        diff = _new_file_diff("pkg/big.py", lines)
        parsed = parse_diff(diff, split_new_files_over=150)
        settings = Settings(partition_strategy=PartitionStrategy.GRAPH, max_loc=150)

        groups = plan_split(parsed, settings)
        dag = PlanDAG(groups)
        assert validate_plan(groups, parsed, dag, settings.max_loc) == []
        assert len(groups) > 1
        assert all(group.estimated_loc <= 150 for group in groups)

        hunk_counts = {pf.path: len(pf) for pf in parsed.patch_set}
        by_id = {g.id: g for g in groups}
        previous = ""
        for gid in dag.topological_order():
            stacked = merge_chain_assignments(
                by_id[gid], [by_id[a] for a in dag.ancestors(gid)], hunk_counts
            )
            content = materialize_group_files(parsed, stacked, "unused")["pkg/big.py"]
            assert content.startswith(previous)
            compile(content, "big.py", "exec")
            previous = content
        assert previous == "".join(f"{line}\n" for line in lines)


class TestPieceOrder:
    @pytest.fixture
    def parsed(self):  # type: ignore[no-untyped-def]
        lines = _python_module(functions=6, body_lines=20)
        parsed = parse_diff(_new_file_diff("big.py", lines), split_new_files_over=50)
        assert len(parsed.patch_set[0]) >= 3
        return parsed

    def test_later_piece_without_the_earlier_one_is_rejected(self, parsed) -> None:  # type: ignore[no-untyped-def]
        groups = [_pieces("pr-1", "big.py", [0]), _pieces("pr-2", "big.py", [1, 2])]
        with pytest.raises(PlanValidationError, match=r"piece 2 of new file 'big\.py'"):
            validate_new_file_pieces(groups, parsed, PlanDAG(groups))

    def test_linking_adds_the_missing_edges(self, parsed) -> None:  # type: ignore[no-untyped-def]
        groups = [
            _pieces("pr-1", "big.py", [2]),
            _pieces("pr-2", "big.py", [0]),
            _pieces("pr-3", "big.py", [1]),
        ]
        link_new_file_pieces(groups, parsed)

        assert groups[2].depends_on == ["pr-2"]
        assert groups[0].depends_on == ["pr-3"]
        validate_new_file_pieces(groups, parsed, PlanDAG(groups))

    def test_edge_that_would_close_a_cycle_is_left_for_validation(self, parsed) -> None:  # type: ignore[no-untyped-def]
        groups = [
            _pieces("pr-1", "big.py", [0], depends_on=["pr-2"]),
            _pieces("pr-2", "big.py", [1, 2]),
        ]
        link_new_file_pieces(groups, parsed)

        assert groups[1].depends_on == []
        with pytest.raises(PlanValidationError, match="does not build on group 'pr-1'"):
            validate_new_file_pieces(groups, parsed, PlanDAG(groups))

    def test_one_group_holding_every_piece_is_fine(self, parsed) -> None:  # type: ignore[no-untyped-def]
        groups = [_whole("pr-1", "big.py")]
        link_new_file_pieces(groups, parsed)
        validate_new_file_pieces(groups, parsed, PlanDAG(groups))
        assert groups[0].depends_on == []


def test_plan_records_the_threshold() -> None:
    plan = SplitPlan(
        dev_branch="d", base_branch="main", max_loc=100, priority="orthogonal", stacked=True
    )
    assert plan.split_new_files_over is None
    restored = SplitPlan.model_validate_json(
        plan.model_copy(update={"split_new_files_over": 100}).model_dump_json()
    )
    assert restored.split_new_files_over == 100


class TestExecuteReparsesPieces:
    def _plan_file(self, groups: list[Group]):  # type: ignore[no-untyped-def]
        from unittest.mock import MagicMock

        lines = _python_module(functions=6, body_lines=20)
        plan_file = MagicMock()
        plan_file.git_state.branches = []
        plan_file.git_state.prs = []
        plan_file.plan = SplitPlan(
            dev_branch="feature",
            base_branch="main",
            max_loc=50,
            priority="orthogonal",
            stacked=True,
            groups=groups,
            merge_base_sha="abc123",
            raw_diff=_new_file_diff("big.py", lines),
            split_new_files_over=50,
        )
        return plan_file

    def _invoke(self, plan_file):  # type: ignore[no-untyped-def]
        from unittest.mock import patch

        from typer.testing import CliRunner

        from pr_split.cli import app

        with (
            patch("pr_split.cli.plan_exists", return_value=True),
            patch("pr_split.cli.load_plan", return_value=plan_file),
            patch("pr_split.cli.branch_exists", return_value=True),
            patch("pr_split.cli.is_worktree_clean", return_value=True),
            patch("pr_split.cli.check_gh_auth", return_value=True),
            patch("pr_split.cli.commit_exists", return_value=True),
            patch("pr_split.cli._require_gh_stack"),
            patch("pr_split.cli._create_branches_and_commits") as create,
        ):
            create.side_effect = RuntimeError("stop after validation")
            result = CliRunner().invoke(app, ["execute"], input="y\n")
        return result, create

    def test_out_of_order_pieces_are_refused_before_any_branch(self) -> None:
        pieces = len(
            parse_diff(self._plan_file([]).plan.raw_diff, split_new_files_over=50).patch_set[0]
        )
        groups = [
            _pieces("pr-1", "big.py", [0]),
            _pieces("pr-2", "big.py", list(range(1, pieces))),
        ]
        result, create = self._invoke(self._plan_file(groups))

        assert result.exit_code == 1
        assert "holds piece 2 of new file 'big.py'" in result.output
        create.assert_not_called()

    def test_pieces_in_stack_order_reach_branch_creation(self) -> None:
        pieces = len(
            parse_diff(self._plan_file([]).plan.raw_diff, split_new_files_over=50).patch_set[0]
        )
        groups = [
            _pieces("pr-1", "big.py", [0]),
            _pieces("pr-2", "big.py", list(range(1, pieces)), depends_on=["pr-1"]),
        ]
        _result, create = self._invoke(self._plan_file(groups))

        create.assert_called_once()
        parsed = create.call_args.args[1]
        assert len(parsed.patch_set[0]) == pieces
