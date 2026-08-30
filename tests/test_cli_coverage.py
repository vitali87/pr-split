"""Tests to improve coverage for pr_split/cli.py.

Covers: _validate_inputs, _handle_loc_bound_warnings, _resolve_fork_ref,
_present_plan, _build_pr_body (template error paths), _send_webhook,
_show_group_detail, _move_assignment, and split command argument validation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from pr_split.cli import (
    _build_pr_body,
    _handle_loc_bound_warnings,
    _move_assignment,
    _present_plan,
    _resolve_fork_ref,
    _send_webhook,
    _show_group_detail,
    _validate_inputs,
    app,
)
from pr_split.constants import AssignmentType
from pr_split.exceptions import PRSplitError
from pr_split.schemas import Group, GroupAssignment
from pr_split.types_defs import ForkPRInfo

runner = CliRunner()


def _group(
    gid: str,
    title: str,
    depends_on: list[str] | None = None,
    files: list[str] | None = None,
    added: int = 10,
    removed: int = 5,
) -> Group:
    assignments = [
        GroupAssignment(file_path=f, assignment_type=AssignmentType.WHOLE_FILE, hunk_indices=[])
        for f in (files or [])
    ]
    return Group(
        id=gid,
        title=title,
        description=f"desc for {gid}",
        depends_on=depends_on or [],
        assignments=assignments,
        estimated_loc=added + removed,
        estimated_added=added,
        estimated_removed=removed,
    )


# ---------------------------------------------------------------------------
# _validate_inputs
# ---------------------------------------------------------------------------
class TestValidateInputs:
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_all_ok_no_exit(
        self, mock_be: MagicMock, mock_auth: MagicMock, mock_clean: MagicMock
    ) -> None:
        _validate_inputs("feature", "main")

    @patch("pr_split.cli.branch_exists", side_effect=[False])
    def test_dev_branch_missing(self, mock_be: MagicMock) -> None:
        with pytest.raises(typer.Exit):
            _validate_inputs("no-such-branch", "main")

    @patch("pr_split.cli.branch_exists", side_effect=[True, False])
    def test_base_branch_missing(self, mock_be: MagicMock) -> None:
        with pytest.raises(typer.Exit):
            _validate_inputs("feature", "no-such-base")

    @patch("pr_split.cli.is_worktree_clean", return_value=False)
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_dirty_worktree(self, mock_be: MagicMock, mock_clean: MagicMock) -> None:
        with pytest.raises(typer.Exit):
            _validate_inputs("feature", "main")

    @patch("pr_split.cli.check_gh_auth", return_value=False)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_gh_auth_failed(
        self, mock_be: MagicMock, mock_clean: MagicMock, mock_auth: MagicMock
    ) -> None:
        with pytest.raises(typer.Exit):
            _validate_inputs("feature", "main")

    @patch("pr_split.cli.check_gh_auth")
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_dry_run_skips_gh_auth(
        self, mock_be: MagicMock, mock_clean: MagicMock, mock_auth: MagicMock
    ) -> None:
        # dry_run=True should not check gh auth
        _validate_inputs("feature", "main", dry_run=True)
        mock_auth.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_loc_bound_warnings
# ---------------------------------------------------------------------------
class TestHandleLocBoundWarningsExtended:
    def test_no_warnings_no_effect(self) -> None:
        # Should not raise even with strict=True when there are no warnings
        _handle_loc_bound_warnings([], strict_loc_bounds=True)

    @patch("pr_split.cli.logger.warning")
    def test_non_strict_logs_multiple(self, mock_warn: MagicMock) -> None:
        _handle_loc_bound_warnings(["w1", "w2"], strict_loc_bounds=False)
        assert mock_warn.call_count == 2

    def test_strict_with_warnings_exits(self) -> None:
        with pytest.raises(typer.Exit):
            _handle_loc_bound_warnings(["too big"], strict_loc_bounds=True)


# ---------------------------------------------------------------------------
# _resolve_fork_ref
# ---------------------------------------------------------------------------
class TestResolveForkRefCoverage:
    def test_plain_branch_returns_none(self) -> None:
        assert _resolve_fork_ref("my-feature") is None

    @patch("pr_split.cli.fetch_fork_pr")
    def test_hash_pr_number(self, mock_fetch: MagicMock) -> None:
        info: ForkPRInfo = {
            "pr_number": 42,
            "local_ref": "refs/pr-split/pr-42",
            "base_branch": "main",
            "author": "Author <a@b.c>",
            "fork_full_name": "user/repo",
        }
        mock_fetch.return_value = info
        result = _resolve_fork_ref("#42")
        mock_fetch.assert_called_once_with(42)
        assert result is not None
        assert result["pr_number"] == 42

    @patch("pr_split.cli.fetch_fork_branch")
    def test_user_colon_branch(self, mock_fetch: MagicMock) -> None:
        info: ForkPRInfo = {
            "pr_number": None,
            "local_ref": "refs/pr-split/fork-user-feat",
            "base_branch": "main",
            "author": "Author <a@b.c>",
            "fork_full_name": "user/repo",
        }
        mock_fetch.return_value = info
        result = _resolve_fork_ref("user:feat")
        mock_fetch.assert_called_once_with("user", "feat")
        assert result is not None

    @patch("pr_split.cli.fetch_fork_pr")
    def test_bare_number_treated_as_pr(self, mock_fetch: MagicMock) -> None:
        info: ForkPRInfo = {
            "pr_number": 7,
            "local_ref": "refs/pr-split/pr-7",
            "base_branch": "main",
            "author": "Author <a@b.c>",
            "fork_full_name": "user/repo",
        }
        mock_fetch.return_value = info
        result = _resolve_fork_ref("7")
        mock_fetch.assert_called_once_with(7)
        assert result is not None


# ---------------------------------------------------------------------------
# _present_plan
# ---------------------------------------------------------------------------
class TestPresentPlan:
    def test_renders_without_error(self) -> None:
        groups = [
            _group("pr-1", "base", files=["a.py"]),
            _group("pr-2", "child", depends_on=["pr-1"], files=["b.py"]),
        ]
        # Should not raise
        _present_plan(groups)

    def test_handles_no_deps(self) -> None:
        groups = [_group("pr-1", "solo", files=["a.py"])]
        _present_plan(groups)


# ---------------------------------------------------------------------------
# _build_pr_body template OSError path
# ---------------------------------------------------------------------------
class TestBuildPrBodyOsError:
    def test_template_os_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        template_file = tmp_path / "template.md"
        template_file.write_text("{description}")

        # Create a mock Path that exists but raises OSError on read_text
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.side_effect = OSError("Permission denied")
        monkeypatch.setattr("pr_split.cli._PR_TEMPLATE_PATH", mock_path)

        group = _group("pr-1", "t", files=["a.py"])
        with pytest.raises(PRSplitError, match="Could not read PR template"):
            _build_pr_body(group, [group])


# ---------------------------------------------------------------------------
# _send_webhook
# ---------------------------------------------------------------------------
class TestSendWebhook:
    @patch("pr_split.cli.urllib.request.urlopen")
    def test_successful_webhook(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        _send_webhook("https://example.com/hook", {"event": "test"})
        mock_urlopen.assert_called_once()

    @patch("pr_split.cli.urllib.request.urlopen", side_effect=Exception("timeout"))
    def test_failed_webhook_logs_warning(self, mock_urlopen: MagicMock) -> None:
        # Should not raise
        _send_webhook("https://example.com/hook", {"event": "test"})


# ---------------------------------------------------------------------------
# _show_group_detail
# ---------------------------------------------------------------------------
class TestShowGroupDetail:
    def test_unknown_group(self) -> None:
        groups = [_group("pr-1", "root")]
        # Should print error but not raise
        _show_group_detail(groups, "nonexistent")

    def test_known_group_renders(self) -> None:
        g = _group("pr-1", "root", files=["a.py"], depends_on=["pr-0"])
        _show_group_detail([g], "pr-1")

    def test_assignment_type_is_printed_not_eaten_as_markup(self) -> None:
        from pr_split.cli import console

        g = _group("pr-1", "Support list[str] [WIP]", files=["a.py"])
        g.description = "uses the [red] tag literally"
        g.assignments.append(
            GroupAssignment(
                file_path="src/[id].py",
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[0, 2],
            )
        )
        with console.capture() as capture:
            _show_group_detail([g], "pr-1")
        out = capture.get()
        assert "pr-1: Support list[str] [WIP]" in out
        assert "Description: uses the [red] tag literally" in out
        assert "a.py [whole_file] hunks: [all]" in out
        assert "src/[id].py [partial_hunks] hunks: [0, 2]" in out

    def test_partial_hunks_rendering(self) -> None:
        g = Group(
            id="pr-1",
            title="partial",
            description="d",
            depends_on=[],
            assignments=[
                GroupAssignment(
                    file_path="x.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0, 2, 4],
                )
            ],
        )
        _show_group_detail([g], "pr-1")


# ---------------------------------------------------------------------------
# _move_assignment
# ---------------------------------------------------------------------------
class TestMoveAssignment:
    def _parsed_diff_with_file(self, path: str, num_hunks: int) -> MagicMock:
        pf = MagicMock()
        pf.path = path
        pf.__len__ = MagicMock(return_value=num_hunks)
        parsed = MagicMock()
        parsed.patch_set = [pf]
        return parsed

    def test_same_src_dst_returns_false(self) -> None:
        g = _group("pr-1", "t", files=["a.py"])
        pd = self._parsed_diff_with_file("a.py", 3)
        result = _move_assignment([g], pd, "a.py", 0, "pr-1", "pr-1")
        assert result is False

    def test_unknown_group_returns_false(self) -> None:
        g = _group("pr-1", "t", files=["a.py"])
        pd = self._parsed_diff_with_file("a.py", 3)
        result = _move_assignment([g], pd, "a.py", 0, "pr-1", "nonexistent")
        assert result is False

    def test_hunk_not_found_returns_false(self) -> None:
        g1 = _group("pr-1", "t", files=["a.py"])
        g2 = _group("pr-2", "t2")
        pd = self._parsed_diff_with_file("a.py", 3)
        # The group has WHOLE_FILE assignment, hunk 99 is out of range
        result = _move_assignment([g1, g2], pd, "a.py", 99, "pr-1", "pr-2")
        assert result is False

    def test_successful_move_whole_file(self) -> None:
        g1 = _group("pr-1", "src", files=["a.py"])
        g2 = _group("pr-2", "dst")
        pd = self._parsed_diff_with_file("a.py", 3)
        result = _move_assignment([g1, g2], pd, "a.py", 1, "pr-1", "pr-2")
        assert result is True
        # Hunk 1 should now be in pr-2
        dst_assignments = [a for a in g2.assignments if a.file_path == "a.py"]
        assert len(dst_assignments) == 1
        assert 1 in dst_assignments[0].hunk_indices

    def test_move_to_existing_partial_assignment(self) -> None:
        g1 = Group(
            id="pr-1",
            title="src",
            description="d",
            depends_on=[],
            assignments=[
                GroupAssignment(
                    file_path="a.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0, 1, 2],
                )
            ],
        )
        g2 = Group(
            id="pr-2",
            title="dst",
            description="d",
            depends_on=[],
            assignments=[
                GroupAssignment(
                    file_path="a.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[3],
                )
            ],
        )
        pd = self._parsed_diff_with_file("a.py", 5)
        result = _move_assignment([g1, g2], pd, "a.py", 1, "pr-1", "pr-2")
        assert result is True
        # Hunk 1 removed from pr-1
        src_a = [a for a in g1.assignments if a.file_path == "a.py"]
        assert 1 not in src_a[0].hunk_indices
        # Hunk 1 added to pr-2
        dst_a = [a for a in g2.assignments if a.file_path == "a.py"]
        assert 1 in dst_a[0].hunk_indices

    def test_move_last_hunk_removes_assignment(self) -> None:
        g1 = Group(
            id="pr-1",
            title="src",
            description="d",
            depends_on=[],
            assignments=[
                GroupAssignment(
                    file_path="a.py",
                    assignment_type=AssignmentType.PARTIAL_HUNKS,
                    hunk_indices=[0],
                )
            ],
        )
        g2 = _group("pr-2", "dst")
        pd = self._parsed_diff_with_file("a.py", 3)
        result = _move_assignment([g1, g2], pd, "a.py", 0, "pr-1", "pr-2")
        assert result is True
        # pr-1 should have no assignments for a.py
        src_a = [a for a in g1.assignments if a.file_path == "a.py"]
        assert len(src_a) == 0


# ---------------------------------------------------------------------------
# split command argument validation
# ---------------------------------------------------------------------------
class TestSplitCommandValidation:
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.extract_diff", return_value="diff --git a/a.py b/a.py\n")
    @patch("pr_split.cli._validate_inputs")
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.plan_exists", return_value=False)
    def test_split_min_loc_ge_max_loc_error(
        self,
        mock_pe: MagicMock,
        mock_be: MagicMock,
        mock_vi: MagicMock,
        mock_extract: MagicMock,
        mock_parse: MagicMock,
    ) -> None:
        """min_loc >= max_loc should cause a validation error."""
        parsed_diff = MagicMock()
        parsed_diff.stats = {
            "total_files": 1,
            "total_added": 10,
            "total_removed": 5,
            "total_loc": 15,
        }
        mock_parse.return_value = parsed_diff
        result = runner.invoke(
            app,
            ["split", "feature", "--dry-run", "--min-loc", "500", "--max-loc", "400"],
            env={"ANTHROPIC_API_KEY": "sk-test"},
        )
        assert result.exit_code != 0

    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli.merge_base", return_value="abc123")
    @patch("pr_split.cli._interactive_edit")
    @patch("pr_split.cli._present_plan")
    @patch("pr_split.cli.validate_plan", return_value=[])
    @patch("pr_split.cli.plan_split")
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.extract_diff", return_value="diff --git a/a.py b/a.py\n")
    @patch("pr_split.cli._validate_inputs")
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.plan_exists", return_value=False)
    def test_split_dry_run_saves_plan(
        self,
        mock_plan_exists: MagicMock,
        mock_branch_exists: MagicMock,
        mock_validate_inputs: MagicMock,
        mock_extract_diff: MagicMock,
        mock_parse_diff: MagicMock,
        mock_plan_split: MagicMock,
        mock_validate_plan: MagicMock,
        mock_present_plan: MagicMock,
        mock_interactive_edit: MagicMock,
        mock_merge_base: MagicMock,
        mock_save_plan: MagicMock,
    ) -> None:
        parsed_diff = MagicMock()
        parsed_diff.stats = {
            "total_files": 1,
            "total_added": 10,
            "total_removed": 5,
            "total_loc": 15,
        }
        mock_parse_diff.return_value = parsed_diff
        group = _group("pr-1", "feat: auth", files=["a.py"])
        mock_plan_split.return_value = [group]
        mock_interactive_edit.return_value = [group]

        result = runner.invoke(
            app,
            ["split", "feature-branch", "--dry-run"],
            env={"ANTHROPIC_API_KEY": "sk-test"},
        )
        assert result.exit_code == 0
        mock_save_plan.assert_called_once()

    @patch("pr_split.cli.check_gh_auth", return_value=False)
    @patch("pr_split.cli.branch_exists", return_value=False)
    def test_split_fork_ref_no_auth(
        self,
        mock_be: MagicMock,
        mock_auth: MagicMock,
    ) -> None:
        result = runner.invoke(app, ["split", "#42", "--dry-run"])
        assert result.exit_code != 0

    @patch("pr_split.cli.is_worktree_clean", return_value=False)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=False)
    def test_split_fork_ref_dirty_worktree(
        self,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
    ) -> None:
        result = runner.invoke(app, ["split", "#42", "--dry-run"])
        assert result.exit_code != 0

    @patch("pr_split.cli._resolve_fork_ref", return_value=None)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=False)
    def test_split_fork_ref_not_found(
        self,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_resolve: MagicMock,
    ) -> None:
        result = runner.invoke(app, ["split", "unknown-ref", "--dry-run"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# execute command
# ---------------------------------------------------------------------------
class TestExecuteCommand:
    @patch("pr_split.cli.plan_exists", return_value=False)
    def test_execute_no_plan(self, mock_pe: MagicMock) -> None:
        result = runner.invoke(app, ["execute"])
        assert result.exit_code != 0

    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_execute_already_has_git_state(self, mock_pe: MagicMock, mock_load: MagicMock) -> None:
        mock_plan_file = MagicMock()
        mock_plan_file.git_state.branches = [MagicMock()]
        mock_plan_file.git_state.prs = []
        mock_load.return_value = mock_plan_file
        result = runner.invoke(app, ["execute"])
        assert result.exit_code != 0

    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_execute_missing_raw_diff(self, mock_pe: MagicMock, mock_load: MagicMock) -> None:
        mock_plan_file = MagicMock()
        mock_plan_file.git_state.branches = []
        mock_plan_file.git_state.prs = []
        mock_plan_file.plan.raw_diff = ""
        mock_load.return_value = mock_plan_file
        result = runner.invoke(app, ["execute"])
        assert result.exit_code != 0

    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_execute_missing_merge_base_sha(
        self, mock_pe: MagicMock, mock_load: MagicMock
    ) -> None:
        mock_plan_file = MagicMock()
        mock_plan_file.git_state.branches = []
        mock_plan_file.git_state.prs = []
        mock_plan_file.plan.raw_diff = "some diff"
        mock_plan_file.plan.merge_base_sha = ""
        mock_load.return_value = mock_plan_file
        result = runner.invoke(app, ["execute"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------
class TestStatusCommand:
    @patch("pr_split.cli.plan_exists", return_value=False)
    def test_status_no_plan(self, mock_pe: MagicMock) -> None:
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0

    @patch("pr_split.cli.plan_exists", return_value=False)
    def test_clean_no_plan(self, mock_pe: MagicMock) -> None:
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0


class TestRichEscapingOfPlanText:
    def test_present_plan_keeps_brackets_in_title_and_paths(self) -> None:
        from pr_split.cli import _present_plan, console

        g = _group("pr-1", "Add [core] typing", files=["src/[id].py"])
        # _render_dag captures on the same console, which would reset an
        # enclosing capture; it has its own test below.
        with patch("pr_split.cli._render_dag", return_value=""), console.capture() as capture:
            _present_plan([g])
        out = capture.get()
        assert "Add [core] typing" in out
        assert "src/[id].py" in out

    def test_render_dag_keeps_brackets(self) -> None:
        from pr_split.cli import _render_dag

        root = _group("pr-1", "Root [v2]")
        child = _group("pr-2", "Child [x]", depends_on=["pr-1"])
        out = _render_dag([root, child])
        assert "pr-1: Root [v2]" in out
        assert "pr-2: Child [x] (depends on: pr-1)" in out

    def test_move_messages_keep_bracketed_paths(self) -> None:
        from pr_split.cli import console

        g1 = _group("pr-1", "src", files=["src/[id].py"])
        g2 = _group("pr-2", "dst")
        pf = MagicMock()
        pf.path = "src/[id].py"
        pf.__len__ = MagicMock(return_value=2)
        parsed = MagicMock()
        parsed.patch_set = [pf]
        with console.capture() as capture:
            assert _move_assignment([g1, g2], parsed, "src/[id].py", 1, "pr-1", "pr-2")
            assert not _move_assignment([g1, g2], parsed, "src/[id].py", 9, "pr-1", "pr-2")
            assert not _move_assignment([g1, g2], parsed, "a.py", 0, "[pr-1]", "pr-2")
            _show_group_detail([g1], "[pr-9]")
        out = capture.get()
        assert "Moved src/[id].py:1 from pr-1 to pr-2" in out
        assert "Hunk src/[id].py:9 not found in pr-1." in out
        assert "Group '[pr-1]' or 'pr-2' not found." in out
        assert "Group '[pr-9]' not found." in out


class TestRichEscapingOfStatusAndSummaries:
    @patch("pr_split.cli.get_pr_state", return_value={"state": "OPEN", "reviewDecision": None})
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_status_table_keeps_bracketed_title(
        self, mock_pe: MagicMock, mock_load: MagicMock, mock_state: MagicMock
    ) -> None:
        from pr_split.constants import Priority
        from pr_split.schemas import BranchRecord, GitState, PlanFile, PRRecord, SplitPlan

        plan = SplitPlan(
            dev_branch="feature",
            base_branch="main",
            max_loc=400,
            priority=Priority.ORTHOGONAL,
            groups=[_group("pr-1", "Add [core] typing")],
        )
        git_state = GitState(
            branches=[BranchRecord(group_id="pr-1", branch_name="b[1]", base_branch="main")],
            prs=[PRRecord(group_id="pr-1", pr_number=7, pr_url="u")],
        )
        mock_load.return_value = PlanFile(plan=plan, git_state=git_state)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Add [core] typing" in result.output
        assert "b[1]" in result.output

    def test_error_echo_keeps_bracketed_path(self) -> None:
        from pr_split.cli import _handle_loc_bound_warnings, console

        # The strict path prints each warning in red (the non-strict path only logs).
        with console.capture() as capture, pytest.raises(typer.Exit):
            _handle_loc_bound_warnings(
                ["Group 'pr-1' touches src/[id].py"], strict_loc_bounds=True
            )
        assert "src/[id].py" in capture.get()

    def test_editor_same_group_message_keeps_bracketed_id(self) -> None:
        from pr_split.cli import console

        with console.capture() as capture:
            assert not _move_assignment([], MagicMock(), "a.py", 0, "[pr-1]", "[pr-1]")
        assert "Source and destination are the same ('[pr-1]')" in capture.get()


class TestRichEscapingOfMergeSummary:
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_summary_lines_keep_bracketed_group_ids(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
    ) -> None:
        from pr_split.constants import Priority
        from pr_split.exceptions import GitOperationError
        from pr_split.schemas import GitState, PlanFile, PRRecord, SplitPlan

        groups = [_group("[pr-1]", "a"), _group("[pr-2]", "b"), _group("[pr-3]", "c")]
        plan = SplitPlan(
            dev_branch="feature",
            base_branch="main",
            max_loc=400,
            priority=Priority.ORTHOGONAL,
            groups=groups,
        )
        prs = [PRRecord(group_id=g.id, pr_number=i + 1, pr_url="u") for i, g in enumerate(groups)]
        mock_load.return_value = PlanFile(plan=plan, git_state=GitState(prs=prs))
        states = {
            1: {"state": "OPEN", "isDraft": False, "reviewDecision": None},
            2: {"state": "OPEN", "isDraft": True, "reviewDecision": None},
            3: {"state": "OPEN", "isDraft": False, "reviewDecision": None},
        }
        mock_state.side_effect = lambda n: states[n]

        def merge(n: int, *, auto: bool = False) -> None:
            if n == 3:
                raise GitOperationError("conflict")

        mock_merge.side_effect = merge

        result = runner.invoke(app, ["merge"])

        flat = " ".join(result.output.split())
        assert "Merged (1): [pr-1]" in flat
        assert "Skipped (1): [pr-2] (draft)" in flat
        assert "Failed (1): [pr-3]" in flat


class TestRichEscapingOfEmptyGroupsError:
    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli.merge_base", return_value="abc123")
    @patch("pr_split.cli._interactive_edit")
    @patch("pr_split.cli._present_plan")
    @patch("pr_split.cli.validate_plan", return_value=[])
    @patch("pr_split.cli.plan_split")
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.extract_diff", return_value="diff --git a/a.py b/a.py\n")
    @patch("pr_split.cli._validate_inputs")
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_empty_groups_message_keeps_bracketed_ids(
        self,
        mock_branch_exists: MagicMock,
        mock_validate_inputs: MagicMock,
        mock_extract_diff: MagicMock,
        mock_parse_diff: MagicMock,
        mock_plan_split: MagicMock,
        mock_validate_plan: MagicMock,
        mock_present_plan: MagicMock,
        mock_interactive_edit: MagicMock,
        mock_merge_base: MagicMock,
        mock_save_plan: MagicMock,
    ) -> None:
        parsed_diff = MagicMock()
        parsed_diff.stats = {
            "total_files": 1,
            "total_added": 1,
            "total_removed": 0,
            "total_loc": 1,
        }
        mock_parse_diff.return_value = parsed_diff
        mock_plan_split.return_value = [_group("[pr-1]", "a", files=["a.py"])]
        mock_interactive_edit.return_value = [_group("[pr-1]", "emptied")]

        result = runner.invoke(
            app, ["split", "feature-branch", "--dry-run"], env={"ANTHROPIC_API_KEY": "sk-test"}
        )

        assert result.exit_code == 1
        assert "Groups ['[pr-1]'] are empty after editing." in " ".join(result.output.split())
