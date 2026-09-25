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
    _drop_empty_groups,
    _handle_loc_bound_warnings,
    _interactive_edit,
    _move_assignment,
    _present_plan,
    _resolve_fork_ref,
    _send_webhook,
    _show_group_detail,
    _validate_inputs,
    app,
)
from pr_split.constants import AssignmentType, Priority
from pr_split.diff_ops.parser import parse_diff
from pr_split.exceptions import GitOperationError, PRSplitError
from pr_split.graph import PlanDAG
from pr_split.schemas import (
    BranchRecord,
    GitState,
    Group,
    GroupAssignment,
    PlanFile,
    PRRecord,
    SplitPlan,
)
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

    @patch("pr_split.cli.check_gh_stack", return_value=False)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_stacked_requires_gh_stack(
        self,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_stack: MagicMock,
    ) -> None:
        with pytest.raises(typer.Exit):
            _validate_inputs("feature", "main", stacked=True)

    @patch("pr_split.cli.check_gh_stack", side_effect=GitOperationError("auth rejected"))
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_stacked_gh_failure_reports_real_error(
        self,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_stack: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(typer.Exit):
            _validate_inputs("feature", "main", stacked=True)
        out = capsys.readouterr().out
        assert "auth rejected" in out
        assert "gh extension install" not in out

    @patch("pr_split.cli.check_gh_stack", return_value=False)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_stacked_dry_run_skips_gh_stack_check(
        self,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_stack: MagicMock,
    ) -> None:
        _validate_inputs("feature", "main", dry_run=True, stacked=True)
        mock_stack.assert_not_called()

    @patch("pr_split.cli.check_gh_stack", return_value=False)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_unstacked_skips_gh_stack_check(
        self,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_stack: MagicMock,
    ) -> None:
        _validate_inputs("feature", "main")
        mock_stack.assert_not_called()

    @patch("pr_split.cli.branch_exists", side_effect=[False])
    def test_dev_branch_missing(self, mock_be: MagicMock) -> None:
        with pytest.raises(typer.Exit):
            _validate_inputs("no-such-branch", "main")

    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists")
    def test_remote_tracking_base_is_rejected(
        self, mock_be: MagicMock, mock_auth: MagicMock, mock_clean: MagicMock
    ) -> None:
        from pr_split.cli import console

        # origin/main resolves as a ref but is not a local branch head.
        mock_be.side_effect = lambda ref: ref != "refs/heads/origin/main"
        with (
            patch("pr_split.cli._remote_names", return_value={"origin"}),
            console.capture() as capture,
            pytest.raises(typer.Exit),
        ):
            _validate_inputs("feature", "origin/main")
        out = " ".join(capture.get().split())
        assert "Base 'origin/main' is not a local branch" in out
        assert "for example 'main'" in out
        mock_clean.assert_not_called()

    @patch("pr_split.cli._remote_names", return_value={"origin"})
    @patch("pr_split.cli.branch_exists")
    def test_tag_or_sha_base_gets_no_bogus_suggestion(
        self, mock_be: MagicMock, mock_remotes: MagicMock
    ) -> None:
        from pr_split.cli import console

        mock_be.side_effect = lambda ref: not ref.startswith("refs/heads/")
        for base in ("v1", "abc123", "feature/topic"):
            with console.capture() as capture, pytest.raises(typer.Exit):
                _validate_inputs("feature", base)
            out = " ".join(capture.get().split())
            assert f"Base '{base}' is not a local branch" in out
            assert "for example" not in out

    @patch("pr_split.cli._remote_names", return_value={"origin"})
    @patch("pr_split.cli.branch_exists")
    def test_full_remote_ref_suggests_the_branch_name(
        self, mock_be: MagicMock, mock_remotes: MagicMock
    ) -> None:
        from pr_split.cli import console

        mock_be.side_effect = lambda ref: not ref.startswith("refs/heads/")
        with console.capture() as capture, pytest.raises(typer.Exit):
            _validate_inputs("feature", "refs/remotes/origin/main")
        assert "for example 'main'" in " ".join(capture.get().split())

    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_local_base_checks_the_branch_head(
        self, mock_be: MagicMock, mock_auth: MagicMock, mock_clean: MagicMock
    ) -> None:
        _validate_inputs("feature", "main")
        assert ("refs/heads/main",) in [c.args for c in mock_be.call_args_list]

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
# _interactive_edit
# ---------------------------------------------------------------------------
class TestInteractiveEditRecomputesLoc:
    @patch("pr_split.cli.recompute_estimated_loc")
    @patch("pr_split.cli._move_assignment", return_value=True)
    @patch("pr_split.cli.typer.prompt", side_effect=["move a.py:0 pr-1 pr-2", "done"])
    def test_successful_move_recomputes_loc(
        self, mock_prompt: MagicMock, mock_move: MagicMock, mock_recompute: MagicMock
    ) -> None:
        groups = [_group("pr-1", "a", files=["a.py"]), _group("pr-2", "b")]
        parsed = MagicMock()
        _interactive_edit(groups, parsed)
        mock_recompute.assert_called_once_with(groups, parsed)

    @patch("pr_split.cli.recompute_estimated_loc")
    @patch("pr_split.cli._move_assignment", return_value=False)
    @patch("pr_split.cli.typer.prompt", side_effect=["move a.py:9 pr-1 pr-2", "done"])
    def test_failed_move_does_not_recompute(
        self, mock_prompt: MagicMock, mock_move: MagicMock, mock_recompute: MagicMock
    ) -> None:
        _interactive_edit([_group("pr-1", "a", files=["a.py"])], MagicMock())
        mock_recompute.assert_not_called()

    @patch("pr_split.cli.typer.prompt", side_effect=["move a.py:0 pr-1 pr-2", "done"])
    def test_destination_keeps_whole_file_loc_after_move(self, mock_prompt: MagicMock) -> None:
        diff = (
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
            "@@ -1,1 +1,3 @@\n x\n+a1\n+a2\n"
            "@@ -20,1 +22,2 @@\n y\n+a3\n"
            "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n"
            "@@ -1,2 +1,4 @@\n-b0\n+b1\n+b2\n+b3\n z\n"
        )
        parsed = parse_diff(diff)
        groups = [
            Group(
                id="pr-1",
                title="a",
                description="a",
                assignments=[
                    GroupAssignment(
                        file_path="a.py",
                        assignment_type=AssignmentType.WHOLE_FILE,
                        hunk_indices=[0, 1],
                    )
                ],
                estimated_loc=3,
            ),
            Group(
                id="pr-2",
                title="b",
                description="b",
                assignments=[
                    GroupAssignment(
                        file_path="b.py",
                        assignment_type=AssignmentType.WHOLE_FILE,
                        hunk_indices=[],
                    )
                ],
                estimated_loc=4,
            ),
        ]

        result = _interactive_edit(groups, parsed)

        by_id = {g.id: g for g in result}
        assert by_id["pr-1"].estimated_loc == 1
        assert by_id["pr-2"].estimated_loc == 6
        assert by_id["pr-2"].estimated_added == 5
        assert by_id["pr-2"].estimated_removed == 1


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

    @patch("pr_split.cli._resolve_fork_ref", side_effect=GitOperationError("PR #999 not found"))
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=False)
    def test_split_fork_ref_error_is_reported_not_raised(
        self,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_resolve: MagicMock,
    ) -> None:
        result = runner.invoke(app, ["split", "#999", "--dry-run"])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "PR #999 not found" in result.output
        assert "Traceback" not in result.output

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

    @patch("pr_split.cli.commit_exists", return_value=False)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_execute_unknown_merge_base_sha_is_a_clean_error(
        self, mock_pe: MagicMock, mock_load: MagicMock, mock_commit: MagicMock
    ) -> None:
        mock_plan_file = MagicMock()
        mock_plan_file.git_state.branches = []
        mock_plan_file.git_state.prs = []
        mock_plan_file.plan.raw_diff = "some diff"
        mock_plan_file.plan.merge_base_sha = "0123456789abcdef"
        mock_load.return_value = mock_plan_file
        result = runner.invoke(app, ["execute"])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "merge base 0123456789abcdef is not in this repository" in result.output
        mock_commit.assert_called_once_with("0123456789abcdef")

    @patch(
        "pr_split.cli._create_branches_and_commits",
        side_effect=PRSplitError("3 branch(es) failed"),
    )
    @patch("pr_split.cli.typer.confirm", return_value=True)
    @patch("pr_split.cli.validate_coverage")
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_execute_branch_creation_failure_is_a_clean_error(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_clean: MagicMock,
        mock_auth: MagicMock,
        mock_commit: MagicMock,
        mock_parse: MagicMock,
        mock_validate: MagicMock,
        mock_confirm: MagicMock,
        mock_create: MagicMock,
    ) -> None:
        mock_plan_file = MagicMock()
        mock_plan_file.git_state.branches = []
        mock_plan_file.git_state.prs = []
        mock_plan_file.plan.raw_diff = "some diff"
        mock_plan_file.plan.merge_base_sha = "abc123"
        mock_plan_file.plan.stacked = False
        mock_plan_file.plan.dev_branch_arg = "feature"
        mock_plan_file.plan.dev_branch = "feature"
        mock_plan_file.plan.base_branch = "main"
        mock_plan_file.plan.groups = [_group("pr-1", "t", files=["a.py"])]
        mock_load.return_value = mock_plan_file
        result = runner.invoke(app, ["execute"])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "3 branch(es) failed" in result.output

    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli._remote_names", return_value={"origin"})
    @patch("pr_split.cli._create_branches_and_commits")
    @patch("pr_split.cli.typer.confirm", return_value=True)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.branch_exists")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_execute_rejects_a_remote_tracking_base_from_an_old_plan(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_clean: MagicMock,
        mock_confirm: MagicMock,
        mock_create: MagicMock,
        mock_remotes: MagicMock,
        mock_commit_exists: MagicMock,
    ) -> None:
        mock_be.side_effect = lambda ref: ref != "refs/heads/origin/main"
        mock_plan_file = MagicMock()
        mock_plan_file.git_state.branches = []
        mock_plan_file.git_state.prs = []
        mock_plan_file.plan.raw_diff = "some diff"
        mock_plan_file.plan.merge_base_sha = "abc123"
        mock_plan_file.plan.base_branch = "origin/main"
        mock_load.return_value = mock_plan_file
        result = runner.invoke(app, ["execute"])
        assert result.exit_code == 1
        assert "Base 'origin/main' is not a local branch" in " ".join(result.output.split())
        mock_confirm.assert_not_called()
        mock_create.assert_not_called()

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

    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli._create_branches_and_commits")
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_execute_rejects_saved_plan_with_binary_files(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_clean: MagicMock,
        mock_auth: MagicMock,
        mock_create: MagicMock,
        mock_commit_exists: MagicMock,
    ) -> None:
        mock_plan_file = MagicMock()
        mock_plan_file.git_state.branches = []
        mock_plan_file.git_state.prs = []
        plan = mock_plan_file.plan
        plan.raw_diff = (
            "diff --git a/img.png b/img.png\n"
            "index 1111111..2222222 100644\n"
            "Binary files a/img.png and b/img.png differ\n"
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        )
        plan.merge_base_sha = "abc123"
        plan.base_branch = "main"
        plan.stacked = False
        plan.groups = [_group("pr-1", "a", files=["a.py"])]
        mock_load.return_value = mock_plan_file
        result = runner.invoke(app, ["execute"])
        assert result.exit_code == 1
        assert "binary files" in result.output
        assert "img.png" in result.output
        mock_create.assert_not_called()

    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli._create_branches_and_commits")
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_execute_rejects_unknown_dependency_before_branch_creation(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_clean: MagicMock,
        mock_auth: MagicMock,
        mock_create: MagicMock,
        mock_commit_exists: MagicMock,
    ) -> None:
        mock_plan_file = MagicMock()
        mock_plan_file.git_state.branches = []
        mock_plan_file.git_state.prs = []
        plan = mock_plan_file.plan
        plan.raw_diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        plan.merge_base_sha = "abc123"
        plan.base_branch = "main"
        plan.stacked = False
        plan.groups = [_group("pr-1", "a", files=["a.py"]), _group("pr-2", "b", ["ghost"])]
        mock_load.return_value = mock_plan_file
        result = runner.invoke(app, ["execute"])
        assert result.exit_code == 1
        assert "depends on unknown group 'ghost'" in result.output
        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------
class TestStatusCommand:
    @patch("pr_split.cli.plan_exists", return_value=False)
    def test_status_no_plan(self, mock_pe: MagicMock) -> None:
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0

    def _plan_file(self) -> PlanFile:
        plan = SplitPlan(
            dev_branch="feature",
            base_branch="main",
            max_loc=400,
            priority=Priority.ORTHOGONAL,
            groups=[_group("pr-1", "one"), _group("pr-2", "two")],
        )
        git_state = GitState(
            branches=[
                BranchRecord(group_id="pr-1", branch_name="b1", base_branch="main"),
                BranchRecord(group_id="pr-2", branch_name="b2", base_branch="main"),
            ],
            prs=[
                PRRecord(group_id="pr-1", pr_number=7, pr_url="u"),
                PRRecord(group_id="pr-2", pr_number=8, pr_url="u"),
            ],
        )
        return PlanFile(plan=plan, git_state=git_state)

    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_unfetchable_state_is_unknown_not_open(
        self, mock_pe: MagicMock, mock_load: MagicMock, mock_state: MagicMock
    ) -> None:
        mock_load.return_value = self._plan_file()
        mock_state.side_effect = lambda n: {} if n == 7 else {"state": "MERGED"}

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "UNKNOWN" in result.output
        assert "MERGED" in result.output
        assert "OPEN" not in result.output
        assert "Could not fetch live state for 1 PR(s): #7" in result.output

    @patch(
        "pr_split.cli.get_pr_state", return_value={"state": "OPEN", "reviewDecision": "APPROVED"}
    )
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_live_state_and_review_are_shown(
        self, mock_pe: MagicMock, mock_load: MagicMock, mock_state: MagicMock
    ) -> None:
        mock_load.return_value = self._plan_file()
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Approved" in result.output
        assert "Could not fetch" not in result.output

    @patch("pr_split.cli.plan_exists", return_value=False)
    def test_clean_no_plan(self, mock_pe: MagicMock) -> None:
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0


TWO_HUNK_DIFF = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,4 @@
 x
+one
 y
 z
@@ -20,3 +21,5 @@
 p
+two
+three
 q
 r
"""


class TestDropEmptyGroups:
    def test_no_empty_groups_returns_same_list(self) -> None:
        groups = [_group("pr-1", "a", files=["a.py"]), _group("pr-2", "b", files=["b.py"])]
        assert _drop_empty_groups(groups, {}, {}) is groups

    def test_empty_group_is_dropped_and_dependants_inherit_its_parents(self) -> None:
        root = _group("pr-1", "root", files=["a.py"])
        emptied = _group("pr-2", "emptied", depends_on=["pr-1"])
        leaf = _group("pr-3", "leaf", depends_on=["pr-2"], files=["c.py"])
        from pr_split.cli import console

        with console.capture() as capture:
            result = _drop_empty_groups([root, emptied, leaf], {}, {})

        assert [g.id for g in result] == ["pr-1", "pr-3"]
        assert result[1].depends_on == ["pr-1"]
        assert "Dropped empty group(s) after editing: pr-2" in capture.get()

    def test_dropping_a_root_leaves_dependants_as_roots(self) -> None:
        emptied = _group("pr-1", "emptied")
        leaf = _group("pr-2", "leaf", depends_on=["pr-1"], files=["b.py"])
        result = _drop_empty_groups([emptied, leaf], {}, {})
        assert [g.id for g in result] == ["pr-2"]
        assert result[0].depends_on == []

    def test_chain_of_empty_groups_collapses_transitively(self) -> None:
        root = _group("pr-1", "root", files=["a.py"])
        e1 = _group("pr-2", "e1", depends_on=["pr-1"])
        e2 = _group("pr-3", "e2", depends_on=["pr-2"])
        leaf = _group("pr-4", "leaf", depends_on=["pr-3"], files=["d.py"])
        result = _drop_empty_groups([root, e1, e2, leaf], {}, {})
        assert [g.id for g in result] == ["pr-1", "pr-4"]
        assert result[1].depends_on == ["pr-1"]

    def test_chain_with_direct_edge_does_not_duplicate(self) -> None:
        root = _group("pr-1", "root", files=["a.py"])
        e1 = _group("pr-2", "e1", depends_on=["pr-1"])
        e2 = _group("pr-3", "e2", depends_on=["pr-2"])
        leaf = _group("pr-4", "leaf", depends_on=["pr-3", "pr-1"], files=["d.py"])
        result = _drop_empty_groups([root, e1, e2, leaf], {}, {})
        assert result[1].depends_on == ["pr-1"]

    def test_diamond_through_dropped_groups(self) -> None:
        a = _group("pr-1", "a", files=["a.py"])
        b = _group("pr-2", "b", files=["b.py"])
        e1 = _group("pr-3", "e1", depends_on=["pr-1"])
        e2 = _group("pr-4", "e2", depends_on=["pr-2"])
        e3 = _group("pr-5", "e3", depends_on=["pr-3", "pr-4"])
        leaf = _group("pr-6", "leaf", depends_on=["pr-5"], files=["f.py"])
        result = _drop_empty_groups([a, b, e1, e2, e3, leaf], {}, {})
        assert [g.id for g in result] == ["pr-1", "pr-2", "pr-6"]
        assert result[2].depends_on == ["pr-1", "pr-2"]

    def test_dependants_of_a_dropped_group_depend_on_its_hunks_new_owner(self) -> None:
        # E's only hunk moved to the unrelated group D; C was planned on top of
        # E, so it must now build on D or its stacked branch misses that hunk.
        partial = AssignmentType.PARTIAL_HUNKS
        d = _group("pr-1", "d")
        d.assignments = [
            GroupAssignment(file_path="a.py", assignment_type=partial, hunk_indices=[0, 1])
        ]
        e = _group("pr-2", "e")
        c = _group("pr-3", "c", depends_on=["pr-2"], files=["c.py"])
        held_before = {"pr-1": {("a.py", 0)}, "pr-2": {("a.py", 1)}, "pr-3": set()}

        result = _drop_empty_groups([d, e, c], held_before, {"a.py": 2})

        assert [g.id for g in result] == ["pr-1", "pr-3"]
        assert result[1].depends_on == ["pr-1"]

    def test_recipient_is_added_after_inherited_ancestors(self) -> None:
        partial = AssignmentType.PARTIAL_HUNKS
        root = _group("pr-1", "root", files=["r.py"])
        d = _group("pr-2", "d")
        d.assignments = [
            GroupAssignment(file_path="a.py", assignment_type=partial, hunk_indices=[0])
        ]
        e = _group("pr-3", "e", depends_on=["pr-1"])
        c = _group("pr-4", "c", depends_on=["pr-3"], files=["c.py"])
        held_before = {"pr-2": set(), "pr-3": {("a.py", 0)}}

        result = _drop_empty_groups([root, d, e, c], held_before, {"a.py": 1})

        assert result[-1].depends_on == ["pr-1", "pr-2"]

    def test_recipient_already_reached_through_an_ancestor_adds_no_edge(self) -> None:
        # pr-3's hunks moved to pr-1; pr-4 already reaches pr-1 through pr-2,
        # so a direct edge would make pr-4 a multi-parent node for no reason.
        partial = AssignmentType.PARTIAL_HUNKS
        root = _group("pr-1", "root")
        root.assignments = [
            GroupAssignment(file_path="a.py", assignment_type=partial, hunk_indices=[0])
        ]
        mid = _group("pr-2", "mid", depends_on=["pr-1"], files=["b.py"])
        emptied = _group("pr-3", "emptied", depends_on=["pr-2"])
        leaf = _group("pr-4", "leaf", depends_on=["pr-3"], files=["d.py"])
        held_before = {"pr-1": set(), "pr-3": {("a.py", 0)}}

        result = _drop_empty_groups([root, mid, emptied, leaf], held_before, {"a.py": 1})

        assert result[-1].depends_on == ["pr-2"]

    def test_move_to_a_downstream_group_is_refused(self) -> None:
        # D already builds on C, so it cannot become C's parent; accepting the
        # edit would build C's stacked branch without the hunk it inherited.
        partial = AssignmentType.PARTIAL_HUNKS
        e = _group("pr-1", "e")
        c = _group("pr-2", "c", depends_on=["pr-1"], files=["c.py"])
        d = _group("pr-3", "d", depends_on=["pr-2"])
        d.assignments = [
            GroupAssignment(file_path="a.py", assignment_type=partial, hunk_indices=[0])
        ]
        held_before = {"pr-1": {("a.py", 0)}, "pr-2": set(), "pr-3": set()}
        from pr_split.cli import console

        with console.capture() as capture, pytest.raises(typer.Exit):
            _drop_empty_groups([e, c, d], held_before, {"a.py": 1})

        assert "Cannot drop emptied group 'pr-1'" in " ".join(capture.get().split())

    def test_crossed_moves_are_refused(self) -> None:
        # C depended on E1 whose hunk moved to D; D depended on E2 whose hunk
        # moved to C. Both edges cannot hold, so the edit is refused.
        partial = AssignmentType.PARTIAL_HUNKS
        e1 = _group("pr-1", "e1")
        e2 = _group("pr-2", "e2")
        c = _group("pr-3", "c", depends_on=["pr-1"])
        c.assignments = [
            GroupAssignment(file_path="a.py", assignment_type=partial, hunk_indices=[1])
        ]
        d = _group("pr-4", "d", depends_on=["pr-2"])
        d.assignments = [
            GroupAssignment(file_path="a.py", assignment_type=partial, hunk_indices=[0])
        ]
        held_before = {"pr-1": {("a.py", 0)}, "pr-2": {("a.py", 1)}}

        with pytest.raises(typer.Exit):
            _drop_empty_groups([e1, e2, c, d], held_before, {"a.py": 2})


class TestEditorEmptiedGroupEndToEnd:
    @patch("pr_split.cli.typer.prompt")
    def test_moving_the_last_hunk_out_yields_a_valid_one_group_plan(
        self, mock_prompt: MagicMock
    ) -> None:
        from pr_split.cli import _drop_empty_groups, _held_hunks, _interactive_edit
        from pr_split.diff_ops.parser import parse_diff
        from pr_split.planner.validator import validate_plan

        parsed = parse_diff(TWO_HUNK_DIFF)
        g1 = _group("pr-1", "first", added=1, removed=0)
        g1.assignments = [
            GroupAssignment(
                file_path="a.py", assignment_type=AssignmentType.PARTIAL_HUNKS, hunk_indices=[0]
            )
        ]
        g2 = _group("pr-2", "second", depends_on=["pr-1"], added=2, removed=0)
        g2.assignments = [
            GroupAssignment(
                file_path="a.py", assignment_type=AssignmentType.PARTIAL_HUNKS, hunk_indices=[1]
            )
        ]
        mock_prompt.side_effect = ["move a.py:1 pr-2 pr-1", "done"]

        hunk_counts = {pf.path: len(pf) for pf in parsed.patch_set}
        held_before = _held_hunks([g1, g2], hunk_counts)
        edited = _interactive_edit([g1, g2], parsed)
        groups = _drop_empty_groups(edited, held_before, hunk_counts)

        assert [g.id for g in groups] == ["pr-1"]
        assert groups[0].assignments[0].hunk_indices == [0, 1]
        assert groups[0].estimated_loc == 3
        assert validate_plan(groups, parsed, PlanDAG(groups), max_loc=400) == []
