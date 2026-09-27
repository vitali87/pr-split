"""Tests to improve coverage for pr_split/cli.py.

Covers: _validate_inputs, _handle_loc_bound_warnings, _resolve_fork_ref,
_present_plan, _build_pr_body (template error paths), _send_webhook,
_show_group_detail, _move_assignment, and split command argument validation.
"""

from __future__ import annotations

import subprocess
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
from pr_split.exceptions import GitOperationError, PRCreationError, PRSplitError
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
        monkeypatch.setattr("pr_split.cli._pr_template_path", lambda: mock_path)

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
    def test_execute_already_has_prs(self, mock_pe: MagicMock, mock_load: MagicMock) -> None:
        mock_plan_file = MagicMock()
        mock_plan_file.git_state.branches = [MagicMock()]
        mock_plan_file.git_state.prs = [MagicMock()]
        mock_load.return_value = mock_plan_file
        result = runner.invoke(app, ["execute"])
        assert result.exit_code != 0
        assert "already has PRs" in result.output

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
        mock_plan_file.plan.max_loc = 400
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


class TestExecuteRetriesAfterFailedPush:
    """Branches without PRs mean a failed push; execute must retry, not refuse.

    Forcing 'clean' + re-plan would delete the plan and require paying for
    LLM planning again after a transient network failure.
    """

    @patch("pr_split.cli.typer.confirm", return_value=True)
    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli._push_and_create_prs", return_value=[])
    @patch("pr_split.cli._create_branches_and_commits", return_value=[])
    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_branches_without_prs_is_retried(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_parse: MagicMock,
        mock_commit: MagicMock,
        mock_create: MagicMock,
        mock_push: MagicMock,
        mock_save: MagicMock,
        mock_confirm: MagicMock,
    ) -> None:
        from pr_split.constants import Priority
        from pr_split.schemas import BranchRecord, GitState, PlanFile, SplitPlan

        plan_file = PlanFile(
            plan=SplitPlan(
                dev_branch="feature-branch",
                base_branch="main",
                max_loc=400,
                priority=Priority.ORTHOGONAL,
                merge_base_sha="0123456789abcdef",
                raw_diff="some diff",
                groups=[_group("pr-1", "feat: auth", files=["a.py"])],
            ),
            git_state=GitState(
                branches=[
                    BranchRecord(
                        group_id="pr-1",
                        branch_name="pr-split/feature-branch/pr-1",
                        base_branch="main",
                        commit_sha="abc123",
                    )
                ]
            ),
        )
        mock_load.return_value = plan_file

        with patch("pr_split.cli.validate_coverage"):
            result = runner.invoke(app, ["execute"])

        assert result.exit_code == 0, result.output
        assert "Recreating them and retrying" in result.output.replace("\n", " ")
        mock_create.assert_called_once()
        mock_push.assert_called_once()
        mock_save.assert_called_once()

    @staticmethod
    def _plan_with_one_of_two_prs(pr_groups: list[str]) -> object:
        from pr_split.constants import Priority
        from pr_split.schemas import BranchRecord, GitState, PlanFile, PRRecord, SplitPlan

        return PlanFile(
            plan=SplitPlan(
                dev_branch="feature-branch",
                base_branch="main",
                max_loc=400,
                priority=Priority.ORTHOGONAL,
                merge_base_sha="0123456789abcdef",
                raw_diff="some diff",
                groups=[
                    _group("pr-1", "feat: a", files=["a.py"]),
                    _group("pr-2", "feat: b", files=["b.py"]),
                ],
            ),
            git_state=GitState(
                branches=[
                    BranchRecord(
                        group_id=gid,
                        branch_name=f"pr-split/feature-branch/{gid}",
                        base_branch="main",
                        commit_sha="abc123",
                    )
                    for gid in ("pr-1", "pr-2")
                ],
                prs=[PRRecord(group_id=gid, pr_number=10, pr_url="u") for gid in pr_groups],
            ),
        )

    @patch("pr_split.cli.typer.confirm", return_value=True)
    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli._push_and_create_prs", return_value=[])
    @patch("pr_split.cli._create_branches_and_commits", return_value=[])
    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.validate_coverage")
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_partial_prs_are_kept_and_the_rest_created(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_validate: MagicMock,
        mock_parse: MagicMock,
        mock_commit: MagicMock,
        mock_create: MagicMock,
        mock_push: MagicMock,
        mock_save: MagicMock,
        mock_confirm: MagicMock,
    ) -> None:
        mock_load.return_value = self._plan_with_one_of_two_prs(["pr-1"])

        result = runner.invoke(app, ["execute"])

        assert result.exit_code == 0, result.output
        assert "opened 1 of 2 PR(s)" in result.output.replace("\n", " ")
        assert set(mock_create.call_args.kwargs["keep"]) == {"pr-1"}
        assert set(mock_push.call_args.kwargs["existing_prs"]) == {"pr-1"}

    @pytest.mark.parametrize("stacked", [True, False])
    @patch("pr_split.cli._link_stacks")
    @patch("pr_split.cli._require_gh_stack")
    @patch("pr_split.cli.typer.confirm", return_value=True)
    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli._push_and_create_prs", return_value=[])
    @patch("pr_split.cli._create_branches_and_commits", return_value=[])
    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.validate_coverage")
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_stacked_resume_keeps_the_parents_of_a_layer_with_a_pr(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_validate: MagicMock,
        mock_parse: MagicMock,
        mock_commit: MagicMock,
        mock_create: MagicMock,
        mock_push: MagicMock,
        mock_save: MagicMock,
        mock_confirm: MagicMock,
        mock_stack: MagicMock,
        mock_link: MagicMock,
        stacked: bool,
    ) -> None:
        """pr-2's PR opened but its parent pr-1's did not: rebuilding pr-1 would
        rewrite the base under pr-2's open PR, so the resume keeps it. A plan
        with dependencies is laid out along its DAG with or without --stack."""
        plan_file = self._plan_with_one_of_two_prs(["pr-2"])
        plan_file.plan.groups[1].depends_on = ["pr-1"]
        plan_file.plan.stacked = stacked
        mock_load.return_value = plan_file

        result = runner.invoke(app, ["execute"])

        assert result.exit_code == 0, result.output
        kept = set(mock_create.call_args.kwargs["keep"])
        assert kept == {"pr-1", "pr-2"}
        # pr-1 still gets its PR opened.
        assert set(mock_push.call_args.kwargs["existing_prs"]) == {"pr-2"}

    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_every_group_with_a_pr_is_refused(
        self, mock_pe: MagicMock, mock_load: MagicMock
    ) -> None:
        mock_load.return_value = self._plan_with_one_of_two_prs(["pr-1", "pr-2"])
        result = runner.invoke(app, ["execute"])
        assert result.exit_code == 1
        assert "already has PRs" in result.output

    @staticmethod
    def _plan_with_stale_pr2() -> object:
        from pr_split.constants import Priority
        from pr_split.schemas import BranchRecord, GitState, PlanFile, SplitPlan

        return PlanFile(
            plan=SplitPlan(
                dev_branch="feature-branch",
                base_branch="main",
                max_loc=400,
                priority=Priority.ORTHOGONAL,
                merge_base_sha="0123456789abcdef",
                raw_diff="some diff",
                groups=[_group("pr-1", "feat: auth", files=["a.py"])],
            ),
            git_state=GitState(
                branches=[
                    BranchRecord(
                        group_id="pr-1",
                        branch_name="pr-split/feature-branch/pr-1",
                        base_branch="main",
                        commit_sha="abc123",
                    ),
                    # pr-2 was merged into pr-1 by an edit between the runs.
                    BranchRecord(
                        group_id="pr-2",
                        branch_name="pr-split/feature-branch/pr-2",
                        base_branch="main",
                        commit_sha="def456",
                    ),
                ]
            ),
        )

    @patch("pr_split.cli.run_git")
    @patch("pr_split.cli.delete_branch")
    @patch("pr_split.cli.typer.confirm", return_value=True)
    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli._push_and_create_prs", return_value=[])
    @patch("pr_split.cli._create_branches_and_commits", return_value=[])
    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_branches_dropped_from_an_edited_plan_are_deleted(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_parse: MagicMock,
        mock_commit: MagicMock,
        mock_create: MagicMock,
        mock_push: MagicMock,
        mock_save: MagicMock,
        mock_confirm: MagicMock,
        mock_delete: MagicMock,
        mock_rg: MagicMock,
    ) -> None:
        """A retry must not orphan branches the edited plan no longer contains."""
        mock_load.return_value = self._plan_with_stale_pr2()

        with patch("pr_split.cli.validate_coverage"):
            result = runner.invoke(app, ["execute"])

        assert result.exit_code == 0
        mock_delete.assert_called_once_with("pr-split/feature-branch/pr-2")
        mock_rg.assert_any_call(
            "ls-remote", "--exit-code", "origin", "refs/heads/pr-split/feature-branch/pr-2"
        )
        mock_rg.assert_any_call("push", "origin", "--delete", "pr-split/feature-branch/pr-2")

    @patch("pr_split.cli.run_git")
    @patch("pr_split.cli.delete_branch")
    @patch("pr_split.cli.typer.confirm", return_value=True)
    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli._push_and_create_prs", return_value=[])
    @patch("pr_split.cli._create_branches_and_commits", return_value=[])
    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_remote_only_stale_branch_is_still_deleted_on_origin(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_parse: MagicMock,
        mock_commit: MagicMock,
        mock_create: MagicMock,
        mock_push: MagicMock,
        mock_save: MagicMock,
        mock_confirm: MagicMock,
        mock_delete: MagicMock,
        mock_rg: MagicMock,
    ) -> None:
        """A stale branch pruned locally must still be removed from origin."""
        mock_be.side_effect = lambda name: name != "pr-split/feature-branch/pr-2"
        mock_load.return_value = self._plan_with_stale_pr2()

        with patch("pr_split.cli.validate_coverage"):
            result = runner.invoke(app, ["execute"])

        assert result.exit_code == 0
        mock_delete.assert_not_called()
        mock_rg.assert_any_call("push", "origin", "--delete", "pr-split/feature-branch/pr-2")


class TestPRCreationErrorIsACleanMessage:
    """A failed push/PR creation must end with a red message, not a traceback."""

    @patch("pr_split.cli.typer.confirm", return_value=True)
    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli._push_and_create_prs")
    @patch("pr_split.cli._create_branches_and_commits", return_value=[])
    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_execute_push_failure_exits_cleanly(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_be: MagicMock,
        mock_auth: MagicMock,
        mock_clean: MagicMock,
        mock_parse: MagicMock,
        mock_commit: MagicMock,
        mock_create: MagicMock,
        mock_push: MagicMock,
        mock_save: MagicMock,
        mock_confirm: MagicMock,
    ) -> None:
        from pr_split.constants import Priority
        from pr_split.schemas import GitState, PlanFile, SplitPlan

        plan_file = PlanFile(
            plan=SplitPlan(
                dev_branch="feature-branch",
                base_branch="main",
                max_loc=400,
                priority=Priority.ORTHOGONAL,
                merge_base_sha="0123456789abcdef",
                raw_diff="some diff",
                groups=[_group("pr-1", "feat: auth", files=["a.py"])],
            ),
            git_state=GitState(branches=[]),
        )
        mock_load.return_value = plan_file
        # git stderr can contain rich-markup-like text; it must be escaped,
        # not interpreted (a stray closing tag would raise MarkupError).
        mock_push.side_effect = PRCreationError(
            "2 PR(s) failed:\n- pr-1: fatal: remote said [/bold] nonsense", []
        )

        with patch("pr_split.cli.validate_coverage"):
            result = runner.invoke(app, ["execute"])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "2 PR(s) failed" in result.output
        assert "[/bold] nonsense" in result.output
        assert "saved to the plan file" in result.output
        mock_save.assert_called_once()

    @patch("pr_split.cli.typer.confirm", return_value=True)
    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli._push_and_create_prs")
    @patch("pr_split.cli._create_branches_and_commits", return_value=[])
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
    def test_split_push_failure_exits_cleanly(
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
        mock_create: MagicMock,
        mock_push: MagicMock,
        mock_save: MagicMock,
        mock_confirm: MagicMock,
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
        mock_push.side_effect = PRCreationError("1 PR(s) failed:\n- pr-1: boom", [])

        result = runner.invoke(
            app,
            ["split", "feature-branch"],
            env={"ANTHROPIC_API_KEY": "sk-test"},
        )

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "1 PR(s) failed" in result.output
        assert "saved to the plan file" in result.output
        mock_save.assert_called_once()


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


class TestAdoptExistingBranches:
    """`adopt` registers already-built branches as a stack and saves them as a plan."""

    def _repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
            monkeypatch.setenv(var, "t")
        for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
            monkeypatch.setenv(var, "t@x")
        monkeypatch.chdir(tmp_path)

        def git(*args: str) -> None:
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

        git("init", "-q", "-b", "main")
        (tmp_path / "a.txt").write_text("a\n")
        git("add", "a.txt")
        git("commit", "-qm", "base")
        git("checkout", "-qb", "test/allowlist")
        (tmp_path / "b.txt").write_text("b\n")
        git("add", "b.txt")
        git("commit", "-qm", "parent")
        git("checkout", "-qb", "test/derive")
        (tmp_path / "c.txt").write_text("c\n")
        git("add", "c.txt")
        git("commit", "-qm", "child")
        git("checkout", "-qb", "unrelated", "main")
        (tmp_path / "d.txt").write_text("d\n")
        git("add", "d.txt")
        git("commit", "-qm", "unrelated")
        git("checkout", "-q", "main")

    @patch("pr_split.cli.find_open_pr")
    @patch("pr_split.cli.link_stack")
    @patch("pr_split.cli.check_gh_stack", return_value=True)
    def test_chain_is_linked_on_base_and_saved_as_a_stacked_plan(
        self,
        mock_stack: MagicMock,
        mock_link: MagicMock,
        mock_find: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._repo(tmp_path, monkeypatch)
        mock_find.side_effect = [(1951, "https://x/1951"), (1952, "https://x/1952")]

        result = runner.invoke(
            app, ["adopt", "test/allowlist", "test/derive", "--base", "main", "--yes"]
        )

        assert result.exit_code == 0, result.output
        mock_link.assert_called_once_with(["test/allowlist", "test/derive"], base="main")
        from pr_split.plan_store import load_plan

        saved = load_plan()
        assert saved.plan.stacked is True
        assert [(g.id, g.depends_on) for g in saved.plan.groups] == [
            ("pr-1", []),
            ("pr-2", ["pr-1"]),
        ]
        assert [(b.branch_name, b.base_branch) for b in saved.git_state.branches] == [
            ("test/allowlist", "main"),
            ("test/derive", "test/allowlist"),
        ]
        assert [p.pr_number for p in saved.git_state.prs] == [1951, 1952]
        # The branches and PRs are the user's own, so clean must never delete them.
        assert all(b.adopted for b in saved.git_state.branches)
        assert all(p.adopted for p in saved.git_state.prs)

    @patch("pr_split.cli.find_open_pr")
    @patch("pr_split.cli.link_stack")
    @patch("pr_split.cli.check_gh_stack", return_value=True)
    def test_another_branchs_plan_does_not_block_adoption(
        self,
        mock_stack: MagicMock,
        mock_link: MagicMock,
        mock_find: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pr_split.constants import Priority
        from pr_split.plan_store import load_plan, save_plan, select_plan
        from pr_split.schemas import BranchRecord, GitState, PlanFile, SplitPlan

        self._repo(tmp_path, monkeypatch)
        for other in ("feat/a", "feat/b"):
            save_plan(
                PlanFile(
                    plan=SplitPlan(
                        dev_branch=other,
                        base_branch="main",
                        max_loc=400,
                        priority=Priority.ORTHOGONAL,
                        groups=[],
                    ),
                    git_state=GitState(
                        branches=[
                            BranchRecord(group_id="pr-1", branch_name=other, base_branch="main")
                        ]
                    ),
                )
            )
        mock_find.side_effect = [(1951, "https://x/1951"), (1952, "https://x/1952")]

        result = runner.invoke(
            app, ["adopt", "test/allowlist", "test/derive", "--base", "main", "--yes"]
        )

        assert result.exit_code == 0, result.output
        select_plan("test/derive")
        assert [p.pr_number for p in load_plan().git_state.prs] == [1951, 1952]

    @patch("pr_split.cli.find_open_pr", return_value=None)
    @patch("pr_split.cli.link_stack")
    @patch("pr_split.cli.check_gh_stack", return_value=True)
    def test_a_failed_lookup_saves_nothing_and_says_to_rerun(
        self,
        mock_stack: MagicMock,
        mock_link: MagicMock,
        mock_find: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._repo(tmp_path, monkeypatch)

        result = runner.invoke(
            app, ["adopt", "test/allowlist", "test/derive", "--base", "main", "--yes"]
        )

        assert result.exit_code == 1
        assert "re-run the same adopt command" in " ".join(result.output.split())
        from pr_split.plan_store import plan_exists

        assert not plan_exists()

    @patch("pr_split.cli.link_stack")
    def test_branch_that_does_not_contain_its_parent_is_refused(
        self, mock_link: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._repo(tmp_path, monkeypatch)

        result = runner.invoke(app, ["adopt", "test/allowlist", "unrelated", "--yes"])

        assert result.exit_code == 1
        assert "unrelated does not contain test/allowlist" in result.output
        mock_link.assert_not_called()

    def test_a_single_branch_is_refused(self) -> None:
        result = runner.invoke(app, ["adopt", "only-one"])
        assert result.exit_code == 1
        assert "at least two branches" in result.output


class TestRetargetMergedBase:
    def _plan(self, stacked: bool = True) -> object:
        from pr_split.constants import Priority
        from pr_split.schemas import BranchRecord, GitState, PlanFile, PRRecord, SplitPlan

        return PlanFile(
            plan=SplitPlan(
                dev_branch="feat/change-signature",
                base_branch="feat/postcondition",
                max_loc=400,
                priority=Priority.ORTHOGONAL,
                stacked=stacked,
                groups=[_group("pr-1", "a"), _group("pr-2", "b", depends_on=["pr-1"])],
            ),
            git_state=GitState(
                branches=[
                    BranchRecord(
                        group_id="pr-1", branch_name="s/pr-1", base_branch="feat/postcondition"
                    ),
                    BranchRecord(group_id="pr-2", branch_name="s/pr-2", base_branch="s/pr-1"),
                ],
                prs=[
                    PRRecord(group_id="pr-1", pr_number=2042, pr_url="u"),
                    PRRecord(group_id="pr-2", pr_number=2043, pr_url="u"),
                ],
            ),
        )

    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli.link_stack")
    @patch("pr_split.cli.set_pr_base")
    @patch("pr_split.cli.unstack")
    @patch("pr_split.cli.stack_numbers_for", return_value={2054})
    @patch("pr_split.cli.default_branch", return_value="main")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_unstacks_retargets_roots_and_relinks_on_the_new_base(
        self,
        mock_pe: MagicMock,
        mock_load: MagicMock,
        mock_default: MagicMock,
        mock_stacks: MagicMock,
        mock_unstack: MagicMock,
        mock_set_base: MagicMock,
        mock_link: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        mock_load.return_value = self._plan()

        result = runner.invoke(app, ["retarget", "--yes"])

        assert result.exit_code == 0, result.output
        mock_unstack.assert_called_once_with(2054)
        mock_set_base.assert_called_once_with(2042, "main")
        mock_link.assert_called_once_with([2042, 2043], base="main")
        saved = mock_save.call_args.args[0]
        assert saved.plan.base_branch == "main"
        assert [b.base_branch for b in saved.git_state.branches] == ["main", "s/pr-1"]

    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_merge_refuses_when_the_base_has_merged(
        self, mock_pe: MagicMock, mock_load: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pr_split.cli._base_has_merged", lambda base: True)
        mock_load.return_value = self._plan()

        result = runner.invoke(app, ["merge"])

        assert result.exit_code == 1
        assert "feat/postcondition has already merged" in " ".join(result.output.split())
        assert "pr-split retarget" in " ".join(result.output.split())


class TestEditorPlanCommands:
    def _partial(self, file_path: str, indices: list[int]) -> GroupAssignment:
        return GroupAssignment(
            file_path=file_path,
            assignment_type=AssignmentType.PARTIAL_HUNKS,
            hunk_indices=indices,
        )

    def _two_groups(self) -> list[Group]:
        g1 = _group("pr-1", "first")
        g1.assignments = [self._partial("a.py", [0])]
        g2 = _group("pr-2", "second")
        g2.assignments = [self._partial("a.py", [1])]
        return [g1, g2]

    @patch("pr_split.cli.typer.prompt")
    def test_dep_title_desc_and_undep(self, mock_prompt: MagicMock) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        mock_prompt.side_effect = [
            "dep pr-2 pr-1",
            "title pr-2 feat: add the second half",
            "desc pr-2 Builds on pr-1.",
            "done",
        ]
        groups = _interactive_edit(self._two_groups(), parsed)
        assert groups[1].depends_on == ["pr-1"]
        assert groups[1].title == "feat: add the second half"
        assert groups[1].description == "Builds on pr-1."

        mock_prompt.side_effect = ["undep pr-2 pr-1", "done"]
        groups = _interactive_edit(groups, parsed)
        assert groups[1].depends_on == []

    @patch("pr_split.cli.typer.prompt")
    def test_dep_that_would_cycle_is_refused(self, mock_prompt: MagicMock) -> None:
        groups = self._two_groups()
        groups[1].depends_on = ["pr-1"]
        mock_prompt.side_effect = ["dep pr-1 pr-2", "done"]
        result = _interactive_edit(groups, parse_diff(TWO_HUNK_DIFF))
        assert result[0].depends_on == []

    @patch("pr_split.cli.typer.prompt")
    def test_movefile_moves_every_hunk_and_recomputes_loc(self, mock_prompt: MagicMock) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        mock_prompt.side_effect = ["movefile a.py pr-1 pr-2", "done"]
        groups = _interactive_edit(self._two_groups(), parsed)
        assert groups[0].assignments == []
        assert groups[1].assignments[0].assignment_type is AssignmentType.WHOLE_FILE
        assert groups[1].assignments[0].hunk_indices == [0, 1]
        assert groups[1].estimated_loc == 3

    @patch("pr_split.cli.typer.prompt")
    def test_new_group_then_move_into_it(self, mock_prompt: MagicMock) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        mock_prompt.side_effect = ["new pr-3", "move a.py:1 pr-2 pr-3", "done"]
        groups = _interactive_edit(self._two_groups(), parsed)
        assert [g.id for g in groups] == ["pr-1", "pr-2", "pr-3"]
        assert groups[2].assignments[0].hunk_indices == [1]

    @patch("pr_split.cli.typer.prompt")
    def test_merge_folds_hunks_parents_and_dependants(self, mock_prompt: MagicMock) -> None:
        parsed = parse_diff(TWO_HUNK_DIFF)
        groups = self._two_groups()
        leaf = _group("pr-3", "leaf", depends_on=["pr-2"], files=["c.py"])
        mock_prompt.side_effect = ["merge pr-1 pr-2", "done"]

        result = _interactive_edit([*groups, leaf], parsed)

        assert [g.id for g in result] == ["pr-1", "pr-3"]
        assert result[0].assignments[0].assignment_type is AssignmentType.WHOLE_FILE
        assert result[0].assignments[0].hunk_indices == [0, 1]
        assert result[1].depends_on == ["pr-1"]
        assert result[0].estimated_loc == 3

    @patch("pr_split.cli.typer.prompt")
    def test_merge_that_would_cycle_is_refused(self, mock_prompt: MagicMock) -> None:
        # pr-3 sits between pr-1 and pr-2; folding pr-2 into pr-1 would make
        # pr-1 depend on pr-3, which depends on pr-1.
        groups = self._two_groups()
        mid = _group("pr-3", "mid", depends_on=["pr-1"], files=["c.py"])
        groups[1].depends_on = ["pr-3"]
        mock_prompt.side_effect = ["merge pr-1 pr-2", "done"]
        result = _interactive_edit([*groups, mid], parse_diff(TWO_HUNK_DIFF))
        assert [g.id for g in result] == ["pr-1", "pr-2", "pr-3"]


class TestExecuteYes:
    @patch("pr_split.cli._create_branches_and_commits", side_effect=PRSplitError("stop here"))
    @patch("pr_split.cli.typer.confirm")
    @patch("pr_split.cli.validate_coverage")
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.commit_exists", return_value=True)
    @patch("pr_split.cli.check_gh_auth", return_value=True)
    @patch("pr_split.cli.is_worktree_clean", return_value=True)
    @patch("pr_split.cli.branch_exists", return_value=True)
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_yes_skips_the_confirmation_prompt(
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
        mock_plan_file.plan.max_loc = 400
        mock_plan_file.plan.groups = [_group("pr-1", "t", files=["a.py"])]
        mock_load.return_value = mock_plan_file

        result = runner.invoke(app, ["execute", "--yes"], input="")

        mock_confirm.assert_not_called()
        mock_create.assert_called_once()
        assert "stop here" in result.output
