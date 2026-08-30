from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from pr_split.cli import (
    _build_pr_body,
    _cleanup_git_state,
    _handle_loc_bound_warnings,
    app,
)
from pr_split.constants import AssignmentType
from pr_split.exceptions import GitOperationError, PRSplitError
from pr_split.git_ops.prs import get_pr_state, merge_pr
from pr_split.schemas import (
    BranchRecord,
    GitState,
    Group,
    GroupAssignment,
    PRRecord,
)

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


class TestBuildPrBody:
    def test_includes_description(self) -> None:
        group = _group("pr-1", "feat: auth", files=["auth.py"])
        body = _build_pr_body(group, [group])
        assert "desc for pr-1" in body

    def test_includes_files_changed(self) -> None:
        group = _group("pr-1", "feat: auth", files=["auth.py", "config.py"])
        body = _build_pr_body(group, [group])
        assert "## Files changed" in body
        assert "`auth.py`" in body
        assert "`config.py`" in body

    def test_includes_diff_stats(self) -> None:
        group = _group("pr-1", "feat: auth", files=["a.py"], added=20, removed=5)
        body = _build_pr_body(group, [group])
        assert "## Diff stats" in body
        assert "+20" in body
        assert "-5" in body
        assert "25 LOC" in body

    def test_includes_dependencies(self) -> None:
        g1 = _group("pr-1", "base")
        g2 = _group("pr-2", "child", depends_on=["pr-1"], files=["b.py"])
        body = _build_pr_body(g2, [g1, g2])
        assert "## Dependencies" in body
        assert "`pr-1`" in body

    def test_no_dependencies_section_for_root(self) -> None:
        group = _group("pr-1", "root", files=["a.py"])
        body = _build_pr_body(group, [group])
        assert "## Dependencies" not in body

    def test_includes_dag_markdown(self) -> None:
        group = _group("pr-1", "root", files=["a.py"])
        body = _build_pr_body(group, [group])
        assert "## Dependency graph" in body

    def test_no_files_section_when_no_assignments(self) -> None:
        group = _group("pr-1", "root")
        body = _build_pr_body(group, [group])
        assert "## Files changed" not in body

    def test_custom_template(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        template_dir = tmp_path / ".pr-split"
        template_dir.mkdir()
        template_file = template_dir / "template.md"
        template_file.write_text("{description}\n\nFiles: {files}\nLOC: {loc}")
        monkeypatch.setattr("pr_split.cli._PR_TEMPLATE_PATH", template_file)

        group = _group("pr-1", "feat: auth", files=["auth.py"], added=10, removed=5)
        body = _build_pr_body(group, [group])
        assert "desc for pr-1" in body
        assert "auth.py" in body
        assert "15" in body

    def test_custom_template_invalid_placeholder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        template_dir = tmp_path / ".pr-split"
        template_dir.mkdir()
        template_file = template_dir / "template.md"
        template_file.write_text("{nonexistent}")
        monkeypatch.setattr("pr_split.cli._PR_TEMPLATE_PATH", template_file)

        group = _group("pr-1", "feat: auth", files=["auth.py"])
        with pytest.raises(PRSplitError, match="Invalid PR template"):
            _build_pr_body(group, [group])


class TestCleanupGitState:
    @patch("pr_split.cli.shutil.rmtree")
    @patch("pr_split.cli.Path")
    @patch("pr_split.cli.delete_branch")
    @patch("pr_split.cli.close_pr")
    def test_closes_prs_and_deletes_branches(
        self,
        mock_close: MagicMock,
        mock_delete: MagicMock,
        mock_path: MagicMock,
        mock_rmtree: MagicMock,
    ) -> None:
        mock_path.return_value.exists.return_value = True
        git_state = GitState(
            branches=[
                BranchRecord(group_id="pr-1", branch_name="pr-split/ns/pr-1", base_branch="main"),
                BranchRecord(group_id="pr-2", branch_name="pr-split/ns/pr-2", base_branch="main"),
            ],
            prs=[
                PRRecord(group_id="pr-1", pr_number=10, pr_url="url1"),
                PRRecord(group_id="pr-2", pr_number=11, pr_url="url2"),
            ],
        )
        closed, deleted = _cleanup_git_state(git_state)
        assert closed == 2
        assert deleted == 2
        mock_close.assert_any_call(10)
        mock_close.assert_any_call(11)
        mock_delete.assert_any_call("pr-split/ns/pr-1", remote=True)
        mock_delete.assert_any_call("pr-split/ns/pr-2", remote=True)

    @patch("pr_split.cli.shutil.rmtree")
    @patch("pr_split.cli.Path")
    @patch("pr_split.cli.delete_branch")
    @patch("pr_split.cli.close_pr")
    def test_handles_partial_failures(
        self,
        mock_close: MagicMock,
        mock_delete: MagicMock,
        mock_path: MagicMock,
        mock_rmtree: MagicMock,
    ) -> None:
        mock_path.return_value.exists.return_value = True
        mock_close.side_effect = [None, PRSplitError("fail")]
        mock_delete.side_effect = [PRSplitError("fail"), None]
        git_state = GitState(
            branches=[
                BranchRecord(group_id="pr-1", branch_name="b1", base_branch="main"),
                BranchRecord(group_id="pr-2", branch_name="b2", base_branch="main"),
            ],
            prs=[
                PRRecord(group_id="pr-1", pr_number=10, pr_url="url1"),
                PRRecord(group_id="pr-2", pr_number=11, pr_url="url2"),
            ],
        )
        closed, deleted = _cleanup_git_state(git_state)
        assert closed == 1
        assert deleted == 1


class TestGetPrState:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_returns_parsed_state(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = '{"state": "OPEN", "reviewDecision": null, "isDraft": false}'
        result = get_pr_state(42)
        assert result["state"] == "OPEN"
        assert result["reviewDecision"] is None
        assert result["isDraft"] is False

    @patch("pr_split.git_ops.prs._run_gh")
    def test_returns_empty_on_error(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = GitOperationError("not found")
        result = get_pr_state(999)
        assert result == {}


class TestMergePr:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_merge_without_auto(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = ""
        merge_pr(42)
        args = mock_gh.call_args[0]
        assert "--auto" not in args
        assert "--merge" in args
        assert "--delete-branch" in args

    @patch("pr_split.git_ops.prs._run_gh")
    def test_merge_with_auto(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = ""
        merge_pr(42, auto=True)
        args = mock_gh.call_args[0]
        assert "--auto" in args
        assert "--merge" in args
        assert "--delete-branch" in args


class TestHandleLocBoundWarnings:
    @patch("pr_split.cli.logger.warning")
    def test_non_strict_only_logs(self, mock_warning: MagicMock) -> None:
        _handle_loc_bound_warnings(["warning 1"], strict_loc_bounds=False)
        mock_warning.assert_called_once_with("warning 1")

    @patch("pr_split.cli.console.print")
    @patch("pr_split.cli.logger.warning")
    def test_strict_raises_exit(self, mock_warning: MagicMock, mock_print: MagicMock) -> None:
        with pytest.raises(typer.Exit):
            _handle_loc_bound_warnings(["warning 1"], strict_loc_bounds=True)
        mock_warning.assert_not_called()


class TestSplitCliEnvVars:
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
    def test_split_uses_max_loc_envvar(
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
        group = _group("pr-1", "t", files=["a.py"])
        mock_parse_diff.return_value = parsed_diff
        mock_plan_split.return_value = [group]
        mock_interactive_edit.return_value = [group]

        result = runner.invoke(
            app,
            ["split", "feature-branch", "--dry-run"],
            env={
                "ANTHROPIC_API_KEY": "sk-test",
                "PR_SPLIT_MAX_LOC": "123",
            },
        )

        assert result.exit_code == 0
        settings = mock_plan_split.call_args[0][1]
        assert settings.max_loc == 123

    @patch("pr_split.cli.save_plan")
    @patch("pr_split.cli.merge_base", return_value="abc123")
    @patch("pr_split.cli._interactive_edit")
    @patch("pr_split.cli._present_plan")
    @patch("pr_split.cli.validate_plan", side_effect=[[], ["warning 1"]])
    @patch("pr_split.cli.plan_split")
    @patch("pr_split.cli.parse_diff")
    @patch("pr_split.cli.extract_diff", return_value="diff --git a/a.py b/a.py\n")
    @patch("pr_split.cli._validate_inputs")
    @patch("pr_split.cli.branch_exists", return_value=True)
    def test_split_revalidates_min_loc_after_interactive_edit(
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
        group = _group("pr-1", "t", files=["a.py"])
        mock_plan_split.return_value = [group]
        mock_interactive_edit.return_value = [group]

        result = runner.invoke(
            app,
            [
                "split",
                "feature-branch",
                "--dry-run",
                "--min-loc",
                "50",
                "--strict-loc-bounds",
            ],
            env={"ANTHROPIC_API_KEY": "sk-test"},
        )

        assert result.exit_code == 1
        assert mock_validate_plan.call_args_list[0].kwargs["min_loc"] == 50
        assert mock_validate_plan.call_args_list[1].kwargs["min_loc"] == 50
        mock_save_plan.assert_not_called()


def _plan_with_prs(groups: list[Group]) -> MagicMock:
    plan_file = MagicMock()
    plan_file.plan.groups = groups
    plan_file.git_state.prs = [
        PRRecord(group_id=g.id, pr_number=index + 1, pr_url=f"url{index + 1}")
        for index, g in enumerate(groups)
    ]
    return plan_file


class TestMergeBlockedDependencies:
    @patch("pr_split.cli._send_webhook")
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_child_of_skipped_parent_is_not_merged(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
        mock_webhook: MagicMock,
    ) -> None:
        groups = [
            _group("pr-1", "parent", files=["a.py"]),
            _group("pr-2", "child", depends_on=["pr-1"], files=["b.py"]),
        ]
        mock_load.return_value = _plan_with_prs(groups)
        mock_state.side_effect = lambda n: (
            {"state": "OPEN", "isDraft": True, "reviewDecision": "APPROVED"}
            if n == 1
            else {"state": "OPEN", "isDraft": False, "reviewDecision": "APPROVED"}
        )

        result = runner.invoke(app, ["merge", "--notify", "https://example.invalid/hook"])

        mock_merge.assert_not_called()
        assert result.exit_code == 1
        assert "pr-2 (dependency pr-1 not merged)" in result.output
        payload = mock_webhook.call_args[0][1]
        assert payload["success"] is False
        assert payload["exit_reason"] == "unmerged_dependency"

    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_child_of_group_without_pr_is_not_merged(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
    ) -> None:
        groups = [
            _group("pr-1", "parent", files=["a.py"]),
            _group("pr-2", "child", depends_on=["pr-1"], files=["b.py"]),
        ]
        plan_file = _plan_with_prs(groups)
        plan_file.git_state.prs = plan_file.git_state.prs[1:]
        mock_load.return_value = plan_file
        mock_state.return_value = {"state": "OPEN", "isDraft": False, "reviewDecision": None}

        result = runner.invoke(app, ["merge"])

        mock_merge.assert_not_called()
        assert result.exit_code == 1
        assert "pr-2 (dependency pr-1 not merged)" in result.output

    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_independent_subtree_still_merges(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
    ) -> None:
        groups = [
            _group("pr-1", "draft parent", files=["a.py"]),
            _group("pr-2", "child", depends_on=["pr-1"], files=["b.py"]),
            _group("pr-3", "independent", files=["c.py"]),
        ]
        mock_load.return_value = _plan_with_prs(groups)
        mock_state.side_effect = lambda n: {
            "state": "OPEN",
            "isDraft": n == 1,
            "reviewDecision": None,
        }

        result = runner.invoke(app, ["merge"])

        mock_merge.assert_called_once_with(3, auto=False)
        assert result.exit_code == 1
        assert "Merged (1): pr-3" in result.output

    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_already_merged_child_under_unmerged_parent_is_not_blocked(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
    ) -> None:
        groups = [
            _group("pr-1", "draft parent", files=["a.py"]),
            _group("pr-2", "child already merged", depends_on=["pr-1"], files=["b.py"]),
            _group("pr-3", "grandchild", depends_on=["pr-2"], files=["c.py"]),
        ]
        mock_load.return_value = _plan_with_prs(groups)
        states = {
            1: {"state": "OPEN", "isDraft": True, "reviewDecision": None},
            2: {"state": "MERGED", "isDraft": False, "reviewDecision": None},
            3: {"state": "OPEN", "isDraft": False, "reviewDecision": None},
        }
        mock_state.side_effect = lambda n: states[n]

        result = runner.invoke(app, ["merge"])

        mock_merge.assert_called_once_with(3, auto=False)
        assert "pr-2 (dependency" not in result.output
        assert "Merged (2): pr-2, pr-3" in result.output

    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_child_of_merged_parent_merges(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
    ) -> None:
        groups = [
            _group("pr-1", "parent", files=["a.py"]),
            _group("pr-2", "child", depends_on=["pr-1"], files=["b.py"]),
        ]
        mock_load.return_value = _plan_with_prs(groups)
        mock_state.return_value = {"state": "OPEN", "isDraft": False, "reviewDecision": None}

        result = runner.invoke(app, ["merge"])

        assert [c.args[0] for c in mock_merge.call_args_list] == [1, 2]
        assert result.exit_code == 0


class TestMergeFetchErrors:
    @patch("pr_split.cli._send_webhook")
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_unfetchable_pr_state_fails_the_run(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
        mock_webhook: MagicMock,
    ) -> None:
        groups = [
            _group("pr-1", "ok", files=["a.py"]),
            _group("pr-2", "unfetchable", files=["b.py"]),
        ]
        mock_load.return_value = _plan_with_prs(groups)
        mock_state.side_effect = lambda n: (
            {"state": "OPEN", "isDraft": False, "reviewDecision": None} if n == 1 else {}
        )

        result = runner.invoke(app, ["merge", "--notify", "https://example.invalid/hook"])

        mock_merge.assert_called_once_with(1, auto=False)
        assert result.exit_code == 1
        assert "Could not fetch PR state" in result.output
        assert "Merge complete" not in result.output
        payload = mock_webhook.call_args[0][1]
        assert payload["success"] is False
        assert payload["exit_reason"] == "fetch_error"

    @patch("pr_split.cli._send_webhook")
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_fetch_error_on_parent_is_reported_as_root_cause(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
        mock_webhook: MagicMock,
    ) -> None:
        groups = [
            _group("pr-1", "unfetchable", files=["a.py"]),
            _group("pr-2", "child", depends_on=["pr-1"], files=["b.py"]),
        ]
        mock_load.return_value = _plan_with_prs(groups)
        mock_state.side_effect = lambda n: (
            {} if n == 1 else {"state": "OPEN", "isDraft": False, "reviewDecision": None}
        )

        result = runner.invoke(app, ["merge", "--notify", "https://example.invalid/hook"])

        assert result.exit_code == 1
        assert mock_webhook.call_args[0][1]["exit_reason"] == "fetch_error"

    @patch("pr_split.cli.time.sleep")
    @patch("pr_split.cli._send_webhook")
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_fetch_error_while_polling_auto_merge_is_reported(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
        mock_webhook: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        groups = [_group("pr-1", "only", files=["a.py"])]
        mock_load.return_value = _plan_with_prs(groups)
        mock_state.side_effect = [{"state": "OPEN", "isDraft": False, "reviewDecision": None}, {}]

        result = runner.invoke(
            app, ["merge", "--auto", "--notify", "https://example.invalid/hook"]
        )

        assert result.exit_code == 1
        assert "Could not fetch PR state" in result.output
        payload = mock_webhook.call_args[0][1]
        assert payload["exit_reason"] == "fetch_error"
        assert [s["id"] for s in payload["skipped"]] == ["pr-1"]


class TestMergeClosedWhilePolling:
    @patch("pr_split.cli.time.sleep")
    @patch("pr_split.cli._send_webhook")
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_closed_during_auto_wait_is_listed_as_skipped(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
        mock_webhook: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        groups = [
            _group("pr-1", "closed later", files=["a.py"]),
            _group("pr-2", "child", depends_on=["pr-1"], files=["b.py"]),
        ]
        mock_load.return_value = _plan_with_prs(groups)
        open_state = {"state": "OPEN", "isDraft": False, "reviewDecision": None}
        mock_state.side_effect = [open_state, {"state": "CLOSED"}, open_state]

        result = runner.invoke(
            app, ["merge", "--auto", "--notify", "https://example.invalid/hook"]
        )

        assert result.exit_code == 1
        assert "pr-1 (CLOSED)" in result.output
        assert "pr-2 (dependency pr-1 not merged)" in result.output
        payload = mock_webhook.call_args[0][1]
        assert [s["id"] for s in payload["skipped"]] == ["pr-1", "pr-2"]
        assert payload["exit_reason"] == "pr_closed"
        assert payload["success"] is False

    @patch("pr_split.cli.time.sleep")
    @patch("pr_split.cli._send_webhook")
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_closed_leaf_during_auto_wait_exits_nonzero(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
        mock_webhook: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        groups = [_group("pr-1", "closed later", files=["a.py"])]
        mock_load.return_value = _plan_with_prs(groups)
        open_state = {"state": "OPEN", "isDraft": False, "reviewDecision": None}
        mock_state.side_effect = [open_state, {"state": "CLOSED"}]

        result = runner.invoke(
            app, ["merge", "--auto", "--notify", "https://example.invalid/hook"]
        )

        assert result.exit_code == 1
        assert "pr-1 (CLOSED)" in result.output
        assert "Closed before auto-merge completed (1): pr-1" in result.output
        assert "Merge complete" not in result.output
        payload = mock_webhook.call_args[0][1]
        assert payload["exit_reason"] == "pr_closed"
        assert payload["success"] is False
        assert payload["merged"] == []


class TestStackedMergeRetargetsBase:
    def _stacked_plan(self) -> MagicMock:
        groups = [
            _group("pr-1", "root", files=["a.py"]),
            _group("pr-2", "child", depends_on=["pr-1"], files=["b.py"]),
        ]
        plan_file = _plan_with_prs(groups)
        plan_file.plan.base_branch = "main"
        plan_file.git_state.branches = [
            BranchRecord(group_id="pr-1", branch_name="pr-split/ns/pr-1", base_branch="main"),
            BranchRecord(
                group_id="pr-2", branch_name="pr-split/ns/pr-2", base_branch="pr-split/ns/pr-1"
            ),
        ]
        return plan_file

    @patch("pr_split.cli.retarget_pr")
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_child_is_retargeted_at_the_plan_base_before_merging(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
        mock_retarget: MagicMock,
    ) -> None:
        mock_load.return_value = self._stacked_plan()
        mock_state.return_value = {"state": "OPEN", "isDraft": False, "reviewDecision": None}
        calls: list[str] = []
        mock_retarget.side_effect = lambda n, base: calls.append(f"retarget {n} -> {base}")
        mock_merge.side_effect = lambda n, *, auto=False: calls.append(f"merge {n}")

        result = runner.invoke(app, ["merge"])

        assert result.exit_code == 0, result.output
        assert calls == ["merge 1", "retarget 2 -> main", "merge 2"]

    @patch("pr_split.cli.retarget_pr")
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_unstacked_prs_are_not_retargeted(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
        mock_retarget: MagicMock,
    ) -> None:
        plan_file = self._stacked_plan()
        plan_file.git_state.branches[1] = BranchRecord(
            group_id="pr-2", branch_name="pr-split/ns/pr-2", base_branch="main"
        )
        mock_load.return_value = plan_file
        mock_state.return_value = {"state": "OPEN", "isDraft": False, "reviewDecision": None}

        result = runner.invoke(app, ["merge"])

        assert result.exit_code == 0, result.output
        mock_retarget.assert_not_called()
        assert mock_merge.call_count == 2

    @patch("pr_split.cli.time.sleep")
    @patch("pr_split.cli.retarget_pr")
    @patch("pr_split.cli.merge_pr")
    @patch("pr_split.cli.get_pr_state")
    @patch("pr_split.cli.load_plan")
    @patch("pr_split.cli.plan_exists", return_value=True)
    def test_auto_mode_retargets_before_queueing(
        self,
        mock_exists: MagicMock,
        mock_load: MagicMock,
        mock_state: MagicMock,
        mock_merge: MagicMock,
        mock_retarget: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        mock_load.return_value = self._stacked_plan()
        open_state = {"state": "OPEN", "isDraft": False, "reviewDecision": None}
        mock_state.side_effect = [open_state, {"state": "MERGED"}, open_state, {"state": "MERGED"}]
        calls: list[str] = []
        mock_retarget.side_effect = lambda n, base: calls.append(f"retarget {n}")
        mock_merge.side_effect = lambda n, *, auto=False: calls.append(f"queue {n}")

        result = runner.invoke(app, ["merge", "--auto"])

        assert result.exit_code == 0, result.output
        assert calls == ["queue 1", "retarget 2", "queue 2"]
