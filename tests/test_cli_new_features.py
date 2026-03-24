from __future__ import annotations

from unittest.mock import MagicMock, patch

from pr_split.cli import _build_pr_body, _cleanup_git_state
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


def _group(
    gid: str,
    title: str,
    depends_on: list[str] | None = None,
    files: list[str] | None = None,
    added: int = 10,
    removed: int = 5,
) -> Group:
    assignments = [
        GroupAssignment(
            file_path=f, assignment_type=AssignmentType.WHOLE_FILE, hunk_indices=[]
        )
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
