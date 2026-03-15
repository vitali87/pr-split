from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pr_split.exceptions import GitOperationError
from pr_split.git_ops.branches import (
    branch_exists,
    checkout_branch,
    checkout_new_branch,
    commit_files,
    create_group_branch,
    delete_branch,
    derive_split_namespace,
    is_worktree_clean,
    merge_base,
    push_branch,
    run_git,
)


class TestRunGit:
    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "status"], returncode=0, stdout="clean\n", stderr=""
        )
        result = run_git("status")
        assert result == "clean"

    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "status"], returncode=1, stdout="", stderr="fatal error"
        )
        with pytest.raises(GitOperationError, match="fatal error"):
            run_git("status")


class TestBranchExists:
    @patch("pr_split.git_ops.branches.run_git")
    def test_exists(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "abc123"
        assert branch_exists("main") is True

    @patch("pr_split.git_ops.branches.run_git")
    def test_not_exists(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = GitOperationError("not found")
        assert branch_exists("nonexistent") is False


class TestIsWorktreeClean:
    @patch("pr_split.git_ops.branches.run_git")
    def test_clean_empty(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        assert is_worktree_clean() is True

    @patch("pr_split.git_ops.branches.run_git")
    def test_clean_with_untracked(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "?? untracked.txt"
        assert is_worktree_clean() is True

    @patch("pr_split.git_ops.branches.run_git")
    def test_dirty(self, mock_git: MagicMock) -> None:
        mock_git.return_value = " M modified.py"
        assert is_worktree_clean() is False


class TestMergeBase:
    @patch("pr_split.git_ops.branches.run_git")
    def test_returns_sha(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "abc123def"
        assert merge_base("main", "feature") == "abc123def"


class TestCommitFiles:
    @patch("pr_split.git_ops.branches.run_git")
    def test_basic_commit(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = ["", "", "abc123"]
        sha = commit_files(["file.py"], "test commit")
        assert sha == "abc123"

    @patch("pr_split.git_ops.branches.run_git")
    def test_commit_with_author(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = ["", "", "abc123"]
        sha = commit_files(["file.py"], "test commit", author="Jane <jane@x.com>")
        assert sha == "abc123"
        commit_call = mock_git.call_args_list[1]
        assert "--author" in commit_call.args[0] or "--author" in commit_call[0]

    @patch("pr_split.git_ops.branches.run_git")
    def test_commit_fallback_on_failure(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = [
            "",
            GitOperationError("nothing to commit"),
            "",
            "",
            "def456",
        ]
        sha = commit_files(["file.py"], "test commit")
        assert sha == "def456"


class TestPushBranch:
    @patch("pr_split.git_ops.branches.run_git")
    def test_calls_push(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        push_branch("pr-split/pr-1")
        mock_git.assert_called_once_with(
            "push", "--force-with-lease", "-u", "origin", "pr-split/pr-1"
        )


class TestDeleteBranch:
    @patch("pr_split.git_ops.branches.run_git")
    def test_local_only(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        delete_branch("pr-split/pr-1")
        mock_git.assert_called_once_with("branch", "-D", "pr-split/pr-1")

    @patch("pr_split.git_ops.branches.run_git")
    def test_with_remote(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        delete_branch("pr-split/pr-1", remote=True)
        assert mock_git.call_count == 2


class TestDeriveSplitNamespace:
    def test_simple_branch(self) -> None:
        result = derive_split_namespace("feat/auth")
        assert "feat" in result
        assert "auth" in result

    def test_pr_number(self) -> None:
        assert derive_split_namespace("#42") == "42"

    def test_fork_ref(self) -> None:
        result = derive_split_namespace("user:feature/branch")
        assert "feature" in result
        assert "branch" in result

    def test_special_chars_sanitized(self) -> None:
        result = derive_split_namespace("feat/some weird@chars!")
        assert "@" not in result
        assert "!" not in result


class TestCreateGroupBranch:
    @patch("pr_split.git_ops.branches.checkout_new_branch")
    @patch("pr_split.git_ops.branches.branch_exists")
    def test_creates_new_branch(self, mock_exists: MagicMock, mock_checkout: MagicMock) -> None:
        mock_exists.return_value = False
        result = create_group_branch("pr-1", "abc123", "my-feat")
        assert result == "pr-split/my-feat/pr-1"
        mock_checkout.assert_called_once()

    @patch("pr_split.git_ops.branches.checkout_new_branch")
    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.branches.checkout_branch")
    @patch("pr_split.git_ops.branches.branch_exists")
    def test_deletes_existing_branch(
        self,
        mock_exists: MagicMock,
        mock_checkout: MagicMock,
        mock_run_git: MagicMock,
        mock_checkout_new: MagicMock,
    ) -> None:
        mock_exists.return_value = True
        mock_run_git.return_value = ""
        create_group_branch("pr-1", "abc123", "my-feat")
        mock_run_git.assert_called_once_with("branch", "-D", "pr-split/my-feat/pr-1")


class TestRunGitExtended:
    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_strips_trailing_whitespace(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="  result  \n\n", stderr=""
        )
        assert run_git("status") == "result"

    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_empty_stderr_on_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=1, stdout="", stderr=""
        )
        with pytest.raises(GitOperationError):
            run_git("fail")

    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_multiple_args_forwarded(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="ok", stderr=""
        )
        run_git("commit", "-m", "message", "--author", "Test <t@t.com>")
        call_args = mock_run.call_args[0][0]
        assert call_args == ["git", "commit", "-m", "message", "--author", "Test <t@t.com>"]


class TestIsWorktreeCleanExtended:
    @patch("pr_split.git_ops.branches.run_git")
    def test_staged_file_is_dirty(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "A  new_file.py"
        assert is_worktree_clean() is False

    @patch("pr_split.git_ops.branches.run_git")
    def test_deleted_file_is_dirty(self, mock_git: MagicMock) -> None:
        mock_git.return_value = " D deleted.py"
        assert is_worktree_clean() is False

    @patch("pr_split.git_ops.branches.run_git")
    def test_renamed_file_is_dirty(self, mock_git: MagicMock) -> None:
        mock_git.return_value = "R  old.py -> new.py"
        assert is_worktree_clean() is False


class TestCheckoutWrappers:
    @patch("pr_split.git_ops.branches.run_git")
    def test_checkout_new_branch_args(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        checkout_new_branch("feature/x", "abc123")
        mock_git.assert_called_once_with("checkout", "-b", "feature/x", "abc123")

    @patch("pr_split.git_ops.branches.run_git")
    def test_checkout_branch_args(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        checkout_branch("main")
        mock_git.assert_called_once_with("checkout", "main")
