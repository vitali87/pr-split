from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pr_split.exceptions import GitOperationError
from pr_split.git_ops.branches import (
    add_worktree,
    branch_exists,
    commit_files_in_dir,
    delete_branch,
    derive_split_namespace,
    is_worktree_clean,
    merge_base,
    push_branch,
    remove_worktree,
    run_git,
    run_git_in_dir,
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

    @patch("pr_split.git_ops.branches.run_git")
    def test_local_failure_still_deletes_remote(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = [GitOperationError("checked out"), ""]
        with pytest.raises(GitOperationError, match="checked out"):
            delete_branch("pr-split/pr-1", remote=True)
        mock_git.assert_any_call("push", "origin", "--delete", "pr-split/pr-1")

    @patch("pr_split.git_ops.branches.run_git")
    def test_local_failure_without_remote_raises_immediately(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = GitOperationError("nope")
        with pytest.raises(GitOperationError):
            delete_branch("pr-split/pr-1")
        mock_git.assert_called_once()


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


class TestRunGitInDir:
    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_passes_cwd(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="ok\n", stderr=""
        )
        result = run_git_in_dir("/tmp/wt", "status")
        assert result == "ok"
        assert mock_run.call_args.kwargs["cwd"] == "/tmp/wt"

    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=1, stdout="", stderr="error"
        )
        with pytest.raises(GitOperationError, match="error"):
            run_git_in_dir("/tmp/wt", "status")


class TestAddWorktree:
    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.branches.branch_exists", return_value=False)
    def test_adds_worktree(self, mock_exists: MagicMock, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        add_worktree("/tmp/wt", "pr-split/ns/pr-1", "abc123")
        args = mock_git.call_args.args
        assert args[0] == "-c" and args[1].startswith("core.hooksPath=")
        assert args[2:] == ("worktree", "add", "-b", "pr-split/ns/pr-1", "/tmp/wt", "abc123")

    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.branches.branch_exists", return_value=True)
    def test_deletes_existing_branch_first(
        self, mock_exists: MagicMock, mock_git: MagicMock
    ) -> None:
        mock_git.side_effect = ["oldsha", "", ""]
        add_worktree("/tmp/wt", "pr-split/ns/pr-1", "abc123")
        assert mock_git.call_count == 3
        mock_git.assert_any_call("branch", "-D", "pr-split/ns/pr-1")

    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.branches.branch_exists", return_value=True)
    def test_restores_branch_on_failure(self, mock_exists: MagicMock, mock_git: MagicMock) -> None:
        mock_git.side_effect = [
            "oldsha",
            "",
            GitOperationError("worktree add failed"),
            "",
        ]
        with pytest.raises(GitOperationError, match="worktree add failed"):
            add_worktree("/tmp/wt", "pr-split/ns/pr-1", "abc123")
        mock_git.assert_any_call("branch", "pr-split/ns/pr-1", "oldsha")


class TestRemoveWorktree:
    @patch("pr_split.git_ops.branches.run_git")
    def test_removes_worktree(self, mock_git: MagicMock) -> None:
        mock_git.return_value = ""
        remove_worktree("/tmp/wt")
        mock_git.assert_called_once_with("worktree", "remove", "--force", "/tmp/wt")


class TestCommitFilesInDir:
    @patch("pr_split.git_ops.branches.run_git_in_dir")
    def test_basic_commit(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = ["", "", "abc123"]
        sha = commit_files_in_dir("/tmp/wt", ["file.py"], "test commit")
        assert sha == "abc123"
        assert mock_git.call_args_list[0].args == ("/tmp/wt", "add", "-A", "-f", "--", "file.py")

    @patch("pr_split.git_ops.branches.run_git_in_dir")
    def test_commit_with_author(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = ["", "", "abc123"]
        sha = commit_files_in_dir("/tmp/wt", ["f.py"], "msg", author="J <j@x.com>")
        assert sha == "abc123"
        commit_call = mock_git.call_args_list[1]
        assert "--author" in commit_call.args

    def test_empty_file_paths_raises(self) -> None:
        with pytest.raises(GitOperationError, match="no file paths"):
            commit_files_in_dir("/tmp/wt", [], "msg")


def _git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI runners have no git identity configured; commits need one."""
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@x")


class TestCommitsSkipHooks:
    @patch("pr_split.git_ops.branches.run_git_in_dir", return_value="sha")
    def test_commit_files_in_dir_passes_no_verify(self, mock_git: MagicMock) -> None:
        commit_files_in_dir("/wt", ["a.py"], "msg", author="A <a@x>")
        commit_call = next(c for c in mock_git.call_args_list if c.args[1] == "commit")
        assert commit_call.args == (
            "/wt",
            "commit",
            "--no-verify",
            "-m",
            "msg",
            "--author",
            "A <a@x>",
        )

    def test_failing_pre_commit_hook_does_not_block_the_sub_pr_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _git_identity(monkeypatch)
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        hook = repo / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\necho 'husky: node_modules missing' >&2\nexit 1\n")
        hook.chmod(0o755)
        (repo / "a.py").write_text("x\n")

        sha = commit_files_in_dir(str(repo), ["a.py"], "feat: a", author="Test <t@x>")

        assert len(sha) == 40
        log = subprocess.run(
            ["git", "log", "--format=%s", "-1"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert log.strip() == "feat: a"


class TestIgnoredPathsAreStillCommitted:
    def test_file_tracked_on_dev_despite_gitignore_is_committed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _git_identity(monkeypatch)
        repo = tmp_path / "repo"
        (repo / "build").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        (repo / ".gitignore").write_text("build/\n")
        (repo / "build" / "out.txt").write_text("artifact\n")

        commit_files_in_dir(str(repo), [".gitignore", "build/out.txt"], "feat: artifact")

        tracked = subprocess.run(
            ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.split()
        assert "build/out.txt" in tracked

    def test_deleted_files_are_still_staged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _git_identity(monkeypatch)
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        (repo / "a.py").write_text("x\n")
        commit_files_in_dir(str(repo), ["a.py"], "add")
        (repo / "a.py").unlink()

        commit_files_in_dir(str(repo), ["a.py"], "remove")

        tracked = subprocess.run(
            ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.split()
        assert tracked == []


class TestWorktreeAddSkipsHooks:
    def test_failing_post_checkout_hook_does_not_block_worktree_creation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _git_identity(monkeypatch)
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", "base"], cwd=repo, check=True
        )
        hook = repo / ".git" / "hooks" / "post-checkout"
        hook.write_text("#!/bin/sh\necho 'post-checkout failing' >&2\nexit 1\n")
        hook.chmod(0o755)
        monkeypatch.chdir(repo)

        add_worktree(str(tmp_path / "wt"), "pr-split/ns/g1", "main")

        assert (tmp_path / "wt" / ".git").exists()

    def test_hooks_path_from_config_is_also_bypassed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _git_identity(monkeypatch)
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", "base"], cwd=repo, check=True
        )
        hooks = tmp_path / "myhooks"
        hooks.mkdir()
        (hooks / "post-checkout").write_text("#!/bin/sh\nexit 1\n")
        (hooks / "post-checkout").chmod(0o755)
        subprocess.run(["git", "config", "core.hooksPath", str(hooks)], cwd=repo, check=True)
        monkeypatch.chdir(repo)

        add_worktree(str(tmp_path / "wt"), "pr-split/ns/g2", "main")

        assert (tmp_path / "wt" / ".git").exists()
