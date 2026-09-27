from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pr_split.constants import PLAN_DIR
from pr_split.exceptions import GitOperationError
from pr_split.git_ops.branches import (
    add_worktree,
    adopt_remote_branch,
    branch_exists,
    commit_exists,
    commit_files_in_dir,
    delete_branch,
    derive_split_namespace,
    diff_base_ref,
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


class TestPushBranchRetries:
    @patch("pr_split.git_ops.branches.time.sleep")
    @patch("pr_split.git_ops.branches.run_git")
    def test_transient_server_error_is_retried(
        self, mock_git: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_git.side_effect = [
            GitOperationError("remote: fatal error in commit_refs"),
            "",
        ]
        push_branch("pr-split/pr-5")
        assert mock_git.call_count == 2
        mock_sleep.assert_called_once()

    @patch("pr_split.git_ops.branches.time.sleep")
    @patch("pr_split.git_ops.branches.run_git")
    def test_rejection_is_not_retried(self, mock_git: MagicMock, mock_sleep: MagicMock) -> None:
        mock_git.side_effect = GitOperationError("! [rejected] (stale info)")
        with pytest.raises(GitOperationError, match="stale info"):
            push_branch("pr-split/pr-5")
        assert mock_git.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pr_split.git_ops.branches.time.sleep")
    @patch("pr_split.git_ops.branches.run_git")
    def test_gives_up_after_the_last_attempt(
        self, mock_git: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_git.side_effect = GitOperationError("fatal: the remote end hung up unexpectedly")
        with pytest.raises(GitOperationError, match="hung up"):
            push_branch("pr-split/pr-5")
        assert mock_git.call_count == 3


class TestPushThatLandedDespiteAnError:
    @patch("pr_split.git_ops.branches.time.sleep")
    @patch("pr_split.git_ops.branches.run_git")
    def test_stale_lease_after_a_landed_push_is_success(
        self, mock_git: MagicMock, mock_sleep: MagicMock
    ) -> None:
        def git(*args: str) -> str:
            if args[0] == "push":
                if git.pushes == 0:
                    git.pushes += 1
                    raise GitOperationError("fatal: the remote end hung up unexpectedly")
                raise GitOperationError("! [rejected] pr-split/pr-5 (stale info)")
            if args[0] == "ls-remote":
                return "abc123\trefs/heads/pr-split/pr-5"
            return "abc123"  # rev-parse

        git.pushes = 0  # type: ignore[attr-defined]
        mock_git.side_effect = git
        push_branch("pr-split/pr-5")

    @patch("pr_split.git_ops.branches.time.sleep")
    @patch("pr_split.git_ops.branches.run_git")
    def test_stale_lease_is_still_an_error_when_the_remote_differs(
        self, mock_git: MagicMock, mock_sleep: MagicMock
    ) -> None:
        def git(*args: str) -> str:
            if args[0] == "push":
                if git.pushes == 0:
                    git.pushes += 1
                    raise GitOperationError("fatal: the remote end hung up unexpectedly")
                raise GitOperationError("! [rejected] pr-split/pr-5 (stale info)")
            if args[0] == "ls-remote":
                return "fff999\trefs/heads/pr-split/pr-5"
            return "abc123"

        git.pushes = 0  # type: ignore[attr-defined]
        mock_git.side_effect = git
        with pytest.raises(GitOperationError, match="stale info"):
            push_branch("pr-split/pr-5")

    @patch("pr_split.git_ops.branches.run_git")
    def test_a_first_attempt_rejection_does_not_consult_the_remote(
        self, mock_git: MagicMock
    ) -> None:
        mock_git.side_effect = GitOperationError("! [rejected] (stale info)")
        with pytest.raises(GitOperationError):
            push_branch("pr-split/pr-5")
        assert [c.args[0] for c in mock_git.call_args_list] == ["push"]


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

    @patch("pr_split.git_ops.branches.run_git")
    def test_branch_already_deleted_locally_counts_as_deleted(self, mock_git: MagicMock) -> None:
        # `pr-split merge` deletes the local branch; cleanup must not fail on it.
        mock_git.side_effect = [GitOperationError("error: branch 'pr-split/pr-1' not found."), ""]
        delete_branch("pr-split/pr-1", remote=True)
        mock_git.assert_any_call("push", "origin", "--delete", "pr-split/pr-1")

    @patch("pr_split.git_ops.branches.run_git")
    def test_branch_already_deleted_on_origin_counts_as_deleted(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = [
            "",
            GitOperationError(
                "error: unable to delete 'pr-split/pr-1': remote ref does not exist"
            ),
        ]
        delete_branch("pr-split/pr-1", remote=True)

    @patch("pr_split.git_ops.branches.run_git")
    def test_branch_gone_everywhere_counts_as_deleted(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = [
            GitOperationError("error: branch 'pr-split/pr-1' not found."),
            GitOperationError(
                "error: unable to delete 'pr-split/pr-1': remote ref does not exist"
            ),
        ]
        delete_branch("pr-split/pr-1", remote=True)

    @patch("pr_split.git_ops.branches.run_git")
    def test_missing_local_branch_without_remote_is_fine(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = GitOperationError("error: branch 'pr-split/pr-1' not found.")
        delete_branch("pr-split/pr-1")

    @patch("pr_split.git_ops.branches.run_git")
    def test_other_remote_failure_still_raises(self, mock_git: MagicMock) -> None:
        mock_git.side_effect = ["", GitOperationError("fatal: could not read from remote")]
        with pytest.raises(GitOperationError, match="could not read from remote"):
            delete_branch("pr-split/pr-1", remote=True)


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


class TestGitLocaleIsPinned:
    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_run_git_forces_the_c_locale(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="", stderr=""
        )
        run_git("status")
        env = mock_run.call_args.kwargs["env"]
        assert env["LC_ALL"] == "C"
        assert env["LANGUAGE"] == "C"

    @patch("pr_split.git_ops.branches.subprocess.run")
    def test_run_git_in_dir_forces_the_c_locale(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="", stderr=""
        )
        run_git_in_dir("/tmp", "status")
        assert mock_run.call_args.kwargs["env"]["LC_ALL"] == "C"

    def test_localised_git_still_reports_already_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        monkeypatch.chdir(repo)
        monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
        monkeypatch.setenv("LANGUAGE", "de")
        # never existed: must be treated as already deleted, not as an error
        delete_branch("pr-split/ns/never")


class TestIsWorktreeCleanIgnoresPlanDir:
    """The check must not trip on pr-split's own `.pr-split/plan.json`."""

    @pytest.fixture
    def repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        env = {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
        (tmp_path / "src.py").write_text("x = 1\n")
        plan = tmp_path / PLAN_DIR / "plan.json"
        plan.parent.mkdir()
        plan.write_text("{}\n")
        subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_modified_tracked_plan_is_ignored(self, repo: Path) -> None:
        (repo / PLAN_DIR / "plan.json").write_text('{"groups": []}\n')
        assert is_worktree_clean() is True

    def test_staged_plan_is_ignored(self, repo: Path) -> None:
        (repo / PLAN_DIR / "plan.json").write_text('{"groups": []}\n')
        subprocess.run(["git", "add", "-A"], check=True)
        assert is_worktree_clean() is True

    def test_plan_dir_is_ignored_from_a_subdirectory(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (repo / PLAN_DIR / "plan.json").write_text('{"groups": []}\n')
        sub = repo / "pkg"
        sub.mkdir()
        monkeypatch.chdir(sub)
        assert is_worktree_clean() is True

    def test_other_modifications_still_count(self, repo: Path) -> None:
        (repo / PLAN_DIR / "plan.json").write_text('{"groups": []}\n')
        (repo / "src.py").write_text("x = 2\n")
        assert is_worktree_clean() is False


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


class TestCommitExists:
    @patch("pr_split.git_ops.branches.run_git", return_value="")
    def test_true_when_object_resolves(self, mock_git: MagicMock) -> None:
        assert commit_exists("abc123") is True
        mock_git.assert_called_once_with("cat-file", "-e", "abc123^{commit}")

    @patch(
        "pr_split.git_ops.branches.run_git", side_effect=GitOperationError("Not a valid object")
    )
    def test_false_when_missing(self, mock_git: MagicMock) -> None:
        assert commit_exists("0123456789abcdef") is False


class TestAdoptRemoteBranch:
    def _clone_with_remote_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *remotes: str
    ) -> Path:
        _git_identity(monkeypatch)
        upstream = tmp_path / "upstream"
        upstream.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=upstream, check=True)
        (upstream / "a.txt").write_text("a\n")
        subprocess.run(["git", "add", "a.txt"], cwd=upstream, check=True)
        subprocess.run(["git", "commit", "-qm", "a"], cwd=upstream, check=True)
        subprocess.run(["git", "branch", "feat/move-op"], cwd=upstream, check=True)
        clone = tmp_path / "clone"
        subprocess.run(["git", "init", "-q", "-b", "other", str(clone)], check=True)
        for remote in remotes:
            subprocess.run(["git", "remote", "add", remote, str(upstream)], cwd=clone, check=True)
            subprocess.run(["git", "fetch", "-q", remote], cwd=clone, check=True)
        monkeypatch.chdir(clone)
        return clone

    def test_branch_on_one_remote_is_created_locally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clone_with_remote_branch(tmp_path, monkeypatch, "origin")
        assert not branch_exists("refs/heads/feat/move-op")

        assert adopt_remote_branch("feat/move-op") is True

        assert run_git("rev-parse", "refs/heads/feat/move-op") == run_git(
            "rev-parse", "origin/feat/move-op"
        )
        # It tracks the remote, so diff_base_ref diffs an adopted base against it.
        assert diff_base_ref("feat/move-op") == "origin/feat/move-op"

    def test_branch_on_two_remotes_is_ambiguous_and_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clone_with_remote_branch(tmp_path, monkeypatch, "origin", "upstream")
        assert adopt_remote_branch("feat/move-op") is False
        assert not branch_exists("refs/heads/feat/move-op")

    def test_existing_local_branch_is_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clone_with_remote_branch(tmp_path, monkeypatch, "origin")
        run_git("branch", "--no-track", "feat/move-op", "origin/main")
        before = run_git("rev-parse", "refs/heads/feat/move-op")
        assert adopt_remote_branch("feat/move-op") is False
        assert run_git("rev-parse", "refs/heads/feat/move-op") == before
