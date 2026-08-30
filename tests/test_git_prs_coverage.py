"""Tests to improve coverage for pr_split/git_ops/prs.py.

Covers: get_pr_state (JSON decode error), merge_pr, fetch_fork_pr (happy path),
fetch_fork_branch (happy path and error paths).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pr_split.exceptions import GitOperationError
from pr_split.git_ops.prs import (
    fetch_fork_branch,
    fetch_fork_pr,
    get_pr_state,
    merge_pr,
)


class TestGetPrStateJsonError:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_returns_empty_on_json_decode_error(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "not valid json"
        result = get_pr_state(42)
        assert result == {}


class TestMergePrExtended:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_merge_passes_pr_number_as_string(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = ""
        merge_pr(99)
        args = mock_gh.call_args[0]
        assert "99" in args
        assert "pr" in args
        assert "merge" in args

    @patch("pr_split.git_ops.prs._run_gh")
    def test_merge_raises_on_failure(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = GitOperationError("conflict")
        with pytest.raises(GitOperationError, match="conflict"):
            merge_pr(42)


class TestFetchForkPrHappyPath:
    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_successful_fetch(self, mock_gh: MagicMock, mock_git: MagicMock) -> None:
        pr_data = {
            "head": {
                "ref": "feature-branch",
                "repo": {
                    "fork": True,
                    "clone_url": "https://github.com/user/repo.git",
                    "full_name": "user/repo",
                },
            },
            "base": {"ref": "main"},
        }
        mock_gh.return_value = json.dumps(pr_data)
        mock_git.side_effect = [
            "",  # fetch
            "Author Name <author@example.com>",  # log
        ]

        result = fetch_fork_pr(42)

        assert result["pr_number"] == 42
        assert result["base_branch"] == "main"
        assert result["author"] == "Author Name <author@example.com>"
        assert result["fork_full_name"] == "user/repo"
        assert "pr-split/pr-42" in result["local_ref"]

    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_fetch_git_failure(self, mock_gh: MagicMock, mock_git: MagicMock) -> None:
        pr_data = {
            "head": {
                "ref": "feature-branch",
                "repo": {
                    "fork": True,
                    "clone_url": "https://github.com/user/repo.git",
                    "full_name": "user/repo",
                },
            },
            "base": {"ref": "main"},
        }
        mock_gh.return_value = json.dumps(pr_data)
        mock_git.side_effect = GitOperationError("fetch failed")

        with pytest.raises(GitOperationError, match="Failed to fetch"):
            fetch_fork_pr(42)

    @patch("pr_split.git_ops.prs._run_gh")
    def test_head_repo_none_raises(self, mock_gh: MagicMock) -> None:
        """When head.repo is None (deleted fork), should raise."""
        pr_data = {
            "head": {
                "ref": "feature",
                "repo": None,
            },
            "base": {"ref": "main"},
        }
        mock_gh.return_value = json.dumps(pr_data)
        with pytest.raises(GitOperationError):
            fetch_fork_pr(42)


class TestFetchForkBranch:
    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_successful_fetch(self, mock_gh: MagicMock, mock_git: MagicMock) -> None:
        # _run_gh is called 3 times:
        # 1. get repo name
        # 2. get fork repo data
        # 3. get default branch
        mock_gh.side_effect = [
            "my-repo",  # repo name
            json.dumps(
                {
                    "clone_url": "https://github.com/user/my-repo.git",
                    "full_name": "user/my-repo",
                }
            ),
            "main",  # default branch
        ]
        mock_git.side_effect = [
            "",  # fetch
            "Author <a@b.c>",  # log
        ]

        result = fetch_fork_branch("user", "feature")
        assert result["pr_number"] is None
        assert result["base_branch"] == "main"
        assert result["author"] == "Author <a@b.c>"
        assert result["fork_full_name"] == "user/my-repo"
        assert "fork-user-feature" in result["local_ref"]

    @patch("pr_split.git_ops.prs._run_gh")
    def test_fork_repo_not_found(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = [
            "my-repo",  # repo name ok
            GitOperationError("Not Found"),  # fork repo fails
        ]
        with pytest.raises(GitOperationError, match="Failed to fetch"):
            fetch_fork_branch("nonexistent-user", "branch")

    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_git_fetch_failure(self, mock_gh: MagicMock, mock_git: MagicMock) -> None:
        mock_gh.side_effect = [
            "my-repo",
            json.dumps(
                {
                    "clone_url": "https://github.com/user/my-repo.git",
                    "full_name": "user/my-repo",
                }
            ),
        ]
        mock_git.side_effect = GitOperationError("fetch failed")

        with pytest.raises(GitOperationError, match="Failed to fetch"):
            fetch_fork_branch("user", "branch")


class TestForkFetchRefspecIsForced:
    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_fetch_fork_pr_uses_forced_refspec(
        self, mock_gh: MagicMock, mock_git: MagicMock
    ) -> None:
        mock_gh.return_value = json.dumps(
            {
                "head": {
                    "ref": "feature",
                    "repo": {"fork": True, "clone_url": "https://x/f.git", "full_name": "u/f"},
                },
                "base": {"ref": "main"},
            }
        )
        mock_git.side_effect = ["", "A <a@x>"]
        fetch_fork_pr(42)
        fetch_call = mock_git.call_args_list[0].args
        assert fetch_call[0] == "fetch"
        assert fetch_call[2] == "+feature:refs/pr-split/pr-42"

    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_fetch_fork_branch_uses_forced_refspec(
        self, mock_gh: MagicMock, mock_git: MagicMock
    ) -> None:
        mock_gh.side_effect = [
            "repo",
            json.dumps({"clone_url": "https://x/f.git", "full_name": "u/repo"}),
            "main",
        ]
        mock_git.side_effect = ["", "A <a@x>"]
        fetch_fork_branch("u", "feature")
        fetch_call = mock_git.call_args_list[0].args
        assert fetch_call[2].startswith("+feature:")

    def test_forced_refspec_survives_amended_fork_head(self, tmp_path: Path) -> None:
        def git(cwd: Path, *args: str) -> str:
            return subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@x", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

        fork = tmp_path / "fork"
        fork.mkdir()
        git(fork, "init", "-q", "-b", "feature")
        (fork / "f.txt").write_text("one\n")
        git(fork, "add", "f.txt")
        git(fork, "commit", "-qm", "first")

        local = tmp_path / "local"
        local.mkdir()
        git(local, "init", "-q", "-b", "main")
        git(local, "fetch", "-q", str(fork), "+feature:refs/pr-split/pr-1")

        # Rewrite the commit the local ref already points at: non-fast-forward.
        (fork / "f.txt").write_text("two\n")
        git(fork, "add", "f.txt")
        git(fork, "commit", "-q", "--amend", "-m", "rewritten")
        new_head = git(fork, "rev-parse", "HEAD")

        # Without "+" git refuses the non-fast-forward update.
        with pytest.raises(subprocess.CalledProcessError):
            git(local, "fetch", "-q", str(fork), "feature:refs/pr-split/pr-1")
        git(local, "fetch", "-q", str(fork), "+feature:refs/pr-split/pr-1")
        assert git(local, "rev-parse", "refs/pr-split/pr-1") == new_head
