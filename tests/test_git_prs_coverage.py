"""Tests to improve coverage for pr_split/git_ops/prs.py.

Covers: get_pr_state (JSON decode error), merge_pr, fetch_fork_pr (happy path),
fetch_fork_branch (happy path and error paths).
"""

from __future__ import annotations

import json
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
