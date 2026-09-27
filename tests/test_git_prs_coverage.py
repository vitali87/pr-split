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
    find_open_pr,
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


def _pr_data(head_repo: str, base_repo: str, *, fork: bool) -> str:
    def repo(full_name: str, *, is_fork: bool) -> dict[str, object]:
        return {
            "fork": is_fork,
            "clone_url": f"https://github.com/{full_name}.git",
            "full_name": full_name,
        }

    return json.dumps(
        {
            "head": {"ref": "feat/1806-constant-node", "repo": repo(head_repo, is_fork=fork)},
            "base": {"ref": "main", "repo": repo(base_repo, is_fork=fork)},
        }
    )


class TestFetchPrHead:
    """A PR's head is fetched as refs/pull/N/head from the repository gh resolved."""

    @pytest.mark.parametrize(
        "origin_url",
        [
            "https://github.com/org/repo.git",
            "git@github.com:Org/Repo.git",
            "ssh://git@github.com/org/repo",
        ],
    )
    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_same_repo_pr_is_fetched_from_origin(
        self, mock_gh: MagicMock, mock_git: MagicMock, origin_url: str
    ) -> None:
        mock_gh.return_value = _pr_data("org/repo", "org/repo", fork=False)
        mock_git.side_effect = [origin_url, "", "A <a@x>"]

        info = fetch_fork_pr(1931)

        assert mock_git.call_args_list[1].args == (
            "fetch",
            "origin",
            "+refs/pull/1931/head:refs/pr-split/pr-1931",
        )
        assert info["local_ref"] == "refs/pr-split/pr-1931"
        assert info["base_branch"] == "main"

    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_internal_pr_of_a_forked_repo_uses_origin(
        self, mock_gh: MagicMock, mock_git: MagicMock
    ) -> None:
        # The repository is itself a fork, so head.repo.fork is true for an internal PR.
        mock_gh.return_value = _pr_data("me/repo", "me/repo", fork=True)
        mock_git.side_effect = ["git@github.com:me/repo.git", "", "A <a@x>"]

        fetch_fork_pr(7)

        assert mock_git.call_args_list[1].args[1] == "origin"

    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_pr_of_another_repo_is_not_fetched_from_origin(
        self, mock_gh: MagicMock, mock_git: MagicMock
    ) -> None:
        # gh resolved upstream while origin is the user's fork: a same-named
        # branch on origin must not stand in for the PR's head.
        mock_gh.return_value = _pr_data("up/repo", "up/repo", fork=False)
        mock_git.side_effect = ["git@github.com:me/repo.git", "", "A <a@x>"]

        fetch_fork_pr(7)

        assert mock_git.call_args_list[1].args == (
            "fetch",
            "https://github.com/up/repo.git",
            "+refs/pull/7/head:refs/pr-split/pr-7",
        )

    @patch("pr_split.git_ops.branches.run_git")
    @patch("pr_split.git_ops.prs._run_gh")
    def test_fetch_failure_names_the_source(self, mock_gh: MagicMock, mock_git: MagicMock) -> None:
        mock_gh.return_value = _pr_data("org/repo", "org/repo", fork=False)
        mock_git.side_effect = ["https://github.com/org/repo", GitOperationError("no ref")]

        with pytest.raises(GitOperationError, match="head of PR #7 from origin: no ref"):
            fetch_fork_pr(7)


class TestFetchForkPrMalformedResponse:
    @pytest.mark.parametrize(
        "raw", ["not json", "[]", '{"head": {}}'], ids=["text", "list", "no-base"]
    )
    @patch("pr_split.git_ops.prs._run_gh")
    def test_malformed_response_is_a_git_operation_error(
        self, mock_gh: MagicMock, raw: str
    ) -> None:
        mock_gh.return_value = raw
        with pytest.raises(GitOperationError, match="Unexpected response from GitHub for PR #42"):
            fetch_fork_pr(42)


class TestFindOpenPr:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_only_an_exact_same_repository_head_counts(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = json.dumps(
            [
                {"number": 1, "url": "u1", "headRefName": "feat/x-2", "isCrossRepository": False},
                {"number": 2, "url": "u2", "headRefName": "feat/x", "isCrossRepository": True},
                {"number": 3, "url": "u3", "headRefName": "feat/x", "isCrossRepository": False},
            ]
        )
        assert find_open_pr("feat/x") == (3, "u3")

    @patch("pr_split.git_ops.prs._run_gh")
    def test_no_exact_match_is_none(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = json.dumps(
            [{"number": 1, "url": "u1", "headRefName": "feat/x-2", "isCrossRepository": False}]
        )
        assert find_open_pr("feat/x") is None
