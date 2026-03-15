from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pr_split.exceptions import GitOperationError
from pr_split.git_ops.prs import (
    _run_gh,
    check_gh_auth,
    close_pr,
    create_pr,
    fetch_fork_pr,
)


class TestRunGh:
    @patch("pr_split.git_ops.prs.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["gh", "auth", "status"], returncode=0, stdout="ok\n", stderr=""
        )
        result = _run_gh("auth", "status")
        assert result == "ok"

    @patch("pr_split.git_ops.prs.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["gh", "auth", "status"], returncode=1, stdout="", stderr="not logged in"
        )
        with pytest.raises(GitOperationError, match="not logged in"):
            _run_gh("auth", "status")


class TestCheckGhAuth:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_auth_ok(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "Logged in"
        assert check_gh_auth() is True

    @patch("pr_split.git_ops.prs._run_gh")
    def test_auth_fail(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = GitOperationError("not logged in")
        assert check_gh_auth() is False


class TestCreatePr:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_creates_pr_and_returns_tuple(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "https://github.com/org/repo/pull/42"
        number, url = create_pr("head-branch", "main", "Title", "Body")
        assert number == 42
        assert url == "https://github.com/org/repo/pull/42"

    @patch("pr_split.git_ops.prs._run_gh")
    def test_create_pr_failure_raises(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = GitOperationError("rate limited")
        with pytest.raises(GitOperationError, match="Failed to create PR"):
            create_pr("head", "main", "Title", "Body")


class TestClosePr:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_close_pr(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = ""
        close_pr(42)
        mock_gh.assert_called_once_with("pr", "close", "42")


class TestRunGhExtended:
    @patch("pr_split.git_ops.prs.subprocess.run")
    def test_strips_output(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["gh"], returncode=0, stdout="  data  \n", stderr=""
        )
        assert _run_gh("test") == "data"


class TestCreatePrUrlParsingExtended:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_extracts_number_from_url_with_trailing_slash(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "https://github.com/org/repo/pull/99/"
        number, url = create_pr("head", "base", "Title", "Body")
        assert number == 99

    @patch("pr_split.git_ops.prs._run_gh")
    def test_multiline_output_uses_last_line(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = (
            "Creating PR...\nhttps://github.com/org/repo/pull/55"
        )
        number, url = create_pr("head", "base", "Title", "Body")
        assert number == 55
        assert url == "https://github.com/org/repo/pull/55"


class TestFetchForkPr:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_non_fork_raises(self, mock_gh: MagicMock) -> None:
        import json

        pr_data = {
            "head": {
                "ref": "feature",
                "repo": {"fork": False, "clone_url": "https://x", "full_name": "u/r"},
            },
            "base": {"ref": "main"},
        }
        mock_gh.return_value = json.dumps(pr_data)
        with pytest.raises(GitOperationError):
            fetch_fork_pr(42)

    @patch("pr_split.git_ops.prs._run_gh")
    def test_api_failure_raises(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = GitOperationError("Not Found")
        with pytest.raises(GitOperationError):
            fetch_fork_pr(999)

    @patch("pr_split.git_ops.prs._run_gh")
    def test_invalid_head_structure_raises(self, mock_gh: MagicMock) -> None:
        import json

        mock_gh.return_value = json.dumps({"head": "not_a_dict", "base": {"ref": "main"}})
        with pytest.raises(GitOperationError):
            fetch_fork_pr(42)
