from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pr_split.exceptions import GitOperationError
from pr_split.git_ops.prs import (
    _run_gh,
    check_gh_auth,
    check_gh_stack,
    close_pr,
    create_pr,
    fetch_fork_pr,
    link_stack,
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
        number, _url = create_pr("head", "base", "Title", "Body")
        assert number == 99

    @patch("pr_split.git_ops.prs._run_gh")
    def test_multiline_output_uses_last_line(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "Creating PR...\nhttps://github.com/org/repo/pull/55"
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


class TestLinkStack:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_links_bottom_to_top(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = ""
        link_stack([12, 34, 56])
        mock_gh.assert_called_once_with("stack", "link", "12", "34", "56")

    @patch("pr_split.git_ops.prs._run_gh")
    def test_failure_raises(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = GitOperationError("unknown command: stack")
        with pytest.raises(GitOperationError, match="Failed to link stack for PRs \\[12, 34\\]"):
            link_stack([12, 34])


class TestCheckGhStack:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_installed(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "gh stack\tgithub/gh-stack\tv0.1.0\ngh dash\tdlvhdr/gh-dash\tv4.0.0"
        assert check_gh_stack() is True
        mock_gh.assert_called_once_with("extension", "list")

    @patch("pr_split.git_ops.prs._run_gh")
    def test_not_installed(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "gh dash\tdlvhdr/gh-dash\tv4.0.0"
        assert check_gh_stack() is False

    @patch("pr_split.git_ops.prs._run_gh")
    def test_lookalike_name_does_not_count(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "gh stack\tsomeone/github-gh-stack-fork\tv1.0.0"
        assert check_gh_stack() is False

    @patch("pr_split.git_ops.prs._run_gh")
    def test_gh_failure_propagates(self, mock_gh: MagicMock) -> None:
        mock_gh.side_effect = GitOperationError("gh not found")
        with pytest.raises(GitOperationError, match="gh not found"):
            check_gh_stack()


class TestCreatePrDraft:
    @patch("pr_split.git_ops.prs._run_gh")
    def test_draft_flag_forwarded(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "https://github.com/org/repo/pull/7"
        create_pr("head", "main", "Title", "Body", draft=True)
        assert "--draft" in mock_gh.call_args.args

    @patch("pr_split.git_ops.prs._run_gh")
    def test_ready_by_default(self, mock_gh: MagicMock) -> None:
        mock_gh.return_value = "https://github.com/org/repo/pull/7"
        create_pr("head", "main", "Title", "Body")
        assert "--draft" not in mock_gh.call_args.args


class TestCreatePrBodyLimit:
    @patch("pr_split.git_ops.prs._run_gh", return_value="https://github.com/o/r/pull/9")
    def test_oversized_body_is_truncated_below_the_limit(self, mock_gh: MagicMock) -> None:
        from pr_split.git_ops.prs import PR_BODY_MAX_CHARS

        create_pr(head="h", base="b", title="t", body="x" * 100_000)
        args = mock_gh.call_args[0]
        sent = args[args.index("--body") + 1]
        assert len(sent) <= PR_BODY_MAX_CHARS
        assert sent.endswith("_(body truncated: GitHub's limit is 65,536 characters)_")

    @patch("pr_split.git_ops.prs._run_gh", return_value="https://github.com/o/r/pull/9")
    def test_truncation_closes_an_open_code_fence_on_a_line_boundary(
        self, mock_gh: MagicMock
    ) -> None:
        from pr_split.git_ops.prs import PR_BODY_MAX_CHARS

        lines = "\n".join(f"pr-{i}: group {i}" for i in range(6000))
        body = f"intro\n\n```\n{lines}\n```"
        create_pr(head="h", base="b", title="t", body=body)
        args = mock_gh.call_args[0]
        sent = args[args.index("--body") + 1]
        assert len(sent) <= PR_BODY_MAX_CHARS
        assert sent.count("```") % 2 == 0
        assert sent.endswith("```\n\n_(body truncated: GitHub's limit is 65,536 characters)_")
        # cut on a line boundary: the line before the closing fence is intact
        before_close = sent.rsplit("\n```", 1)[0]
        last_line = before_close.rsplit("\n", 1)[-1]
        assert last_line.startswith("pr-")
        assert (
            last_line
            == f"{last_line.split(':')[0]}: group {last_line.split('-')[1].split(':')[0]}"
        )

    @patch("pr_split.git_ops.prs._run_gh", return_value="https://github.com/o/r/pull/9")
    def test_normal_body_is_sent_verbatim(self, mock_gh: MagicMock) -> None:
        create_pr(head="h", base="b", title="t", body="hello")
        args = mock_gh.call_args[0]
        assert args[args.index("--body") + 1] == "hello"
