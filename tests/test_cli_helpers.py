from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pr_split.cli import _render_dag, _render_dag_markdown, _resolve_fork_ref
from pr_split.schemas import Group


def _group(gid: str, title: str, depends_on: list[str] | None = None) -> Group:
    return Group(
        id=gid,
        title=title,
        description=f"desc for {gid}",
        depends_on=depends_on or [],
    )


class TestRenderDag:
    def test_single_root(self) -> None:
        groups = [_group("pr-1", "feat: auth")]
        result = _render_dag(groups)
        assert "pr-1" in result

    def test_linear_chain(self) -> None:
        groups = [
            _group("pr-1", "feat: auth"),
            _group("pr-2", "feat: api", depends_on=["pr-1"]),
        ]
        result = _render_dag(groups)
        assert "pr-1" in result
        assert "pr-2" in result

    def test_diamond(self) -> None:
        groups = [
            _group("pr-1", "base"),
            _group("pr-2", "left", depends_on=["pr-1"]),
            _group("pr-3", "right", depends_on=["pr-1"]),
            _group("pr-4", "merge", depends_on=["pr-2", "pr-3"]),
        ]
        result = _render_dag(groups)
        assert "pr-4" in result


class TestRenderDagMarkdown:
    def test_marks_current_pr(self) -> None:
        groups = [
            _group("pr-1", "base"),
            _group("pr-2", "child", depends_on=["pr-1"]),
        ]
        result = _render_dag_markdown(groups, "pr-2")
        assert "<-- this PR" in result

    def test_root_current_pr(self) -> None:
        groups = [_group("pr-1", "root")]
        result = _render_dag_markdown(groups, "pr-1")
        assert "<-- this PR" in result

    def test_header_present(self) -> None:
        groups = [_group("pr-1", "root")]
        result = _render_dag_markdown(groups, "pr-1")
        assert "## Dependency graph" in result
        assert "Merge in this order" in result


class TestResolveForkRef:
    def test_non_fork_returns_none(self) -> None:
        result = _resolve_fork_ref("regular-branch")
        assert result is None

    def test_pr_number_format(self) -> None:
        with pytest.raises(Exception):
            _resolve_fork_ref("#42")

    def test_colon_format(self) -> None:
        with pytest.raises(Exception):
            _resolve_fork_ref("user:branch")


class TestRenderDagMarkdownExtended:
    def test_linear_chain_marks_only_current(self) -> None:
        g1 = _group("pr-1", "first")
        g2 = _group("pr-2", "second", depends_on=["pr-1"])
        result = _render_dag_markdown([g1, g2], "pr-2")
        assert "<-- this PR" in result
        lines = result.splitlines()
        pr1_lines = [l for l in lines if "pr-1" in l]
        for line in pr1_lines:
            assert "<-- this PR" not in line

    def test_code_block_present(self) -> None:
        groups = [_group("pr-1", "t")]
        result = _render_dag_markdown(groups, "pr-1")
        assert "```" in result


class TestRenderDagRichTreeExtended:
    def test_contains_tree_label(self) -> None:
        groups = [_group("pr-1", "root")]
        result = _render_dag(groups)
        assert "Split Plan" in result

    def test_deps_shown(self) -> None:
        g1 = _group("pr-1", "root")
        g2 = _group("pr-2", "child", depends_on=["pr-1"])
        result = _render_dag([g1, g2])
        assert "pr-2" in result
        assert "depends on" in result


class TestResolveForkRefExtended:
    @patch("pr_split.cli.fetch_fork_pr")
    def test_pr_number_fork_ref(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"pr_number": 42, "local_ref": "ref", "base_branch": "main", "author": "a", "fork_full_name": "u/r"}
        result = _resolve_fork_ref("#42")
        mock_fetch.assert_called_once_with(42)
        assert result is not None

    @patch("pr_split.cli.fetch_fork_branch")
    def test_colon_fork_ref(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"pr_number": None, "local_ref": "ref", "base_branch": "main", "author": "a", "fork_full_name": "u/r"}
        result = _resolve_fork_ref("user:branch")
        mock_fetch.assert_called_once_with("user", "branch")
        assert result is not None

    @patch("pr_split.cli.fetch_fork_pr")
    def test_bare_number_treated_as_pr(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"pr_number": 7, "local_ref": "ref", "base_branch": "main", "author": "a", "fork_full_name": "u/r"}
        result = _resolve_fork_ref("7")
        mock_fetch.assert_called_once_with(7)
