from __future__ import annotations

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
