from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pr_split.cli import (
    _push_and_create_prs,
    _render_dag,
    _render_dag_markdown,
    _resolve_fork_ref,
)
from pr_split.schemas import BranchRecord, Group, PRRecord


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


def _branch_record(group_id: str, branch_name: str) -> BranchRecord:
    return BranchRecord(
        group_id=group_id,
        branch_name=branch_name,
        base_branch="main",
        commit_sha="abc123",
    )


class TestPushAndCreatePrs:
    @patch("pr_split.cli.create_pr", return_value=(1, "https://github.com/pr/1"))
    @patch("pr_split.cli.push_branch")
    def test_returns_records_for_all_groups(
        self, mock_push: MagicMock, mock_create: MagicMock
    ) -> None:
        groups = [_group("pr-1", "feat: a"), _group("pr-2", "feat: b")]
        records = [
            _branch_record("pr-1", "pr-split/ns/pr-1"),
            _branch_record("pr-2", "pr-split/ns/pr-2"),
        ]
        result = _push_and_create_prs(groups, records)
        assert len(result) == 2
        assert result[0].group_id == "pr-1"
        assert result[1].group_id == "pr-2"

    @patch("pr_split.cli.create_pr", return_value=(1, "https://github.com/pr/1"))
    @patch("pr_split.cli.push_branch")
    def test_pushes_all_branches(
        self, mock_push: MagicMock, mock_create: MagicMock
    ) -> None:
        groups = [_group("pr-1", "feat: a"), _group("pr-2", "feat: b")]
        records = [
            _branch_record("pr-1", "pr-split/ns/pr-1"),
            _branch_record("pr-2", "pr-split/ns/pr-2"),
        ]
        _push_and_create_prs(groups, records)
        pushed = {call.args[0] for call in mock_push.call_args_list}
        assert pushed == {"pr-split/ns/pr-1", "pr-split/ns/pr-2"}

    @patch("pr_split.cli.create_pr", return_value=(5, "https://github.com/pr/5"))
    @patch("pr_split.cli.push_branch")
    def test_preserves_group_order(
        self, mock_push: MagicMock, mock_create: MagicMock
    ) -> None:
        groups = [
            _group("pr-3", "c"),
            _group("pr-1", "a"),
            _group("pr-2", "b"),
        ]
        records = [
            _branch_record("pr-3", "pr-split/ns/pr-3"),
            _branch_record("pr-1", "pr-split/ns/pr-1"),
            _branch_record("pr-2", "pr-split/ns/pr-2"),
        ]
        result = _push_and_create_prs(groups, records)
        assert [r.group_id for r in result] == ["pr-3", "pr-1", "pr-2"]

    @patch("pr_split.cli.create_pr")
    @patch("pr_split.cli.push_branch")
    def test_concurrent_execution(
        self, mock_push: MagicMock, mock_create: MagicMock
    ) -> None:
        """Verify that multiple groups are processed concurrently."""
        import threading

        # Barrier for 3 parties proves at least 3 threads run simultaneously
        # (matches _GH_API_CONCURRENCY=3). Timeout prevents hanging if
        # execution is actually sequential.
        barrier = threading.Barrier(3, timeout=5)
        max_concurrent = {"value": 0}
        lock = threading.Lock()
        active = {"count": 0}

        def barrier_create(**kwargs) -> tuple[int, str]:
            with lock:
                active["count"] += 1
                if active["count"] > max_concurrent["value"]:
                    max_concurrent["value"] = active["count"]
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            with lock:
                active["count"] -= 1
            return (1, "https://github.com/pr/1")

        mock_create.side_effect = barrier_create

        groups = [_group(f"pr-{i}", f"feat: {i}") for i in range(1, 6)]
        records = [_branch_record(f"pr-{i}", f"pr-split/ns/pr-{i}") for i in range(1, 6)]
        _push_and_create_prs(groups, records)

        assert mock_push.call_count == 5
        assert mock_create.call_count == 5
        assert max_concurrent["value"] >= 3
