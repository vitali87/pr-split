from __future__ import annotations

import contextlib
import threading
from unittest.mock import MagicMock, patch

import pytest

from pr_split.cli import (
    _create_branches_and_commits,
    _link_stacks,
    _push_and_create_prs,
    _render_dag,
    _render_dag_markdown,
    _resolve_fork_ref,
    _stacked_batch_args,
)
from pr_split.constants import AssignmentType
from pr_split.exceptions import GitOperationError, PRSplitError
from pr_split.graph import PlanDAG
from pr_split.schemas import BranchRecord, Group, GroupAssignment, PRRecord


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
        with pytest.raises((Exception, SystemExit)):
            _resolve_fork_ref("#42")

    def test_colon_format(self) -> None:
        with pytest.raises((Exception, SystemExit)):
            _resolve_fork_ref("user:branch")


class TestRenderDagMarkdownExtended:
    def test_linear_chain_marks_only_current(self) -> None:
        g1 = _group("pr-1", "first")
        g2 = _group("pr-2", "second", depends_on=["pr-1"])
        result = _render_dag_markdown([g1, g2], "pr-2")
        assert "<-- this PR" in result
        lines = result.splitlines()
        pr1_lines = [line for line in lines if "pr-1" in line]
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


_FORK_INFO = {
    "pr_number": 42,
    "local_ref": "ref",
    "base_branch": "main",
    "author": "a",
    "fork_full_name": "u/r",
}


class TestResolveForkRefExtended:
    @patch("pr_split.cli.fetch_fork_pr")
    def test_pr_number_fork_ref(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = _FORK_INFO
        result = _resolve_fork_ref("#42")
        mock_fetch.assert_called_once_with(42)
        assert result is not None

    @patch("pr_split.cli.fetch_fork_branch")
    def test_colon_fork_ref(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {**_FORK_INFO, "pr_number": None}
        result = _resolve_fork_ref("user:branch")
        mock_fetch.assert_called_once_with("user", "branch")
        assert result is not None

    @patch("pr_split.cli.fetch_fork_pr")
    def test_bare_number_treated_as_pr(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {**_FORK_INFO, "pr_number": 7}
        _resolve_fork_ref("7")
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
        barrier = threading.Barrier(3, timeout=5)
        lock = threading.Lock()
        max_concurrent_val = 0
        active_count = 0

        def barrier_create(**kwargs) -> tuple[int, str]:
            nonlocal max_concurrent_val, active_count
            with lock:
                active_count += 1
                if active_count > max_concurrent_val:
                    max_concurrent_val = active_count
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait()
            with lock:
                active_count -= 1
            return (1, "https://github.com/pr/1")

        mock_create.side_effect = barrier_create

        groups = [_group(f"pr-{i}", f"feat: {i}") for i in range(1, 7)]
        records = [_branch_record(f"pr-{i}", f"pr-split/ns/pr-{i}") for i in range(1, 7)]
        _push_and_create_prs(groups, records)

        assert mock_push.call_count == 6
        assert mock_create.call_count == 6
        assert max_concurrent_val >= 3


class TestCreateBranchesAndCommitsStacked:
    def _stacked_groups(self) -> list[Group]:
        return [_group("pr-2", "feat: base"), _group("pr-3", "feat: top", ["pr-2"])]

    @patch("pr_split.cli.commit_files_in_dir", return_value="sha1")
    @patch("pr_split.cli.materialize_group_files", return_value={})
    @patch("pr_split.cli.remove_worktree")
    @patch("pr_split.cli.add_worktree")
    def test_child_branch_starts_from_parent_branch(
        self,
        mock_add: MagicMock,
        mock_remove: MagicMock,
        mock_mat: MagicMock,
        mock_commit: MagicMock,
    ) -> None:
        _create_branches_and_commits(
            self._stacked_groups(), MagicMock(), "main", "base_sha", "ns", stacked=True
        )
        start_points = {call.args[1]: call.args[2] for call in mock_add.call_args_list}
        assert start_points["pr-split/ns/pr-2"] == "base_sha"
        assert start_points["pr-split/ns/pr-3"] == "pr-split/ns/pr-2"

    @patch("pr_split.cli.commit_files_in_dir", return_value="sha1")
    @patch("pr_split.cli.materialize_group_files", return_value={})
    @patch("pr_split.cli.remove_worktree")
    @patch("pr_split.cli.add_worktree")
    def test_child_pr_base_is_parent_branch(
        self,
        mock_add: MagicMock,
        mock_remove: MagicMock,
        mock_mat: MagicMock,
        mock_commit: MagicMock,
    ) -> None:
        records = _create_branches_and_commits(
            self._stacked_groups(), MagicMock(), "main", "base_sha", "ns", stacked=True
        )
        bases = {r.group_id: r.base_branch for r in records}
        assert bases == {"pr-2": "main", "pr-3": "pr-split/ns/pr-2"}

    @patch("pr_split.cli.commit_files_in_dir", return_value="sha1")
    @patch("pr_split.cli.materialize_group_files", return_value={})
    @patch("pr_split.cli.remove_worktree")
    @patch("pr_split.cli.add_worktree")
    def test_merge_node_falls_back_to_base_branch(
        self,
        mock_add: MagicMock,
        mock_remove: MagicMock,
        mock_mat: MagicMock,
        mock_commit: MagicMock,
    ) -> None:
        groups = [
            _group("pr-1", "a"),
            _group("pr-2", "b"),
            _group("pr-3", "c", ["pr-1", "pr-2"]),
        ]
        records = _create_branches_and_commits(
            groups, MagicMock(), "main", "base_sha", "ns", stacked=True
        )
        bases = {r.group_id: r.base_branch for r in records}
        assert bases["pr-3"] == "main"
        start_points = {call.args[1]: call.args[2] for call in mock_add.call_args_list}
        assert start_points["pr-split/ns/pr-3"] == "base_sha"

    @patch("pr_split.cli.commit_files_in_dir", return_value="sha1")
    @patch("pr_split.cli.materialize_group_files", return_value={})
    @patch("pr_split.cli.remove_worktree")
    @patch("pr_split.cli.add_worktree")
    def test_child_materializes_with_ancestor_hunks(
        self,
        mock_add: MagicMock,
        mock_remove: MagicMock,
        mock_mat: MagicMock,
        mock_commit: MagicMock,
    ) -> None:
        parent = _group("pr-2", "feat: base")
        parent.assignments = [
            GroupAssignment(
                file_path="shared.py",
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[0],
            )
        ]
        child = _group("pr-3", "feat: top", ["pr-2"])
        child.assignments = [
            GroupAssignment(
                file_path="shared.py",
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[1],
            )
        ]
        _create_branches_and_commits(
            [parent, child], MagicMock(), "main", "base_sha", "ns", stacked=True
        )
        child_calls = [
            call for call in mock_mat.call_args_list if call.args[1].id == "pr-3"
        ]
        assert child_calls[0].args[1].assignments[0].hunk_indices == [0, 1]
        assert child_calls[0].args[2] == "base_sha"

    @patch("pr_split.cli.commit_files_in_dir", return_value="sha1")
    @patch("pr_split.cli.materialize_group_files", return_value={})
    @patch("pr_split.cli.remove_worktree")
    @patch("pr_split.cli.add_worktree")
    def test_flat_mode_unchanged(
        self,
        mock_add: MagicMock,
        mock_remove: MagicMock,
        mock_mat: MagicMock,
        mock_commit: MagicMock,
    ) -> None:
        records = _create_branches_and_commits(
            self._stacked_groups(), MagicMock(), "main", "base_sha", "ns"
        )
        start_points = {call.args[1]: call.args[2] for call in mock_add.call_args_list}
        assert set(start_points.values()) == {"base_sha"}
        assert {r.base_branch for r in records} == {"main"}


class TestLinkStacks:
    @patch("pr_split.cli.link_stack")
    def test_links_only_chains_of_two_or_more(self, mock_link: MagicMock) -> None:
        groups = [
            _group("pr-1", "a"),
            _group("pr-2", "b"),
            _group("pr-3", "c", ["pr-2"]),
        ]
        prs = [
            PRRecord(group_id="pr-1", pr_number=11, pr_url="u"),
            PRRecord(group_id="pr-2", pr_number=12, pr_url="u"),
            PRRecord(group_id="pr-3", pr_number=13, pr_url="u"),
        ]
        _link_stacks(PlanDAG(groups), prs)
        mock_link.assert_called_once_with([12, 13])


class TestPushAndCreatePrsDraft:
    @patch("pr_split.cli.create_pr", return_value=(1, "https://github.com/pr/1"))
    @patch("pr_split.cli.push_branch")
    def test_draft_forwarded_to_every_pr(
        self, mock_push: MagicMock, mock_create: MagicMock
    ) -> None:
        groups = [_group("pr-1", "feat: a"), _group("pr-2", "feat: b")]
        records = [
            _branch_record("pr-1", "pr-split/ns/pr-1"),
            _branch_record("pr-2", "pr-split/ns/pr-2"),
        ]
        _push_and_create_prs(groups, records, draft=True)
        assert [call.kwargs["draft"] for call in mock_create.call_args_list] == [True, True]


class TestStackedBatchArgsMergeNode:
    def _diamond(self) -> list[Group]:
        left = _group("pr-1", "left")
        left.assignments = [
            GroupAssignment(
                file_path="left.py",
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[0],
            )
        ]
        right = _group("pr-2", "right")
        right.assignments = [
            GroupAssignment(
                file_path="right.py",
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[0],
            )
        ]
        child = _group("pr-3", "merge", ["pr-1", "pr-2"])
        child.assignments = [
            GroupAssignment(
                file_path="child.py",
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[0],
            )
        ]
        return [left, right, child]

    def _merge_node_args(self) -> tuple[Group, str, str]:
        groups = self._diamond()
        batches = _stacked_batch_args(
            PlanDAG(groups),
            {g.id: g for g in groups},
            {g.id: f"pr-split/ns/{g.id}" for g in groups},
            "main",
            "base_sha",
            {"left.py": 1, "right.py": 1, "child.py": 1},
        )
        return next(
            (merged, base, start)
            for batch in batches
            for merged, base, start in batch
            if merged.id == "pr-3"
        )

    def test_merge_node_carries_both_parents_changes(self) -> None:
        merged, _, _ = self._merge_node_args()
        assert {a.file_path for a in merged.assignments} == {
            "left.py",
            "right.py",
            "child.py",
        }

    def test_merge_node_still_builds_from_merge_base(self) -> None:
        _, base, start = self._merge_node_args()
        assert (base, start) == ("main", "base_sha")


class TestPushFailureGating:
    @patch("pr_split.cli.create_pr", return_value=(1, "https://github.com/pr/1"))
    @patch("pr_split.cli.push_branch")
    def test_child_pr_skipped_when_parent_push_fails(
        self, mock_push: MagicMock, mock_create: MagicMock
    ) -> None:
        groups = [_group("pr-1", "feat: a"), _group("pr-2", "feat: b", ["pr-1"])]
        records = [
            _branch_record("pr-1", "pr-split/ns/pr-1"),
            BranchRecord(
                group_id="pr-2",
                branch_name="pr-split/ns/pr-2",
                base_branch="pr-split/ns/pr-1",
                commit_sha="abc123",
            ),
        ]

        def push(branch: str) -> None:
            if branch == "pr-split/ns/pr-1":
                raise GitOperationError("push rejected")

        mock_push.side_effect = push
        with pytest.raises(PRSplitError):
            _push_and_create_prs(groups, records)
        assert mock_create.call_count == 0

    @patch("pr_split.cli.create_pr")
    @patch("pr_split.cli.push_branch")
    def test_partial_pr_records_ride_on_the_error(
        self, mock_push: MagicMock, mock_create: MagicMock
    ) -> None:
        from pr_split.exceptions import PRCreationError

        groups = [_group("pr-1", "feat: a"), _group("pr-2", "feat: b")]
        records = [
            _branch_record("pr-1", "pr-split/ns/pr-1"),
            _branch_record("pr-2", "pr-split/ns/pr-2"),
        ]

        def create(
            *, head: str, base: str, title: str, body: str, draft: bool = False
        ) -> tuple[int, str]:
            if head == "pr-split/ns/pr-2":
                raise GitOperationError("boom")
            return (11, "https://github.com/pr/11")

        mock_create.side_effect = create
        with pytest.raises(PRCreationError) as excinfo:
            _push_and_create_prs(groups, records)
        assert [r.pr_number for r in excinfo.value.pr_records] == [11]


class TestStackedTransitiveChain:
    @patch("pr_split.cli.commit_files_in_dir", return_value="sha1")
    @patch("pr_split.cli.materialize_group_files", return_value={})
    @patch("pr_split.cli.remove_worktree")
    @patch("pr_split.cli.add_worktree")
    def test_grandchild_materializes_with_all_ancestor_hunks(
        self,
        mock_add: MagicMock,
        mock_remove: MagicMock,
        mock_mat: MagicMock,
        mock_commit: MagicMock,
    ) -> None:
        grandparent = _group("pr-1", "feat: base")
        grandparent.assignments = [
            GroupAssignment(
                file_path="shared.py",
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[0],
            )
        ]
        parent = _group("pr-2", "feat: mid", ["pr-1"])
        parent.assignments = [
            GroupAssignment(
                file_path="shared.py",
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[1],
            )
        ]
        child = _group("pr-3", "feat: top", ["pr-2"])
        child.assignments = [
            GroupAssignment(
                file_path="shared.py",
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[2],
            )
        ]
        _create_branches_and_commits(
            [grandparent, parent, child], MagicMock(), "main", "base_sha", "ns", stacked=True
        )
        child_calls = [
            call for call in mock_mat.call_args_list if call.args[1].id == "pr-3"
        ]
        assert child_calls[0].args[1].assignments[0].hunk_indices == [0, 1, 2]
