"""Carry a change on a lower stacked layer up to every layer above it.

After ``execute --stack`` each child branch was cut from its parent's head at
that time. A review fix committed on a parent later is missing from every
layer above it, whose diff against its parent then shows the fix reversed.
Restacking rebases each child onto its parent's current head, in plan order,
and pushes the rewritten branches with ``--force-with-lease``.
"""

from __future__ import annotations

import contextlib
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from . import logs
from .exceptions import ErrorMsg, GitOperationError, PRSplitError
from .git_ops import push_branch, remove_worktree
from .git_ops.branches import commit_exists, run_git, run_git_in_dir
from .graph import PlanDAG

if TYPE_CHECKING:
    from .schemas import PlanFile

REMOTE = "origin"


@dataclass
class LayerResult:
    group_id: str
    branch: str
    action: str  # "restacked", "up to date", "pushed" or "skipped: <why>"


def _rev(ref: str) -> str | None:
    if not commit_exists(ref):
        return None
    return run_git("rev-parse", "--verify", f"{ref}^{{commit}}")


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    try:
        run_git("merge-base", "--is-ancestor", ancestor, descendant)
    except GitOperationError:
        return False
    return True


def _remote_ref(branch: str) -> str:
    return f"refs/remotes/{REMOTE}/{branch}"


def _checked_out_branches() -> set[str]:
    out = run_git("worktree", "list", "--porcelain")
    return {
        line.removeprefix("branch refs/heads/")
        for line in out.splitlines()
        if line.startswith("branch refs/heads/")
    }


def _fetch(branches: list[str]) -> None:
    for branch in branches:
        try:
            run_git("fetch", "--quiet", REMOTE, f"+refs/heads/{branch}:{_remote_ref(branch)}")
        except GitOperationError as exc:
            # A layer whose PR merged may have had its branch deleted.
            logger.warning(logs.RESTACK_FETCH_FAILED.format(branch=branch, detail=exc))


def _sync_with_remote(branch: str) -> None:
    """Bring the local branch up to its remote copy (a fix pushed from elsewhere)."""
    local, remote = _rev(f"refs/heads/{branch}"), _rev(_remote_ref(branch))
    if remote is None or local == remote:
        return
    if local is None:
        run_git("branch", branch, remote)
    elif _is_ancestor(local, remote):
        run_git("update-ref", f"refs/heads/{branch}", remote, local)
    elif not _is_ancestor(remote, local):
        raise PRSplitError(ErrorMsg.RESTACK_DIVERGED(branch=branch, remote=REMOTE))


def _rebase(child: str, onto: str, upstream: str) -> str:
    """Replay child's commits after upstream onto ``onto``; return the new head."""
    worktree = tempfile.mkdtemp(prefix="pr-split-restack-")
    run_git("worktree", "add", "--detach", worktree, f"refs/heads/{child}")
    try:
        try:
            # Hooks belong to the user's checkout; the throwaway worktree has
            # none of their tooling, so skip them as execute does.
            run_git_in_dir(
                worktree, "-c", "core.hooksPath=/dev/null", "rebase", "--onto", onto, upstream
            )
        except GitOperationError as exc:
            with contextlib.suppress(GitOperationError):
                run_git_in_dir(worktree, "rebase", "--abort")
            raise PRSplitError(
                ErrorMsg.RESTACK_CONFLICT(branch=child, parent=onto, detail=exc)
            ) from exc
        return run_git_in_dir(worktree, "rev-parse", "HEAD")
    finally:
        remove_worktree(worktree)


def _layers(plan_file: PlanFile) -> list[tuple[str, str, str]]:
    """(group id, branch, parent branch) for every layer stacked on one parent."""
    branches = {r.group_id: r.branch_name for r in plan_file.git_state.branches}
    dag = PlanDAG(plan_file.plan.groups)
    layers: list[tuple[str, str, str]] = []
    for gid in dag.topological_order():
        parents = dag.parents(gid)
        # Roots build on the base branch and merge nodes are rebuilt from the
        # merge base; only a single-parent child is stacked on a branch.
        if len(parents) == 1 and gid in branches and parents[0] in branches:
            layers.append((gid, branches[gid], branches[parents[0]]))
    return layers


def stale_layers(plan_file: PlanFile) -> list[tuple[str, str, str]]:
    """Layers whose branch (local or remote copy) lacks its parent's head. No fetch."""
    if not plan_file.plan.stacked:
        return []
    stale: list[tuple[str, str, str]] = []
    for gid, branch, parent in _layers(plan_file):
        for prefix in ("refs/heads/", f"refs/remotes/{REMOTE}/"):
            child_ref, parent_ref = prefix + branch, prefix + parent
            if (
                commit_exists(child_ref)
                and commit_exists(parent_ref)
                and not _is_ancestor(parent_ref, child_ref)
            ):
                stale.append((gid, branch, parent))
                break
    return stale


def restack(plan_file: PlanFile, *, dry_run: bool = False) -> list[LayerResult]:
    if not plan_file.plan.stacked:
        raise PRSplitError(ErrorMsg.RESTACK_NOT_STACKED())
    if not plan_file.git_state.branches:
        raise PRSplitError(ErrorMsg.RESTACK_NO_BRANCHES())

    order = {g: i for i, g in enumerate(PlanDAG(plan_file.plan.groups).topological_order())}
    records = sorted(plan_file.git_state.branches, key=lambda r: order.get(r.group_id, 0))
    all_branches = [r.branch_name for r in records]
    layers = _layers(plan_file)

    _fetch(all_branches)
    if not dry_run:
        busy = _checked_out_branches() & {branch for _, branch, _ in layers}
        if busy:
            raise PRSplitError(ErrorMsg.RESTACK_CHECKED_OUT(branches=", ".join(sorted(busy))))
        for branch in all_branches:
            if commit_exists(f"refs/heads/{branch}") or commit_exists(_remote_ref(branch)):
                _sync_with_remote(branch)

    results: dict[str, LayerResult] = {}
    old_heads: dict[str, str] = {}
    for gid, branch, parent in layers:
        child_head = _rev(f"refs/heads/{branch}")
        parent_head = _rev(f"refs/heads/{parent}")
        if child_head is None or parent_head is None:
            missing = branch if child_head is None else parent
            results[gid] = LayerResult(gid, branch, f"skipped: branch '{missing}' not found")
            continue
        if _is_ancestor(parent_head, child_head):
            results[gid] = LayerResult(gid, branch, "up to date")
            continue
        if dry_run:
            results[gid] = LayerResult(gid, branch, f"would restack onto {parent}")
            continue
        # The child was cut from the parent's head before the parent was
        # rebased in this run, or from the point where the two diverge.
        upstream = old_heads.get(parent) or run_git("merge-base", child_head, parent_head)
        new_head = _rebase(branch, parent, upstream)
        run_git("update-ref", f"refs/heads/{branch}", new_head, child_head)
        old_heads[branch] = child_head
        logger.info(logs.RESTACKED_LAYER.format(branch=branch, parent=parent))
        results[gid] = LayerResult(gid, branch, "restacked")

    if not dry_run:
        # Push parents before children: a pushed fix on a layer that needed no
        # rebase must reach GitHub too, or the child's diff shows it.
        for record in records:
            branch = record.branch_name
            local, remote = _rev(f"refs/heads/{branch}"), _rev(_remote_ref(branch))
            if local is None or local == remote:
                continue
            push_branch(branch)
            result = results.get(record.group_id)
            if result is None or result.action == "up to date":
                results[record.group_id] = LayerResult(record.group_id, branch, "pushed")

    return [results[r.group_id] for r in records if r.group_id in results]
