"""Move one hunk between layers of an executed stack, keeping every PR open.

The hunk is reverted on the source layer and the revert is carried up the
stack by a restack, so the layers in between lose it too. The hunk is then
applied again on the target layer and the layers above it are restacked.
Only branches whose content changed are pushed, so no PR is closed and no
review thread is lost. Each step patches the layer's current content with
``git apply``, so changes that reached a layer by other routes, such as a
base merge, are left as they are.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from . import logs
from .constants import AssignmentType
from .diff_ops import parse_diff
from .exceptions import ErrorMsg, GitOperationError, PRSplitError
from .git_ops import remove_worktree
from .git_ops.branches import run_git, run_git_in_dir
from .graph import PlanDAG
from .planner.partitioning import refresh_generated_description
from .restack import LayerResult, restack
from .schemas import GroupAssignment

if TYPE_CHECKING:
    from unidiff import PatchedFile

    from .schemas import Group, PlanFile


def _hunk_patch(patch_file: PatchedFile, index: int) -> str:
    path = patch_file.path
    if patch_file.is_added_file:
        header = (
            f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
        )
    else:
        header = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
    return header + str(patch_file[index])


def _commit_patch(branch: str, patch: str, *, reverse: bool, message: str) -> None:
    """Apply the patch to the layer's current tip and commit it to the branch."""
    worktree = tempfile.mkdtemp(prefix="pr-split-move-")
    run_git("worktree", "add", worktree, branch)
    try:
        patch_path = Path(worktree).parent / f"{Path(worktree).name}.patch"
        patch_path.write_text(patch, encoding="utf-8", errors="surrogateescape")
        try:
            args = ["apply", "--index", *(["--reverse"] if reverse else []), str(patch_path)]
            run_git_in_dir(worktree, *args)
        except GitOperationError as exc:
            raise PRSplitError(
                ErrorMsg.MOVE_HUNK_DOES_NOT_APPLY(branch=branch, detail=exc)
            ) from exc
        finally:
            patch_path.unlink(missing_ok=True)
        run_git_in_dir(
            worktree, "-c", "core.hooksPath=/dev/null", "commit", "--no-verify", "-m", message
        )
    finally:
        remove_worktree(worktree)


def _move_in_plan(
    groups: list[Group], path: str, index: int, source: Group, target: Group, hunk_count: int
) -> None:
    for assignment in list(source.assignments):
        if assignment.file_path != path:
            continue
        covered = assignment.covered_indices(hunk_count)
        assignment.assignment_type = AssignmentType.PARTIAL_HUNKS
        assignment.hunk_indices = [i for i in covered if i != index]
        if not assignment.hunk_indices:
            source.assignments.remove(assignment)
    existing = next((a for a in target.assignments if a.file_path == path), None)
    if existing is None:
        target.assignments.append(
            GroupAssignment(
                file_path=path, assignment_type=AssignmentType.PARTIAL_HUNKS, hunk_indices=[index]
            )
        )
    elif index not in existing.covered_indices(hunk_count):
        existing.hunk_indices = sorted({*existing.hunk_indices, index})
    for group in (source, target):
        refresh_generated_description(group)


def move_hunk(
    plan_file: PlanFile, path: str, index: int, source_id: str, target_id: str
) -> list[LayerResult]:
    plan = plan_file.plan
    if not plan.stacked:
        raise PRSplitError(ErrorMsg.RESTACK_NOT_STACKED())
    if plan.raw_diff is None:
        raise PRSplitError(ErrorMsg.MOVE_NO_DIFF())
    groups = {g.id: g for g in plan.groups}
    branches = {r.group_id: r.branch_name for r in plan_file.git_state.branches}
    for gid in (source_id, target_id):
        if gid not in groups or gid not in branches:
            raise PRSplitError(ErrorMsg.MOVE_UNKNOWN_LAYER(group=gid))
    dag = PlanDAG(plan.groups)
    if source_id not in dag.ancestors(target_id):
        raise PRSplitError(ErrorMsg.MOVE_NOT_UPWARD(source=source_id, target=target_id))
    # restack only rebuilds single-parent layers, so the path must be a chain.
    node = target_id
    while node != source_id:
        parents = dag.parents(node)
        if len(parents) != 1:
            raise PRSplitError(ErrorMsg.MOVE_NOT_A_CHAIN(group=node))
        node = parents[0]

    parsed = parse_diff(plan.raw_diff, split_new_files_over=plan.split_new_files_over)
    patch_file = next((pf for pf in parsed.patch_set if pf.path == path), None)
    if patch_file is None or not 0 <= index < len(patch_file):
        raise PRSplitError(ErrorMsg.MOVE_UNKNOWN_HUNK(file=path, index=index))
    if patch_file.is_removed_file or (patch_file.is_added_file and len(patch_file) > 1):
        raise PRSplitError(ErrorMsg.MOVE_UNSUPPORTED_FILE(file=path))
    held = {
        i
        for a in groups[source_id].assignments
        if a.file_path == path
        for i in a.covered_indices(len(patch_file))
    }
    if index not in held:
        raise PRSplitError(
            ErrorMsg.MOVE_HUNK_NOT_IN_LAYER(file=path, index=index, group=source_id)
        )

    patch = _hunk_patch(patch_file, index)
    _commit_patch(
        branches[source_id],
        patch,
        reverse=True,
        message=f"Move {path} hunk {index} up to {target_id}",
    )
    logger.info(logs.MOVE_REVERTED.format(file=path, index=index, branch=branches[source_id]))
    first = restack(plan_file)
    _commit_patch(
        branches[target_id],
        patch,
        reverse=False,
        message=f"Take {path} hunk {index} from {source_id}",
    )
    logger.info(logs.MOVE_APPLIED.format(file=path, index=index, branch=branches[target_id]))
    second = restack(plan_file)

    _move_in_plan(plan.groups, path, index, groups[source_id], groups[target_id], len(patch_file))
    # Report each layer once, with its most significant outcome.
    rank = {"restacked": 2, "pushed": 1}
    merged: dict[str, LayerResult] = {}
    for result in [*first, *second]:
        current = merged.get(result.group_id)
        if current is None or rank.get(result.action, 0) > rank.get(current.action, 0):
            merged[result.group_id] = result
    return [merged[r.group_id] for r in plan_file.git_state.branches if r.group_id in merged]
