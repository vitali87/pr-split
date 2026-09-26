"""Rebuild a lost plan from the branches and PRs of a stack that still exists.

The plan lives in ``.pr-split/plan.json`` of the checkout where ``split`` ran.
When that checkout is gone, every branch and PR of the stack is still on the
remote; this recovers the groups, their dependencies, the branches and the
PRs from them, so ``status``, ``merge``, ``restack`` and ``clean`` work again.

A recovered plan holds no diff and no hunk assignments: commands that rebuild
branches from the plan's hunks (``execute``, ``move``) cannot use it.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass

from loguru import logger

from . import logs
from .constants import BRANCH_PREFIX, DEFAULT_MAX_LOC, Priority, PRState
from .exceptions import ErrorMsg, GitOperationError, PRSplitError
from .git_ops.branches import derive_split_namespace, merge_base, run_git
from .git_ops.prs import list_prs_with_head_prefix
from .graph import PlanDAG
from .schemas import BranchRecord, GitState, Group, PlanFile, PRRecord, SplitPlan

REMOTE = "origin"
_DEPENDS_ON = re.compile(r"^This PR depends on: (.+)$", re.MULTILINE)
_BACKTICKED = re.compile(r"`([^`]+)`")
# An open PR wins over a merged one for the same head, and both over a closed one.
_STATE_RANK = {PRState.OPEN: 0, PRState.MERGED: 1, PRState.CLOSED: 2}


@dataclass(frozen=True)
class StackPR:
    number: int
    url: str
    state: PRState
    head: str
    base: str
    title: str
    body: str

    @classmethod
    def from_gh(cls, raw: dict[str, object]) -> StackPR:
        state = str(raw.get("state") or "").lower()
        return cls(
            number=int(str(raw["number"])),
            url=str(raw.get("url") or ""),
            state=PRState(state) if state in PRState.__members__.values() else PRState.CLOSED,
            head=str(raw.get("headRefName") or ""),
            base=str(raw.get("baseRefName") or ""),
            title=str(raw.get("title") or ""),
            body=str(raw.get("body") or ""),
        )


def stack_prefix(dev_branch_arg: str) -> str:
    return f"{BRANCH_PREFIX}{derive_split_namespace(dev_branch_arg)}/"


def _description(body: str) -> str:
    # pr-split's own sections follow the description; keep only the description.
    return body.split("\n## ", 1)[0].strip()


def _body_dependencies(body: str) -> list[str]:
    match = _DEPENDS_ON.search(body)
    return _BACKTICKED.findall(match.group(1)) if match else []


def _pick_prs(prs: list[StackPR]) -> dict[str, StackPR]:
    """One PR per head branch: the open one, else the merged one, else the newest."""
    picked: dict[str, StackPR] = {}
    for pr in sorted(prs, key=lambda p: (_STATE_RANK[p.state], -p.number)):
        picked.setdefault(pr.head, pr)
    return picked


def _base_branch(prefix: str, prs: list[StackPR], base: str | None) -> str:
    if base:
        return base
    bases = sorted({pr.base for pr in prs if not pr.base.startswith(prefix)})
    if len(bases) != 1:
        raise PRSplitError(
            ErrorMsg.RECOVER_BASE_UNKNOWN(prefix=prefix, bases=", ".join(bases) or "no PRs")
        )
    return bases[0]


def rebuild_plan(
    dev_branch_arg: str,
    *,
    prs: list[StackPR],
    branches: dict[str, str],
    base: str | None = None,
    stacked: bool = False,
    merge_base_sha: str | None = None,
) -> PlanFile:
    """Build a plan from a stack's PRs and its branches (name -> head sha)."""
    prefix = stack_prefix(dev_branch_arg)
    picked = _pick_prs([pr for pr in prs if pr.head.startswith(prefix)])
    heads = sorted(set(picked) | {b for b in branches if b.startswith(prefix)})
    if not heads:
        raise PRSplitError(ErrorMsg.RECOVER_NOTHING_FOUND(prefix=prefix))
    base_branch = _base_branch(prefix, list(picked.values()), base)
    ids = {head: head.removeprefix(prefix) for head in heads}
    known = set(ids.values())

    groups: list[Group] = []
    branch_records: list[BranchRecord] = []
    pr_records: list[PRRecord] = []
    for head in sorted(heads, key=lambda h: (picked[h].number if h in picked else 0, h)):
        gid = ids[head]
        pr = picked.get(head)
        depends_on = _body_dependencies(pr.body) if pr else []
        if pr and pr.base.startswith(prefix):
            # A stacked child targets its parent's branch.
            stacked = True
            depends_on.append(pr.base.removeprefix(prefix))
        depends_on = sorted({d for d in depends_on if d in known and d != gid})
        groups.append(
            Group(
                id=gid,
                title=pr.title if pr else gid,
                description=_description(pr.body) if pr else "",
                depends_on=depends_on,
            )
        )
        branch_records.append(
            BranchRecord(
                group_id=gid,
                branch_name=head,
                base_branch=pr.base if pr else base_branch,
                commit_sha=branches.get(head, ""),
            )
        )
        if pr:
            pr_records.append(
                PRRecord(group_id=gid, pr_number=pr.number, pr_url=pr.url, state=pr.state)
            )

    PlanDAG(groups).validate_acyclic()
    return PlanFile(
        plan=SplitPlan(
            dev_branch=dev_branch_arg,
            base_branch=base_branch,
            max_loc=DEFAULT_MAX_LOC,
            stacked=stacked,
            priority=Priority.ORTHOGONAL,
            groups=groups,
            merge_base_sha=merge_base_sha,
            dev_branch_arg=dev_branch_arg,
        ),
        git_state=GitState(branches=branch_records, prs=pr_records),
    )


def _stack_branches(prefix: str) -> dict[str, str]:
    """Head sha of every branch under ``prefix``: the remote copy, else the local one."""
    remote_refs = f"refs/remotes/{REMOTE}/"
    try:
        run_git(
            "fetch", "--quiet", "--prune", REMOTE, f"+refs/heads/{prefix}*:{remote_refs}{prefix}*"
        )
    except GitOperationError as exc:
        logger.warning(logs.RECOVER_FETCH_FAILED.format(remote=REMOTE, detail=exc))
    found: dict[str, str] = {}
    # Remote refs come last, so the copy the PRs show wins over a local one.
    for refs in ("refs/heads/", remote_refs):
        out = run_git("for-each-ref", "--format=%(refname) %(objectname)", refs + prefix)
        for line in out.splitlines():
            ref, sha = line.split()
            found[ref.removeprefix(refs)] = sha
    return found


def recover_plan(
    dev_branch_arg: str, *, base: str | None = None, stacked: bool = False
) -> PlanFile:
    prefix = stack_prefix(dev_branch_arg)
    try:
        prs = [StackPR.from_gh(raw) for raw in list_prs_with_head_prefix(prefix)]
    except GitOperationError as exc:
        raise PRSplitError(ErrorMsg.RECOVER_PR_LIST_FAILED(detail=exc)) from exc
    plan_file = rebuild_plan(
        dev_branch_arg, prs=prs, branches=_stack_branches(prefix), base=base, stacked=stacked
    )
    plan = plan_file.plan
    # The dev branch may be gone as well; nothing that reads a recovered plan needs this.
    with contextlib.suppress(GitOperationError):
        plan.merge_base_sha = merge_base(plan.dev_branch, plan.base_branch)
    return plan_file
