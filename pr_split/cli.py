from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from . import logs
from .config import Settings
from .constants import DEFAULT_MAX_LOC, PLAN_DIR, Priority
from .diff_ops import ParsedDiff, extract_diff, materialize_group_files, parse_diff
from .exceptions import ErrorMsg, PRSplitError
from .git_ops import (
    branch_exists,
    check_gh_auth,
    checkout_branch,
    commit_files,
    create_group_branch,
    delete_branch,
    derive_split_namespace,
    fetch_fork_branch,
    fetch_fork_pr,
    is_worktree_clean,
    merge_base,
    push_branch,
)
from .git_ops.prs import close_pr, create_pr
from .graph import PlanDAG
from .plan_store import load_plan, plan_exists, save_plan
from .planner import plan_split, validate_plan
from .schemas import (
    BranchRecord,
    GitState,
    Group,
    PlanFile,
    PRRecord,
    SplitPlan,
)
from .types_defs import ForkPRInfo

app = typer.Typer(name="pr-split", help="Decompose large PRs into reviewable stacked PRs")
console = Console()


def _render_dag(groups: list[Group]) -> str:
    roots = [g for g in groups if not g.depends_on]
    tree = Tree("Split Plan")

    def _add_children(parent_tree: Tree, parent_id: str) -> None:
        children = [g for g in groups if parent_id in g.depends_on]
        for child in children:
            deps_label = ", ".join(child.depends_on)
            branch = parent_tree.add(f"{child.id}: {child.title} (depends on: {deps_label})")
            _add_children(branch, child.id)

    for root in roots:
        root_branch = tree.add(f"{root.id}: {root.title}")
        _add_children(root_branch, root.id)

    with console.capture() as capture:
        console.print(tree)
    return capture.get()


def _render_dag_markdown(groups: list[Group], current_id: str) -> str:
    roots = [g for g in groups if not g.depends_on]
    lines: list[str] = []

    def _add_children(parent_id: str, prefix: str) -> None:
        children = [g for g in groups if parent_id in g.depends_on]
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "\u2514\u2500\u2500" if is_last else "\u251c\u2500\u2500"
            marker = "  <-- this PR" if child.id == current_id else ""
            lines.append(f"{prefix}{connector} {child.id}: {child.title}{marker}")
            extension = "    " if is_last else "\u2502   "
            _add_children(child.id, prefix + extension)

    for root in roots:
        marker = "  <-- this PR" if root.id == current_id else ""
        lines.append(f"{root.id}: {root.title}{marker}")
        _add_children(root.id, "")

    tree_block = "\n".join(lines)
    return f"## Dependency graph\n\nMerge in this order:\n\n```\n{tree_block}\n```"


def _validate_inputs(dev_branch: str, base: str) -> None:
    if not branch_exists(dev_branch):
        console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=dev_branch)}[/red]")
        raise typer.Exit(1)
    if not branch_exists(base):
        console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=base)}[/red]")
        raise typer.Exit(1)
    if not is_worktree_clean():
        console.print(f"[red]{ErrorMsg.DIRTY_WORKTREE()}[/red]")
        raise typer.Exit(1)
    if not check_gh_auth():
        console.print(f"[red]{ErrorMsg.GH_AUTH_FAILED()}[/red]")
        raise typer.Exit(1)


def _present_plan(groups: list[Group], max_loc: int) -> None:
    table = Table(title="Split Plan")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Diff", justify="right")
    table.add_column("Depends On")
    table.add_column("Files")

    for group in groups:
        files = ", ".join(a.file_path for a in group.assignments)
        deps = ", ".join(group.depends_on) if group.depends_on else ""
        diff_str = f"+{group.estimated_added}/-{group.estimated_removed}"
        table.add_row(
            group.id,
            group.title,
            diff_str,
            deps,
            files,
        )

    console.print(table)
    dag_text = _render_dag(groups)
    console.print(Panel(dag_text, title="Dependency Graph"))


def _create_branches_and_commits(
    groups: list[Group],
    parsed_diff: ParsedDiff,
    base_branch: str,
    merge_base_ref: str,
    namespace: str,
    *,
    author: str | None = None,
) -> list[BranchRecord]:
    branch_records: list[BranchRecord] = []

    for group in groups:
        branch_name = create_group_branch(group.id, merge_base_ref, namespace)
        record = BranchRecord(
            group_id=group.id,
            branch_name=branch_name,
            base_branch=base_branch,
        )

        materialized = materialize_group_files(parsed_diff, group, merge_base_ref)
        for file_path, content in materialized.items():
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

        logger.info(logs.COMMITTING_GROUP.format(group=group.id, title=group.title))
        commit_sha = commit_files(
            list(materialized.keys()),
            group.title,
            author=author,
        )
        record.commit_sha = commit_sha
        branch_records.append(record)

    return branch_records


def _push_and_create_prs(
    groups: list[Group],
    branch_records: list[BranchRecord],
) -> list[PRRecord]:
    pr_records: list[PRRecord] = []
    record_map = {r.group_id: r for r in branch_records}

    for group in groups:
        record = record_map[group.id]

        push_branch(record.branch_name)
        logger.info(logs.CREATING_PR.format(group=group.id))
        dag_md = _render_dag_markdown(groups, group.id)
        body = f"{group.description}\n\n{dag_md}"
        pr_number, pr_url = create_pr(
            head=record.branch_name,
            base=record.base_branch,
            title=group.title,
            body=body,
        )
        pr_records.append(
            PRRecord(
                group_id=group.id,
                pr_number=pr_number,
                pr_url=pr_url,
            )
        )

    return pr_records


def _resolve_fork_ref(dev_branch: str) -> ForkPRInfo | None:
    cleaned = dev_branch.lstrip("#")
    if cleaned.isdigit():
        return fetch_fork_pr(int(cleaned))
    if ":" in dev_branch:
        user, branch = dev_branch.split(":", 1)
        return fetch_fork_branch(user, branch)
    return None


@app.command()
def split(
    dev_branch: Annotated[str, typer.Argument(help="Branch name, PR number, or user:branch")],
    base: Annotated[str, typer.Option(help="Base branch")] = "main",
    max_loc: Annotated[
        int, typer.Option(help="Soft limit on diff lines per sub-PR")
    ] = DEFAULT_MAX_LOC,
    priority: Annotated[Priority, typer.Option(help="Grouping priority")] = Priority.ORTHOGONAL,
) -> None:
    dev_branch_arg = dev_branch
    author: str | None = None
    fork_info: ForkPRInfo | None = None

    if not branch_exists(dev_branch):
        if not check_gh_auth():
            console.print(f"[red]{ErrorMsg.GH_AUTH_FAILED()}[/red]")
            raise typer.Exit(1)
        if not is_worktree_clean():
            console.print(f"[red]{ErrorMsg.DIRTY_WORKTREE()}[/red]")
            raise typer.Exit(1)
        fork_info = _resolve_fork_ref(dev_branch)
        if not fork_info:
            console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=dev_branch)}[/red]")
            raise typer.Exit(1)
        dev_branch = fork_info["local_ref"]
        base = fork_info["base_branch"]
        author = fork_info["author"]

    _validate_inputs(dev_branch, base)

    raw_diff = extract_diff(dev_branch, base)
    parsed_diff = parse_diff(raw_diff)
    stats = parsed_diff.stats
    logger.info(
        logs.DIFF_STATS.format(
            files=stats["total_files"],
            added=stats["total_added"],
            removed=stats["total_removed"],
            loc=stats["total_loc"],
        )
    )

    settings = Settings(max_loc=max_loc, priority=priority)
    groups = plan_split(parsed_diff, settings)

    logger.info(logs.VALIDATING_PLAN)
    dag = PlanDAG(groups)
    warnings = validate_plan(groups, parsed_diff, dag, max_loc)
    for warning in warnings:
        logger.warning(warning)
    logger.success(logs.VALIDATION_PASSED)

    logger.info(logs.PRESENTING_PLAN)
    _present_plan(groups, max_loc)

    typer.confirm("Proceed with creating branches and PRs?", abort=True)

    original_branch = dev_branch
    namespace = derive_split_namespace(dev_branch_arg)
    merge_base_ref = merge_base(base, dev_branch)
    checkout_branch(merge_base_ref)
    branch_records = _create_branches_and_commits(
        groups, parsed_diff, base, merge_base_ref, namespace, author=author
    )
    pr_records = _push_and_create_prs(groups, branch_records)

    checkout_branch(original_branch)

    plan_file = PlanFile(
        plan=SplitPlan(
            dev_branch=dev_branch,
            base_branch=base,
            max_loc=max_loc,
            priority=priority,
            groups=groups,
            author=author,
        ),
        git_state=GitState(
            branches=branch_records,
            prs=pr_records,
        ),
    )
    save_plan(plan_file)
    logger.success(f"Split complete: {len(groups)} PRs created")


@app.command()
def status() -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(0)

    plan_file = load_plan()
    plan = plan_file.plan
    git_state = plan_file.git_state

    branch_map = {r.group_id: r.branch_name for r in git_state.branches}
    pr_map = {r.group_id: r for r in git_state.prs}

    table = Table(title="PR Split Status")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Branch")
    table.add_column("PR")
    table.add_column("State")

    for group in plan.groups:
        branch_name = branch_map.get(group.id, "")
        pr_record = pr_map.get(group.id)
        pr_info = f"#{pr_record.pr_number}" if pr_record else ""
        pr_state = pr_record.state.value if pr_record else ""
        table.add_row(group.id, group.title, branch_name, pr_info, pr_state)

    console.print(table)


@app.command()
def clean() -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(0)

    plan_file = load_plan()
    git_state = plan_file.git_state

    typer.confirm("Delete all pr-split branches and close PRs?", abort=True)

    closed_prs = 0
    for pr_record in git_state.prs:
        try:
            close_pr(pr_record.pr_number)
            closed_prs += 1
        except PRSplitError:
            logger.warning(f"Could not close PR #{pr_record.pr_number}")

    logger.info(logs.CLEANING_BRANCHES)
    deleted_branches = 0
    for branch_record in git_state.branches:
        try:
            delete_branch(branch_record.branch_name, remote=True)
            deleted_branches += 1
        except PRSplitError:
            logger.warning(f"Could not delete branch {branch_record.branch_name}")

    plan_dir = Path(PLAN_DIR)
    if plan_dir.exists():
        shutil.rmtree(plan_dir)

    logger.success(logs.CLEAN_COMPLETE.format(branches=deleted_branches, prs=closed_prs))
