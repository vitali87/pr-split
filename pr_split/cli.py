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
    create_merge_base_branch,
    delete_branch,
    is_worktree_clean,
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
    table.add_column("LOC", justify="right")
    table.add_column("Depends On")
    table.add_column("Files")

    for group in groups:
        files = ", ".join(a.file_path for a in group.assignments)
        deps = ", ".join(group.depends_on) if group.depends_on else ""
        table.add_row(
            group.id,
            group.title,
            str(group.estimated_loc),
            deps,
            files,
        )

    console.print(table)
    dag_text = _render_dag(groups)
    console.print(Panel(dag_text, title="Dependency Graph"))


def _create_branches_and_commits(
    groups: list[Group],
    dag: PlanDAG,
    parsed_diff: ParsedDiff,
    base_branch: str,
) -> list[BranchRecord]:
    branch_records: list[BranchRecord] = []
    branch_map: dict[str, str] = {}

    for batch in dag.iter_ready():
        for group_id in batch:
            group = next(g for g in groups if g.id == group_id)
            parents = dag.parents(group_id)

            if dag.is_merge_node(group_id):
                parent_branches = [branch_map[p] for p in parents]
                merge_base_name = create_merge_base_branch(group_id, parent_branches)
                branch_name = create_group_branch(group_id, merge_base_name)
                record = BranchRecord(
                    group_id=group_id,
                    branch_name=branch_name,
                    base_branch=merge_base_name,
                    merge_base_branch=merge_base_name,
                    merge_base_parents=parent_branches,
                )
            elif not parents:
                branch_name = create_group_branch(group_id, base_branch)
                record = BranchRecord(
                    group_id=group_id,
                    branch_name=branch_name,
                    base_branch=base_branch,
                )
            else:
                parent_branch = branch_map[parents[0]]
                branch_name = create_group_branch(group_id, parent_branch)
                record = BranchRecord(
                    group_id=group_id,
                    branch_name=branch_name,
                    base_branch=parent_branch,
                )

            materialized = materialize_group_files(parsed_diff, group, base_branch)
            for file_path, content in materialized.items():
                p = Path(file_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)

            logger.info(logs.COMMITTING_GROUP.format(group=group.id, title=group.title))
            commit_sha = commit_files(
                list(materialized.keys()),
                group.title,
            )
            record.commit_sha = commit_sha

            branch_map[group_id] = branch_name
            branch_records.append(record)

    return branch_records


def _push_and_create_prs(
    groups: list[Group],
    dag: PlanDAG,
    branch_records: list[BranchRecord],
) -> list[PRRecord]:
    pr_records: list[PRRecord] = []
    record_map = {r.group_id: r for r in branch_records}

    for group_id in dag.topological_order():
        group = next(g for g in groups if g.id == group_id)
        record = record_map[group_id]

        push_branch(record.branch_name)
        logger.info(logs.CREATING_PR.format(group=group.id))
        pr_number, pr_url = create_pr(
            head=record.branch_name,
            base=record.base_branch,
            title=group.title,
            body=group.description,
        )
        pr_records.append(
            PRRecord(
                group_id=group_id,
                pr_number=pr_number,
                pr_url=pr_url,
            )
        )

    return pr_records


@app.command()
def split(
    dev_branch: Annotated[str, typer.Argument(help="Development branch to split")],
    base: Annotated[str, typer.Option(help="Base branch")] = "main",
    max_loc: Annotated[int, typer.Option(help="Max lines of code per sub-PR")] = DEFAULT_MAX_LOC,
    priority: Annotated[Priority, typer.Option(help="Grouping priority")] = Priority.ORTHOGONAL,
) -> None:
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
    checkout_branch(base)
    branch_records = _create_branches_and_commits(groups, dag, parsed_diff, base)
    pr_records = _push_and_create_prs(groups, dag, branch_records)

    checkout_branch(original_branch)

    plan_file = PlanFile(
        plan=SplitPlan(
            dev_branch=dev_branch,
            base_branch=base,
            max_loc=max_loc,
            priority=priority,
            groups=groups,
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
