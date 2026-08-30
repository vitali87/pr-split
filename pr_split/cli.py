from __future__ import annotations

import json as json_mod
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, Semaphore
from typing import Annotated

import typer
from loguru import logger
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from . import logs
from .config import Settings
from .constants import (
    BRANCH_PREFIX,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_CP_SAT_TIMEOUT_SECONDS,
    DEFAULT_MAX_LOC,
    DEFAULT_MAX_REFINEMENT_ITERATIONS,
    DEFAULT_MIN_LOC,
    DEFAULT_PARTITION_STRATEGY,
    DEFAULT_STRICT_LOC_BOUNDS,
    PLAN_DIR,
    PLAN_FILE,
    AssignmentType,
    ChunkStrategy,
    PartitionStrategy,
    Priority,
)
from .diff_ops import (
    ParsedDiff,
    extract_diff,
    materialize_group_files,
    merge_chain_assignments,
    parse_diff,
)
from .exceptions import ErrorMsg, PlanValidationError, PRCreationError, PRSplitError
from .git_ops import (
    add_worktree,
    branch_exists,
    check_gh_auth,
    commit_files_in_dir,
    delete_branch,
    derive_split_namespace,
    fetch_fork_branch,
    fetch_fork_pr,
    is_worktree_clean,
    merge_base,
    push_branch,
    remove_worktree,
)
from .git_ops.branches import run_git
from .git_ops.prs import close_pr, create_pr, get_pr_state, link_stack, merge_pr
from .graph import PlanDAG
from .plan_store import load_plan, plan_exists, save_plan
from .planner import plan_split, validate_coverage, validate_plan
from .schemas import (
    BranchRecord,
    GitState,
    Group,
    GroupAssignment,
    PlanFile,
    PRRecord,
    SplitPlan,
)
from .types_defs import ForkPRInfo

app = typer.Typer(
    name="pr-split",
    help="Decompose large PRs into reviewable dependency-ordered PRs",
)
console = Console()


def _render_dag(groups: list[Group]) -> str:
    roots = [g for g in groups if not g.depends_on]
    tree = Tree("Split Plan")

    def _add_children(parent_tree: Tree, parent_id: str) -> None:
        children = [g for g in groups if parent_id in g.depends_on]
        for child in children:
            deps_label = ", ".join(child.depends_on)
            branch = parent_tree.add(
                escape(f"{child.id}: {child.title} (depends on: {deps_label})")
            )
            _add_children(branch, child.id)

    for root in roots:
        root_branch = tree.add(escape(f"{root.id}: {root.title}"))
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


def _validate_inputs(dev_branch: str, base: str, *, dry_run: bool = False) -> None:
    if not branch_exists(dev_branch):
        console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=dev_branch)}[/red]")
        raise typer.Exit(1)
    if not branch_exists(base):
        console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=base)}[/red]")
        raise typer.Exit(1)
    if not is_worktree_clean():
        console.print(f"[red]{ErrorMsg.DIRTY_WORKTREE()}[/red]")
        raise typer.Exit(1)
    if not dry_run and not check_gh_auth():
        console.print(f"[red]{ErrorMsg.GH_AUTH_FAILED()}[/red]")
        raise typer.Exit(1)


def _handle_loc_bound_warnings(warnings: list[str], *, strict_loc_bounds: bool) -> None:
    if strict_loc_bounds and warnings:
        console.print(f"[red]{ErrorMsg.LOC_BOUNDS_STRICT_FAILED()}[/red]")
        for warning in warnings:
            console.print(f"[red]- {escape(warning)}[/red]")
        raise typer.Exit(1)

    for warning in warnings:
        logger.warning(warning)


def _present_plan(groups: list[Group]) -> None:
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
        # Plan text is LLM/user-written; "[...]" in it is Rich markup unless escaped.
        table.add_row(
            escape(group.id),
            escape(group.title),
            diff_str,
            escape(deps),
            escape(files),
        )

    console.print(table)
    dag_text = _render_dag(groups)
    console.print(Panel(dag_text, title="Dependency Graph"))


_WORKTREE_MAX_WORKERS = 4
_worktree_ref_lock = Lock()


def _create_single_branch_and_commit(
    group: Group,
    parsed_diff: ParsedDiff,
    base_branch: str,
    merge_base_ref: str,
    namespace: str,
    worktree_base: Path,
    *,
    author: str | None = None,
    start_point: str | None = None,
) -> BranchRecord:
    branch_name = f"{BRANCH_PREFIX}{namespace}/{group.id}"
    worktree_path = str(worktree_base / group.id)
    commit_sha: str = ""

    with _worktree_ref_lock:
        add_worktree(worktree_path, branch_name, start_point or merge_base_ref)
    try:
        materialized = materialize_group_files(parsed_diff, group, merge_base_ref)
        for file_path, content in materialized.items():
            p = Path(worktree_path) / file_path
            if content is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
            elif p.exists():
                p.unlink()

        logger.info(logs.COMMITTING_GROUP.format(group=group.id, title=group.title))
        commit_sha = commit_files_in_dir(
            worktree_path,
            list(materialized.keys()),
            group.title,
            author=author,
        )
    finally:
        try:
            remove_worktree(worktree_path)
        except PRSplitError as exc:
            logger.warning(f"Failed to remove worktree {worktree_path}: {exc}")

    return BranchRecord(
        group_id=group.id,
        branch_name=branch_name,
        base_branch=base_branch,
        commit_sha=commit_sha,
    )


def _stacked_batch_args(
    dag: PlanDAG,
    groups_by_id: dict[str, Group],
    branch_names: dict[str, str],
    base_branch: str,
    merge_base_ref: str,
    hunk_counts: dict[str, int],
) -> Generator[list[tuple[Group, str, str]], None, None]:
    effective: dict[str, Group] = {}
    for batch in dag.iter_ready():
        batch_args: list[tuple[Group, str, str]] = []
        for gid in batch:
            group = groups_by_id[gid]
            parents = dag.parents(gid)
            if len(parents) == 1:
                merged = merge_chain_assignments(group, [effective[parents[0]]], hunk_counts)
                start_point = branch_names[parents[0]]
                group_base = branch_names[parents[0]]
            elif len(parents) > 1:
                # Native stacks are linear, so a merge node builds from the
                # merge base and carries every ancestor's changes itself.
                logger.warning(logs.MERGE_NODE_NOT_STACKED.format(group=gid))
                merged = merge_chain_assignments(
                    group,
                    [groups_by_id[a] for a in sorted(dag.ancestors(gid))],
                    hunk_counts,
                    carry_ancestor_files=True,
                )
                start_point = merge_base_ref
                group_base = base_branch
            else:
                merged = group
                start_point = merge_base_ref
                group_base = base_branch
            effective[gid] = merged
            batch_args.append((merged, group_base, start_point))
        yield batch_args


def _create_branches_and_commits(
    groups: list[Group],
    parsed_diff: ParsedDiff,
    base_branch: str,
    merge_base_ref: str,
    namespace: str,
    *,
    author: str | None = None,
    stacked: bool = False,
) -> list[BranchRecord]:
    worktree_base = Path(tempfile.mkdtemp(prefix="pr-split-worktrees-"))

    if stacked:
        dag = PlanDAG(groups)
        groups_by_id = {g.id: g for g in groups}
        branch_names = {g.id: f"{BRANCH_PREFIX}{namespace}/{g.id}" for g in groups}
        hunk_counts = {pf.path: len(pf) for pf in parsed_diff.patch_set}
        batches = _stacked_batch_args(
            dag, groups_by_id, branch_names, base_branch, merge_base_ref, hunk_counts
        )
    else:
        batches = iter([[(group, base_branch, merge_base_ref) for group in groups]])

    try:
        results: dict[str, BranchRecord] = {}
        errors: list[tuple[str, Exception]] = []
        for batch_args in batches:
            with ThreadPoolExecutor(max_workers=_WORKTREE_MAX_WORKERS) as executor:
                future_to_group_id = {
                    executor.submit(
                        _create_single_branch_and_commit,
                        group,
                        parsed_diff,
                        group_base,
                        merge_base_ref,
                        namespace,
                        worktree_base,
                        author=author,
                        start_point=start_point,
                    ): group.id
                    for group, group_base, start_point in batch_args
                }
                for future in as_completed(future_to_group_id):
                    group_id = future_to_group_id[future]
                    try:
                        results[group_id] = future.result()
                    except Exception as exc:
                        logger.error(f"Failed to create branch for {group_id}: {exc}")
                        errors.append((group_id, exc))
            if errors:
                break

        if errors:
            for record in results.values():
                try:
                    delete_branch(record.branch_name)
                except PRSplitError as exc:
                    logger.warning(f"Could not clean up branch {record.branch_name}: {exc}")
            error_details = "\n".join([f"- {gid}: {exc}" for gid, exc in errors])
            raise PRSplitError(f"{len(errors)} branch(es) failed:\n{error_details}")
    finally:
        shutil.rmtree(worktree_base, ignore_errors=True)
        try:
            run_git("worktree", "prune")
        except PRSplitError as exc:
            logger.warning(f"Failed to prune worktrees: {exc}")

    return [results[g.id] for g in groups]


_PUSH_MAX_WORKERS = 5
_GH_API_CONCURRENCY = 3
_gh_semaphore = Semaphore(_GH_API_CONCURRENCY)


_PR_TEMPLATE_PATH = Path(PLAN_DIR) / "template.md"


def _build_pr_body(group: Group, all_groups: list[Group]) -> str:
    if _PR_TEMPLATE_PATH.exists():
        files = [a.file_path for a in group.assignments]
        template_vars = {
            "description": group.description,
            "files": "\n".join(f"- `{f}`" for f in files),
            "added": group.estimated_added,
            "removed": group.estimated_removed,
            "loc": group.estimated_loc,
            "dependencies": ", ".join(f"`{d}`" for d in group.depends_on),
            "dag": _render_dag_markdown(all_groups, group.id),
            "id": group.id,
            "title": group.title,
        }
        try:
            template = _PR_TEMPLATE_PATH.read_text(encoding="utf-8")
            return template.format(**template_vars)
        except (KeyError, ValueError, IndexError) as exc:
            available = ", ".join(f"{{{k}}}" for k in sorted(template_vars))
            raise PRSplitError(
                f"Invalid PR template at {_PR_TEMPLATE_PATH}: {exc}. "
                f"Available placeholders: {available}. "
                "Escape literal braces with {{ and }}."
            ) from exc
        except OSError as exc:
            raise PRSplitError(
                f"Could not read PR template at {_PR_TEMPLATE_PATH}: {exc}"
            ) from exc

    files = [a.file_path for a in group.assignments]
    sections = [group.description]
    if files:
        file_list = "\n".join(f"- `{f}`" for f in files)
        sections.append(f"## Files changed\n\n{file_list}")
    sections.append(
        f"## Diff stats\n\n"
        f"**+{group.estimated_added}** additions, "
        f"**-{group.estimated_removed}** deletions "
        f"({group.estimated_loc} LOC)"
    )
    if group.depends_on:
        dep_list = ", ".join(f"`{d}`" for d in group.depends_on)
        sections.append(f"## Dependencies\n\nThis PR depends on: {dep_list}")
    sections.append(_render_dag_markdown(all_groups, group.id))
    return "\n\n".join(sections)


def _create_single_pr(
    group: Group,
    record: BranchRecord,
    all_groups: list[Group],
    *,
    draft: bool = False,
) -> PRRecord:
    logger.info(logs.CREATING_PR.format(group=group.id))
    body = _build_pr_body(group, all_groups)
    with _gh_semaphore:
        pr_number, pr_url = create_pr(
            head=record.branch_name,
            base=record.base_branch,
            title=group.title,
            body=body,
            draft=draft,
        )
    return PRRecord(
        group_id=group.id,
        pr_number=pr_number,
        pr_url=pr_url,
    )


def _push_and_create_prs(
    groups: list[Group],
    branch_records: list[BranchRecord],
    *,
    draft: bool = False,
) -> list[PRRecord]:
    record_map = {r.group_id: r for r in branch_records}
    errors: list[tuple[str, Exception]] = []

    # Children target parent branches, so every branch is pushed before any PR opens.
    with ThreadPoolExecutor(max_workers=_PUSH_MAX_WORKERS) as executor:
        push_futures = {
            executor.submit(push_branch, record_map[group.id].branch_name): group.id
            for group in groups
        }
        pushed: set[str] = set()
        for future in as_completed(push_futures):
            group_id = push_futures[future]
            try:
                future.result()
                pushed.add(group_id)
            except Exception as exc:
                logger.error(f"Failed to push branch for {group_id}: {exc}")
                errors.append((group_id, exc))

    branch_owner = {record_map[g.id].branch_name: g.id for g in groups}

    def _base_pushed(group: Group) -> bool:
        # Walk the whole base chain: a pushed leaf must not open a PR when
        # any ancestor branch in its stack failed to push.
        gid = group.id
        while True:
            owner = branch_owner.get(record_map[gid].base_branch)
            if owner is None:
                return True
            if owner not in pushed:
                logger.warning(
                    logs.PR_SKIPPED_BASE_NOT_PUSHED.format(
                        group=group.id, base=record_map[gid].base_branch
                    )
                )
                return False
            gid = owner

    with ThreadPoolExecutor(max_workers=_PUSH_MAX_WORKERS) as executor:
        future_to_group_id = {
            executor.submit(
                _create_single_pr, group, record_map[group.id], groups, draft=draft
            ): group.id
            for group in groups
            if group.id in pushed and _base_pushed(group)
        }
        results: dict[str, PRRecord] = {}
        for future in as_completed(future_to_group_id):
            group_id = future_to_group_id[future]
            try:
                results[group_id] = future.result()
            except Exception as exc:
                logger.error(f"Failed to create PR for {group_id}: {exc}")
                errors.append((group_id, exc))

    if errors:
        error_details = "\n".join([f"- {gid}: {exc}" for gid, exc in errors])
        raise PRCreationError(
            f"{len(errors)} PR(s) failed:\n{error_details}",
            pr_records=[results[g.id] for g in groups if g.id in results],
        )

    return [results[g.id] for g in groups]


def _link_stacks(dag: PlanDAG, pr_records: list[PRRecord]) -> None:
    pr_by_group = {r.group_id: r.pr_number for r in pr_records}
    for chain in dag.linear_chains():
        if len(chain) < 2:
            continue
        link_stack([pr_by_group[gid] for gid in chain])


def _move_assignment(
    groups: list[Group],
    parsed_diff: ParsedDiff,
    file_path: str,
    hunk_index: int,
    from_id: str,
    to_id: str,
) -> bool:
    if from_id == to_id:
        console.print(
            f"[yellow]Source and destination are the same"
            f" ('{escape(from_id)}'). No move performed.[/yellow]"
        )
        return False

    group_map = {g.id: g for g in groups}
    src = group_map.get(from_id)
    dst = group_map.get(to_id)
    if not src or not dst:
        console.print(f"[red]Group '{escape(from_id)}' or '{escape(to_id)}' not found.[/red]")
        return False

    pf_map = {pf.path: pf for pf in parsed_diff.patch_set}

    found = False
    for assignment in src.assignments:
        if assignment.file_path != file_path:
            continue
        # For WHOLE_FILE, check hunk validity before expanding
        if assignment.assignment_type == AssignmentType.WHOLE_FILE:
            pf = pf_map.get(file_path)
            if pf is None:
                continue
            all_indices = list(range(len(pf)))
            if hunk_index not in all_indices:
                continue
            assignment.hunk_indices = all_indices
            assignment.assignment_type = AssignmentType.PARTIAL_HUNKS
        if hunk_index in assignment.hunk_indices:
            assignment.hunk_indices.remove(hunk_index)
            if not assignment.hunk_indices:
                src.assignments.remove(assignment)
            found = True
            break

    if not found:
        console.print(
            f"[red]Hunk {escape(file_path)}:{hunk_index} not found in {escape(from_id)}.[/red]"
        )
        return False

    dst_assignment = next((a for a in dst.assignments if a.file_path == file_path), None)
    if dst_assignment:
        if dst_assignment.assignment_type == AssignmentType.WHOLE_FILE:
            pf = pf_map.get(file_path)
            if pf is not None:
                dst_assignment.hunk_indices = list(range(len(pf)))
                dst_assignment.assignment_type = AssignmentType.PARTIAL_HUNKS
        if hunk_index not in dst_assignment.hunk_indices:
            dst_assignment.hunk_indices.append(hunk_index)
            dst_assignment.hunk_indices.sort()
    else:
        dst.assignments.append(
            GroupAssignment(
                file_path=file_path,
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=[hunk_index],
            )
        )

    console.print(
        f"[green]Moved {escape(file_path)}:{hunk_index} from {escape(from_id)} "
        f"to {escape(to_id)}[/green]"
    )
    return True


def _show_group_detail(groups: list[Group], group_id: str) -> None:
    group_map = {g.id: g for g in groups}
    group = group_map.get(group_id)
    if not group:
        console.print(f"[red]Group '{escape(group_id)}' not found.[/red]")
        return
    # Titles, descriptions and paths come from the plan (LLM-written); any
    # "[...]" in them would be swallowed as Rich markup unless escaped.
    console.print(f"\n[bold]{escape(group.id)}[/bold]: {escape(group.title)}")
    console.print(f"  Description: {escape(group.description)}")
    console.print(f"  Depends on: {escape(', '.join(group.depends_on) or 'none')}")
    console.print(
        f"  Estimated: +{group.estimated_added}/-{group.estimated_removed}"
        f" ({group.estimated_loc} LOC)"
    )
    for a in group.assignments:
        if a.assignment_type == AssignmentType.WHOLE_FILE:
            hunks_str = "all"
        else:
            hunks_str = ", ".join(str(i) for i in a.hunk_indices)
        # Square brackets are Rich markup; without escaping "[whole_file]"
        # is treated as a style tag and silently dropped from the output.
        console.print(escape(f"  {a.file_path} [{a.assignment_type.value}] hunks: [{hunks_str}]"))
    console.print()


def _interactive_edit(groups: list[Group], parsed_diff: ParsedDiff) -> list[Group]:
    console.print(
        "\n[cyan]Interactive editor. Commands:[/cyan]\n"
        "  [bold]move[/bold] <file>:<hunk> <from_group> <to_group>\n"
        "  [bold]show[/bold] <group_id>\n"
        "  [bold]plan[/bold]  — redisplay the plan table\n"
        "  [bold]done[/bold]  — proceed\n"
        "  [bold]abort[/bold] — cancel\n"
    )
    while True:
        try:
            cmd = typer.prompt("edit", default="done")
        except (KeyboardInterrupt, EOFError) as exc:
            raise typer.Abort() from exc

        parts = cmd.strip().split()
        if not parts:
            continue

        action = parts[0].lower()

        if action == "done":
            return groups
        elif action == "abort":
            raise typer.Abort()
        elif action == "plan":
            _present_plan(groups)
        elif action == "show":
            if len(parts) == 2:
                _show_group_detail(groups, parts[1])
            else:
                console.print("[red]Usage: show <group_id>[/red]")
        elif action == "move":
            if len(parts) != 4:
                console.print("[red]Usage: move <file>:<hunk_index> <from_group> <to_group>[/red]")
                continue
            ref, from_id, to_id = parts[1], parts[2], parts[3]
            if ":" not in ref:
                console.print("[red]Usage: move <file>:<hunk_index> <from_group> <to_group>[/red]")
                continue
            file_path, hunk_str = ref.rsplit(":", 1)
            try:
                hunk_index = int(hunk_str)
            except ValueError:
                console.print("[red]Hunk index must be an integer.[/red]")
                continue
            if hunk_index < 0:
                console.print("[red]Hunk index must be non-negative.[/red]")
                continue
            _move_assignment(groups, parsed_diff, file_path, hunk_index, from_id, to_id)
        else:
            console.print(
                "[yellow]Unknown command. Type 'done' to proceed or 'abort' to cancel.[/yellow]"
            )


def _resolve_fork_ref(dev_branch: str) -> ForkPRInfo | None:
    cleaned = dev_branch.lstrip("#")
    if cleaned.isdigit():
        return fetch_fork_pr(int(cleaned))
    if ":" in dev_branch:
        user, branch = dev_branch.split(":", 1)
        return fetch_fork_branch(user, branch)
    return None


@app.command(help="Split a large PR into smaller dependency-ordered PRs.")
def split(
    dev_branch: Annotated[str, typer.Argument(help="Branch name, PR number, or user:branch")],
    base: Annotated[str, typer.Option(help="Base branch")] = "main",
    min_loc: Annotated[
        int | None,
        typer.Option(
            "--min-loc",
            envvar="PR_SPLIT_MIN_LOC",
            help="Minimum target diff lines per sub-PR",
        ),
    ] = DEFAULT_MIN_LOC,
    max_loc: Annotated[
        int,
        typer.Option(
            "--max-loc",
            envvar="PR_SPLIT_MAX_LOC",
            help="Maximum target diff lines per sub-PR",
        ),
    ] = DEFAULT_MAX_LOC,
    strict_loc_bounds: Annotated[
        bool,
        typer.Option(
            "--strict-loc-bounds",
            envvar="PR_SPLIT_STRICT_LOC_BOUNDS",
            help="Fail if the final plan violates configured LOC bounds",
        ),
    ] = DEFAULT_STRICT_LOC_BOUNDS,
    max_refinement_iterations: Annotated[
        int,
        typer.Option(
            "--max-refinement-iterations",
            envvar="PR_SPLIT_MAX_REFINEMENT_ITERATIONS",
            help="Maximum LLM refinement iterations to fix LOC bound violations (0 = disabled)",
        ),
    ] = DEFAULT_MAX_REFINEMENT_ITERATIONS,
    priority: Annotated[Priority, typer.Option(help="Grouping priority")] = Priority.ORTHOGONAL,
    chunk_strategy: Annotated[
        ChunkStrategy, typer.Option(help="Chunking strategy for large diffs")
    ] = DEFAULT_CHUNK_STRATEGY,
    partition_strategy: Annotated[
        PartitionStrategy, typer.Option(help="Backend for hunk-to-PR partitioning")
    ] = DEFAULT_PARTITION_STRATEGY,
    cp_sat_timeout: Annotated[
        float, typer.Option(help="Maximum seconds to spend in the CP-SAT solver")
    ] = DEFAULT_CP_SAT_TIMEOUT_SECONDS,
    stack: Annotated[
        bool,
        typer.Option(
            "--stack",
            envvar="PR_SPLIT_STACK",
            help="Stack dependent PRs: each child branches from and targets its parent's branch",
        ),
    ] = False,
    draft: Annotated[
        bool,
        typer.Option(
            "--draft",
            envvar="PR_SPLIT_DRAFT",
            help="Open every sub-PR as a draft",
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview plan without creating branches or PRs")
    ] = False,
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

    _validate_inputs(dev_branch, base, dry_run=dry_run)

    if plan_exists():
        existing = load_plan()
        has_git_state = existing.git_state.branches or existing.git_state.prs
        if has_git_state:
            console.print("[yellow]An existing split plan with branches/PRs was found.[/yellow]")
            console.print(
                "[red]Warning: this will permanently close PRs and delete remote branches.[/red]"
            )
            if typer.confirm("Clean up and proceed with re-splitting?"):
                closed_prs, deleted_branches = _cleanup_git_state(existing.git_state)
                logger.success(
                    logs.CLEAN_COMPLETE.format(branches=deleted_branches, prs=closed_prs)
                )
            else:
                console.print("[red]Aborting. Run 'pr-split clean' manually first.[/red]")
                raise typer.Exit(1)
        else:
            logger.info("Overwriting existing dry-run plan")

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

    try:
        settings = Settings(
            min_loc=min_loc,
            max_loc=max_loc,
            strict_loc_bounds=strict_loc_bounds,
            max_refinement_iterations=max_refinement_iterations,
            cp_sat_timeout=cp_sat_timeout,
            priority=priority,
            chunk_strategy=chunk_strategy,
            partition_strategy=partition_strategy,
        )
    except (ValidationError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    groups = plan_split(parsed_diff, settings)

    logger.info(logs.VALIDATING_PLAN)
    dag = PlanDAG(groups)
    warnings = validate_plan(groups, parsed_diff, dag, settings.max_loc, min_loc=settings.min_loc)
    _handle_loc_bound_warnings(warnings, strict_loc_bounds=settings.strict_loc_bounds)
    logger.success(logs.VALIDATION_PASSED)

    logger.info(logs.PRESENTING_PLAN)
    _present_plan(groups)

    groups = _interactive_edit(groups, parsed_diff)

    # Re-validate after user edits
    empty_groups = [g for g in groups if not g.assignments]
    if empty_groups:
        empty_ids = [g.id for g in empty_groups]
        console.print(f"[red]Groups {escape(str(empty_ids))} are empty after editing.[/red]")
        raise typer.Exit(1)
    try:
        dag = PlanDAG(groups)
        warnings = validate_plan(
            groups,
            parsed_diff,
            dag,
            settings.max_loc,
            min_loc=settings.min_loc,
        )
        _handle_loc_bound_warnings(warnings, strict_loc_bounds=settings.strict_loc_bounds)
        logger.success("Edited plan validation passed")
    except PRSplitError as exc:
        console.print(f"[red]Edited plan is invalid: {escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    merge_base_ref = merge_base(base, dev_branch)

    split_plan = SplitPlan(
        dev_branch=dev_branch,
        base_branch=base,
        min_loc=settings.min_loc,
        max_loc=settings.max_loc,
        strict_loc_bounds=settings.strict_loc_bounds,
        stacked=stack,
        draft=draft,
        priority=priority,
        groups=groups,
        author=author,
        merge_base_sha=merge_base_ref,
        dev_branch_arg=dev_branch_arg,
        raw_diff=raw_diff,
    )

    if dry_run:
        save_plan(PlanFile(plan=split_plan, git_state=GitState(branches=[], prs=[])))
        logger.success(f"Dry run complete: plan with {len(groups)} groups saved to {PLAN_FILE}")
        return

    typer.confirm("Proceed with creating branches and PRs?", abort=True)

    namespace = derive_split_namespace(dev_branch_arg)
    branch_records = _create_branches_and_commits(
        groups, parsed_diff, base, merge_base_ref, namespace, author=author, stacked=stack
    )
    try:
        pr_records = _push_and_create_prs(groups, branch_records, draft=draft)
    except PRCreationError as exc:
        save_plan(
            PlanFile(
                plan=split_plan,
                git_state=GitState(branches=branch_records, prs=exc.pr_records),
            )
        )
        raise
    if stack:
        _link_stacks(dag, pr_records)

    save_plan(
        PlanFile(
            plan=split_plan,
            git_state=GitState(branches=branch_records, prs=pr_records),
        )
    )
    logger.success(f"Split complete: {len(groups)} PRs created")


@app.command(help="Show the current split plan with live PR state and review status.")
def status() -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(0)

    plan_file = load_plan()
    plan = plan_file.plan
    git_state = plan_file.git_state

    branch_map = {r.group_id: r.branch_name for r in git_state.branches}
    pr_map = {r.group_id: r for r in git_state.prs}

    live_states: dict[int, dict[str, str | bool | None]] = {}
    pr_numbers = [r.pr_number for r in git_state.prs]
    if pr_numbers:
        with ThreadPoolExecutor(max_workers=_GH_API_CONCURRENCY) as executor:
            futures = {executor.submit(get_pr_state, n): n for n in pr_numbers}
            for future in as_completed(futures):
                pr_num = futures[future]
                try:
                    live_states[pr_num] = future.result()
                except Exception:
                    live_states[pr_num] = {}

    table = Table(title="PR Split Status")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Branch")
    table.add_column("PR")
    table.add_column("State")
    table.add_column("Review")

    for group in plan.groups:
        branch_name = branch_map.get(group.id, "")
        pr_record = pr_map.get(group.id)
        pr_info = f"#{pr_record.pr_number}" if pr_record else ""
        pr_state = ""
        review = ""
        if pr_record:
            live = live_states.get(pr_record.pr_number, {})
            pr_state = live.get("state", pr_record.state.value).upper()
            review = (live.get("reviewDecision") or "").replace("_", " ").title()
        table.add_row(
            escape(group.id), escape(group.title), escape(branch_name), pr_info, pr_state, review
        )

    console.print(table)


def _cleanup_git_state(git_state: GitState) -> tuple[int, int]:
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

    plan_path = Path(PLAN_FILE)
    if plan_path.exists():
        plan_path.unlink()

    return closed_prs, deleted_branches


@app.command(help="Close all split PRs and delete their branches.")
def clean() -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(0)

    plan_file = load_plan()
    git_state = plan_file.git_state

    typer.confirm("Delete all pr-split branches and close PRs?", abort=True)

    closed_prs, deleted_branches = _cleanup_git_state(git_state)
    logger.success(logs.CLEAN_COMPLETE.format(branches=deleted_branches, prs=closed_prs))


@app.command(
    help="Execute a previously saved dry-run plan, creating branches and PRs.",
)
def execute(
    stack: Annotated[
        bool,
        typer.Option(
            "--stack",
            envvar="PR_SPLIT_STACK",
            help="Stack dependent PRs even if the saved plan was not created with --stack",
        ),
    ] = False,
    draft: Annotated[
        bool,
        typer.Option(
            "--draft",
            envvar="PR_SPLIT_DRAFT",
            help="Open every sub-PR as a draft even if the plan was not saved with --draft",
        ),
    ] = False,
) -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(1)

    plan_file = load_plan()
    plan = plan_file.plan
    if stack and not plan.stacked:
        plan = plan.model_copy(update={"stacked": True})
    if draft and not plan.draft:
        plan = plan.model_copy(update={"draft": True})

    if plan_file.git_state.branches or plan_file.git_state.prs:
        console.print("[red]This plan already has branches/PRs. Use 'pr-split clean' first.[/red]")
        raise typer.Exit(1)

    if not plan.raw_diff:
        console.print(
            "[red]Plan is missing saved diff data."
            " Re-run 'pr-split split --dry-run' to regenerate.[/red]"
        )
        raise typer.Exit(1)

    if not plan.merge_base_sha:
        console.print(
            "[red]Plan is missing merge base SHA."
            " Re-run 'pr-split split --dry-run' to regenerate.[/red]"
        )
        raise typer.Exit(1)

    if not branch_exists(plan.base_branch):
        console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=plan.base_branch)}[/red]")
        raise typer.Exit(1)
    if not is_worktree_clean():
        console.print(f"[red]{ErrorMsg.DIRTY_WORKTREE()}[/red]")
        raise typer.Exit(1)
    if not check_gh_auth():
        console.print(f"[red]{ErrorMsg.GH_AUTH_FAILED()}[/red]")
        raise typer.Exit(1)

    parsed_diff = parse_diff(plan.raw_diff)

    try:
        validate_coverage(plan.groups, parsed_diff)
    except PlanValidationError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    _present_plan(plan.groups)
    typer.confirm("Proceed with creating branches and PRs?", abort=True)

    namespace = derive_split_namespace(plan.dev_branch_arg or plan.dev_branch)
    branch_records = _create_branches_and_commits(
        plan.groups,
        parsed_diff,
        plan.base_branch,
        plan.merge_base_sha,
        namespace,
        author=plan.author,
        stacked=plan.stacked,
    )
    try:
        pr_records = _push_and_create_prs(plan.groups, branch_records, draft=plan.draft)
    except PRCreationError as exc:
        save_plan(
            PlanFile(
                plan=plan,
                git_state=GitState(branches=branch_records, prs=exc.pr_records),
            )
        )
        raise
    if plan.stacked:
        _link_stacks(PlanDAG(plan.groups), pr_records)

    save_plan(
        PlanFile(
            plan=plan,
            git_state=GitState(branches=branch_records, prs=pr_records),
        )
    )
    logger.success(f"Execute complete: {len(plan.groups)} PRs created from saved plan")


_AUTO_MERGE_POLL_INTERVAL = 10
_AUTO_MERGE_POLL_TIMEOUT = 600


def _poll_for_merged(group_ids: list[str], pr_map: dict[str, PRRecord]) -> set[str]:
    pending = set(group_ids)
    actually_merged: set[str] = set()
    deadline = time.monotonic() + _AUTO_MERGE_POLL_TIMEOUT
    while pending and time.monotonic() < deadline:
        time.sleep(_AUTO_MERGE_POLL_INTERVAL)
        for gid in list(pending):
            pr_record = pr_map[gid]
            live = get_pr_state(pr_record.pr_number)
            state = (live.get("state") or "").upper()
            if state == "MERGED":
                logger.info(f"PR #{pr_record.pr_number} ({gid}) merged")
                actually_merged.add(gid)
                pending.discard(gid)
            elif state in ("CLOSED", ""):
                reason = "closed" if state == "CLOSED" else "fetch error"
                logger.warning(
                    f"PR #{pr_record.pr_number} ({gid}) {reason} while polling, aborting wait"
                )
                pending.discard(gid)
    if pending:
        remaining = ", ".join(pending)
        logger.warning(f"Timed out waiting for auto-merge: {remaining}")
    return actually_merged


def _send_webhook(url: str, payload: dict[str, object]) -> None:
    try:
        data = json_mod.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        logger.info(f"Webhook notification sent to {url}")
    except Exception as exc:
        logger.warning(f"Failed to send webhook notification: {exc}")


@app.command(
    name="merge",
    help="Merge all split PRs in dependency order. Skips already-merged PRs.",
)
def merge_all(
    auto: Annotated[
        bool, typer.Option("--auto", help="Queue merges to run after CI checks pass")
    ] = False,
    notify: Annotated[
        str | None,
        typer.Option(
            "--notify",
            help="Webhook URL to POST merge results to",
            envvar="PR_SPLIT_WEBHOOK_URL",
        ),
    ] = None,
) -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(0)

    plan_file = load_plan()
    plan = plan_file.plan
    git_state = plan_file.git_state
    pr_map = {r.group_id: r for r in git_state.prs}

    if not pr_map:
        console.print("[yellow]No PRs found in plan. Nothing to merge.[/yellow]")
        raise typer.Exit(0)

    dag = PlanDAG(plan.groups)
    merged: list[str] = []
    skipped: list[str] = []
    skipped_ids: set[str] = set()
    failed: list[str] = []

    stopped = False
    exited_early = False
    for batch in dag.iter_ready():
        for group_id in batch:
            pr_record = pr_map.get(group_id)
            if not pr_record:
                skipped_ids.add(group_id)
                skipped.append(f"{group_id} (no PR)")
                continue

            live = get_pr_state(pr_record.pr_number)
            if not live:
                logger.warning(
                    f"PR #{pr_record.pr_number} ({group_id}) state could not be fetched, skipping"
                )
                skipped_ids.add(group_id)
                skipped.append(f"{group_id} (fetch error)")
                continue

            state = (live.get("state") or "").upper()

            if state == "MERGED":
                logger.info(f"PR #{pr_record.pr_number} ({group_id}) already merged")
                merged.append(group_id)
                continue

            if state != "OPEN":
                logger.warning(f"PR #{pr_record.pr_number} ({group_id}) is {state}, skipping")
                skipped_ids.add(group_id)
                skipped.append(f"{group_id} ({state})")
                continue

            if live.get("isDraft"):
                logger.warning(f"PR #{pr_record.pr_number} ({group_id}) is a draft, skipping")
                skipped_ids.add(group_id)
                skipped.append(f"{group_id} (draft)")
                continue

            review = live.get("reviewDecision") or ""
            if review in ("CHANGES_REQUESTED", "REVIEW_REQUIRED"):
                label = review.lower().replace("_", " ")
                logger.warning(
                    f"PR #{pr_record.pr_number} ({group_id}) "
                    f"review not approved ({label}), skipping"
                )
                skipped_ids.add(group_id)
                skipped.append(f"{group_id} ({label})")
                continue

            try:
                merge_pr(pr_record.pr_number, auto=auto)
                if not auto:
                    merged.append(group_id)
            except PRSplitError as exc:
                logger.error(f"Failed to merge PR #{pr_record.pr_number} ({group_id}): {exc}")
                failed.append(group_id)
                stopped = True
                break

        if auto and not stopped:
            queued = [gid for gid in batch if gid not in merged and gid not in skipped_ids]
            if queued:
                logger.info(f"Waiting for auto-merge to complete: {', '.join(queued)}")
                actually_merged = _poll_for_merged(queued, pr_map)
                merged.extend(actually_merged)

        if stopped or any(gid not in merged and gid not in skipped_ids for gid in batch):
            if not stopped:
                console.print(
                    "[yellow]Some PRs in this batch were not merged. "
                    "Stopping to avoid merging dependent PRs out of order.[/yellow]"
                )
            else:
                console.print(
                    "[red]Merge failed. "
                    "Stopping to avoid merging dependent PRs out of order.[/red]"
                )
            exited_early = True
            break

    console.print()
    if merged:
        console.print(f"[green]Merged ({len(merged)}): {escape(', '.join(merged))}[/green]")
    if skipped:
        console.print(f"[yellow]Skipped ({len(skipped)}): {escape(', '.join(skipped))}[/yellow]")
    if failed:
        console.print(f"[red]Failed ({len(failed)}): {escape(', '.join(failed))}[/red]")
    if notify:
        exit_reason = (
            "merge_error" if stopped else "incomplete_batch" if exited_early else "success"
        )
        skipped_structured = [{"id": s.split(" (")[0], "reason": s} for s in skipped]
        _send_webhook(
            notify,
            {
                "event": "merge_complete",
                "merged": merged,
                "skipped": skipped_structured,
                "failed": failed,
                "success": not (failed or stopped or exited_early),
                "exit_reason": exit_reason,
            },
        )

    if failed or stopped or exited_early:
        raise typer.Exit(1)
    logger.success(f"Merge complete: {len(merged)} PRs merged")
