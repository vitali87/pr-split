from __future__ import annotations

import json as json_mod
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Callable, Generator
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
    DEFAULT_STRICT_LOC_BOUNDS,
    NO_BACKEND_STRATEGY,
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
from .exceptions import (
    ErrorMsg,
    GitOperationError,
    PlanValidationError,
    PRCreationError,
    PRSplitError,
)
from .git_ops import (
    add_worktree,
    adopt_remote_branch,
    branch_exists,
    check_gh_auth,
    check_gh_stack,
    commit_files_in_dir,
    delete_branch,
    derive_split_namespace,
    diff_base_ref,
    fetch_fork_branch,
    fetch_fork_pr,
    is_worktree_clean,
    merge_base,
    push_branch,
    remove_worktree,
)
from .git_ops.branches import commit_exists, run_git
from .git_ops.prs import (
    branch_has_merged,
    close_pr,
    create_pr,
    default_branch,
    find_open_pr,
    get_pr_state,
    link_stack,
    merge_pr,
    set_pr_base,
    stack_numbers_for,
    unstack,
)
from .graph import PlanDAG
from .per_group import PerGroupStep, load_per_group_step, run_per_group_step
from .plan_store import load_plan, plan_dir, plan_exists, plan_path, save_plan, select_plan
from .planner import plan_split, validate_coverage, validate_no_binary_files, validate_plan
from .planner.chunker import recompute_estimated_loc
from .planner.new_file_pieces import link_new_file_pieces
from .planner.partitioning import refresh_generated_description
from .planner.validator import validate_new_file_pieces
from .recover import recover_plan
from .restack import restack as restack_layers
from .restack import stale_layers
from .schemas import (
    BranchRecord,
    GitState,
    Group,
    GroupAssignment,
    PlanFile,
    PRRecord,
    SplitPlan,
)
from .stack_move import move_hunk as move_stack_hunk
from .types_defs import ForkPRInfo

app = typer.Typer(
    name="pr-split",
    help="Decompose large PRs into reviewable dependency-ordered PRs",
)
console = Console()


@app.callback()
def _choose_plan(
    branch: Annotated[
        str | None,
        typer.Option(
            "--branch",
            envvar="PR_SPLIT_BRANCH",
            help="Dev branch whose saved plan to use when several splits are in progress",
        ),
    ] = None,
) -> None:
    # Plans are saved per dev branch; without --branch the only saved plan is used.
    select_plan(branch)


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


def _require_local_branch(base: str) -> None:
    """Exit unless ``base`` names a local branch head.

    A remote-tracking ref such as origin/main (or a tag or SHA) resolves
    locally, so the split would run to completion -- branches created and
    pushed -- and only fail when GitHub is asked to open PRs against a
    branch it does not have.
    """
    if branch_exists(f"refs/heads/{base}"):
        return
    remote_prefix = "refs/remotes/" if base.startswith("refs/remotes/") else ""
    stripped = base.removeprefix(remote_prefix)
    suggestion = ""
    if remote_prefix or "/" in stripped:
        remote, _, name = stripped.partition("/")
        if name and remote in _remote_names():
            suggestion = f" (for example '{name}')"
    console.print(
        f"[red]{ErrorMsg.BASE_NOT_A_LOCAL_BRANCH(base=base, suggestion=suggestion)}[/red]"
    )
    raise typer.Exit(1)


def _remote_names() -> set[str]:
    try:
        return set(run_git("remote").split())
    except PRSplitError:
        return set()


def _require_gh_stack() -> None:
    try:
        installed = check_gh_stack()
    except GitOperationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not installed:
        console.print(f"[red]{ErrorMsg.GH_STACK_MISSING()}[/red]")
        raise typer.Exit(1)


def _validate_inputs(
    dev_branch: str, base: str, *, dry_run: bool = False, stacked: bool = False
) -> None:
    if not branch_exists(dev_branch):
        console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=dev_branch)}[/red]")
        raise typer.Exit(1)
    if not branch_exists(base):
        console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=base)}[/red]")
        raise typer.Exit(1)
    _require_local_branch(base)
    if not is_worktree_clean():
        console.print(f"[red]{ErrorMsg.DIRTY_WORKTREE()}[/red]")
        raise typer.Exit(1)
    if not dry_run and not check_gh_auth():
        console.print(f"[red]{ErrorMsg.GH_AUTH_FAILED()}[/red]")
        raise typer.Exit(1)
    if not dry_run and stacked:
        _require_gh_stack()


def _load_plan_or_exit() -> PlanFile:
    try:
        return load_plan()
    except PRSplitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _handle_loc_bound_warnings(warnings: list[str], *, strict_loc_bounds: bool) -> None:
    if strict_loc_bounds and warnings:
        console.print(f"[red]{ErrorMsg.LOC_BOUNDS_STRICT_FAILED()}[/red]")
        for warning in warnings:
            console.print(f"[red]- {escape(warning)}[/red]")
        raise typer.Exit(1)

    for warning in warnings:
        logger.warning(warning)


def _plan_provenance(plan: SplitPlan) -> str:
    """One line naming how the plan was made; llm and cp_sat plans vary run to run."""
    how = plan.partition_strategy or "unknown"
    if plan.provider:
        how += f" ({plan.provider}{' ' + plan.model if plan.model else ''})"
    if plan.partition_strategy == NO_BACKEND_STRATEGY:
        return "[dim]Kept as one group: the diff is within --max-loc, so no backend ran.[/dim]"
    note = (
        ""
        if plan.partition_strategy == PartitionStrategy.GRAPH.value
        else "; re-running may give a different plan, so keep the saved plan file"
    )
    return f"[dim]Planned with {escape(how)}{note}.[/dim]"


def _oversized_group_ids(groups: list[Group], max_loc: int) -> list[str]:
    return [g.id for g in groups if g.estimated_loc > max_loc]


def _report_oversized_groups(
    groups: list[Group], max_loc: int, hunk_counts: dict[str, int]
) -> None:
    """One visible line when groups miss the --max-loc target, so it is never silent.

    ``hunk_counts`` maps each file to its parsed hunk count, so a WHOLE_FILE
    assignment is counted by the hunks it covers, not the indices it lists.
    """
    oversized = [g for g in groups if g.estimated_loc > max_loc]
    if not oversized:
        return
    largest = max(oversized, key=lambda g: g.estimated_loc)
    # A group holding one hunk (e.g. a whole new file) cannot be split by any
    # plan; say so rather than leave it looking like a planning miss.
    single_hunk = [
        g.id
        for g in oversized
        if sum(len(a.covered_indices(hunk_counts.get(a.file_path, 0))) for a in g.assignments) == 1
    ]
    irreducible = (
        f" {', '.join(escape(gid) for gid in single_hunk)} hold a single hunk each and"
        " cannot be split below the limit."
        if single_hunk
        else ""
    )
    console.print(
        f"[yellow]{len(oversized)} of {len(groups)} groups exceed --max-loc {max_loc}"
        f" (largest: {escape(largest.id)} at {largest.estimated_loc} LOC).{irreducible}"
        " Use the editor, --max-refinement-iterations or --strict-loc-bounds to act on it."
        "[/yellow]"
    )


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


def _discard_worktree(worktree_path: str) -> None:
    try:
        remove_worktree(worktree_path)
    except PRSplitError as exc:
        logger.warning(f"Failed to remove worktree {worktree_path}: {exc}")


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
    per_group: tuple[PerGroupStep, int] | None = None,
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
                # newline="" keeps CRLF from the reconstructed content intact.
                p.write_text(content, encoding="utf-8", errors="surrogateescape", newline="")
            elif p.exists():
                p.unlink()

        logger.info(logs.COMMITTING_GROUP.format(group=group.id, title=group.title))
        commit_sha = commit_files_in_dir(
            worktree_path,
            list(materialized.keys()),
            group.title,
            author=author,
        )
        if per_group is not None:
            step, index = per_group
            commit_sha = (
                run_per_group_step(
                    step,
                    worktree_path,
                    group,
                    index=index,
                    pr_base=base_branch,
                    parent_ref=start_point or merge_base_ref,
                    author=author,
                )
                or commit_sha
            )
    except Exception:
        # add_worktree succeeded, so this run created branch_name (a
        # pre-existing branch of that name was already replaced). Delete it
        # here, where that is known for certain: if add_worktree itself had
        # failed it restores the previous branch and never reaches this path.
        _discard_worktree(worktree_path)
        try:
            delete_branch(branch_name)
        except PRSplitError as exc:
            logger.warning(f"Could not clean up branch {branch_name}: {exc}")
        raise
    _discard_worktree(worktree_path)

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
    for batch in dag.iter_ready():
        batch_args: list[tuple[Group, str, str]] = []
        for gid in batch:
            group = groups_by_id[gid]
            parents = dag.parents(gid)
            if len(parents) == 1:
                # Files are rebuilt from the merge base, so a child must carry
                # every ancestor's hunks for the files it touches - not only
                # its direct parent's - or it silently reverts them.
                merged = merge_chain_assignments(
                    group,
                    [groups_by_id[a] for a in sorted(dag.ancestors(gid))],
                    hunk_counts,
                )
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
    keep: dict[str, BranchRecord] | None = None,
) -> list[BranchRecord]:
    """Create a branch and commit per group; groups in ``keep`` reuse their record.

    ``keep`` holds groups whose PRs already exist from an earlier, partly
    failed run: their branches are left as they are (recreating them would
    rewrite the open PR's head).
    """
    kept = keep or {}
    worktree_base = Path(tempfile.mkdtemp(prefix="pr-split-worktrees-"))
    step = load_per_group_step()
    order = {gid: i for i, gid in enumerate(PlanDAG(groups).topological_order(), start=1)}

    # A plan with dependency edges is laid out along its DAG whether or not
    # native stacking is on: each dependant builds on (and targets) its
    # parent's branch, so it is reviewable and buildable against the code it
    # depends on. ``stacked`` only adds native gh-stack registration on top.
    if stacked or any(g.depends_on for g in groups):
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
        results: dict[str, BranchRecord] = dict(kept)
        errors: list[tuple[str, Exception]] = []
        for batch_args in batches:
            batch_args = [args for args in batch_args if args[0].id not in kept]
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
                        per_group=(step, order[group.id]) if step else None,
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
            # Failed groups already removed their own branch inside the
            # worker; only the successful ones remain to roll back.
            for gid, record in results.items():
                if gid in kept:
                    continue
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


def _pr_template_path() -> Path:
    return plan_dir() / "template.md"


def _build_pr_body(group: Group, all_groups: list[Group]) -> str:
    template_path = _pr_template_path()
    if template_path.exists():
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
            template = template_path.read_text(encoding="utf-8")
            return template.format(**template_vars)
        except (KeyError, ValueError, IndexError) as exc:
            available = ", ".join(f"{{{k}}}" for k in sorted(template_vars))
            raise PRSplitError(
                f"Invalid PR template at {template_path}: {exc}. "
                f"Available placeholders: {available}. "
                "Escape literal braces with {{ and }}."
            ) from exc
        except OSError as exc:
            raise PRSplitError(f"Could not read PR template at {template_path}: {exc}") from exc

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
        dependencies = f"## Dependencies\n\nThis PR depends on: {dep_list}"
        if len(set(group.depends_on)) > 1:
            # A merge node targets the base branch, so its diff also shows
            # every ancestor's changes beyond the files listed above.
            ancestors = sorted(PlanDAG(all_groups).ancestors(group.id))
            carried = ", ".join(f"`{a}`" for a in ancestors)
            dependencies += (
                f"\n\nIt targets the base branch, so its diff also includes the changes of: "
                f"{carried}."
            )
        sections.append(dependencies)
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
    existing_prs: dict[str, PRRecord] | None = None,
) -> list[PRRecord]:
    """Push each group's branch and open its PR; groups in ``existing_prs`` are done."""
    done = existing_prs or {}
    record_map = {r.group_id: r for r in branch_records}
    errors: list[tuple[str, Exception]] = []

    # Children target parent branches, so every branch is pushed before any PR opens.
    with ThreadPoolExecutor(max_workers=_PUSH_MAX_WORKERS) as executor:
        push_futures = {
            executor.submit(push_branch, record_map[group.id].branch_name): group.id
            for group in groups
            if group.id not in done
        }
        pushed: set[str] = set(done)
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
            if group.id not in done and group.id in pushed and _base_pushed(group)
        }
        results: dict[str, PRRecord] = dict(done)
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


def _link_stacks(
    dag: PlanDAG, pr_records: list[PRRecord], branch_records: list[BranchRecord]
) -> None:
    pr_by_group = {r.group_id: r.pr_number for r in pr_records}
    base_by_group = {r.group_id: r.base_branch for r in branch_records}
    for chain in dag.linear_chains():
        if len(chain) < 2:
            continue
        link_stack([pr_by_group[gid] for gid in chain], base=base_by_group[chain[0]])


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

    refresh_generated_description(src)
    refresh_generated_description(dst)
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


def _held_hunks(
    groups: list[Group], hunk_counts: dict[str, int]
) -> dict[str, set[tuple[str, int]]]:
    """Map each group id to the (file, hunk index) pairs its assignments cover."""
    return {
        g.id: {
            (a.file_path, idx)
            for a in g.assignments
            for idx in a.covered_indices(hunk_counts.get(a.file_path, 0))
        }
        for g in groups
    }


def _drop_empty_groups(
    groups: list[Group],
    held_before: dict[str, set[tuple[str, int]]],
    hunk_counts: dict[str, int],
) -> list[Group]:
    """Remove groups the user emptied in the editor and unlink them from the DAG.

    Moving every hunk out of a group is a legitimate way to dissolve it;
    aborting the session there would throw away all the other edits. A
    dropped group's dependants inherit its own dependencies and every kept
    group that now holds one of its former hunks (``held_before`` is the
    coverage snapshot taken before editing), so a stacked dependant still
    builds on the code it was planned against. A move that cannot keep that
    guarantee (the hunk went to a group that itself builds on the dependant)
    is refused, since the dependant's stacked branch would lose the hunk.
    """
    dropped = {g.id: list(g.depends_on) for g in groups if not g.assignments}
    if not dropped:
        return groups

    held_now = _held_hunks(groups, hunk_counts)
    recipients = {
        gid: [
            g.id
            for g in groups
            if g.id not in dropped and held_before.get(gid, set()) & held_now[g.id]
        ]
        for gid in dropped
    }

    def _surviving(dep: str, seen: set[str]) -> tuple[list[str], list[tuple[str, str]]]:
        # Walk through chains of dropped groups to the nearest kept ancestors,
        # collecting the kept groups that took over the dropped groups' hunks.
        if dep not in dropped:
            return [dep], []
        ancestors: list[str] = []
        taken_over = [(recipient, dep) for recipient in recipients[dep]]
        for parent in dropped[dep]:
            if parent not in seen:
                seen.add(parent)
                more_ancestors, more_taken = _surviving(parent, seen)
                ancestors.extend(more_ancestors)
                taken_over.extend(more_taken)
        return ancestors, taken_over

    # Inherited ancestors first: they are transitive ancestors in the original
    # acyclic plan, so rewiring to them cannot create a cycle.
    deps: dict[str, list[str]] = {}
    wanted: dict[str, list[tuple[str, str]]] = {}
    for group in groups:
        if group.id in dropped:
            continue
        deps[group.id] = []
        wanted[group.id] = []
        for dep in group.depends_on:
            ancestors, taken_over = _surviving(dep, set())
            for candidate in ancestors:
                if candidate not in deps[group.id] and candidate != group.id:
                    deps[group.id].append(candidate)
            wanted[group.id].extend(taken_over)

    def _depends_on(start: str, target: str) -> bool:
        stack, seen = [start], set()
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                stack.extend(deps.get(node, []))
        return False

    # Then the groups that took over dropped hunks. One that already builds on
    # the dependant cannot become its parent without a cycle, and leaving it out
    # would build the dependant's branch without the moved hunk: refuse.
    for gid, candidates in wanted.items():
        for candidate, source in candidates:
            # Already reachable through an inherited ancestor: a direct edge
            # would only turn a linear stack into a multi-parent node.
            if candidate == gid or _depends_on(gid, candidate):
                continue
            if _depends_on(candidate, gid):
                console.print(
                    f"[red]Cannot drop emptied group '{source}': its hunks moved to"
                    f" '{candidate}', which builds on '{gid}', so '{gid}' would lose code"
                    " it was planned on. Move them to a group it can depend on.[/red]"
                )
                raise typer.Exit(1)
            deps[gid].append(candidate)

    kept = [g.model_copy(update={"depends_on": deps[g.id]}) for g in groups if g.id not in dropped]
    console.print(f"[yellow]Dropped empty group(s) after editing: {', '.join(dropped)}[/yellow]")
    return kept


def _find_group(groups: list[Group], group_id: str) -> Group | None:
    group = next((g for g in groups if g.id == group_id), None)
    if group is None:
        console.print(f"[red]Group '{group_id}' not found.[/red]")
    return group


def _creates_cycle(groups: list[Group]) -> bool:
    try:
        PlanDAG(groups).validate_acyclic()
    except PlanValidationError:
        return True
    return False


def _add_dependency(groups: list[Group], child_id: str, parent_id: str) -> bool:
    child, parent = _find_group(groups, child_id), _find_group(groups, parent_id)
    if child is None or parent is None:
        return False
    if child_id == parent_id:
        console.print("[red]A group cannot depend on itself.[/red]")
        return False
    if parent_id in child.depends_on:
        console.print(f"[yellow]{child_id} already depends on {parent_id}.[/yellow]")
        return False
    child.depends_on.append(parent_id)
    if _creates_cycle(groups):
        child.depends_on.remove(parent_id)
        console.print(f"[red]{child_id} -> {parent_id} would create a dependency cycle.[/red]")
        return False
    console.print(f"[green]{child_id} now depends on {parent_id}[/green]")
    return True


def _remove_dependency(groups: list[Group], child_id: str, parent_id: str) -> bool:
    child = _find_group(groups, child_id)
    if child is None:
        return False
    if parent_id not in child.depends_on:
        console.print(f"[yellow]{child_id} does not depend on {parent_id}.[/yellow]")
        return False
    child.depends_on.remove(parent_id)
    console.print(f"[green]{child_id} no longer depends on {parent_id}[/green]")
    return True


def _move_file(
    groups: list[Group], parsed_diff: ParsedDiff, file_path: str, from_id: str, to_id: str
) -> bool:
    """Move every hunk of ``file_path`` that ``from_id`` holds into ``to_id``."""
    src, dst = _find_group(groups, from_id), _find_group(groups, to_id)
    if src is None or dst is None:
        return False
    if from_id == to_id:
        console.print("[yellow]Source and destination are the same. No move performed.[/yellow]")
        return False
    hunk_count = next((len(pf) for pf in parsed_diff.patch_set if pf.path == file_path), 0)
    moving = sorted(
        {
            idx
            for a in src.assignments
            if a.file_path == file_path
            for idx in a.covered_indices(hunk_count)
        }
    )
    if not moving:
        console.print(f"[red]{from_id} holds no hunks of {file_path}.[/red]")
        return False
    src.assignments = [a for a in src.assignments if a.file_path != file_path]
    dst.assignments = _combine_assignments(
        [
            *dst.assignments,
            GroupAssignment(
                file_path=file_path,
                assignment_type=AssignmentType.PARTIAL_HUNKS,
                hunk_indices=moving,
            ),
        ],
        {pf.path: len(pf) for pf in parsed_diff.patch_set},
    )
    console.print(
        f"[green]Moved {len(moving)} hunk(s) of {file_path} from {from_id} to {to_id}[/green]"
    )
    return True


def _new_group(groups: list[Group], group_id: str) -> bool:
    if any(g.id == group_id for g in groups):
        console.print(f"[red]Group '{group_id}' already exists.[/red]")
        return False
    groups.append(Group(id=group_id, title=group_id, description=""))
    console.print(f"[green]Created empty group {group_id}[/green]")
    return True


def _combine_assignments(
    assignments: list[GroupAssignment], hunk_counts: dict[str, int]
) -> list[GroupAssignment]:
    """One assignment per file covering the union of the given hunks."""
    by_file: dict[str, set[int]] = {}
    for a in assignments:
        by_file.setdefault(a.file_path, set()).update(
            a.covered_indices(hunk_counts.get(a.file_path, 0))
        )
    combined: list[GroupAssignment] = []
    for path, indices in by_file.items():
        whole = sorted(indices) == list(range(hunk_counts.get(path, 0)))
        combined.append(
            GroupAssignment(
                file_path=path,
                assignment_type=AssignmentType.WHOLE_FILE
                if whole
                else AssignmentType.PARTIAL_HUNKS,
                hunk_indices=sorted(indices),
            )
        )
    return combined


def _merge_groups(
    groups: list[Group], parsed_diff: ParsedDiff, keep_id: str, absorb_id: str
) -> bool:
    """Fold ``absorb_id`` into ``keep_id``: its hunks, its parents and its dependants."""
    keep, absorb = _find_group(groups, keep_id), _find_group(groups, absorb_id)
    if keep is None or absorb is None:
        return False
    if keep_id == absorb_id:
        console.print("[yellow]Cannot merge a group into itself.[/yellow]")
        return False
    merged: list[Group] = []
    for g in groups:
        if g.id == absorb_id:
            continue
        deps = [keep_id if d == absorb_id else d for d in g.depends_on]
        if g.id == keep_id:
            deps += absorb.depends_on
        deps = [d for i, d in enumerate(deps) if d != g.id and d not in deps[:i]]
        assignments = list(g.assignments)
        if g.id == keep_id:
            hunk_counts = {pf.path: len(pf) for pf in parsed_diff.patch_set}
            assignments = _combine_assignments(assignments + absorb.assignments, hunk_counts)
        merged.append(g.model_copy(update={"depends_on": deps, "assignments": assignments}))
    if _creates_cycle(merged):
        console.print(
            f"[red]Merging {absorb_id} into {keep_id} would create a dependency cycle.[/red]"
        )
        return False
    groups[:] = merged
    console.print(f"[green]Merged {absorb_id} into {keep_id}[/green]")
    return True


def _interactive_edit(groups: list[Group], parsed_diff: ParsedDiff) -> list[Group]:
    console.print(
        "\n[cyan]Interactive editor. Commands:[/cyan]\n"
        "  [bold]move[/bold] <file>:<hunk> <from_group> <to_group>\n"
        "  [bold]movefile[/bold] <file> <from_group> <to_group>\n"
        "  [bold]dep[/bold] <child> <parent>  /  [bold]undep[/bold] <child> <parent>\n"
        "  [bold]title[/bold] <group_id> <text>  /  [bold]desc[/bold] <group_id> <text>\n"
        "  [bold]new[/bold] <group_id>  /  [bold]merge[/bold] <keep_id> <absorb_id>\n"
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
            if _move_assignment(groups, parsed_diff, file_path, hunk_index, from_id, to_id):
                # Keep per-group LOC in step with the new assignments so the
                # plan table, strict LOC bounds, plan.json and PR bodies are
                # accurate.
                recompute_estimated_loc(groups, parsed_diff)
        elif action in ("dep", "undep"):
            if len(parts) != 3:
                console.print(f"[red]Usage: {action} <child_group> <parent_group>[/red]")
                continue
            if action == "dep":
                _add_dependency(groups, parts[1], parts[2])
            else:
                _remove_dependency(groups, parts[1], parts[2])
        elif action in ("title", "desc"):
            text_parts = cmd.strip().split(maxsplit=2)
            if len(text_parts) != 3:
                console.print(f"[red]Usage: {action} <group_id> <text>[/red]")
                continue
            group = _find_group(groups, text_parts[1])
            if group is not None:
                if action == "title":
                    group.title = text_parts[2]
                else:
                    group.description = text_parts[2]
                console.print(f"[green]Updated {action} of {group.id}[/green]")
        elif action == "movefile":
            if len(parts) != 4:
                console.print("[red]Usage: movefile <file> <from_group> <to_group>[/red]")
                continue
            if _move_file(groups, parsed_diff, parts[1], parts[2], parts[3]):
                recompute_estimated_loc(groups, parsed_diff)
        elif action == "new":
            if len(parts) != 2:
                console.print("[red]Usage: new <group_id>[/red]")
                continue
            _new_group(groups, parts[1])
        elif action == "merge":
            if len(parts) != 3:
                console.print("[red]Usage: merge <keep_group> <absorb_group>[/red]")
                continue
            if _merge_groups(groups, parsed_diff, parts[1], parts[2]):
                recompute_estimated_loc(groups, parsed_diff)
        else:
            console.print(
                "[yellow]Unknown command. Type 'done' to proceed or 'abort' to cancel.[/yellow]"
            )


def _split_settings(
    partition_strategy: PartitionStrategy | None,
    build: Callable[[PartitionStrategy], Settings],
) -> Settings:
    """Settings for split; with no strategy chosen, llm if usable, else graph.

    A machine without an API key (and no keyless provider) can still split
    with the graph backend, which needs no model at all. An explicit
    --partition-strategy llm keeps failing loudly when its key is missing.
    """
    if partition_strategy is not None:
        return build(partition_strategy)
    try:
        return build(PartitionStrategy.LLM)
    except (ValidationError, ValueError) as llm_error:
        try:
            settings = build(PartitionStrategy.GRAPH)
        except (ValidationError, ValueError):
            raise llm_error from None
        reason = (
            llm_error.errors()[0]["msg"].removeprefix("Value error, ")
            if isinstance(llm_error, ValidationError)
            else str(llm_error)
        )
        logger.warning(logs.LLM_UNAVAILABLE_USING_GRAPH.format(reason=reason))
        return settings


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
    priority: Annotated[
        Priority,
        typer.Option("--priority", envvar="PR_SPLIT_PRIORITY", help="Grouping priority"),
    ] = Priority.ORTHOGONAL,
    chunk_strategy: Annotated[
        ChunkStrategy,
        typer.Option(
            "--chunk-strategy",
            envvar="PR_SPLIT_CHUNK_STRATEGY",
            help="Chunking strategy for large diffs",
        ),
    ] = DEFAULT_CHUNK_STRATEGY,
    partition_strategy: Annotated[
        PartitionStrategy | None,
        typer.Option(
            "--partition-strategy",
            envvar="PR_SPLIT_PARTITION_STRATEGY",
            help="Backend for hunk-to-PR partitioning"
            " [default: llm when its provider is configured, else graph]",
            show_default=False,
        ),
    ] = None,
    cp_sat_timeout: Annotated[
        float,
        typer.Option(
            "--cp-sat-timeout",
            envvar="PR_SPLIT_CP_SAT_TIMEOUT",
            help="Maximum seconds to spend in the CP-SAT solver",
        ),
    ] = DEFAULT_CP_SAT_TIMEOUT_SECONDS,
    stack: Annotated[
        bool,
        typer.Option(
            "--stack",
            envvar="PR_SPLIT_STACK",
            help="Register dependent PR chains as native GitHub stacks",
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
    select_plan(dev_branch_arg)

    # A branch that exists only as origin/<name> (fresh clone or worktree)
    # is adopted as a local branch, as `git checkout <name>` would.
    for name in (dev_branch, base):
        if not (name.lstrip("#").isdigit() or ":" in name):
            adopt_remote_branch(name)

    if not branch_exists(dev_branch):
        if not check_gh_auth():
            console.print(f"[red]{ErrorMsg.GH_AUTH_FAILED()}[/red]")
            raise typer.Exit(1)
        if not is_worktree_clean():
            console.print(f"[red]{ErrorMsg.DIRTY_WORKTREE()}[/red]")
            raise typer.Exit(1)
        try:
            fork_info = _resolve_fork_ref(dev_branch)
        except PRSplitError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        if not fork_info:
            console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=dev_branch)}[/red]")
            raise typer.Exit(1)
        dev_branch = fork_info["local_ref"]
        base = fork_info["base_branch"]
        author = fork_info["author"]

    _validate_inputs(dev_branch, base, dry_run=dry_run, stacked=stack)

    if plan_exists():
        existing = _load_plan_or_exit()
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

    # The sub-PRs target the remote's base, which a local base may lag behind.
    diff_base = diff_base_ref(base)
    raw_diff = extract_diff(dev_branch, diff_base)
    # Only a stacked child builds on the PR holding a file's earlier pieces.
    split_new_files_over = max_loc if stack else None
    parsed_diff = parse_diff(raw_diff, split_new_files_over=split_new_files_over)
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
        settings = _split_settings(
            partition_strategy,
            lambda strategy: Settings(
                min_loc=min_loc,
                max_loc=max_loc,
                strict_loc_bounds=strict_loc_bounds,
                max_refinement_iterations=max_refinement_iterations,
                cp_sat_timeout=cp_sat_timeout,
                priority=priority,
                chunk_strategy=chunk_strategy,
                partition_strategy=strategy,
            ),
        )
    except (ValidationError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    try:
        validate_no_binary_files(parsed_diff)
    except PlanValidationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    try:
        groups = plan_split(parsed_diff, settings)
        link_new_file_pieces(groups, parsed_diff)
    except PRSplitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    logger.info(logs.VALIDATING_PLAN)
    try:
        dag = PlanDAG(groups)
        warnings = validate_plan(
            groups, parsed_diff, dag, settings.max_loc, min_loc=settings.min_loc
        )
    except PRSplitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    _handle_loc_bound_warnings(warnings, strict_loc_bounds=settings.strict_loc_bounds)
    logger.success(logs.VALIDATION_PASSED)

    logger.info(logs.PRESENTING_PLAN)
    _present_plan(groups)

    hunk_counts = {pf.path: len(pf) for pf in parsed_diff.patch_set}
    held_before = _held_hunks(groups, hunk_counts)
    groups = _interactive_edit(groups, parsed_diff)

    # Re-validate after user edits
    groups = _drop_empty_groups(groups, held_before, hunk_counts)
    if not groups:
        console.print("[red]Every group is empty after editing; nothing to split.[/red]")
        raise typer.Exit(1)
    link_new_file_pieces(groups, parsed_diff)
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
        _report_oversized_groups(groups, settings.max_loc, hunk_counts)
    except PRSplitError as exc:
        console.print(f"[red]Edited plan is invalid: {escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    merge_base_ref = merge_base(diff_base, dev_branch)

    backend_ran = parsed_diff.stats["total_loc"] > settings.max_loc
    llm_ran = backend_ran and settings.partition_strategy is PartitionStrategy.LLM
    split_plan = SplitPlan(
        dev_branch=dev_branch,
        base_branch=base,
        min_loc=settings.min_loc,
        max_loc=settings.max_loc,
        strict_loc_bounds=settings.strict_loc_bounds,
        stacked=stack,
        split_new_files_over=split_new_files_over,
        draft=draft,
        priority=priority,
        groups=groups,
        author=author,
        merge_base_sha=merge_base_ref,
        dev_branch_arg=dev_branch_arg,
        raw_diff=raw_diff,
        # A diff within --max-loc is kept as one group without any backend.
        partition_strategy=(
            settings.partition_strategy.value if backend_ran else NO_BACKEND_STRATEGY
        ),
        chunk_strategy=settings.chunk_strategy.value,
        provider=settings.provider.value if llm_ran else None,
        model=settings.model if llm_ran else None,
        oversized_groups=_oversized_group_ids(groups, settings.max_loc),
    )
    console.print(_plan_provenance(split_plan))

    if dry_run:
        save_plan(PlanFile(plan=split_plan, git_state=GitState(branches=[], prs=[])))
        logger.success(f"Dry run complete: plan with {len(groups)} groups saved to {plan_path()}")
        return

    typer.confirm("Proceed with creating branches and PRs?", abort=True)

    namespace = derive_split_namespace(dev_branch_arg)
    try:
        branch_records = _create_branches_and_commits(
            groups, parsed_diff, base, merge_base_ref, namespace, author=author, stacked=stack
        )
    except PRSplitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    try:
        pr_records = _push_and_create_prs(groups, branch_records, draft=draft)
    except PRCreationError as exc:
        save_plan(
            PlanFile(
                plan=split_plan,
                git_state=GitState(branches=branch_records, prs=exc.pr_records),
            )
        )
        console.print(f"[red]{escape(str(exc))}[/red]")
        console.print("[yellow]Created branches and PRs were saved to the plan file.[/yellow]")
        raise typer.Exit(1) from exc
    save_plan(
        PlanFile(
            plan=split_plan,
            git_state=GitState(branches=branch_records, prs=pr_records),
        )
    )
    if stack:
        _link_stacks(dag, pr_records, branch_records)
    logger.success(f"Split complete: {len(groups)} PRs created")


@app.command(help="Show the current split plan with live PR state and review status.")
def status() -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(0)

    plan_file = _load_plan_or_exit()
    plan = plan_file.plan
    git_state = plan_file.git_state

    if git_state.prs and _base_has_merged(plan.base_branch):
        console.print(
            f"[yellow]{logs.BASE_ALREADY_MERGED.format(base=escape(plan.base_branch))}[/yellow]"
        )

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

    unverified: list[int] = []
    for group in plan.groups:
        branch_name = branch_map.get(group.id, "")
        pr_record = pr_map.get(group.id)
        pr_info = f"#{pr_record.pr_number}" if pr_record else ""
        pr_state = ""
        review = ""
        if pr_record:
            live = live_states.get(pr_record.pr_number) or {}
            if live:
                pr_state = str(live.get("state") or "").upper()
                review = str(live.get("reviewDecision") or "").replace("_", " ").title()
            else:
                # The recorded state is never written back after merge/close,
                # so showing it would claim OPEN for a PR that may be gone.
                pr_state = "UNKNOWN"
                unverified.append(pr_record.pr_number)
        table.add_row(
            escape(group.id), escape(group.title), escape(branch_name), pr_info, pr_state, review
        )

    console.print(table)
    for gid, branch, parent in stale_layers(plan_file):
        behind = logs.LAYER_BEHIND_PARENT.format(group=gid, branch=branch, parent=parent)
        console.print(f"[yellow]{behind}[/yellow]")
    if unverified:
        console.print(
            f"[yellow]Could not fetch live state for {len(unverified)} PR(s): "
            f"{', '.join(f'#{n}' for n in unverified)}. Check 'gh auth status'.[/yellow]"
        )


def _cleanup_git_state(git_state: GitState) -> tuple[int, int]:
    closed_prs = 0
    for pr_record in git_state.prs:
        if pr_record.adopted:
            # The user's own PR, only registered by adopt: forget it, never close it.
            logger.info(logs.ADOPTED_PR_KEPT.format(number=pr_record.pr_number))
            closed_prs += 1
            continue
        # gh refuses to close a merged PR and silently succeeds on a closed
        # one; neither needs a warning nor should count as newly closed, but
        # both are "done" for the purpose of removing the plan.
        state = str(get_pr_state(pr_record.pr_number).get("state") or "").upper()
        if state in ("MERGED", "CLOSED"):
            logger.info(
                logs.PR_ALREADY_DONE.format(number=pr_record.pr_number, state=state.lower())
            )
            closed_prs += 1
            continue
        try:
            close_pr(pr_record.pr_number)
            closed_prs += 1
        except PRSplitError as exc:
            logger.warning(f"Could not close PR #{pr_record.pr_number}: {exc}")

    logger.info(logs.CLEANING_BRANCHES)
    deleted_branches = 0
    for branch_record in git_state.branches:
        if branch_record.adopted:
            logger.info(logs.ADOPTED_BRANCH_KEPT.format(branch=branch_record.branch_name))
            deleted_branches += 1
            continue
        try:
            delete_branch(branch_record.branch_name, remote=True)
            deleted_branches += 1
        except PRSplitError:
            logger.warning(f"Could not delete branch {branch_record.branch_name}")

    complete = closed_prs == len(git_state.prs) and deleted_branches == len(git_state.branches)
    saved_plan = plan_path()
    if complete and saved_plan.exists():
        saved_plan.unlink()

    return closed_prs, deleted_branches


# Nothing is saved until every step succeeds, so a failed adopt never blocks
# a re-run; the stack itself may already be partly registered on GitHub.
ADOPT_RERUN_HINT = (
    "[yellow]The stack may be partly registered on GitHub; nothing was saved, so"
    " re-run the same adopt command to finish it.[/yellow]"
)


@app.command(
    help="Register branches you already built as a native stack (bottom to top) and "
    "save them as a plan, so 'status' and 'merge' work on them."
)
def adopt(
    branches: Annotated[
        list[str], typer.Argument(help="Branches in stack order, bottom first (at least two)")
    ],
    base: Annotated[str, typer.Option(help="Branch the bottom of the stack targets")] = "main",
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Register the stack without asking to confirm")
    ] = False,
) -> None:
    if len(branches) < 2:
        console.print("[red]adopt needs at least two branches, bottom of the stack first.[/red]")
        raise typer.Exit(1)
    if len(set(branches)) != len(branches):
        console.print("[red]Each branch may appear only once.[/red]")
        raise typer.Exit(1)
    # The adopted stack's plan is filed under its top branch, like a split's dev branch.
    select_plan(branches[-1])
    if plan_exists():
        existing = _load_plan_or_exit()
        if existing.git_state.branches or existing.git_state.prs:
            console.print(
                "[red]A split plan with branches/PRs already exists."
                " Run 'pr-split clean' (or merge it) first.[/red]"
            )
            raise typer.Exit(1)
    for name in (base, *branches):
        if not branch_exists(f"refs/heads/{name}"):
            console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=name)}[/red]")
            raise typer.Exit(1)
    # A stacked PR shows only its own diff on top of its parent, so each
    # branch must actually contain the one below it.
    for parent, child in zip((base, *branches), branches, strict=False):
        try:
            run_git("merge-base", "--is-ancestor", parent, child)
        except GitOperationError as exc:
            console.print(
                f"[red]{child} does not contain {parent}; rebase it onto {parent} first.[/red]"
            )
            raise typer.Exit(1) from exc

    console.print(f"Stack ({base}) <- " + " <- ".join(escape(b) for b in branches))
    if not yes:
        typer.confirm("Register these branches as a native stack?", abort=True)

    try:
        _require_gh_stack()
        link_stack(list(branches), base=base)
        prs = [find_open_pr(name) for name in branches]
    except PRSplitError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        console.print(ADOPT_RERUN_HINT)
        raise typer.Exit(1) from exc
    missing = [name for name, pr in zip(branches, prs, strict=True) if pr is None]
    if missing:
        console.print(f"[red]No open PR found for: {', '.join(missing)}[/red]")
        console.print(ADOPT_RERUN_HINT)
        raise typer.Exit(1)

    groups: list[Group] = []
    branch_records: list[BranchRecord] = []
    pr_records: list[PRRecord] = []
    for i, (name, pr) in enumerate(zip(branches, prs, strict=True), start=1):
        assert pr is not None
        gid = f"pr-{i}"
        parent = base if i == 1 else branches[i - 2]
        groups.append(
            Group(
                id=gid,
                title=name,
                description=f"Adopted branch {name}",
                depends_on=[] if i == 1 else [f"pr-{i - 1}"],
            )
        )
        branch_records.append(
            BranchRecord(
                group_id=gid,
                branch_name=name,
                base_branch=parent,
                commit_sha=run_git("rev-parse", name),
                adopted=True,
            )
        )
        pr_records.append(PRRecord(group_id=gid, pr_number=pr[0], pr_url=pr[1], adopted=True))
    save_plan(
        PlanFile(
            plan=SplitPlan(
                dev_branch=branches[-1],
                base_branch=base,
                max_loc=DEFAULT_MAX_LOC,
                stacked=True,
                priority=Priority.ORTHOGONAL,
                groups=groups,
                merge_base_sha=run_git("merge-base", base, branches[-1]),
            ),
            git_state=GitState(branches=branch_records, prs=pr_records),
        )
    )
    logger.success(f"Adopted {len(branches)} branches as a stack on {base}")


def _base_has_merged(base: str) -> bool:
    """True when the plan's base is itself a merged feature branch."""
    try:
        return base != default_branch() and branch_has_merged(base)
    except PRSplitError:
        return False


@app.command(
    help="Move the split onto a new base, e.g. after its --base branch merged: "
    "retargets the root PRs and relinks native stacks on the new base."
)
def retarget(
    to: Annotated[
        str | None,
        typer.Option("--to", help="New base branch (default: the repository's default branch)"),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Retarget without confirming")] = False,
) -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(1)
    plan_file = _load_plan_or_exit()
    plan, git_state = plan_file.plan, plan_file.git_state
    if not git_state.prs:
        console.print("[yellow]The plan has no PRs; nothing to retarget.[/yellow]")
        raise typer.Exit(0)
    try:
        new_base = to or default_branch()
    except PRSplitError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    old_base = plan.base_branch
    if new_base == old_base:
        console.print(f"[yellow]The split already targets {escape(new_base)}.[/yellow]")
        raise typer.Exit(0)

    pr_by_group = {r.group_id: r.pr_number for r in git_state.prs}
    roots = [r for r in git_state.branches if r.base_branch == old_base]
    console.print(
        f"Retarget {len(roots)} root PR(s) from {escape(old_base)} to {escape(new_base)}"
        + (" and relink the native stacks" if plan.stacked else "")
    )
    if not yes:
        typer.confirm("Proceed?", abort=True)

    try:
        # GitHub refuses to change the base of a PR that is in a stack, so
        # unstack first, retarget the roots, then link the chains again.
        if plan.stacked:
            for number in sorted(stack_numbers_for(list(pr_by_group.values()))):
                unstack(number)
        for record in roots:
            if record.group_id in pr_by_group:
                set_pr_base(pr_by_group[record.group_id], new_base)
        branches = [
            r.model_copy(update={"base_branch": new_base}) if r.base_branch == old_base else r
            for r in git_state.branches
        ]
        if plan.stacked:
            _link_stacks(PlanDAG(plan.groups), git_state.prs, branches)
    except PRSplitError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    save_plan(
        PlanFile(
            plan=plan.model_copy(update={"base_branch": new_base}),
            git_state=git_state.model_copy(update={"branches": branches}),
        )
    )
    logger.success(f"Retargeted the split from {old_base} to {new_base}")


@app.command(help="Close all split PRs and delete their branches.")
def clean() -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(0)

    plan_file = _load_plan_or_exit()
    git_state = plan_file.git_state

    typer.confirm("Delete all pr-split branches and close PRs?", abort=True)

    closed_prs, deleted_branches = _cleanup_git_state(git_state)
    if closed_prs < len(git_state.prs) or deleted_branches < len(git_state.branches):
        console.print(f"[yellow]{logs.CLEAN_INCOMPLETE}[/yellow]")
        raise typer.Exit(1)
    logger.success(logs.CLEAN_COMPLETE.format(branches=deleted_branches, prs=closed_prs))


def _delete_stale_recorded_branches(
    records: list[BranchRecord], groups: list[Group], namespace: str
) -> None:
    """Remove branches from a previous failed run that the plan no longer contains.

    A retried execute recreates only the current groups and then overwrites
    git_state; without this, a branch created before the plan was edited would
    survive locally (and on origin, if its push succeeded) with no record left
    for `pr-split clean` to delete.
    """
    expected = {f"{BRANCH_PREFIX}{namespace}/{group.id}" for group in groups}
    for record in records:
        if record.branch_name in expected:
            continue
        if branch_exists(record.branch_name):
            try:
                delete_branch(record.branch_name)
            except GitOperationError as exc:
                console.print(
                    f"[yellow]Could not delete stale branch"
                    f" {escape(record.branch_name)}: {escape(str(exc))}[/yellow]"
                )
        # The remote copy is handled independently: the local branch may be
        # gone (manual prune, an earlier retry) while the pushed one survives,
        # and delete_branch would raise before reaching the remote delete.
        try:
            run_git("ls-remote", "--exit-code", "origin", f"refs/heads/{record.branch_name}")
        except GitOperationError:
            continue  # never pushed (or remote unreachable): nothing to delete
        try:
            run_git("push", "origin", "--delete", record.branch_name)
        except GitOperationError as exc:
            console.print(
                f"[yellow]Could not delete stale remote branch"
                f" {escape(record.branch_name)}: {escape(str(exc))}[/yellow]"
            )


@app.command(
    help="Execute a previously saved dry-run plan, creating branches and PRs.",
)
def execute(
    stack: Annotated[
        bool,
        typer.Option(
            "--stack",
            envvar="PR_SPLIT_STACK",
            help="Register native GitHub stacks even if the plan was saved without --stack",
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
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Create the branches and PRs without asking to confirm"),
    ] = False,
) -> None:
    if not plan_exists():
        console.print(ErrorMsg.NO_PLAN())
        raise typer.Exit(1)

    plan_file = _load_plan_or_exit()
    plan = plan_file.plan
    if stack and not plan.stacked:
        plan = plan.model_copy(update={"stacked": True})
    if draft and not plan.draft:
        plan = plan.model_copy(update={"draft": True})

    existing_prs = {r.group_id: r for r in plan_file.git_state.prs}
    recorded = {r.group_id: r for r in plan_file.git_state.branches}
    kept_branches = {gid: recorded[gid] for gid in existing_prs if gid in recorded}
    group_ids = {g.id for g in plan.groups}
    if existing_prs and (
        set(existing_prs) >= group_ids
        or not set(existing_prs) <= group_ids
        or set(kept_branches) != set(existing_prs)
    ):
        # Every group already has a PR, or the recorded PRs no longer match
        # the plan's groups: there is nothing safe to resume.
        console.print("[red]This plan already has PRs. Use 'pr-split clean' first.[/red]")
        raise typer.Exit(1)
    if existing_prs:
        console.print(
            f"[yellow]A previous run opened {len(existing_prs)} of {len(plan.groups)} PR(s)."
            " Keeping those and creating the rest.[/yellow]"
        )
    elif plan_file.git_state.branches:
        # A previous execute created branches but no PRs - a failed push or PR
        # creation. Branch creation is idempotent (add_worktree recreates an
        # existing branch), so retry instead of forcing 'clean' + a full
        # re-plan, which for the LLM backend means paying for planning again.
        console.print(
            "[yellow]A previous run created branches but no PRs"
            " (the push or PR creation failed). Recreating them and retrying.[/yellow]"
        )

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
    if not commit_exists(plan.merge_base_sha):
        console.print(
            f"[red]Plan's merge base {plan.merge_base_sha} is not in this repository "
            "(plan copied from another checkout, or history rewritten). "
            "Fetch it or re-run 'pr-split split --dry-run' to regenerate.[/red]"
        )
        raise typer.Exit(1)

    if not branch_exists(plan.base_branch):
        console.print(f"[red]{ErrorMsg.BRANCH_NOT_FOUND(branch=plan.base_branch)}[/red]")
        raise typer.Exit(1)
    # A plan saved by an older version may record a remote-tracking base.
    _require_local_branch(plan.base_branch)
    if not is_worktree_clean():
        console.print(f"[red]{ErrorMsg.DIRTY_WORKTREE()}[/red]")
        raise typer.Exit(1)
    if not check_gh_auth():
        console.print(f"[red]{ErrorMsg.GH_AUTH_FAILED()}[/red]")
        raise typer.Exit(1)
    if plan.stacked:
        _require_gh_stack()

    parsed_diff = parse_diff(plan.raw_diff, split_new_files_over=plan.split_new_files_over)

    try:
        validate_no_binary_files(parsed_diff)
        # Building the DAG rejects unknown dependency ids; do it here so a
        # malformed saved plan fails before any branch is created.
        dag = PlanDAG(plan.groups)
        dag.validate_acyclic()
        validate_coverage(plan.groups, parsed_diff)
        validate_new_file_pieces(plan.groups, parsed_diff, dag)
    except PlanValidationError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    # Any plan with dependency edges is laid out along its DAG (stacked or not),
    # so a kept layer's branch sits on its ancestors' current commits; rebuilding
    # an ancestor would rewrite the base under the kept layer's open PR, which
    # would then show the ancestor's changes again. Their PRs are still opened.
    if plan.stacked or any(g.depends_on for g in plan.groups):
        for gid in list(kept_branches):
            for ancestor in dag.ancestors(gid):
                if ancestor in recorded:
                    kept_branches.setdefault(ancestor, recorded[ancestor])

    _present_plan(plan.groups)
    _report_oversized_groups(
        plan.groups, plan.max_loc, {pf.path: len(pf) for pf in parsed_diff.patch_set}
    )
    if not yes:
        typer.confirm("Proceed with creating branches and PRs?", abort=True)

    namespace = derive_split_namespace(plan.dev_branch_arg or plan.dev_branch)
    _delete_stale_recorded_branches(plan_file.git_state.branches, plan.groups, namespace)
    try:
        branch_records = _create_branches_and_commits(
            plan.groups,
            parsed_diff,
            plan.base_branch,
            plan.merge_base_sha,
            namespace,
            author=plan.author,
            stacked=plan.stacked,
            keep=kept_branches,
        )
    except PRSplitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    try:
        pr_records = _push_and_create_prs(
            plan.groups, branch_records, draft=plan.draft, existing_prs=existing_prs
        )
    except PRCreationError as exc:
        save_plan(
            PlanFile(
                plan=plan,
                git_state=GitState(branches=branch_records, prs=exc.pr_records),
            )
        )
        console.print(f"[red]{escape(str(exc))}[/red]")
        console.print("[yellow]Created branches and PRs were saved to the plan file.[/yellow]")
        raise typer.Exit(1) from exc
    save_plan(
        PlanFile(
            plan=plan,
            git_state=GitState(branches=branch_records, prs=pr_records),
        )
    )
    if plan.stacked:
        _link_stacks(PlanDAG(plan.groups), pr_records, branch_records)
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
        bool,
        typer.Option(
            "--auto",
            help=(
                "Queue merges to run after CI checks pass, waiting up to 10 minutes per "
                "batch for them to land before merging dependent PRs"
            ),
        ),
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

    plan_file = _load_plan_or_exit()
    plan = plan_file.plan
    git_state = plan_file.git_state
    pr_map = {r.group_id: r for r in git_state.prs}

    if not pr_map:
        console.print("[yellow]No PRs found in plan. Nothing to merge.[/yellow]")
        raise typer.Exit(0)

    if _base_has_merged(plan.base_branch):
        console.print(
            f"[red]{logs.BASE_ALREADY_MERGED.format(base=escape(plan.base_branch))}[/red]"
        )
        raise typer.Exit(1)

    # A hand-edited plan.json can carry unknown or cyclic dependencies;
    # report that instead of a traceback from the DAG walk.
    try:
        dag = PlanDAG(plan.groups)
        dag.validate_acyclic()
    except PlanValidationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    merged: list[str] = []
    skipped: list[str] = []
    skipped_ids: set[str] = set()
    blocked: list[str] = []
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

            # Checked after the live state so a PR that already merged on GitHub
            # (e.g. into a still-open parent branch) counts as merged rather
            # than being reported as blocked.
            unmerged_parents = [dep for dep in dag.parents(group_id) if dep not in merged]
            if unmerged_parents:
                deps = ", ".join(unmerged_parents)
                logger.warning(f"{group_id} depends on unmerged {deps}, skipping")
                skipped_ids.add(group_id)
                blocked.append(group_id)
                skipped.append(f"{group_id} (dependency {deps} not merged)")
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
    if blocked:
        console.print(
            f"[yellow]Blocked by unmerged dependencies ({len(blocked)}): "
            f"{escape(', '.join(blocked))}. Re-run once those PRs are merged.[/yellow]"
        )
    if notify:
        exit_reason = (
            "merge_error"
            if stopped
            else "incomplete_batch"
            if exited_early
            else "unmerged_dependency"
            if blocked
            else "success"
        )
        skipped_structured = [{"id": s.split(" (")[0], "reason": s} for s in skipped]
        _send_webhook(
            notify,
            {
                "event": "merge_complete",
                "merged": merged,
                "skipped": skipped_structured,
                "failed": failed,
                "success": not (failed or stopped or exited_early or blocked),
                "exit_reason": exit_reason,
            },
        )

    if failed or stopped or exited_early or blocked:
        raise typer.Exit(1)
    logger.success(f"Merge complete: {len(merged)} PRs merged")


@app.command(
    help="Rebase every stacked layer onto its parent's current head and push it, so a fix on"
    " a lower layer reaches the layers above it."
)
def restack(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report the layers that need restacking only")
    ] = False,
    onto_base: Annotated[
        bool,
        typer.Option(
            "--onto-base",
            help="Also rebase the layers that target the base branch onto its current head,"
            " so the whole stack catches up after the base moves",
        ),
    ] = False,
) -> None:
    if not plan_exists():
        console.print(f"[red]{ErrorMsg.NO_PLAN()}[/red]")
        raise typer.Exit(1)
    try:
        results = restack_layers(_load_plan_or_exit(), dry_run=dry_run, onto_base=onto_base)
    except PRSplitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    table = Table(title="Restack")
    table.add_column("ID")
    table.add_column("Branch")
    table.add_column("Result")
    for result in results:
        table.add_row(result.group_id, result.branch, result.action)
    console.print(table)


@app.command(
    name="move",
    help="Move one hunk from a layer of an executed stack up to a layer above it, keeping"
    " every PR open. HUNK is <file>:<index>, as the plan editor's 'show' prints it.",
)
def move_hunk_command(
    hunk: Annotated[str, typer.Argument(help="<file>:<hunk index>")],
    source: Annotated[str, typer.Option("--from", help="Group id that holds the hunk now")],
    target: Annotated[str, typer.Option("--to", help="Group id to move it to")],
) -> None:
    path, _, index_text = hunk.rpartition(":")
    if not path or not index_text.isdigit():
        console.print(f"[red]Expected <file>:<hunk index>, got '{hunk}'[/red]")
        raise typer.Exit(1)
    if not plan_exists():
        console.print(f"[red]{ErrorMsg.NO_PLAN()}[/red]")
        raise typer.Exit(1)
    plan_file = _load_plan_or_exit()
    try:
        results = move_stack_hunk(plan_file, path, int(index_text), source, target)
    except PRSplitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    save_plan(plan_file)
    table = Table(title=f"Moved {hunk} from {source} to {target}")
    table.add_column("ID")
    table.add_column("Branch")
    table.add_column("Result")
    for result in results:
        table.add_row(result.group_id, result.branch, result.action)
    console.print(table)


@app.command(
    help="Rebuild a lost plan from the branches and PRs of a stack that still exists, so"
    " status, merge, restack and clean work again. DEV_BRANCH is the branch the stack was"
    " split from."
)
def recover(
    dev_branch: Annotated[str, typer.Argument(help="Branch (or PR) the stack was split from")],
    base: Annotated[
        str | None,
        typer.Option("--base", help="Base branch of the stack; read from its PRs by default"),
    ] = None,
    stack: Annotated[
        bool,
        typer.Option(
            "--stack",
            help="Mark the plan stacked even when no PR targets another layer any more",
        ),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing plan")] = False,
) -> None:
    # The recovered plan is this dev branch's; other branches' plans are left alone.
    select_plan(dev_branch)
    if plan_exists() and not force:
        console.print(f"[red]{ErrorMsg.RECOVER_PLAN_EXISTS(path=plan_path())}[/red]")
        raise typer.Exit(1)
    try:
        plan_file = recover_plan(dev_branch, base=base, stacked=stack)
    except PRSplitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    save_plan(plan_file)
    git_state = plan_file.git_state
    table = Table(title=f"Recovered plan for {dev_branch} onto {plan_file.plan.base_branch}")
    table.add_column("ID")
    table.add_column("Depends on")
    table.add_column("Branch")
    table.add_column("PR")
    prs = {r.group_id: r for r in git_state.prs}
    branches = {r.group_id: r.branch_name for r in git_state.branches}
    for group in plan_file.plan.groups:
        pr = prs.get(group.id)
        table.add_row(
            group.id,
            ", ".join(group.depends_on),
            branches.get(group.id, ""),
            f"#{pr.pr_number} ({pr.state})" if pr else "",
        )
    console.print(table)
    logger.success(
        logs.RECOVERED_PLAN.format(
            branches=len(git_state.branches), prs=len(git_state.prs), path=plan_path()
        )
    )
