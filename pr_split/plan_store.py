import json
import os
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from . import logs
from .constants import PLAN_DIR, PLAN_FILE
from .exceptions import ErrorMsg, GitOperationError, PRSplitError
from .git_ops.branches import derive_split_namespace, run_git
from .schemas import PlanFile


def repo_root() -> Path:
    """The working tree's top level, or the cwd outside a git repository.

    Every git call in the tool is cwd-independent, so the plan and template
    must be too: a `split --dry-run` from `src/` and an `execute` from the
    repository root have to see the same `.pr-split/plan.json`.
    """
    try:
        return Path(run_git("rev-parse", "--show-toplevel"))
    except GitOperationError:
        return Path.cwd()


# The plan the current command works on, keyed by dev-branch slug. Set by
# `split` (its own branch) or the global --branch option; otherwise the only
# saved plan is used. None with several plans saved means "ambiguous".
_selected: str | None = None


def select_plan(branch_slug: str | None) -> None:
    global _selected
    _selected = branch_slug


def plan_dir() -> Path:
    return repo_root() / PLAN_DIR


def _plans_dir() -> Path:
    return plan_dir() / "plans"


def saved_plan_slugs() -> list[str]:
    folder = _plans_dir()
    return sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []


def _migrate_legacy_plan() -> None:
    """Move a pre-per-branch .pr-split/plan.json to plans/<its branch>.json.

    A plan that cannot be read stays where it is, so loading it still
    reports the problem instead of silently ignoring it.
    """
    legacy = repo_root() / PLAN_FILE
    if not legacy.exists():
        return
    try:
        plan = json.loads(legacy.read_text()).get("plan", {})
        branch = plan.get("dev_branch_arg") or plan["dev_branch"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return
    target = _plans_dir() / f"{derive_split_namespace(str(branch))}.json"
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(legacy, target)


def plan_path() -> Path:
    """Where the selected plan lives: per branch, or the legacy single file."""
    _migrate_legacy_plan()
    legacy = repo_root() / PLAN_FILE
    if _selected is not None:
        selected = _plans_dir() / f"{_selected}.json"
        # An unreadable legacy plan could not be migrated; keep using it so
        # its error surfaces rather than being bypassed.
        return legacy if not selected.exists() and legacy.exists() else selected
    slugs = saved_plan_slugs()
    if len(slugs) == 1:
        return _plans_dir() / f"{slugs[0]}.json"
    if len(slugs) > 1:
        raise PRSplitError(ErrorMsg.PLAN_AMBIGUOUS(branches=", ".join(slugs)))
    return repo_root() / PLAN_FILE


def _exclude_plan_dir() -> None:
    """Keep .pr-split/ out of `git status` and `git add -A` in the target repo.

    The plan holds the whole raw diff; listing it in info/exclude (shared by
    every worktree) stops it from being committed by accident.
    """
    try:
        common = Path(run_git("rev-parse", "--git-common-dir"))
    except GitOperationError:
        return
    if not common.is_absolute():
        common = repo_root() / common
    exclude = common / "info" / "exclude"
    entry = f"/{PLAN_DIR}/"
    try:
        existing = exclude.read_text().splitlines() if exclude.exists() else []
        if entry in existing:
            return
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a") as fh:
            prefix = "" if not existing or existing[-1] == "" else "\n"
            fh.write(f"{prefix}{entry}\n")
    except OSError:
        return


def save_plan(plan_file: PlanFile) -> None:
    """Atomically replace the plan file.

    The most important save happens right after PRs are created; a crash or
    full disk mid-write must not leave truncated JSON that destroys the only
    record of those PRs and branches. The plan is therefore written to a
    temporary file in the same directory and renamed over the target, which
    is atomic on POSIX and Windows alike.
    """
    if _selected is None:
        # No branch chosen for this command: file the plan under its own branch.
        branch = plan_file.plan.dev_branch_arg or plan_file.plan.dev_branch
        target = _plans_dir() / f"{derive_split_namespace(branch)}.json"
    else:
        target = plan_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    _exclude_plan_dir()
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(plan_file.model_dump_json(indent=2))
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info(logs.SAVING_PLAN.format(path=target))


def load_plan() -> PlanFile:
    target = plan_path()
    if not target.exists():
        raise PRSplitError(ErrorMsg.NO_PLAN())
    try:
        plan_file = PlanFile.model_validate_json(target.read_text())
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PRSplitError(ErrorMsg.PLAN_LOAD_FAILED(path=target, detail=exc)) from exc
    logger.info(logs.PLAN_LOADED.format(count=len(plan_file.plan.groups), path=target))
    return plan_file


def plan_exists() -> bool:
    try:
        return plan_path().exists()
    except PRSplitError:
        return True  # several plans: let load_plan report which to pick
