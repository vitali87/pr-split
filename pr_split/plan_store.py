import json
import os
from pathlib import Path
from urllib.parse import quote, unquote

from loguru import logger
from pydantic import ValidationError

from . import logs
from .constants import PLAN_DIR, PLAN_FILE
from .exceptions import ErrorMsg, GitOperationError, PRSplitError
from .git_ops.branches import run_git
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


# The plan the current command works on, keyed by its plan file's stem. Set
# by `split` (its own branch) or the global --branch option; otherwise the only
# saved plan is used. None with several plans saved means "ambiguous".
_selected: str | None = None


def plan_key(dev_branch: str) -> str:
    """The plan file stem for a dev branch: the name percent-encoded, so it is
    reversible and `feature/x` and `feature-x` do not share a file."""
    return quote(dev_branch, safe="")


def select_plan(dev_branch: str | None) -> None:
    global _selected
    _selected = plan_key(dev_branch) if dev_branch else None


def plan_dir() -> Path:
    return repo_root() / PLAN_DIR


def _plans_dir() -> Path:
    return plan_dir() / "plans"


def _saved_plan_keys() -> list[str]:
    folder = _plans_dir()
    return sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []


def saved_plan_branches() -> list[str]:
    """The dev branches that have a saved plan."""
    return [unquote(key) for key in _saved_plan_keys()]


def _migrate_legacy_plan() -> bool:
    """Move a pre-per-branch .pr-split/plan.json to plans/<its branch>.json.

    A plan that cannot be read stays where it is, so loading it still
    reports the problem instead of silently ignoring it. Returns True only
    when such an unreadable legacy plan is left behind.
    """
    legacy = repo_root() / PLAN_FILE
    if not legacy.exists():
        return False
    try:
        plan = json.loads(legacy.read_text()).get("plan", {})
        branch = plan.get("dev_branch_arg") or plan["dev_branch"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return True
    target = _plans_dir() / f"{plan_key(str(branch))}.json"
    if target.exists():
        # That branch already has a per-branch plan; the legacy file is left
        # alone and never used in its place.
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    _exclude_plan_dir()
    os.replace(legacy, target)
    return False


def plan_path() -> Path:
    """Where the selected plan lives: per branch, or the legacy single file."""
    legacy_unreadable = _migrate_legacy_plan()
    legacy = repo_root() / PLAN_FILE
    if _selected is not None:
        selected = _plans_dir() / f"{_selected}.json"
        # An unreadable legacy plan could not be migrated; keep using it so
        # its error surfaces rather than being bypassed. A readable one belongs
        # to one branch and must not stand in for another's.
        return legacy if legacy_unreadable and not selected.exists() else selected
    keys = _saved_plan_keys()
    if len(keys) == 1:
        return _plans_dir() / f"{keys[0]}.json"
    if len(keys) > 1:
        raise PRSplitError(ErrorMsg.PLAN_AMBIGUOUS(branches=", ".join(saved_plan_branches())))
    return legacy


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
        # rev-parse prints a relative path relative to the cwd, not the top level.
        common = (Path.cwd() / common).resolve()
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
        target = _plans_dir() / f"{plan_key(branch)}.json"
    else:
        target = plan_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    _exclude_plan_dir()
    tmp = target.with_name(target.name + ".tmp")
    # The raw diff may carry surrogate-escaped bytes from non-UTF-8 files;
    # json.dumps escapes those as \udcXX and loads them back losslessly,
    # which pydantic's own JSON writer refuses to do.
    payload = json.dumps(plan_file.model_dump(mode="json"), indent=2, ensure_ascii=True)
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info(logs.SAVING_PLAN.format(path=target))


def load_plan() -> PlanFile:
    target = plan_path()
    if not target.exists():
        raise PRSplitError(ErrorMsg.NO_PLAN())
    try:
        plan_file = PlanFile.model_validate(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise PRSplitError(ErrorMsg.PLAN_LOAD_FAILED(path=target, detail=exc)) from exc
    logger.info(logs.PLAN_LOADED.format(count=len(plan_file.plan.groups), path=target))
    return plan_file


def plan_exists() -> bool:
    try:
        return plan_path().exists()
    except PRSplitError:
        return True  # several plans: let load_plan report which to pick
