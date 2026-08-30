from pathlib import Path

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


def plan_dir() -> Path:
    return repo_root() / PLAN_DIR


def plan_path() -> Path:
    return repo_root() / PLAN_FILE


def save_plan(plan_file: PlanFile) -> None:
    plan_dir().mkdir(parents=True, exist_ok=True)
    target = plan_path()
    target.write_text(plan_file.model_dump_json(indent=2))
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
    return plan_path().exists()
