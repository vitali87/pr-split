from pathlib import Path

from loguru import logger

from . import logs
from .constants import PLAN_DIR, PLAN_FILE
from .exceptions import ErrorMsg, PRSplitError
from .schemas import BranchRecord, GitState, PlanFile, PRRecord


def save_plan(plan_file: PlanFile) -> Path:
    path = Path(PLAN_DIR)
    path.mkdir(parents=True, exist_ok=True)
    plan_path = Path(PLAN_FILE)
    plan_path.write_text(plan_file.model_dump_json(indent=2))
    logger.info(logs.SAVING_PLAN.format(path=plan_path))
    return plan_path


def load_plan() -> PlanFile:
    plan_path = Path(PLAN_FILE)
    if not plan_path.exists():
        raise PRSplitError(ErrorMsg.NO_PLAN())
    plan_file = PlanFile.model_validate_json(plan_path.read_text())
    logger.info(logs.PLAN_LOADED.format(count=len(plan_file.plan.groups), path=plan_path))
    return plan_file


def plan_exists() -> bool:
    return Path(PLAN_FILE).exists()


def update_git_state(git_state: GitState) -> None:
    plan_file = load_plan()
    plan_file.git_state = git_state
    save_plan(plan_file)


def add_branch_record(record: BranchRecord) -> None:
    plan_file = load_plan()
    plan_file.git_state.branches.append(record)
    save_plan(plan_file)


def add_pr_record(record: PRRecord) -> None:
    plan_file = load_plan()
    plan_file.git_state.prs.append(record)
    save_plan(plan_file)
