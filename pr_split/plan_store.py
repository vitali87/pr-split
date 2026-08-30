import json
from pathlib import Path

from loguru import logger

from . import logs
from .constants import PLAN_DIR, PLAN_FILE
from .exceptions import ErrorMsg, PRSplitError
from .schemas import PlanFile


def save_plan(plan_file: PlanFile) -> None:
    path = Path(PLAN_DIR)
    path.mkdir(parents=True, exist_ok=True)
    plan_path = Path(PLAN_FILE)
    # The raw diff may carry surrogate-escaped bytes from non-UTF-8 files;
    # json.dumps escapes those as \udcXX and loads them back losslessly,
    # which pydantic's own JSON writer refuses to do.
    payload = json.dumps(plan_file.model_dump(mode="json"), indent=2, ensure_ascii=True)
    plan_path.write_text(payload, encoding="utf-8")
    logger.info(logs.SAVING_PLAN.format(path=plan_path))


def load_plan() -> PlanFile:
    plan_path = Path(PLAN_FILE)
    if not plan_path.exists():
        raise PRSplitError(ErrorMsg.NO_PLAN())
    plan_file = PlanFile.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
    logger.info(logs.PLAN_LOADED.format(count=len(plan_file.plan.groups), path=plan_path))
    return plan_file


def plan_exists() -> bool:
    return Path(PLAN_FILE).exists()
