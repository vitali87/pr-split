from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pr_split.constants import AssignmentType, Priority
from pr_split.plan_store import PLAN_FILE
from pr_split.schemas import GitState, Group, GroupAssignment, PlanFile, SplitPlan

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_pr.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("score_pr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["score_pr"] = module
    spec.loader.exec_module(module)
    return module


def _plan_file() -> PlanFile:
    group = Group(
        id="pr-1",
        title="t",
        description="d",
        assignments=[
            GroupAssignment(
                file_path="a.py",
                assignment_type=AssignmentType.WHOLE_FILE,
                hunk_indices=[0],
            )
        ],
        estimated_loc=3,
    )
    plan = SplitPlan(
        dev_branch="feature",
        base_branch="main",
        max_loc=400,
        priority=Priority.ORTHOGONAL,
        groups=[group, group.model_copy(update={"id": "pr-2", "depends_on": ["pr-1"]})],
    )
    return PlanFile(plan=plan, git_state=GitState())


class TestLoadPlanGroups:
    def test_reads_groups_from_saved_plan_file(self, tmp_path: Path) -> None:
        path = tmp_path / Path(PLAN_FILE).name
        path.write_text(_plan_file().model_dump_json())
        groups = _load_script().load_plan_groups(str(path))
        assert [g["id"] for g in groups] == ["pr-1", "pr-2"]
        assert groups[0]["assignments"][0]["file_path"] == "a.py"
        assert groups[1]["depends_on"] == ["pr-1"]

    def test_accepts_bare_plan_document(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.json"
        path.write_text(_plan_file().plan.model_dump_json())
        assert len(_load_script().load_plan_groups(str(path))) == 2
