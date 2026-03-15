from __future__ import annotations

import json
from pathlib import Path

import pytest

from pr_split.constants import Priority
from pr_split.exceptions import PRSplitError
from pr_split.plan_store import load_plan, plan_exists, save_plan
from pr_split.schemas import (
    BranchRecord,
    GitState,
    Group,
    PlanFile,
    SplitPlan,
)


def _make_plan_file() -> PlanFile:
    return PlanFile(
        plan=SplitPlan(
            dev_branch="feat/big",
            base_branch="main",
            max_loc=400,
            priority=Priority.ORTHOGONAL,
            groups=[
                Group(
                    id="pr-1",
                    title="feat: add auth",
                    description="Auth module",
                ),
            ],
        ),
        git_state=GitState(
            branches=[
                BranchRecord(
                    group_id="pr-1",
                    branch_name="pr-split/pr-1",
                    base_branch="main",
                    commit_sha="abc123",
                ),
            ],
        ),
    )


class TestPlanStore:
    def test_save_and_load_roundtrip(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        plan_file = _make_plan_file()
        save_plan(plan_file)
        loaded = load_plan()
        assert loaded.plan.dev_branch == "feat/big"
        assert len(loaded.plan.groups) == 1
        assert loaded.plan.groups[0].id == "pr-1"

    def test_plan_exists_false(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert plan_exists() is False

    def test_plan_exists_true(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        save_plan(_make_plan_file())
        assert plan_exists() is True

    def test_load_missing_raises(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(PRSplitError, match="No split plan"):
            load_plan()

    def test_git_state_preserved(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan_file = _make_plan_file()
        save_plan(plan_file)
        loaded = load_plan()
        assert len(loaded.git_state.branches) == 1
        assert loaded.git_state.branches[0].commit_sha == "abc123"


class TestPlanStoreJson:
    def test_saved_file_is_valid_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan_file = PlanFile(
            plan=SplitPlan(
                dev_branch="dev",
                base_branch="main",
                max_loc=400,
                priority=Priority.LOGICAL,
                groups=[],
            ),
        )
        save_plan(plan_file)
        raw = json.loads((tmp_path / ".pr-split" / "plan.json").read_text())
        assert "plan" in raw
        assert raw["plan"]["priority"] == "logical"
