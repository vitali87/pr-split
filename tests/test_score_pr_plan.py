from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from pr_split.constants import AssignmentType, Priority
from pr_split.plan_store import save_plan
from pr_split.schemas import Group, GroupAssignment, PlanFile, SplitPlan

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_pr.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_pr_plan", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_pr_plan"] = module
    spec.loader.exec_module(module)
    return module


def _outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


def _group(gid: str, loc: int, files: list[str], depends_on: list[str] | None = None) -> Group:
    return Group(
        id=gid,
        title=f"Group {gid}",
        description="d",
        assignments=[
            GroupAssignment(file_path=f, assignment_type=AssignmentType.WHOLE_FILE) for f in files
        ],
        depends_on=depends_on or [],
        estimated_loc=loc,
        estimated_added=loc,
        estimated_removed=0,
    )


def _saved_plan(groups: list[Group]) -> PlanFile:
    return PlanFile(
        plan=SplitPlan(
            dev_branch="feature",
            base_branch="main",
            max_loc=400,
            priority=Priority.ORTHOGONAL,
            groups=groups,
        )
    )


class TestScorePrReadsSavedPlan:
    def _prepare(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path]:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "feature")
        monkeypatch.setenv("MAX_LOC", "400")
        monkeypatch.chdir(tmp_path)
        return _load_script(), output_file

    def _run_main(self, module: ModuleType, plan_file: PlanFile | None) -> None:
        def fake_run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
            stdout = "900\t100\ta.py\n200\t0\tb.py\n" if cmd[:2] == ["git", "diff"] else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        def fake_pr_split(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            assert cmd[:2] == ["pr-split", "split"]
            if plan_file is not None:
                save_plan(plan_file)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(module, "_run", side_effect=fake_run),
            patch.object(module.subprocess, "run", side_effect=fake_pr_split),
        ):
            module.main()

    def test_groups_are_read_from_the_nested_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module, output_file = self._prepare(tmp_path, monkeypatch)
        groups = [_group("pr-1", 1000, ["a.py"]), _group("pr-2", 200, ["b.py"], ["pr-1"])]

        self._run_main(module, _saved_plan(groups))

        outputs = _outputs(output_file)
        assert outputs["total_groups"] == "2"
        assert outputs["should_split"] == "true"
        # overflow (1000 - 400) * 1000 + scatter 0 * 50 + 2 groups
        assert outputs["objective"] == "600002"
        comment = Path(outputs["comment_path"]).read_text()
        assert "could be split into **2 smaller PRs**" in comment
        assert "| pr-2 | Group pr-2 | +200/-0 | pr-1 | `b.py` |" in comment
        assert "Split plan: 2 groups" in capsys.readouterr().out

    def test_plan_without_groups_is_a_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module, output_file = self._prepare(tmp_path, monkeypatch)

        self._run_main(module, _saved_plan([]))

        outputs = _outputs(output_file)
        assert outputs["should_split"] == "false"
        assert "comment_path" not in outputs
        assert "Plan file contains no groups." in capsys.readouterr().out
