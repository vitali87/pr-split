from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_pr.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_pr_events", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_pr_events"] = module
    spec.loader.exec_module(module)
    return module


def _outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


class TestNonPullRequestEvents:
    @pytest.mark.parametrize(
        "env",
        [
            pytest.param({}, id="unset"),
            pytest.param({"BASE_BRANCH": "", "HEAD_BRANCH": ""}, id="empty"),
            pytest.param({"BASE_BRANCH": "main", "HEAD_BRANCH": ""}, id="head-missing"),
            pytest.param({"BASE_BRANCH": "", "HEAD_BRANCH": "feature"}, id="base-missing"),
        ],
    )
    def test_skips_without_running_git(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.delenv("BASE_BRANCH", raising=False)
        monkeypatch.delenv("HEAD_BRANCH", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        module = _load_script()

        with patch.object(module.subprocess, "run") as mock_run:
            module.main()

        mock_run.assert_not_called()
        outputs = _outputs(output_file)
        assert outputs == {
            "total_loc": "0",
            "total_groups": "1",
            "objective": "0",
            "should_split": "false",
        }

    def test_pull_request_event_still_fetches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "feature")
        monkeypatch.setenv("PR_NUMBER", "7")
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        module = _load_script()

        with patch.object(module, "_run") as mock_run:
            mock_run.return_value.stdout = "1\t0\ta.py\n"
            module.main()

        fetches = [call.args[0] for call in mock_run.call_args_list if call.args[0][1] == "fetch"]
        assert ["git", "fetch", "origin", "main"] in fetches
        assert ["git", "fetch", "origin", "refs/pull/7/head:pr-split/head-7"] in fetches
        assert _outputs(output_file)["total_loc"] == "1"


class TestOversizedVerdictDoesNotDependOnThePlanner:
    """A PR over max-loc is flagged whatever the planner produced."""

    def _run_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        returncode: int,
        plan: dict | None,
    ) -> tuple[dict[str, str], str]:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "feature")
        monkeypatch.setenv("PR_NUMBER", "7")
        monkeypatch.setenv("MAX_LOC", "400")
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
        module = _load_script()

        def fake_planner(cmd: list[str], **kwargs: object) -> object:
            if plan is not None:
                (tmp_path / ".pr-split").mkdir(exist_ok=True)
                (tmp_path / ".pr-split" / "plan.json").write_text(json.dumps(plan))
            return subprocess.CompletedProcess(cmd, returncode, "", "boom")

        with (
            patch.object(module, "_run") as mock_run,
            patch.object(module.subprocess, "run", side_effect=fake_planner),
        ):
            mock_run.return_value.stdout = "600\t300\tbig.py\n"
            module.main()

        outputs = _outputs(output_file)
        return outputs, Path(outputs["comment_path"]).read_text()

    def test_planner_failure_still_flags_an_oversized_pr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outputs, comment = self._run_main(tmp_path, monkeypatch, returncode=1, plan=None)
        assert outputs["should_split"] == "true"
        assert "**900 LOC**, over the **400 LOC** limit" in comment
        assert "pr-split exited with an error" in comment

    def test_zero_group_plan_still_flags_an_oversized_pr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outputs, comment = self._run_main(
            tmp_path, monkeypatch, returncode=0, plan={"plan": {"groups": []}, "git_state": {}}
        )
        assert outputs["should_split"] == "true"
        assert outputs["total_groups"] == "0"
        assert "plan has 0 group(s)" in comment
        assert "within acceptable size limits" not in comment

    def test_single_group_plan_is_below_threshold_but_still_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        group = {"id": "pr-1", "title": "t", "estimated_loc": 900, "assignments": []}
        outputs, comment = self._run_main(
            tmp_path, monkeypatch, returncode=0, plan={"plan": {"groups": [group]}}
        )
        assert outputs["should_split"] == "true"
        assert "fewer than the 2 needed" in comment

    def test_missing_plan_file_still_flags_an_oversized_pr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outputs, comment = self._run_main(tmp_path, monkeypatch, returncode=0, plan=None)
        assert outputs["should_split"] == "true"
        assert "wrote no plan file" in comment
