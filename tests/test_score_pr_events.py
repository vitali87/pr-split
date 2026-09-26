from __future__ import annotations

import importlib.util
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
