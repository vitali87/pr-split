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
    spec = importlib.util.spec_from_file_location("score_pr_comment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_pr_comment"] = module
    spec.loader.exec_module(module)
    return module


def _outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


class TestSkipWritesRefreshableComment:
    def _prepare(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path]:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
        return _load_script(), output_file

    def test_under_threshold_skip_writes_marker_comment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, output_file = self._prepare(tmp_path, monkeypatch)

        module._skip("PR has 12 LOC — under the 400 threshold.", within_limits=True)

        outputs = _outputs(output_file)
        assert outputs["should_split"] == "false"
        body = Path(outputs["comment_path"]).read_text()
        assert body.startswith(module.COMMENT_MARKER)
        assert "within acceptable size limits" in body
        assert "under the 400 threshold" in body

    def test_failure_skip_writes_no_comment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, output_file = self._prepare(tmp_path, monkeypatch)

        module._skip("pr-split failed to generate a plan.")

        outputs = _outputs(output_file)
        assert outputs["should_split"] == "false"
        assert "comment_path" not in outputs

    def test_main_failure_path_leaves_existing_comment_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, output_file = self._prepare(tmp_path, monkeypatch)
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "feature")
        monkeypatch.setenv("MAX_LOC", "10")

        def fake_run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
            stdout = "400\t100\ta.py\n" if cmd[:2] == ["git", "diff"] else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        def failing_pr_split(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

        with (
            patch.object(module, "_run", side_effect=fake_run),
            patch.object(module.subprocess, "run", side_effect=failing_pr_split),
        ):
            module.main()

        outputs = _outputs(output_file)
        assert outputs["total_loc"] == "500"
        assert outputs["should_split"] == "false"
        assert "comment_path" not in outputs

    def test_main_split_path_comment_starts_with_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, output_file = self._prepare(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "feature")
        monkeypatch.setenv("MAX_LOC", "10")
        monkeypatch.setenv("THRESHOLD_GROUPS", "2")

        def fake_run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
            stdout = "400\t100\ta.py\n" if cmd[:2] == ["git", "diff"] else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        def stub_pr_split(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            plan_dir = tmp_path / ".pr-split"
            plan_dir.mkdir(exist_ok=True)
            groups = [
                {
                    "id": f"pr-{i}",
                    "title": f"t{i}",
                    "depends_on": [],
                    "estimated_loc": 5,
                    "estimated_added": 5,
                    "estimated_removed": 0,
                    "assignments": [{"file_path": "a.py"}],
                }
                for i in (1, 2)
            ]
            (plan_dir / "plan.json").write_text(json.dumps({"groups": groups}))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(module, "_run", side_effect=fake_run),
            patch.object(module.subprocess, "run", side_effect=stub_pr_split),
        ):
            module.main()

        outputs = _outputs(output_file)
        assert outputs["should_split"] == "true"
        body = Path(outputs["comment_path"]).read_text()
        assert body.startswith(module.COMMENT_MARKER)
        assert "| pr-1 |" in body and "| pr-2 |" in body
