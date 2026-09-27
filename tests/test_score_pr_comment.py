from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_pr.py"
MARKER = "<!-- pr-split-score -->"


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
        assert body.startswith(MARKER)
        assert "within acceptable size limits" in body
        assert "under the 400 threshold" in body

    def test_plain_skip_writes_no_comment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, output_file = self._prepare(tmp_path, monkeypatch)

        module._skip("Not a pull_request event; skipping.")

        outputs = _outputs(output_file)
        assert outputs["should_split"] == "false"
        assert "comment_path" not in outputs

    def test_main_under_threshold_writes_within_limits_comment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, output_file = self._prepare(tmp_path, monkeypatch)
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "feature")
        monkeypatch.setenv("PR_NUMBER", "7")
        monkeypatch.setenv("MAX_LOC", "400")

        with (
            patch.object(module, "_run") as mock_run,
            patch.object(module.subprocess, "run") as mock_planner,
        ):
            mock_run.return_value.stdout = "10\t2\ta.py\n"
            module.main()

        mock_planner.assert_not_called()
        outputs = _outputs(output_file)
        assert outputs["total_loc"] == "12"
        assert outputs["should_split"] == "false"
        body = Path(outputs["comment_path"]).read_text()
        assert body.startswith(MARKER)
        assert "PR has 12 LOC" in body
        assert "within acceptable size limits" in body
