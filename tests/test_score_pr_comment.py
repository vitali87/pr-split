from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
    def test_skip_writes_marker_comment_and_path_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
        module = _load_script()

        module._skip("PR has 12 LOC — under the 400 threshold, no split needed.")

        outputs = _outputs(output_file)
        assert outputs["should_split"] == "false"
        body = Path(outputs["comment_path"]).read_text()
        assert body.startswith(module.COMMENT_MARKER)
        assert "within acceptable size limits" in body
        assert "under the 400 threshold" in body

    def test_split_comment_uses_same_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
        module = _load_script()

        module._write_comment(f"{module.COMMENT_MARKER}\n## pr-split analysis\n")

        body = Path(_outputs(output_file)["comment_path"]).read_text()
        assert body.startswith("<!-- pr-split-score -->")
