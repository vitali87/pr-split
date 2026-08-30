from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_pr.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_pr_table", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_pr_table"] = module
    spec.loader.exec_module(module)
    return module


def _outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


class TestMdEscape:
    def test_pipes_and_newlines_are_neutralised(self) -> None:
        module = _load_script()
        assert module._md_escape("a | b") == "a \\| b"
        assert module._md_escape("first\nsecond\r\nthird") == "first second third"


class TestCommentTable:
    def _prepare(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path]:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "feature")
        monkeypatch.setenv("MAX_LOC", "10")
        monkeypatch.chdir(tmp_path)
        return _load_script(), output_file

    def test_dependency_ids_and_multiline_titles_stay_in_their_cells(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, output_file = self._prepare(tmp_path, monkeypatch)
        groups = [
            {
                "id": "pr-1|x",
                "title": "a | b\nsecond line",
                "depends_on": [],
                "assignments": [{"file_path": "a.py"}],
                "estimated_loc": 300,
                "estimated_added": 300,
                "estimated_removed": 0,
            },
            {
                "id": "pr-2",
                "title": "t",
                "depends_on": ["pr-1|x"],
                "assignments": [{"file_path": "b.py"}],
                "estimated_loc": 300,
                "estimated_added": 300,
                "estimated_removed": 0,
            },
        ]

        def fake_run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
            stdout = "600\t0\ta.py\n" if cmd[:2] == ["git", "diff"] else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        def fake_pr_split(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            plan_dir = tmp_path / ".pr-split"
            plan_dir.mkdir(exist_ok=True)
            (plan_dir / "plan.json").write_text(json.dumps({"groups": groups}))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(module, "_run", side_effect=fake_run),
            patch.object(module.subprocess, "run", side_effect=fake_pr_split),
        ):
            module.main()

        comment = Path(_outputs(output_file)["comment_path"]).read_text()
        rows = [line for line in comment.splitlines() if line.startswith("| pr-")]
        assert len(rows) == 2
        assert rows[0].startswith("| pr-1\\|x | a \\| b second line |")
        assert "| pr-1\\|x |" in rows[1]
        # five columns => exactly six unescaped cell separators per row
        assert all(len(re.findall(r"(?<!\\)\|", row)) == 6 for row in rows)


class TestMinLocValidation:
    def test_non_numeric_min_loc_is_rejected_up_front(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "feature")
        monkeypatch.setenv("MIN_LOC", "abc")
        module = _load_script()
        with patch.object(module, "_run") as mock_run, pytest.raises(SystemExit):
            module.main()
        assert "MIN_LOC must be an integer, got 'abc'" in capsys.readouterr().err
        mock_run.assert_not_called()
