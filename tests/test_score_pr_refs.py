from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_pr.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_pr_refs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_pr_refs"] = module
    spec.loader.exec_module(module)
    return module


class TestScoreUsesFetchedRefsDirectly:
    def test_head_named_like_base_does_not_clobber_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "main")  # fork PR opened from the fork's main
        monkeypatch.setenv("PR_NUMBER", "7")
        monkeypatch.setenv("MAX_LOC", "1")
        module = _load_script()

        git_calls: list[list[str]] = []

        def fake_run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
            git_calls.append(cmd)
            stdout = "5\t3\ta.py\n" if cmd[:2] == ["git", "diff"] else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        split_cmds: list[list[str]] = []

        def fake_subprocess_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            split_cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stub")

        with (
            patch.object(module, "_run", side_effect=fake_run),
            patch.object(module.subprocess, "run", side_effect=fake_subprocess_run),
        ):
            module.main()

        assert not any(c[:2] == ["git", "branch"] for c in git_calls)
        assert len(split_cmds) == 1
        split = split_cmds[0]
        assert split[:3] == ["pr-split", "split", "pr-split/head-7"]
        assert split[split.index("--base") + 1] == "origin/main"

    def test_branch_event_without_pr_number_uses_origin_refs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_file = tmp_path / "output.txt"
        output_file.touch()
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
        monkeypatch.setenv("BASE_BRANCH", "main")
        monkeypatch.setenv("HEAD_BRANCH", "feature")
        monkeypatch.delenv("PR_NUMBER", raising=False)
        monkeypatch.setenv("MAX_LOC", "1")
        module = _load_script()

        def fake_run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
            stdout = "5\t3\ta.py\n" if cmd[:2] == ["git", "diff"] else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        split_cmds: list[list[str]] = []

        def fake_subprocess_run(
            cmd: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            split_cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stub")

        with (
            patch.object(module, "_run", side_effect=fake_run),
            patch.object(module.subprocess, "run", side_effect=fake_subprocess_run),
        ):
            module.main()

        split = split_cmds[0]
        assert split[2] == "origin/feature"
        assert split[split.index("--base") + 1] == "origin/main"
