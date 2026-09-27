from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pr_split.git_ops import diff_base_ref


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, path: str, text: str, message: str) -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(clone, upstream work tree); the clone's main is 1 commit behind origin/main."""
    for key, value in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }.items():
        monkeypatch.setenv(key, value)
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    upstream = tmp_path / "upstream"
    _git(tmp_path, "clone", "-q", str(origin), str(upstream))
    _commit(upstream, "README.md", "hello\n", "init")
    _git(upstream, "push", "-q", "origin", "main")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "checkout", "-q", "-b", "feature")
    _commit(clone, "src/feature.py", "".join(f"x{i} = {i}\n" for i in range(5)), "feature")
    _git(clone, "checkout", "-q", "main")

    # Upstream moves on: a release commit the local main has not seen.
    _commit(upstream, "Cargo.toml", 'version = "0.2.0"\n', "release")
    _git(upstream, "push", "-q", "origin", "main")
    _git(clone, "checkout", "-q", "feature")
    _git(clone, "merge", "-q", "--no-edit", "origin/main")  # stale: origin/main not fetched yet
    monkeypatch.chdir(clone)
    return clone, upstream


def test_fetches_and_diffs_against_the_upstream(repos: tuple[Path, Path]) -> None:
    clone, upstream = repos
    assert diff_base_ref("main") == "origin/main"
    assert _git(clone, "rev-parse", "origin/main") == _git(upstream, "rev-parse", "main")


def test_up_to_date_base_uses_the_tracking_ref(repos: tuple[Path, Path]) -> None:
    clone, _ = repos
    _git(clone, "fetch", "-q")
    _git(clone, "branch", "-f", "main", "origin/main")
    assert diff_base_ref("main") == "origin/main"


def test_branch_without_upstream_is_used_as_is(repos: tuple[Path, Path]) -> None:
    clone, _ = repos
    _git(clone, "branch", "-q", "local-only", "main")
    assert diff_base_ref("local-only") == "local-only"


def test_failed_fetch_falls_back_to_the_last_fetched_ref(repos: tuple[Path, Path]) -> None:
    clone, upstream = repos
    _git(clone, "remote", "set-url", "origin", str(clone.parent / "gone.git"))
    assert diff_base_ref("main") == "origin/main"
    # Nothing new arrived: origin/main is still the state from clone time.
    assert _git(clone, "rev-parse", "origin/main") != _git(upstream, "rev-parse", "main")


def test_split_does_not_count_upstream_commits_as_branch_work(
    repos: tuple[Path, Path],
) -> None:
    from unittest.mock import patch

    from typer.testing import CliRunner

    from pr_split.cli import app
    from pr_split.git_ops import diff_base_ref as real_diff_base_ref

    clone, _ = repos
    _git(clone, "fetch", "-q")
    _git(clone, "merge", "-q", "--no-edit", "origin/main")
    # Local main is now 1 commit behind the branch's own base.
    with patch("pr_split.cli.diff_base_ref", side_effect=real_diff_base_ref):
        result = CliRunner().invoke(
            app,
            ["split", "feature", "--base", "main", "--dry-run", "--partition-strategy", "graph"],
            input="done\n",
        )

    assert result.exit_code == 0, result.output
    plan = json.loads((clone / ".pr-split" / "plan.json").read_text())
    plan = plan.get("plan", plan)
    files = {a["file_path"] for g in plan["groups"] for a in g["assignments"]}
    assert files == {"src/feature.py"}
    assert plan["base_branch"] == "main"
    assert plan["merge_base_sha"] == _git(clone, "rev-parse", "origin/main")


def test_local_commits_missing_upstream_are_warned_about(repos: tuple[Path, Path]) -> None:
    from unittest.mock import patch

    clone, _ = repos
    _git(clone, "checkout", "-q", "main")
    _commit(clone, "local.txt", "unpushed\n", "unpushed")
    _git(clone, "checkout", "-q", "feature")

    with patch("pr_split.git_ops.branches.logger") as mock_logger:
        assert diff_base_ref("main") == "origin/main"

    messages = [call.args[0] for call in mock_logger.warning.call_args_list]
    assert any("1 commit(s) behind 'origin/main'" in m for m in messages)
    assert any("1 commit(s) not on 'origin/main'" in m for m in messages)
