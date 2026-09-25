from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pr_split.cli import _create_branches_and_commits, app
from pr_split.constants import AssignmentType
from pr_split.diff_ops import extract_diff, parse_diff
from pr_split.exceptions import PRSplitError
from pr_split.schemas import GitState, Group, GroupAssignment, PlanFile, SplitPlan
from pr_split.stack_move import move_hunk

DOCS_BASE = "".join(f"line {i}\n" for i in range(1, 31))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(repo: Path, path: str, text: str) -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)


def _assign(path: str, *indices: int) -> GroupAssignment:
    return GroupAssignment(
        file_path=path, assignment_type=AssignmentType.PARTIAL_HUNKS, hunk_indices=list(indices)
    )


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, PlanFile]:
    """An executed, pushed stack pr-1 <- pr-2 <- pr-3; pr-1 holds both docs hunks."""
    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "t")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "t@t")
    monkeypatch.delenv("PR_SPLIT_PER_GROUP_RUN", raising=False)
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    repo = tmp_path / "repo"
    _git(tmp_path, "clone", "-q", str(origin), str(repo))
    _write(repo, "a.py", "A = 1\n")
    _write(repo, "docs.md", DOCS_BASE)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "push", "-q", "origin", "main")

    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "a.py", "A = 2\n")
    _write(
        repo,
        "docs.md",
        DOCS_BASE.replace("line 2\n", "intro\n").replace(
            "line 28\n", "--new-flag enables the thing\n"
        ),
    )
    _write(repo, "b.py", "B = 1\n")
    _write(repo, "flag.py", "NEW_FLAG = '--new-flag'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "feature")
    _git(repo, "checkout", "-q", "main")
    monkeypatch.chdir(repo)

    raw_diff = extract_diff("feature", "main")
    parsed = parse_diff(raw_diff)
    assert len(next(pf for pf in parsed.patch_set if pf.path == "docs.md")) == 2
    groups = [
        Group(
            id="pr-1",
            title="a and docs",
            description="",
            assignments=[_assign("a.py", 0), _assign("docs.md", 0, 1)],
        ),
        Group(
            id="pr-2",
            title="b",
            description="",
            depends_on=["pr-1"],
            assignments=[_assign("b.py", 0)],
        ),
        Group(
            id="pr-3",
            title="flag",
            description="",
            depends_on=["pr-2"],
            assignments=[_assign("flag.py", 0)],
        ),
    ]
    merge_base = _git(repo, "merge-base", "main", "feature")
    records = _create_branches_and_commits(groups, parsed, "main", merge_base, "x", stacked=True)
    for record in records:
        _git(repo, "push", "-q", "origin", record.branch_name)
    plan_file = PlanFile(
        plan=SplitPlan(
            dev_branch="feature",
            base_branch="main",
            max_loc=400,
            priority="orthogonal",
            stacked=True,
            groups=groups,
            merge_base_sha=merge_base,
            raw_diff=raw_diff,
        ),
        git_state=GitState(branches=records),
    )
    return repo, plan_file


def _changed(repo: Path, parent: str, branch: str) -> str:
    return _git(repo, "diff", parent, branch)


def test_hunk_moves_to_the_target_layer_only(stack: tuple[Path, PlanFile]) -> None:
    repo, plan_file = stack
    b = {r.group_id: r.branch_name for r in plan_file.git_state.branches}
    # A base change reaches the stack through a merge, as in the report.
    _write(repo, "base.py", "BASE_CONSTANT = 7\n")
    _git(repo, "add", "base.py")
    _git(repo, "commit", "-q", "-m", "base constant")
    _git(repo, "checkout", "-q", b["pr-1"])
    _git(repo, "merge", "-q", "--no-edit", "main")
    _git(repo, "checkout", "-q", "main")
    from pr_split.restack import restack

    restack(plan_file)

    results = move_hunk(plan_file, "docs.md", 1, "pr-1", "pr-3")

    assert {r.group_id for r in results} == {"pr-1", "pr-2", "pr-3"}
    layer1 = _changed(repo, "main", b["pr-1"])
    assert "+intro" in layer1 and "--new-flag" not in layer1
    assert "--new-flag" not in _changed(repo, b["pr-1"], b["pr-2"])
    layer3 = _changed(repo, b["pr-2"], b["pr-3"])
    assert "+--new-flag enables the thing" in layer3
    assert set(_git(repo, "diff", "--name-only", b["pr-2"], b["pr-3"]).split()) == {
        "docs.md",
        "flag.py",
    }
    # Nothing else changed: the top layer is the dev branch plus the base merge.
    for path in ("a.py", "b.py", "flag.py", "docs.md"):
        assert _git(repo, "show", f"{b['pr-3']}:{path}") == _git(repo, "show", f"feature:{path}")
    for branch in b.values():
        assert _git(repo, "show", f"{branch}:base.py") == "BASE_CONSTANT = 7"
    for branch in b.values():
        assert _git(repo, "rev-parse", branch) == _git(repo, "rev-parse", f"origin/{branch}")
    # The plan follows the move.
    groups = {g.id: g for g in plan_file.plan.groups}
    assert [a.hunk_indices for a in groups["pr-1"].assignments if a.file_path == "docs.md"] == [
        [0]
    ]
    assert any(
        a.file_path == "docs.md" and a.hunk_indices == [1] for a in groups["pr-3"].assignments
    )


def test_moving_down_the_stack_is_refused(stack: tuple[Path, PlanFile]) -> None:
    _, plan_file = stack
    with pytest.raises(PRSplitError, match="only move up the stack"):
        move_hunk(plan_file, "b.py", 0, "pr-2", "pr-1")


def test_hunk_the_source_does_not_hold_is_refused(stack: tuple[Path, PlanFile]) -> None:
    _, plan_file = stack
    with pytest.raises(PRSplitError, match=r"does not hold hunk b\.py:0"):
        move_hunk(plan_file, "b.py", 0, "pr-1", "pr-3")


def test_unknown_hunk_is_refused(stack: tuple[Path, PlanFile]) -> None:
    _, plan_file = stack
    with pytest.raises(PRSplitError, match=r"no hunk docs\.md:9"):
        move_hunk(plan_file, "docs.md", 9, "pr-1", "pr-3")


def test_whole_new_file_moves_up(stack: tuple[Path, PlanFile]) -> None:
    repo, plan_file = stack
    b = {r.group_id: r.branch_name for r in plan_file.git_state.branches}

    move_hunk(plan_file, "b.py", 0, "pr-2", "pr-3")

    assert _git(repo, "diff", "--name-only", b["pr-1"], b["pr-2"]) == ""
    assert "b.py" in _git(repo, "diff", "--name-only", b["pr-2"], b["pr-3"]).split()
    assert _git(repo, "show", f"{b['pr-3']}:b.py") == "B = 1"


def test_cli_move_saves_the_plan(stack: tuple[Path, PlanFile]) -> None:
    _, plan_file = stack
    with (
        patch("pr_split.cli.plan_exists", return_value=True),
        patch("pr_split.cli.load_plan", return_value=plan_file),
        patch("pr_split.cli.save_plan") as save,
    ):
        result = CliRunner().invoke(app, ["move", "docs.md:1", "--from", "pr-1", "--to", "pr-3"])

    assert result.exit_code == 0, result.output
    save.assert_called_once_with(plan_file)
    assert "restacked" in result.output


def test_cli_rejects_a_malformed_hunk() -> None:
    result = CliRunner().invoke(app, ["move", "docs.md", "--from", "pr-1", "--to", "pr-3"])
    assert result.exit_code == 1
    assert "Expected <file>:<hunk index>" in result.output
