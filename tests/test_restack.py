from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pr_split.cli import app
from pr_split.exceptions import PRSplitError
from pr_split.restack import restack, stale_layers
from pr_split.schemas import BranchRecord, GitState, Group, PlanFile, SplitPlan

LAYERS = ["pr-1", "pr-2", "pr-3"]


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, path: str, text: str, message: str) -> None:
    (repo / path).write_text(text)
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", message)


def _branch(gid: str) -> str:
    return f"pr-split/x/{gid}"


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A pushed stack main <- pr-1 <- pr-2 <- pr-3, each layer adding its own file."""
    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "t")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "t@t")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    repo = tmp_path / "repo"
    _git(tmp_path, "clone", "-q", str(origin), str(repo))
    _commit(repo, "base.txt", "base\n", "init")
    _git(repo, "push", "-q", "origin", "main")
    parent = "main"
    for gid in LAYERS:
        _git(repo, "checkout", "-q", "-b", _branch(gid), parent)
        _commit(repo, f"{gid}.txt", f"{gid} line 1\n{gid} line 2\n", f"{gid} work")
        _git(repo, "push", "-q", "origin", _branch(gid))
        parent = _branch(gid)
    _git(repo, "checkout", "-q", "main")
    monkeypatch.chdir(repo)
    return repo


def _plan_file(*, stacked: bool = True) -> PlanFile:
    groups = [
        Group(id=gid, title=gid, description="", depends_on=[LAYERS[i - 1]] if i else [])
        for i, gid in enumerate(LAYERS)
    ]
    return PlanFile(
        plan=SplitPlan(
            dev_branch="feature",
            base_branch="main",
            max_loc=400,
            priority="orthogonal",
            stacked=stacked,
            groups=groups,
        ),
        git_state=GitState(
            branches=[
                BranchRecord(
                    group_id=gid,
                    branch_name=_branch(gid),
                    base_branch=_branch(LAYERS[i - 1]) if i else "main",
                )
                for i, gid in enumerate(LAYERS)
            ]
        ),
    )


def _fix_layer(repo: Path, gid: str, path: str = "fix.txt", text: str = "fix\n") -> None:
    _git(repo, "checkout", "-q", _branch(gid))
    _commit(repo, path, text, f"review fix on {gid}")
    _git(repo, "checkout", "-q", "main")


def _contains(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=repo
        ).returncode
        == 0
    )


def _own_diff(repo: Path, gid: str, parent: str) -> list[str]:
    return _git(repo, "diff", "--name-only", parent, _branch(gid)).splitlines()


def test_fix_on_the_bottom_layer_reaches_every_layer_above(repo: Path) -> None:
    _fix_layer(repo, "pr-1")
    assert [s[0] for s in stale_layers(_plan_file())] == ["pr-2"]

    results = restack(_plan_file())

    assert [(r.group_id, r.action) for r in results] == [
        ("pr-1", "pushed"),
        ("pr-2", "restacked"),
        ("pr-3", "restacked"),
    ]
    for lower, upper in [("pr-1", "pr-2"), ("pr-2", "pr-3")]:
        assert _contains(repo, _branch(lower), _branch(upper))
        # Each layer still shows only its own change against its parent.
        assert _own_diff(repo, upper, _branch(lower)) == [f"{upper}.txt"]
    # Each layer's work is replayed once, not duplicated.
    assert _git(repo, "log", "--format=%s", _branch("pr-3")).splitlines() == [
        "pr-3 work",
        "pr-2 work",
        "review fix on pr-1",
        "pr-1 work",
        "init",
    ]
    for gid in LAYERS:
        assert _git(repo, "rev-parse", _branch(gid)) == _git(
            repo, "rev-parse", f"origin/{_branch(gid)}"
        )
    assert stale_layers(_plan_file()) == []


def test_fix_pushed_from_another_clone_is_fetched_first(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    _git(other, "checkout", "-q", _branch("pr-1"))
    _commit(other, "fix.txt", "fix\n", "fix from the web")
    _git(other, "push", "-q", "origin", _branch("pr-1"))

    restack(_plan_file())

    assert _git(repo, "log", "-1", "--format=%s", _branch("pr-1")) == "fix from the web"
    assert _contains(repo, _branch("pr-1"), _branch("pr-3"))
    assert _contains(repo, _branch("pr-1"), f"origin/{_branch('pr-3')}")


def test_fix_in_the_middle_only_moves_the_layers_above(repo: Path) -> None:
    pr1_before = _git(repo, "rev-parse", _branch("pr-1"))
    _fix_layer(repo, "pr-2")

    results = restack(_plan_file())

    assert {r.group_id: r.action for r in results} == {
        "pr-2": "pushed",
        "pr-3": "restacked",
    }
    assert _git(repo, "rev-parse", _branch("pr-1")) == pr1_before
    assert _contains(repo, _branch("pr-2"), _branch("pr-3"))


def test_conflict_stops_at_the_layer_and_leaves_it_alone(repo: Path) -> None:
    # The fix rewrites the file pr-2 adds, so replaying pr-2 conflicts.
    _fix_layer(repo, "pr-1", path="pr-2.txt", text="clashing\n")
    before = {gid: _git(repo, "rev-parse", _branch(gid)) for gid in ("pr-2", "pr-3")}

    with pytest.raises(PRSplitError, match=r"Rebasing 'pr-split/x/pr-2' onto"):
        restack(_plan_file())

    assert {gid: _git(repo, "rev-parse", _branch(gid)) for gid in before} == before
    assert "pr-split-restack" not in _git(repo, "worktree", "list")


def test_dry_run_changes_nothing(repo: Path) -> None:
    _fix_layer(repo, "pr-1")
    before = {gid: _git(repo, "rev-parse", _branch(gid)) for gid in LAYERS}

    results = restack(_plan_file(), dry_run=True)

    assert [r.action for r in results] == [
        "would restack onto pr-split/x/pr-1",
        "up to date",
    ]
    assert {gid: _git(repo, "rev-parse", _branch(gid)) for gid in LAYERS} == before


def test_checked_out_layer_is_refused(repo: Path) -> None:
    _fix_layer(repo, "pr-1")
    _git(repo, "checkout", "-q", _branch("pr-2"))
    with pytest.raises(PRSplitError, match="checked out"):
        restack(_plan_file())


def test_diverged_layer_is_refused(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    _git(other, "checkout", "-q", _branch("pr-1"))
    _commit(other, "remote.txt", "r\n", "remote fix")
    _git(other, "push", "-q", "origin", _branch("pr-1"))
    _fix_layer(repo, "pr-1", path="local.txt")

    with pytest.raises(PRSplitError, match="both have commits"):
        restack(_plan_file())


def test_unstacked_plan_is_refused(repo: Path) -> None:
    with pytest.raises(PRSplitError, match="--stack"):
        restack(_plan_file(stacked=False))


def test_cli_restack_and_status_warning(repo: Path) -> None:
    _fix_layer(repo, "pr-1")
    plan_file = _plan_file()
    runner = CliRunner()
    with (
        patch("pr_split.cli.plan_exists", return_value=True),
        patch("pr_split.cli.load_plan", return_value=plan_file),
        patch("pr_split.cli.get_pr_state", return_value={}),
    ):
        status = runner.invoke(app, ["status"])
        result = runner.invoke(app, ["restack"])
        status_after = runner.invoke(app, ["status"])

    assert "pr-split restack" in status.output
    assert result.exit_code == 0, result.output
    assert "restacked" in result.output
    assert "pr-split restack" not in status_after.output


def _advance_main(repo: Path, path: str, text: str) -> None:
    _git(repo, "checkout", "-q", "main")
    _commit(repo, path, text, f"main: {path}")
    _git(repo, "push", "-q", "origin", "main")


def test_onto_base_carries_the_whole_stack_onto_a_moved_base(repo: Path) -> None:
    _advance_main(repo, "release.txt", "0.2.0\n")

    results = restack(_plan_file(), onto_base=True)

    assert [(r.group_id, r.action) for r in results] == [
        ("pr-1", "restacked"),
        ("pr-2", "restacked"),
        ("pr-3", "restacked"),
    ]
    assert _contains(repo, "origin/main", _branch("pr-1"))
    parent = "origin/main"
    for gid in LAYERS:
        assert _own_diff(repo, gid, parent) == [f"{gid}.txt"]
        assert _git(repo, "rev-parse", _branch(gid)) == _git(
            repo, "rev-parse", f"origin/{_branch(gid)}"
        )
        parent = _branch(gid)


def test_without_onto_base_a_moved_base_is_left_alone(repo: Path) -> None:
    _advance_main(repo, "release.txt", "0.2.0\n")
    before = _git(repo, "rev-parse", _branch("pr-1"))

    results = restack(_plan_file())

    assert {r.action for r in results} == {"up to date"}
    assert _git(repo, "rev-parse", _branch("pr-1")) == before


def test_onto_base_conflict_names_the_bottom_layer(repo: Path) -> None:
    # main adds the file pr-1 adds, with different content.
    _advance_main(repo, "pr-1.txt", "from main\n")
    before = {gid: _git(repo, "rev-parse", _branch(gid)) for gid in LAYERS}

    with pytest.raises(PRSplitError, match=r"Rebasing 'pr-split/x/pr-1' onto"):
        restack(_plan_file(), onto_base=True)

    assert {gid: _git(repo, "rev-parse", _branch(gid)) for gid in LAYERS} == before


def test_onto_base_dry_run_reports_the_bottom_layer(repo: Path) -> None:
    _advance_main(repo, "release.txt", "0.2.0\n")

    results = restack(_plan_file(), onto_base=True, dry_run=True)

    assert results[0].action == "would restack onto origin/main"


def test_cli_onto_base(repo: Path) -> None:
    _advance_main(repo, "release.txt", "0.2.0\n")
    with (
        patch("pr_split.cli.plan_exists", return_value=True),
        patch("pr_split.cli.load_plan", return_value=_plan_file()),
    ):
        result = CliRunner().invoke(app, ["restack", "--onto-base"])
    assert result.exit_code == 0, result.output
    assert _contains(repo, "origin/main", f"origin/{_branch('pr-3')}")


def _local_heads(repo: Path) -> dict[str, str]:
    return {gid: _git(repo, "rev-parse", _branch(gid)) for gid in LAYERS}


def test_conflict_above_a_rebased_layer_leaves_every_branch_as_found(repo: Path) -> None:
    # main moves: pr-1 rebases cleanly, then pr-2's file clashes with main's.
    _advance_main(repo, "pr-2.txt", "from main\n")
    before = _local_heads(repo)

    with pytest.raises(PRSplitError, match=r"Rebasing 'pr-split/x/pr-2' onto"):
        restack(_plan_file(), onto_base=True)

    assert _local_heads(repo) == before
    # A rerun meets the same conflict, not a "diverged" refusal on pr-1.
    with pytest.raises(PRSplitError, match=r"Rebasing 'pr-split/x/pr-2' onto"):
        restack(_plan_file(), onto_base=True)


def test_conflict_undoes_a_fast_forward_to_the_remote(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    _git(other, "checkout", "-q", _branch("pr-1"))
    _commit(other, "pr-2.txt", "clashes with pr-2\n", "fix from the web")
    _git(other, "push", "-q", "origin", _branch("pr-1"))
    before = _local_heads(repo)

    with pytest.raises(PRSplitError, match="conflicts"):
        restack(_plan_file())

    assert _local_heads(repo) == before


def test_failed_push_keeps_what_was_pushed_and_can_be_rerun(repo: Path) -> None:
    from pr_split.git_ops import push_branch as real_push

    _fix_layer(repo, "pr-1")
    calls: list[str] = []

    def flaky_push(branch: str) -> None:
        calls.append(branch)
        if len(calls) == 2:
            raise PRSplitError("remote: fatal error in commit_refs")
        real_push(branch)

    with (
        patch("pr_split.restack.push_branch", side_effect=flaky_push),
        pytest.raises(PRSplitError, match="commit_refs"),
    ):
        restack(_plan_file())

    # pr-1 reached the remote and stays there; pr-2 was put back.
    assert _git(repo, "rev-parse", _branch("pr-1")) == _git(
        repo, "rev-parse", f"origin/{_branch('pr-1')}"
    )
    assert not _contains(repo, _branch("pr-1"), _branch("pr-2"))

    results = restack(_plan_file())

    assert {r.group_id: r.action for r in results} == {"pr-2": "restacked", "pr-3": "restacked"}
    assert _contains(repo, _branch("pr-1"), f"origin/{_branch('pr-3')}")


def test_onto_base_follows_the_base_branchs_configured_remote(repo: Path) -> None:
    origin_url = _git(repo, "remote", "get-url", "origin")
    _git(repo, "remote", "add", "upstream", origin_url)
    _git(repo, "config", "branch.main.remote", "upstream")
    _git(repo, "config", "branch.main.merge", "refs/heads/main")
    _advance_main(repo, "release.txt", "0.2.0\n")

    results = restack(_plan_file(), onto_base=True, dry_run=True)

    assert results[0].action == "would restack onto upstream/main"
    restack(_plan_file(), onto_base=True)
    assert _contains(repo, "upstream/main", _branch("pr-1"))
