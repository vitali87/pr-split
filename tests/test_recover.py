from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pr_split.cli import app
from pr_split.constants import PLAN_FILE, PRState
from pr_split.exceptions import GitOperationError, PRSplitError
from pr_split.git_ops.prs import list_prs_with_head_prefix
from pr_split.plan_store import load_plan, save_plan
from pr_split.recover import StackPR, rebuild_plan
from pr_split.restack import stale_layers
from pr_split.schemas import PlanFile
from tests.test_stack_move import _git, stack  # noqa: F401  (fixture)

P = "pr-split/feature/"


def _pr(number: int, gid: str, base: str, *, state: str = "OPEN", body: str = "") -> StackPR:
    return StackPR.from_gh(
        {
            "number": number,
            "url": f"https://github.com/o/r/pull/{number}",
            "state": state,
            "headRefName": P + gid,
            "baseRefName": base,
            "title": f"title {gid}",
            "body": body,
        }
    )


def _deps(plan_file: PlanFile) -> dict[str, list[str]]:
    return {g.id: g.depends_on for g in plan_file.plan.groups}


def test_stacked_layers_depend_on_the_branch_their_pr_targets() -> None:
    prs = [_pr(1, "pr-1", "main"), _pr(2, "pr-2", P + "pr-1"), _pr(3, "pr-3", P + "pr-2")]
    plan_file = rebuild_plan("feature", prs=prs, branches={P + "pr-1": "abc"})

    assert _deps(plan_file) == {"pr-1": [], "pr-2": ["pr-1"], "pr-3": ["pr-2"]}
    plan = plan_file.plan
    assert (plan.base_branch, plan.stacked, plan.dev_branch) == ("main", True, "feature")
    records = {r.group_id: r for r in plan_file.git_state.branches}
    assert records["pr-2"].base_branch == P + "pr-1"
    assert records["pr-1"].commit_sha == "abc"
    assert [r.pr_number for r in plan_file.git_state.prs] == [1, 2, 3]


def test_dependencies_come_from_the_pr_body_once_a_pr_is_retargeted() -> None:
    # pr-1 merged and its branch was deleted, so GitHub retargeted pr-2 at main.
    body = "Adds b.\n\n## Dependencies\n\nThis PR depends on: `pr-1`\n\n## Dependency graph"
    prs = [_pr(1, "pr-1", "main", state="MERGED"), _pr(2, "pr-2", "main", body=body)]
    plan_file = rebuild_plan("feature", prs=prs, branches={}, stacked=True)

    assert _deps(plan_file) == {"pr-1": [], "pr-2": ["pr-1"]}
    assert plan_file.plan.groups[1].description == "Adds b."
    assert plan_file.git_state.prs[0].state is PRState.MERGED


def test_a_merge_node_keeps_every_parent_from_its_body() -> None:
    body = "Joins.\n\n## Dependencies\n\nThis PR depends on: `pr-1`, `pr-2`, `gone`"
    prs = [_pr(1, "pr-1", "main"), _pr(2, "pr-2", "main"), _pr(3, "pr-3", "main", body=body)]
    plan_file = rebuild_plan("feature", prs=prs, branches={})

    # A group that no longer exists is dropped rather than failing the whole plan.
    assert _deps(plan_file)["pr-3"] == ["pr-1", "pr-2"]
    assert plan_file.plan.stacked is False


def test_open_pr_wins_over_a_closed_one_for_the_same_branch() -> None:
    prs = [_pr(9, "pr-1", "main", state="CLOSED"), _pr(4, "pr-1", "main")]
    plan_file = rebuild_plan("feature", prs=prs, branches={})
    assert [r.pr_number for r in plan_file.git_state.prs] == [4]


def test_a_pushed_branch_without_a_pr_is_kept() -> None:
    plan_file = rebuild_plan("feature", prs=[_pr(1, "pr-1", "main")], branches={P + "pr-2": "def"})
    assert [g.id for g in plan_file.plan.groups] == ["pr-2", "pr-1"]
    assert [r.group_id for r in plan_file.git_state.prs] == ["pr-1"]
    assert plan_file.git_state.branches[0].base_branch == "main"


def test_other_stacks_are_ignored() -> None:
    other = StackPR.from_gh(
        {"number": 5, "headRefName": "pr-split/feature-2/pr-1", "baseRefName": "dev"}
    )
    plan_file = rebuild_plan("feature", prs=[_pr(1, "pr-1", "main"), other], branches={})
    assert [g.id for g in plan_file.plan.groups] == ["pr-1"]


def test_ambiguous_base_asks_for_base() -> None:
    prs = [_pr(1, "pr-1", "main"), _pr(2, "pr-2", "dev")]
    with pytest.raises(PRSplitError, match=r"dev, main.*pass --base"):
        rebuild_plan("feature", prs=prs, branches={})
    assert rebuild_plan("feature", prs=prs, branches={}, base="main").plan.base_branch == "main"


def test_nothing_found() -> None:
    with pytest.raises(PRSplitError, match="No branch or PR under 'pr-split/feature/'"):
        rebuild_plan("feature", prs=[], branches={"pr-split/other/pr-1": "x"})


@patch("pr_split.git_ops.prs._run_gh")
def test_pr_listing_keeps_only_the_stack_in_this_repository(mock_gh: MagicMock) -> None:
    mock_gh.return_value = json.dumps(
        [
            {"number": 1, "headRefName": P + "pr-1", "isCrossRepository": False},
            {"number": 2, "headRefName": "fix/x", "isCrossRepository": False},
            {"number": 3, "headRefName": P + "pr-2", "isCrossRepository": True},
        ]
    )
    assert [pr["number"] for pr in list_prs_with_head_prefix(P)] == [1]
    args = mock_gh.call_args.args
    # The prefix narrows the search before the limit applies.
    assert args[args.index("--search") + 1] == f"head:{P}"
    assert args[args.index("--state") + 1] == "all"


def _gh_prs(plan_file: PlanFile) -> list[dict[str, object]]:
    """The PRs GitHub would list for the fixture's stack."""
    records = {r.group_id: r for r in plan_file.git_state.branches}
    return [
        {
            "number": n,
            "url": f"https://github.com/o/r/pull/{n}",
            "state": "OPEN",
            "headRefName": records[g.id].branch_name,
            "baseRefName": records[g.id].base_branch,
            "title": g.title,
            "body": g.description,
        }
        for n, g in enumerate(plan_file.plan.groups, start=10)
    ]


def test_cli_rebuilds_an_executed_stack_whose_plan_is_gone(
    stack: tuple[Path, PlanFile],  # noqa: F811
) -> None:
    repo, original = stack
    # The fixture split under the namespace "x"; a dev branch of that name matches it.
    _git(repo, "branch", "x", "feature")
    listed = _gh_prs(original)
    # A new checkout: none of the stack's branches exist locally.
    for record in original.git_state.branches:
        _git(repo, "branch", "-D", record.branch_name)
        _git(repo, "update-ref", "-d", f"refs/remotes/origin/{record.branch_name}")
    assert not Path(PLAN_FILE).exists()

    with patch("pr_split.recover.list_prs_with_head_prefix", return_value=listed):
        result = CliRunner().invoke(app, ["recover", "x"])

    assert result.exit_code == 0, result.output
    recovered = load_plan()
    assert _deps(recovered) == {"pr-1": [], "pr-2": ["pr-1"], "pr-3": ["pr-2"]}
    assert recovered.plan.stacked and recovered.plan.base_branch == "main"
    assert recovered.plan.merge_base_sha == original.plan.merge_base_sha
    for record in recovered.git_state.branches:
        assert record.commit_sha == _git(repo, "rev-parse", f"origin/{record.branch_name}")
    assert [r.pr_number for r in recovered.git_state.prs] == [10, 11, 12]
    # The recovered plan drives the stack commands again.
    assert stale_layers(recovered) == []
    with patch("pr_split.cli.get_pr_state", return_value={"state": "OPEN"}):
        status = CliRunner().invoke(app, ["status"])
    assert status.exit_code == 0, status.output
    assert "#12" in status.output


def test_cli_keeps_an_existing_plan_without_force(
    stack: tuple[Path, PlanFile],  # noqa: F811
) -> None:
    _, original = stack
    save_plan(original)
    listed = _gh_prs(original)
    with patch("pr_split.recover.list_prs_with_head_prefix", return_value=listed) as gh:
        refused = CliRunner().invoke(app, ["recover", "x"])
        assert refused.exit_code == 1
        assert "pass --force" in refused.output
        gh.assert_not_called()
        replaced = CliRunner().invoke(app, ["recover", "x", "--force"])
    assert replaced.exit_code == 0, replaced.output
    assert [r.pr_number for r in load_plan().git_state.prs] == [10, 11, 12]


def test_cli_reports_a_gh_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with patch("pr_split.git_ops.prs._run_gh", side_effect=GitOperationError("gh: not logged in")):
        result = CliRunner().invoke(app, ["recover", "feature"])
    assert result.exit_code == 1
    assert "Cannot list PRs" in result.output
