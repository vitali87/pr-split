from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pr_split import plan_store
from pr_split.cli import app
from pr_split.exceptions import PRSplitError
from pr_split.plan_store import load_plan, plan_exists, save_plan, select_plan
from pr_split.schemas import PlanFile
from tests.test_plan_store import _make_plan_file

# 141 characters, but every "/" is three bytes of the percent-encoded file name.
SLASHES = "a/" * 70 + "a"
TOO_LONG = ["A" * 300, SLASHES]


@pytest.fixture(autouse=True)
def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plan_store, "_selected", None)


def _flat(output: str) -> str:
    """Console output without the line breaks rich wraps it with."""
    return "".join(output.split())


def _plan_for(branch: str) -> PlanFile:
    plan_file = _make_plan_file()
    plan_file.plan.dev_branch = plan_file.plan.dev_branch_arg = branch
    return plan_file


@pytest.mark.parametrize("branch", TOO_LONG)
def test_a_branch_whose_plan_file_name_cannot_fit_is_refused(branch: str) -> None:
    with pytest.raises(PRSplitError, match="too long to keep a split plan"):
        select_plan(branch)


def test_the_longest_name_that_fits_is_saved_and_loaded() -> None:
    # The limit is the temporary file's name, which save_plan really writes.
    fits = "b" * (plan_store._max_file_name() - len(".json.tmp"))
    select_plan(fits)
    assert plan_exists() is False
    save_plan(_plan_for(fits))
    assert load_plan().plan.dev_branch == fits
    with pytest.raises(PRSplitError, match="too long"):
        select_plan(fits + "b")


@pytest.mark.parametrize(
    "argv",
    [
        ["recover", SLASHES],
        ["recover", SLASHES, "--force"],
        ["split", SLASHES],
        ["adopt", "bottom", SLASHES],
        ["--branch", SLASHES, "status"],
    ],
)
def test_commands_stop_before_doing_anything(argv: list[str], tmp_path: Path) -> None:
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "toolongtokeepasplitplan" in _flat(result.output)
    assert not (tmp_path / ".pr-split").exists()


def test_an_old_plan_that_cannot_be_filed_is_named_not_hidden(tmp_path: Path) -> None:
    save_plan(_plan_for("feature"))
    legacy = tmp_path / ".pr-split" / "plan.json"
    legacy.write_text(_plan_for("z" * 300).model_dump_json())
    select_plan("feature")
    # Not "no plan": the caller goes on to load_plan, which says what is wrong.
    assert plan_exists() is True
    with pytest.raises(PRSplitError, match=r"plan\.json.*too long"):
        load_plan()
    with patch("pr_split.cli.recover_plan", return_value=_plan_for("feature")):
        result = CliRunner().invoke(app, ["recover", "feature", "--force"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.output
    assert "plan.json" in _flat(result.output)
    legacy.unlink()
    assert load_plan().plan.dev_branch == "feature"
