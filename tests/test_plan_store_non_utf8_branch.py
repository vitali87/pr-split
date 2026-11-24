from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pr_split import plan_store
from pr_split.cli import app
from pr_split.exceptions import PRSplitError
from pr_split.plan_store import plan_key, select_plan

# What `pr-split recover $'caf\xe9'` receives: the byte that is not UTF-8
# arrives as a lone surrogate.
NOT_UTF8 = "caf\udce9"


@pytest.fixture(autouse=True)
def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plan_store, "_selected", None)


@pytest.mark.parametrize(
    ("name", "shown"),
    [
        (NOT_UTF8, r"'caf\xe9'"),
        # A surrogate no byte stands behind, which a hand-edited plan file can hold.
        ("caf\ud800", r"'caf\ud800'"),
        (NOT_UTF8 + "\ud800", r"'caf\udce9\ud800'"),
    ],
)
def test_a_branch_name_that_is_not_utf8_is_refused_with_a_printable_message(
    name: str, shown: str
) -> None:
    with pytest.raises(PRSplitError, match="not valid UTF-8") as refused:
        select_plan(name)
    message = str(refused.value)
    assert shown in message
    message.encode("utf-8")  # the name is shown as escapes, so this can be printed


def test_a_name_with_non_ascii_characters_is_still_a_plan_key() -> None:
    select_plan("café/ünïcode")
    assert plan_key("café") == "caf%C3%A9"


@pytest.mark.parametrize(
    "argv",
    [
        ["recover", NOT_UTF8],
        ["recover", NOT_UTF8, "--force"],
        ["split", NOT_UTF8],
        ["adopt", "bottom", NOT_UTF8],
        ["--branch", NOT_UTF8, "status"],
    ],
)
def test_commands_stop_before_doing_anything(argv: list[str], tmp_path: Path) -> None:
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "notvalidUTF-8" in "".join(result.output.split())
    assert not (tmp_path / ".pr-split").exists()
