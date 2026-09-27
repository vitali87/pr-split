from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_base_merge_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """status and merge ask GitHub whether the plan's base branch has merged;
    keep that off the network in tests. Tests that exercise it patch it again."""
    monkeypatch.setattr("pr_split.cli._base_has_merged", lambda base: False)


@pytest.fixture(autouse=True)
def _cli_tools_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI checks that git and gh are on PATH before doing anything; the
    tests mock every git/gh call, so they must not depend on the machine
    having gh installed. Tests of the missing-tool path patch it again."""
    monkeypatch.setattr("pr_split.cli.require_tools", lambda *tools: None)
