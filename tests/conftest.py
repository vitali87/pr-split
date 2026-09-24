from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_base_merge_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """status and merge ask GitHub whether the plan's base branch has merged;
    keep that off the network in tests. Tests that exercise it patch it again."""
    monkeypatch.setattr("pr_split.cli._base_has_merged", lambda base: False)
