from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_base_fetch() -> Iterator[None]:
    """Keep CLI tests off the network: split would fetch the base's upstream."""
    with patch("pr_split.cli.diff_base_ref", side_effect=lambda base: base):
        yield


@pytest.fixture(autouse=True)
def _no_selected_plan() -> Iterator[None]:
    """A command's plan selection is process-wide; start each test without one."""
    from pr_split.plan_store import select_plan

    select_plan(None)
    yield
    select_plan(None)
