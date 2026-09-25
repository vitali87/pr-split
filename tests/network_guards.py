from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_base_fetch() -> Iterator[None]:
    """Keep CLI tests off the network: split would fetch the base's upstream."""
    with patch("pr_split.cli.diff_base_ref", side_effect=lambda base: base):
        yield
