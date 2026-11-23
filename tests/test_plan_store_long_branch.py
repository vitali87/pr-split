from __future__ import annotations

from pathlib import Path

import pytest

from pr_split.plan_store import plan_exists, select_plan


def test_no_plan_exists_for_a_branch_name_too_long_for_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pr-split" / "plans").mkdir(parents=True)
    select_plan("A" * 300)
    try:
        assert plan_exists() is False
    finally:
        select_plan(None)
