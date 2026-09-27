from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from pr_split.cli import _split_settings
from pr_split.config import Settings
from pr_split.constants import PartitionStrategy


def _build(strategy: PartitionStrategy, **options: int) -> Settings:
    return Settings(partition_strategy=strategy, **options)


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PR_SPLIT_PROVIDER", "PR_SPLIT_MODEL"):
        monkeypatch.delenv(key, raising=False)


def test_no_key_falls_back_to_graph_and_says_why() -> None:
    with patch("pr_split.cli.logger") as mock_logger:
        settings = _split_settings(None, _build)

    assert settings.partition_strategy is PartitionStrategy.GRAPH
    message = mock_logger.warning.call_args.args[0]
    assert "ANTHROPIC_API_KEY must be set" in message
    assert "graph backend" in message


def test_key_present_keeps_the_llm_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert _split_settings(None, _build).partition_strategy is PartitionStrategy.LLM


def test_local_provider_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PR_SPLIT_PROVIDER", "local")
    monkeypatch.setenv("PR_SPLIT_MODEL", "qwen2.5-coder:7b")
    assert _split_settings(None, _build).partition_strategy is PartitionStrategy.LLM


def test_explicit_llm_without_a_key_still_fails() -> None:
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY must be set"):
        _split_settings(PartitionStrategy.LLM, _build)


def test_explicit_backend_is_used_as_is() -> None:
    assert _split_settings(PartitionStrategy.CP_SAT, _build).partition_strategy == "cp_sat"


def test_an_error_both_backends_share_is_reported_unchanged() -> None:
    def build(strategy: PartitionStrategy) -> Settings:
        return _build(strategy, min_loc=500, max_loc=100)

    with pytest.raises(ValidationError, match=r"ANTHROPIC_API_KEY|min_loc"):
        _split_settings(None, build)
