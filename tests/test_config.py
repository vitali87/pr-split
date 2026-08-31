from __future__ import annotations

import pytest
from pydantic import ValidationError

from pr_split.config import Settings
from pr_split.constants import (
    ANTHROPIC_MAX_CONTEXT_TOKENS,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_CP_SAT_TIMEOUT_SECONDS,
    DEFAULT_MIN_LOC_RATIO,
    DEFAULT_MODEL,
    DEFAULT_PARTITION_STRATEGY,
    DEFAULT_STRICT_LOC_BOUNDS,
    OPENAI_MAX_CONTEXT_TOKENS,
    OPENAI_MODEL,
    PartitionStrategy,
    Provider,
)


class TestSettingsDefaults:
    def test_anthropic_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        s = Settings(provider=Provider.ANTHROPIC)
        assert s.model == DEFAULT_MODEL

    def test_openai_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        s = Settings(provider=Provider.OPENAI)
        assert s.model == OPENAI_MODEL

    def test_explicit_model_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        s = Settings(provider=Provider.ANTHROPIC, model="custom-model")
        assert s.model == "custom-model"

    def test_default_strategies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        s = Settings(provider=Provider.ANTHROPIC)
        assert s.chunk_strategy == DEFAULT_CHUNK_STRATEGY
        assert s.partition_strategy == DEFAULT_PARTITION_STRATEGY
        assert s.cp_sat_timeout == DEFAULT_CP_SAT_TIMEOUT_SECONDS
        assert s.min_loc is None
        assert s.strict_loc_bounds == DEFAULT_STRICT_LOC_BOUNDS


class TestSettingsApiKeyValidation:
    def test_anthropic_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
            Settings(provider=Provider.ANTHROPIC)

    def test_openai_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            Settings(provider=Provider.OPENAI)

    def test_anthropic_key_present_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        s = Settings(provider=Provider.ANTHROPIC)
        assert s.api_key == "sk-test-key"

    def test_openai_key_present_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        s = Settings(provider=Provider.OPENAI)
        assert s.api_key == "sk-test-key"

    def test_graph_partition_does_not_require_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(partition_strategy=PartitionStrategy.GRAPH)
        assert s.partition_strategy == PartitionStrategy.GRAPH

    def test_cp_sat_partition_does_not_require_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(partition_strategy=PartitionStrategy.CP_SAT)
        assert s.partition_strategy == PartitionStrategy.CP_SAT


class TestSettingsLocBoundsValidation:
    def test_min_loc_less_than_max_loc_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(partition_strategy=PartitionStrategy.GRAPH, min_loc=50, max_loc=100)
        assert s.min_loc == 50
        assert s.max_loc == 100

    def test_min_loc_equal_to_max_loc_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="min_loc 100 must be less than max_loc 100"):
            Settings(partition_strategy=PartitionStrategy.GRAPH, min_loc=100, max_loc=100)

    def test_min_loc_greater_than_max_loc_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="min_loc 101 must be less than max_loc 100"):
            Settings(partition_strategy=PartitionStrategy.GRAPH, min_loc=101, max_loc=100)

    def test_min_loc_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("PR_SPLIT_MIN_LOC", "75")
        s = Settings(partition_strategy=PartitionStrategy.GRAPH)
        assert s.min_loc == 75


class TestSettingsAutoDerivMinLoc:
    def test_min_loc_derived_when_refinement_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            max_loc=400,
            max_refinement_iterations=2,
        )
        assert s.min_loc == 400 // DEFAULT_MIN_LOC_RATIO

    def test_min_loc_not_derived_when_refinement_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            max_loc=400,
            max_refinement_iterations=0,
        )
        assert s.min_loc is None

    def test_explicit_min_loc_not_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            min_loc=50,
            max_loc=400,
            max_refinement_iterations=2,
        )
        assert s.min_loc == 50

    def test_derived_min_loc_uses_integer_division(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(
            partition_strategy=PartitionStrategy.GRAPH,
            max_loc=401,
            max_refinement_iterations=1,
        )
        assert s.min_loc == 401 // DEFAULT_MIN_LOC_RATIO


class TestSettingsMaxContextTokens:
    def test_anthropic_max_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        s = Settings(provider=Provider.ANTHROPIC)
        assert s.max_context_tokens == ANTHROPIC_MAX_CONTEXT_TOKENS

    def test_openai_max_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        s = Settings(provider=Provider.OPENAI)
        assert s.max_context_tokens == OPENAI_MAX_CONTEXT_TOKENS


class TestSettingsEmptyKeyValidation:
    def test_empty_string_key_raises_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
            Settings(provider=Provider.ANTHROPIC)

    def test_empty_string_key_raises_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            Settings(provider=Provider.OPENAI)


class TestDerivedMinLocFloor:
    @pytest.mark.parametrize("max_loc", [2, 3])
    def test_tiny_max_loc_derives_min_loc_of_one(
        self, monkeypatch: pytest.MonkeyPatch, max_loc: int
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        settings = Settings(max_loc=max_loc, max_refinement_iterations=1)
        assert settings.min_loc == 1

    def test_max_loc_of_one_with_refinement_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with pytest.raises(ValueError, match="min_loc 1 must be less than max_loc 1"):
            Settings(max_loc=1, max_refinement_iterations=1)

    def test_normal_max_loc_uses_ratio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert Settings(max_loc=400, max_refinement_iterations=1).min_loc == 100
