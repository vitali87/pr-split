from __future__ import annotations

import pytest

from pr_split.config import Settings
from pr_split.constants import (
    ANTHROPIC_MAX_CONTEXT_TOKENS,
    DEFAULT_MODEL,
    OPENAI_MAX_CONTEXT_TOKENS,
    OPENAI_MODEL,
    Provider,
)

ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "PR_SPLIT_PROVIDER",
    "PR_SPLIT_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestSettingsDefaults:
    def test_anthropic_default_model(self) -> None:
        s = Settings(ANTHROPIC_API_KEY="sk-test")
        assert s.model == DEFAULT_MODEL
        assert s.provider == Provider.ANTHROPIC

    def test_openai_default_model(self) -> None:
        s = Settings(provider=Provider.OPENAI, OPENAI_API_KEY="sk-test")
        assert s.model == OPENAI_MODEL

    def test_custom_model_not_overridden(self) -> None:
        s = Settings(ANTHROPIC_API_KEY="sk-test", model="custom-model")
        assert s.model == "custom-model"

    def test_custom_model_openai(self) -> None:
        s = Settings(provider=Provider.OPENAI, OPENAI_API_KEY="sk-test", model="gpt-4o")
        assert s.model == "gpt-4o"


class TestSettingsApiKey:
    def test_anthropic_api_key_property(self) -> None:
        s = Settings(ANTHROPIC_API_KEY="sk-ant-test")
        assert s.api_key == "sk-ant-test"

    def test_openai_api_key_property(self) -> None:
        s = Settings(provider=Provider.OPENAI, OPENAI_API_KEY="sk-oai-test")
        assert s.api_key == "sk-oai-test"


class TestSettingsContextTokens:
    def test_anthropic_context_tokens(self) -> None:
        s = Settings(ANTHROPIC_API_KEY="sk-test")
        assert s.max_context_tokens == ANTHROPIC_MAX_CONTEXT_TOKENS

    def test_openai_context_tokens(self) -> None:
        s = Settings(provider=Provider.OPENAI, OPENAI_API_KEY="sk-test")
        assert s.max_context_tokens == OPENAI_MAX_CONTEXT_TOKENS


class TestSettingsEnvVars:
    def test_provider_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PR_SPLIT_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        s = Settings()
        assert s.provider == Provider.OPENAI
        assert s.model == OPENAI_MODEL

    def test_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("PR_SPLIT_MODEL", "custom-from-env")
        s = Settings()
        assert s.model == "custom-from-env"
