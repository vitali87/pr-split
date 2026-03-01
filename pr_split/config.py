from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from .constants import (
    ANTHROPIC_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_LOC,
    DEFAULT_MODEL,
    OPENAI_MAX_CONTEXT_TOKENS,
    OPENAI_MODEL,
    Priority,
    Provider,
)


class Settings(BaseSettings):
    model_config = {"env_prefix": "PR_SPLIT_"}

    anthropic_api_key: str = Field(
        default="",
        validation_alias="ANTHROPIC_API_KEY",
    )
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
    )
    provider: Provider = Provider.ANTHROPIC
    model: str = ""
    max_loc: int = DEFAULT_MAX_LOC
    priority: Priority = Priority.ORTHOGONAL

    @model_validator(mode="after")
    def set_default_model(self):
        if not self.model:
            match self.provider:
                case Provider.ANTHROPIC:
                    self.model = DEFAULT_MODEL
                case Provider.OPENAI:
                    self.model = OPENAI_MODEL
                case _:
                    raise NotImplementedError(f"No default model for provider '{self.provider}'")
        return self

    @model_validator(mode="after")
    def check_api_key_is_present(self):
        match self.provider:
            case Provider.ANTHROPIC:
                if not self.anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY must be set when provider is 'anthropic'")
            case Provider.OPENAI:
                if not self.openai_api_key:
                    raise ValueError("OPENAI_API_KEY must be set when provider is 'openai'")
        return self

    @property
    def api_key(self) -> str:
        match self.provider:
            case Provider.ANTHROPIC:
                return self.anthropic_api_key
            case Provider.OPENAI:
                return self.openai_api_key
            case _:
                raise NotImplementedError(f"Provider '{self.provider}' is not supported")

    @property
    def max_context_tokens(self) -> int:
        match self.provider:
            case Provider.ANTHROPIC:
                return ANTHROPIC_MAX_CONTEXT_TOKENS
            case Provider.OPENAI:
                return OPENAI_MAX_CONTEXT_TOKENS
            case _:
                raise NotImplementedError(f"Provider '{self.provider}' is not supported")
