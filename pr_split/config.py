from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from .constants import (
    ANTHROPIC_MAX_CONTEXT_TOKENS,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_CP_SAT_TIMEOUT_SECONDS,
    DEFAULT_MAX_LOC,
    DEFAULT_MAX_REFINEMENT_ITERATIONS,
    DEFAULT_MIN_LOC,
    DEFAULT_MIN_LOC_RATIO,
    DEFAULT_MODEL,
    DEFAULT_PARTITION_STRATEGY,
    DEFAULT_STRICT_LOC_BOUNDS,
    OPENAI_MAX_CONTEXT_TOKENS,
    OPENAI_MODEL,
    ChunkStrategy,
    PartitionStrategy,
    Priority,
    Provider,
)
from .exceptions import ErrorMsg


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
    min_loc: int | None = Field(default=DEFAULT_MIN_LOC, ge=1)
    max_loc: int = Field(default=DEFAULT_MAX_LOC, gt=0)
    strict_loc_bounds: bool = DEFAULT_STRICT_LOC_BOUNDS
    max_refinement_iterations: int = Field(default=DEFAULT_MAX_REFINEMENT_ITERATIONS, ge=0)
    cp_sat_timeout: float = DEFAULT_CP_SAT_TIMEOUT_SECONDS
    priority: Priority = Priority.ORTHOGONAL
    chunk_strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY
    partition_strategy: PartitionStrategy = DEFAULT_PARTITION_STRATEGY

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
    def auto_derive_min_loc(self):
        if self.min_loc is None and self.max_refinement_iterations > 0:
            # Floor at 1 so a tiny max_loc cannot derive a min_loc that the
            # explicit field constraint (ge=1) would have rejected.
            self.min_loc = max(1, self.max_loc // DEFAULT_MIN_LOC_RATIO)
        return self

    @model_validator(mode="after")
    def validate_loc_bounds(self):
        if self.min_loc is not None and self.min_loc >= self.max_loc:
            raise ValueError(
                ErrorMsg.MIN_LOC_GE_MAX_LOC(min_loc=self.min_loc, max_loc=self.max_loc)
            )
        return self

    @model_validator(mode="after")
    def check_api_key_is_present(self):
        if self.partition_strategy != PartitionStrategy.LLM:
            return self
        match self.provider:
            case Provider.ANTHROPIC:
                if not self.anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY must be set when provider is 'anthropic'")
            case Provider.OPENAI:
                if not self.openai_api_key:
                    raise ValueError("OPENAI_API_KEY must be set when provider is 'openai'")
            case _:
                raise NotImplementedError(
                    f"API key check not implemented for provider '{self.provider}'"
                )
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
