from pydantic_settings import BaseSettings

from .constants import CLAUDE_MODEL, DEFAULT_MAX_LOC, Priority


class Settings(BaseSettings):
    model_config = {"env_prefix": "PR_SPLIT_"}

    anthropic_api_key: str = ""
    claude_model: str = CLAUDE_MODEL
    max_loc: int = DEFAULT_MAX_LOC
    priority: Priority = Priority.ORTHOGONAL
