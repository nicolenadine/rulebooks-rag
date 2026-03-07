"""Centralized configuration using Pydantic Settings.

All environment variables are loaded and validated here.
Other modules import settings from this module instead of using os.getenv().

Usage:
    from src.config import settings

    api_key = settings.openai_api_key
    db_url = settings.postgres_url
"""

from pathlib import Path
from typing import Optional

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to project root (directory containing main.py / pyproject.toml)
# so the key is found regardless of current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # OpenAI API
    # -------------------------------------------------------------------------
    openai_api_key: str

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_openai_api_key(cls, v: object) -> object:
        """Strip whitespace so trailing space in .env doesn't invalidate the key."""
        if isinstance(v, str):
            return v.strip()
        return v

    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"
 
    # -------------------------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rulebook_rag"
    postgres_user: str = "rulebook"
    postgres_password: str = "rulebook"

    @computed_field
    @property
    def postgres_url(self) -> str:
        """Build PostgreSQL connection URL from components."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # -------------------------------------------------------------------------
    # Vector Index Settings
    # -------------------------------------------------------------------------
    embedding_dimensions: int = 1536
    default_top_k: int = 5

    # -------------------------------------------------------------------------
    # Adaptive retrieval (post-filter after initial top-k)
    # -------------------------------------------------------------------------
    retrieval_initial_top_k: int = 5
    retrieval_similarity_floor: float = 0.70
    retrieval_relative_margin: float = 0.08
    retrieval_max_final_chunks: int = 3

    # -------------------------------------------------------------------------
    # Chunking Settings
    # -------------------------------------------------------------------------
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 50
    chunk_overlap_tokens: int = 50
    # Bump this when chunking logic changes so embedding reuse uses a new key.
    chunking_version: str = "1"

    # -------------------------------------------------------------------------
    # Reducto API (PDF parsing)
    # -------------------------------------------------------------------------
    reducto_api_key: Optional[str] = None

    # -------------------------------------------------------------------------
    # Traceability (query/chunk logging for debugging)
    # -------------------------------------------------------------------------
    trace_last_query_path: str = "data/processed/last_qa_trace.json"

    # -------------------------------------------------------------------------
    # Environment & Logging
    # -------------------------------------------------------------------------
    environment: str = "dev"
    log_level: str = "INFO"
    debug: bool = False


settings = Settings()

# One-time diagnostic: show which key prefix and .env path are in use.
# Set RULEBOOK_DEBUG_CONFIG=1 to print this on every run (helps debug "invalid key" when .env is correct).
if __import__("os").environ.get("RULEBOOK_DEBUG_CONFIG", "").strip() in ("1", "true", "yes"):
    _key_preview = (
        (settings.openai_api_key[:12] + "...")
        if settings.openai_api_key and len(settings.openai_api_key) > 12
        else "(empty or short)"
    )
    import sys
    print(f"[config] OpenAI key prefix: {_key_preview} | env_file: {_ENV_FILE}", file=sys.stderr)
