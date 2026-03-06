"""Centralized configuration using Pydantic Settings.

All environment variables are loaded and validated here.
Other modules import settings from this module instead of using os.getenv().

Usage:
    from src.config import settings

    api_key = settings.openai_api_key
    db_url = settings.postgres_url
"""

from typing import Optional

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # OpenAI API
    # -------------------------------------------------------------------------
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"

    # -------------------------------------------------------------------------
    # PostgreSQL Database
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
    # Chunking Settings
    # -------------------------------------------------------------------------
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 50
    chunk_overlap_tokens: int = 50

    # -------------------------------------------------------------------------
    # Reducto API (PDF parsing)
    # -------------------------------------------------------------------------
    reducto_api_key: Optional[str] = None

    # -------------------------------------------------------------------------
    # Environment & Logging
    # -------------------------------------------------------------------------
    environment: str = "dev"
    log_level: str = "INFO"
    debug: bool = False


settings = Settings()
