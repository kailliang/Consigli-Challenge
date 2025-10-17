"""Application configuration powered by Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Strongly typed application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    project_name: str = Field(default="Annual Report Analyst API")
    version: str = Field(default="0.1.0")

    openai_api_key: str | None = Field(default=None, min_length=1)
    openai_api_base: str | None = None

    embeddings_model: str = Field(default="text-embedding-3-small")
    llm_model: str = Field(default="gpt-5-mini")

    chroma_persist_dir: str = Field(default="./data/chroma")
    structured_db_url: str = Field(default="sqlite+aiosqlite:///./data/metrics.db")

    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    langsmith_project: str = Field(default="consigli-annual-report-analyst")

    memory_max_turns: int = Field(default=10, ge=1, le=20)

    vector_collection_name: str = Field(default="annual-reports")
    retriever_k: int = Field(default=6, ge=1, le=20)

    ingest_bucket_path: str = Field(default="./data/ingest")
    enable_tracing: bool = Field(default=True)


@lru_cache
def get_settings() -> AppSettings:
    """Provide a cached singleton settings instance."""

    return AppSettings()
