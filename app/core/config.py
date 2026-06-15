"""Runtime configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_provider: Literal["openai", "openrouter"] = Field(default="openai", alias="LLM_PROVIDER")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="openai/gpt-4o", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    llama_cloud_api_key: str | None = Field(default=None, alias="LLAMA_CLOUD_API_KEY")
    llama_parse_tier: str = Field(default="agentic", alias="LLAMA_PARSE_TIER")
    llama_parse_version: str = Field(default="latest", alias="LLAMA_PARSE_VERSION")
    llama_parse_result_type: str = Field(default="markdown", alias="LLAMA_PARSE_RESULT_TYPE")
    image_directory: Path = Field(default=Path("./img"), alias="IMAGE_DIRECTORY")
    gcs_credentials: str | None = Field(default=None, alias="GCS_CREDENTIALS")
    gcs_default_bucket: str | None = Field(default=None, alias="GCS_DEFAULT_BUCKET")
    allowed_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".pdf"),
        alias="ALLOWED_EXTENSIONS",
    )
    ocr_preview_chars: int = Field(default=240, alias="OCR_PREVIEW_CHARS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
