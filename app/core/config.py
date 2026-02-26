"""Runtime configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    image_directory: Path = Field(default=Path("./img"), alias="IMAGE_DIRECTORY")
    allowed_extensions: tuple[str, ...] = Field(
        default=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"),
        alias="ALLOWED_EXTENSIONS",
    )
    ocr_preview_chars: int = Field(default=240, alias="OCR_PREVIEW_CHARS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
