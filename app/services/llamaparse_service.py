"""LlamaParse OCR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import LlamaParseServiceError


class LlamaParseAdapter:
    """Convert identity document images/PDFs to OCR text using LlamaParse."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._settings.llama_cloud_api_key:
            raise LlamaParseServiceError("LLAMA_CLOUD_API_KEY is required for v3 extraction")
        try:
            from llama_cloud import LlamaCloud
        except Exception as exc:  # pragma: no cover - import guard
            raise LlamaParseServiceError(f"Unable to import llama_cloud: {exc}") from exc

        self._client = LlamaCloud(api_key=self._settings.llama_cloud_api_key)
        return self._client

    def extract_text(self, image_path: Path) -> str:
        """Run OCR and return plain text."""

        client = self._get_client()

        try:
            uploaded_file = client.files.create(file=image_path, purpose="parse")
            parse_result = client.parsing.parse(
                file_id=uploaded_file.id,
                tier=self._settings.llama_parse_tier,
                version=self._settings.llama_parse_version,
                expand=[self._settings.llama_parse_result_type],
            )
            return self._extract_text(parse_result)
        except LlamaParseServiceError:
            raise
        except Exception as exc:
            raise LlamaParseServiceError(f"LlamaParse OCR failed: {exc}") from exc

    def _extract_text(self, parse_result: Any) -> str:
        result_type = self._settings.llama_parse_result_type.lower()
        if result_type == "markdown":
            chunks = self._extract_markdown_chunks(parse_result)
        elif result_type == "text":
            chunks = self._extract_text_chunks(parse_result)
        else:
            raise LlamaParseServiceError(
                "Unsupported LLAMA_PARSE_RESULT_TYPE. Use 'markdown' or 'text'."
            )

        text = "\n\n".join(chunk for chunk in chunks if chunk)
        if not text:
            raise LlamaParseServiceError("LlamaParse returned empty OCR output")
        return text

    def _extract_markdown_chunks(self, parse_result: Any) -> list[str]:
        markdown = getattr(parse_result, "markdown", None)
        pages = getattr(markdown, "pages", None)
        if not pages:
            raise LlamaParseServiceError("LlamaParse markdown output is missing pages")

        chunks: list[str] = []
        for page in pages:
            value = getattr(page, "markdown", None)
            if isinstance(value, str):
                chunks.append(value)
        return chunks

    def _extract_text_chunks(self, parse_result: Any) -> list[str]:
        text_payload = getattr(parse_result, "text", None)
        pages = getattr(text_payload, "pages", None)
        if not pages:
            raise LlamaParseServiceError("LlamaParse text output is missing pages")

        chunks: list[str] = []
        for page in pages:
            value = getattr(page, "text", None)
            if isinstance(value, str):
                chunks.append(value)
        return chunks
