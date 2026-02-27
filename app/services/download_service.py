"""Document download service for URL-based extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import DocumentDownloadError, InvalidDocumentURLError


class DocumentDownloader:
    """Download a remote document to a local temporary file."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 30.0,
        max_download_bytes: int = 20 * 1024 * 1024,
        chunk_size: int = 64 * 1024,
    ) -> None:
        self.settings = settings or get_settings()
        self.http_client = http_client
        self.timeout_seconds = timeout_seconds
        self.max_download_bytes = max_download_bytes
        self.chunk_size = chunk_size

    def download(self, url: str) -> Path:
        self._validate_url(url)
        suffix = self._resolve_suffix(url)

        temp_file = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=suffix)
        temp_path = Path(temp_file.name)

        try:
            with temp_file:
                self._stream_to_file(url, temp_file)
            return temp_path
        except DocumentDownloadError:
            self._safe_unlink(temp_path)
            raise
        except OSError as exc:
            self._safe_unlink(temp_path)
            raise DocumentDownloadError(f"Unable to write downloaded document: {exc}") from exc

    def _stream_to_file(self, url: str, output_file: BinaryIO) -> None:
        if self.http_client is not None:
            self._read_stream(self.http_client, url, output_file)
            return

        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            self._read_stream(client, url, output_file)

    def _read_stream(self, client: httpx.Client, url: str, output_file: BinaryIO) -> None:
        total_bytes = 0
        try:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=self.chunk_size):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > self.max_download_bytes:
                        raise DocumentDownloadError(
                            f"Downloaded file exceeds limit of {self.max_download_bytes} bytes"
                        )
                    output_file.write(chunk)
        except httpx.TimeoutException as exc:
            raise DocumentDownloadError("Document download timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise DocumentDownloadError(
                f"Document download failed with status {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise DocumentDownloadError(f"Document download failed: {exc}") from exc

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise InvalidDocumentURLError("Document URL must use http or https")

        if not parsed.hostname:
            raise InvalidDocumentURLError("Document URL must include a hostname")

    def _resolve_suffix(self, url: str) -> str:
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix in self.settings.allowed_extensions:
            return suffix
        return ".png"

    def _safe_unlink(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
