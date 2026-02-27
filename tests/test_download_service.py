from __future__ import annotations

import tempfile
from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import DocumentDownloadError, InvalidDocumentURLError
from app.services.download_service import DocumentDownloader


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        IMAGE_DIRECTORY=tmp_path,
        ALLOWED_EXTENSIONS=(".png", ".jpg"),
    )


def test_rejects_invalid_scheme(tmp_path: Path) -> None:
    downloader = DocumentDownloader(settings=_build_settings(tmp_path))

    with pytest.raises(InvalidDocumentURLError):
        downloader.download("file:///tmp/sample.png")


def test_rejects_missing_hostname(tmp_path: Path) -> None:
    downloader = DocumentDownloader(settings=_build_settings(tmp_path))

    with pytest.raises(InvalidDocumentURLError):
        downloader.download("https:///sample.png")


def test_allows_localhost_and_downloads(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"abc123"))
    with httpx.Client(transport=transport) as client:
        downloader = DocumentDownloader(settings=_build_settings(tmp_path), http_client=client)
        downloaded_path = downloader.download("http://localhost/sample.jpg")

    assert downloaded_path.suffix == ".jpg"
    assert downloaded_path.read_bytes() == b"abc123"
    downloaded_path.unlink(missing_ok=True)


def test_http_error_raises_domain_error(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404, content=b"missing"))
    with httpx.Client(transport=transport) as client:
        downloader = DocumentDownloader(settings=_build_settings(tmp_path), http_client=client)
        with pytest.raises(DocumentDownloadError):
            downloader.download("https://example.com/missing.png")


def test_timeout_raises_domain_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        downloader = DocumentDownloader(settings=_build_settings(tmp_path), http_client=client)
        with pytest.raises(DocumentDownloadError):
            downloader.download("https://example.com/sample.png")


def test_size_limit_enforced_and_temp_file_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_named_temp_file = tempfile.NamedTemporaryFile

    def named_temp_file(*args: object, **kwargs: object):
        kwargs.setdefault("dir", tmp_path)
        return original_named_temp_file(*args, **kwargs)

    monkeypatch.setattr("app.services.download_service.tempfile.NamedTemporaryFile", named_temp_file)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"0123456789"))
    with httpx.Client(transport=transport) as client:
        downloader = DocumentDownloader(
            settings=_build_settings(tmp_path),
            http_client=client,
            max_download_bytes=5,
        )
        with pytest.raises(DocumentDownloadError):
            downloader.download("https://example.com/large.jpg")

    assert list(tmp_path.iterdir()) == []


def test_unsupported_extension_falls_back_to_png(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"abc123"))
    with httpx.Client(transport=transport) as client:
        downloader = DocumentDownloader(settings=_build_settings(tmp_path), http_client=client)
        downloaded_path = downloader.download("https://example.com/sample.bin")

    assert downloaded_path.suffix == ".png"
    downloaded_path.unlink(missing_ok=True)
