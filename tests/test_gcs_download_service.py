from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import GCSDownloadError
from app.services.gcs_download_service import GCSDownloader


class FakeBlob:
    def __init__(
        self,
        *,
        size: int | None = 6,
        content: bytes = b"abc123",
        reload_exc: Exception | None = None,
        download_exc: Exception | None = None,
    ) -> None:
        self.size = size
        self._content = content
        self._reload_exc = reload_exc
        self._download_exc = download_exc

    def reload(self, client: object | None = None) -> None:
        if self._reload_exc is not None:
            raise self._reload_exc

    def download_to_filename(self, filename: str) -> None:
        if self._download_exc is not None:
            raise self._download_exc
        Path(filename).write_bytes(self._content)


class FakeBucket:
    def __init__(self, blob: FakeBlob) -> None:
        self._blob = blob

    def blob(self, key: str) -> FakeBlob:
        return self._blob


class FakeClient:
    def __init__(self, blob: FakeBlob) -> None:
        self._blob = blob

    def bucket(self, bucket_name: str) -> FakeBucket:
        return FakeBucket(self._blob)


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        OPENAI_API_KEY="test-key",
        IMAGE_DIRECTORY=tmp_path,
        ALLOWED_EXTENSIONS=(".png", ".jpg"),
    )


def test_download_success(tmp_path: Path) -> None:
    blob = FakeBlob(size=6, content=b"abc123")
    downloader = GCSDownloader(
        settings=_build_settings(tmp_path),
        gcs_client=FakeClient(blob),
    )

    downloaded_path = downloader.download("hstay_kyc", "uploads/sample.png")

    assert downloaded_path.suffix == ".png"
    assert downloaded_path.read_bytes() == b"abc123"
    downloaded_path.unlink(missing_ok=True)


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    downloader = GCSDownloader(
        settings=_build_settings(tmp_path),
        gcs_client=FakeClient(FakeBlob()),
    )

    with pytest.raises(GCSDownloadError):
        downloader.download("hstay_kyc", "uploads/sample.bin")


def test_rejects_oversized_blob(tmp_path: Path) -> None:
    blob = FakeBlob(size=30)
    downloader = GCSDownloader(
        settings=_build_settings(tmp_path),
        gcs_client=FakeClient(blob),
        max_download_bytes=20,
    )

    with pytest.raises(GCSDownloadError):
        downloader.download("hstay_kyc", "uploads/sample.jpg")


def test_blob_metadata_failure_maps_to_gcs_error(tmp_path: Path) -> None:
    blob = FakeBlob(reload_exc=RuntimeError("not found"))
    downloader = GCSDownloader(
        settings=_build_settings(tmp_path),
        gcs_client=FakeClient(blob),
    )

    with pytest.raises(GCSDownloadError):
        downloader.download("hstay_kyc", "uploads/sample.png")


def test_failed_download_cleans_up_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_named_temp_file = tempfile.NamedTemporaryFile

    def named_temp_file(*args: object, **kwargs: object):
        kwargs.setdefault("dir", tmp_path)
        return original_named_temp_file(*args, **kwargs)

    monkeypatch.setattr("app.services.gcs_download_service.tempfile.NamedTemporaryFile", named_temp_file)

    blob = FakeBlob(download_exc=RuntimeError("download failed"))
    downloader = GCSDownloader(
        settings=_build_settings(tmp_path),
        gcs_client=FakeClient(blob),
    )

    with pytest.raises(GCSDownloadError):
        downloader.download("hstay_kyc", "uploads/sample.png")

    assert list(tmp_path.iterdir()) == []


def test_invalid_base64_credentials_raises_error(tmp_path: Path) -> None:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        IMAGE_DIRECTORY=tmp_path,
        ALLOWED_EXTENSIONS=(".png", ".jpg"),
        GCS_CREDENTIALS="not-valid-base64",
    )
    downloader = GCSDownloader(settings=settings)

    with pytest.raises(GCSDownloadError):
        downloader.download("hstay_kyc", "uploads/sample.png")
