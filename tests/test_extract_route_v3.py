from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes.extract import (
    get_document_downloader,
    get_gcs_downloader,
    get_llamaparse_extraction_service,
)
from app.core.config import Settings, get_settings
from app.core.errors import (
    EmptyOCRTextError,
    GCSDownloadError,
    InvalidDocumentSourceError,
    LlamaParseServiceError,
)
from app.main import app
from app.models.schemas import DocumentType, OcrPayload, OtherFields, TimingsMs


@dataclass
class FakeExtractionResult:
    document_type_requested: DocumentType | None
    document_type_detected: DocumentType
    ocr: OcrPayload
    fields: OtherFields
    extractions: list | None
    issues: list
    timings_ms: TimingsMs


class FakeService:
    def __init__(self, result: FakeExtractionResult | None = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc
        self.last_call: dict[str, object] | None = None

    def process_from_path(
        self,
        path: Path,
        *,
        document_type: DocumentType | None,
        include_ocr_text: bool,
        include_extractions: bool,
    ) -> FakeExtractionResult:
        self.last_call = {
            "path": path,
            "document_type": document_type,
            "include_ocr_text": include_ocr_text,
            "include_extractions": include_extractions,
        }
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result


class FakeURLDownloader:
    def __init__(self, path: Path | None = None, exc: Exception | None = None) -> None:
        self.path = path
        self.exc = exc

    def download(self, _: str) -> Path:
        if self.exc is not None:
            raise self.exc
        assert self.path is not None
        return self.path


class FakeGCSDownloader:
    def __init__(self, path: Path | None = None, exc: Exception | None = None) -> None:
        self.path = path
        self.exc = exc

    def download(self, _: str, __: str) -> Path:
        if self.exc is not None:
            raise self.exc
        assert self.path is not None
        return self.path


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _sample_result() -> FakeExtractionResult:
    return FakeExtractionResult(
        document_type_requested=None,
        document_type_detected=DocumentType.OTHER,
        ocr=OcrPayload(text="sample text", text_preview="sample text", char_count=11),
        fields=OtherFields(),
        extractions=[],
        issues=[],
        timings_ms=TimingsMs(validation=None, ocr=1, detection=1, extraction=1, total=3),
    )


def _payload() -> dict[str, object]:
    return {
        "document_id": "doc1",
        "organization_id": "org1",
        "property_id": "prop1",
        "document_url": "https://example.com/sample.png",
    }


def test_extract_v3_success(client: TestClient, tmp_path: Path) -> None:
    downloaded_path = tmp_path / "downloaded.png"
    downloaded_path.write_bytes(b"fake")

    app.dependency_overrides[get_llamaparse_extraction_service] = lambda: FakeService(
        result=_sample_result()
    )
    app.dependency_overrides[get_document_downloader] = lambda: FakeURLDownloader(path=downloaded_path)
    app.dependency_overrides[get_gcs_downloader] = lambda: FakeGCSDownloader(path=tmp_path / "unused.png")

    response = client.post("/v3/extract", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc1"
    assert body["timings_ms"]["download"] is not None
    assert not downloaded_path.exists()


@pytest.mark.parametrize(
    ("service_exc", "expected_status", "expected_code"),
    [
        (LlamaParseServiceError("llamaparse failed"), 502, LlamaParseServiceError.error_code),
        (EmptyOCRTextError("empty"), 422, EmptyOCRTextError.error_code),
    ],
)
def test_extract_v3_error_mapping(
    client: TestClient,
    tmp_path: Path,
    service_exc: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    downloaded_path = tmp_path / "downloaded.png"
    downloaded_path.write_bytes(b"fake")

    app.dependency_overrides[get_llamaparse_extraction_service] = lambda: FakeService(
        result=_sample_result(),
        exc=service_exc,
    )
    app.dependency_overrides[get_document_downloader] = lambda: FakeURLDownloader(path=downloaded_path)
    app.dependency_overrides[get_gcs_downloader] = lambda: FakeGCSDownloader(path=tmp_path / "unused.png")

    response = client.post("/v3/extract", json=_payload())

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert not downloaded_path.exists()


def test_extract_v3_gcs_missing_bucket_returns_400(client: TestClient, tmp_path: Path) -> None:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        IMAGE_DIRECTORY=tmp_path,
        GCS_CREDENTIALS="e30=",
        GCS_DEFAULT_BUCKET=None,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llamaparse_extraction_service] = lambda: FakeService(result=_sample_result())
    app.dependency_overrides[get_document_downloader] = lambda: FakeURLDownloader(path=tmp_path / "unused.png")
    app.dependency_overrides[get_gcs_downloader] = lambda: FakeGCSDownloader(path=tmp_path / "unused2.png")

    response = client.post(
        "/v3/extract",
        json={
            "document_id": "doc1",
            "organization_id": "org1",
            "property_id": "prop1",
            "object_key": "uploads/sample.png",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == InvalidDocumentSourceError.error_code


def test_extract_v3_gcs_unavailable_returns_502(client: TestClient, tmp_path: Path) -> None:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        IMAGE_DIRECTORY=tmp_path,
        GCS_CREDENTIALS="e30=",
        GCS_DEFAULT_BUCKET="default-bucket",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llamaparse_extraction_service] = lambda: FakeService(result=_sample_result())
    app.dependency_overrides[get_document_downloader] = lambda: FakeURLDownloader(path=tmp_path / "unused.png")
    app.dependency_overrides[get_gcs_downloader] = lambda: None

    response = client.post(
        "/v3/extract",
        json={
            "document_id": "doc1",
            "organization_id": "org1",
            "property_id": "prop1",
            "object_key": "uploads/sample.png",
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == GCSDownloadError.error_code
