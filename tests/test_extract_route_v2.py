from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes.extract import (
    get_document_downloader,
    get_extraction_service,
    get_gcs_downloader,
)
from app.core.config import Settings, get_settings
from app.core.errors import (
    DocumentDownloadError,
    EmptyOCRTextError,
    GCSDownloadError,
    InvalidDocumentSourceError,
    InvalidDocumentURLError,
    LangExtractServiceError,
)
from app.main import app
from app.models.schemas import (
    DocumentType,
    OcrPayload,
    OtherFields,
    TimingsMs,
)


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
        self.calls: list[str] = []

    def download(self, url: str) -> Path:
        self.calls.append(url)
        if self.exc is not None:
            raise self.exc
        assert self.path is not None
        return self.path


class FakeGCSDownloader:
    def __init__(self, path: Path | None = None, exc: Exception | None = None) -> None:
        self.path = path
        self.exc = exc
        self.calls: list[tuple[str, str]] = []

    def download(self, bucket: str, object_key: str) -> Path:
        self.calls.append((bucket, object_key))
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


def test_extract_v2_success(client: TestClient, tmp_path: Path) -> None:
    downloaded_path = tmp_path / "downloaded.png"
    downloaded_path.write_bytes(b"fake")

    service = FakeService(result=_sample_result())
    downloader = FakeURLDownloader(path=downloaded_path)
    gcs_downloader = FakeGCSDownloader(path=tmp_path / "unused.png")
    app.dependency_overrides[get_extraction_service] = lambda: service
    app.dependency_overrides[get_document_downloader] = lambda: downloader
    app.dependency_overrides[get_gcs_downloader] = lambda: gcs_downloader

    payload = {
        "document_id": "doc1",
        "organization_id": "org1",
        "property_id": "prop1",
        "document_url": "https://example.com/sample.png",
        "include_ocr_text": True,
        "include_extractions": True,
    }
    response = client.post("/v2/extract", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc1"
    assert body["organization_id"] == "org1"
    assert body["property_id"] == "prop1"
    assert body["document_url"] == "https://example.com/sample.png"
    assert body["bucket"] is None
    assert body["object_key"] is None
    assert body["timings_ms"]["download"] is not None
    assert body["timings_ms"]["validation"] is None
    assert service.last_call is not None
    assert service.last_call["path"] == downloaded_path
    assert downloader.calls == ["https://example.com/sample.png"]
    assert gcs_downloader.calls == []
    assert not downloaded_path.exists()


def test_extract_v2_gcs_success(client: TestClient, tmp_path: Path) -> None:
    downloaded_path = tmp_path / "downloaded.png"
    downloaded_path.write_bytes(b"fake")

    service = FakeService(result=_sample_result())
    downloader = FakeURLDownloader(path=tmp_path / "unused.png")
    gcs_downloader = FakeGCSDownloader(path=downloaded_path)
    app.dependency_overrides[get_extraction_service] = lambda: service
    app.dependency_overrides[get_document_downloader] = lambda: downloader
    app.dependency_overrides[get_gcs_downloader] = lambda: gcs_downloader

    response = client.post(
        "/v2/extract",
        json={
            "document_id": "doc1",
            "organization_id": "org1",
            "property_id": "prop1",
            "bucket": "hstay_kyc",
            "object_key": "uploads/sample.png",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_url"] is None
    assert body["bucket"] == "hstay_kyc"
    assert body["object_key"] == "uploads/sample.png"
    assert gcs_downloader.calls == [("hstay_kyc", "uploads/sample.png")]
    assert downloader.calls == []
    assert not downloaded_path.exists()


def test_extract_v2_gcs_default_bucket_success(client: TestClient, tmp_path: Path) -> None:
    downloaded_path = tmp_path / "downloaded.png"
    downloaded_path.write_bytes(b"fake")
    settings = Settings(
        OPENAI_API_KEY="test-key",
        IMAGE_DIRECTORY=tmp_path,
        GCS_CREDENTIALS="e30=",
        GCS_DEFAULT_BUCKET="default-bucket",
    )

    service = FakeService(result=_sample_result())
    downloader = FakeURLDownloader(path=tmp_path / "unused.png")
    gcs_downloader = FakeGCSDownloader(path=downloaded_path)
    app.dependency_overrides[get_extraction_service] = lambda: service
    app.dependency_overrides[get_document_downloader] = lambda: downloader
    app.dependency_overrides[get_gcs_downloader] = lambda: gcs_downloader
    app.dependency_overrides[get_settings] = lambda: settings

    response = client.post(
        "/v2/extract",
        json={
            "document_id": "doc1",
            "organization_id": "org1",
            "property_id": "prop1",
            "object_key": "uploads/sample.png",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["bucket"] == "default-bucket"
    assert body["object_key"] == "uploads/sample.png"
    assert gcs_downloader.calls == [("default-bucket", "uploads/sample.png")]
    assert not downloaded_path.exists()


def test_extract_v2_source_required_validation(client: TestClient) -> None:
    response = client.post(
        "/v2/extract",
        json={
            "document_id": "doc1",
            "organization_id": "org1",
            "property_id": "prop1",
        },
    )

    assert response.status_code == 422


def test_extract_v2_missing_bucket_returns_400(client: TestClient, tmp_path: Path) -> None:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        IMAGE_DIRECTORY=tmp_path,
        GCS_CREDENTIALS="e30=",
        GCS_DEFAULT_BUCKET=None,
    )
    service = FakeService(result=_sample_result())
    downloader = FakeURLDownloader(path=tmp_path / "unused.png")
    gcs_downloader = FakeGCSDownloader(path=tmp_path / "unused2.png")
    app.dependency_overrides[get_extraction_service] = lambda: service
    app.dependency_overrides[get_document_downloader] = lambda: downloader
    app.dependency_overrides[get_gcs_downloader] = lambda: gcs_downloader
    app.dependency_overrides[get_settings] = lambda: settings

    response = client.post(
        "/v2/extract",
        json={
            "document_id": "doc1",
            "organization_id": "org1",
            "property_id": "prop1",
            "object_key": "uploads/sample.png",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == InvalidDocumentSourceError.error_code


def test_extract_v2_gcs_unavailable_returns_502(client: TestClient, tmp_path: Path) -> None:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        IMAGE_DIRECTORY=tmp_path,
        GCS_CREDENTIALS="e30=",
        GCS_DEFAULT_BUCKET="default-bucket",
    )
    service = FakeService(result=_sample_result())
    downloader = FakeURLDownloader(path=tmp_path / "unused.png")
    app.dependency_overrides[get_extraction_service] = lambda: service
    app.dependency_overrides[get_document_downloader] = lambda: downloader
    app.dependency_overrides[get_gcs_downloader] = lambda: None
    app.dependency_overrides[get_settings] = lambda: settings

    response = client.post(
        "/v2/extract",
        json={
            "document_id": "doc1",
            "organization_id": "org1",
            "property_id": "prop1",
            "object_key": "uploads/sample.png",
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == GCSDownloadError.error_code


def test_extract_v2_object_key_precedence_over_url(client: TestClient, tmp_path: Path) -> None:
    downloaded_path = tmp_path / "downloaded.png"
    downloaded_path.write_bytes(b"fake")
    settings = Settings(
        OPENAI_API_KEY="test-key",
        IMAGE_DIRECTORY=tmp_path,
        GCS_CREDENTIALS="e30=",
        GCS_DEFAULT_BUCKET="default-bucket",
    )

    service = FakeService(result=_sample_result())
    downloader = FakeURLDownloader(path=tmp_path / "unused.png")
    gcs_downloader = FakeGCSDownloader(path=downloaded_path)
    app.dependency_overrides[get_extraction_service] = lambda: service
    app.dependency_overrides[get_document_downloader] = lambda: downloader
    app.dependency_overrides[get_gcs_downloader] = lambda: gcs_downloader
    app.dependency_overrides[get_settings] = lambda: settings

    response = client.post(
        "/v2/extract",
        json={
            "document_id": "doc1",
            "organization_id": "org1",
            "property_id": "prop1",
            "document_url": "https://example.com/sample.png",
            "object_key": "uploads/sample.png",
        },
    )

    assert response.status_code == 200
    assert gcs_downloader.calls == [("default-bucket", "uploads/sample.png")]
    assert downloader.calls == []
    assert not downloaded_path.exists()


def test_extract_v2_accepts_object_key_alias(client: TestClient, tmp_path: Path) -> None:
    downloaded_path = tmp_path / "downloaded.png"
    downloaded_path.write_bytes(b"fake")

    service = FakeService(result=_sample_result())
    downloader = FakeURLDownloader(path=tmp_path / "unused.png")
    gcs_downloader = FakeGCSDownloader(path=downloaded_path)
    app.dependency_overrides[get_extraction_service] = lambda: service
    app.dependency_overrides[get_document_downloader] = lambda: downloader
    app.dependency_overrides[get_gcs_downloader] = lambda: gcs_downloader

    response = client.post(
        "/v2/extract",
        json={
            "document_id": "doc1",
            "organization_id": "org1",
            "property_id": "prop1",
            "bucket": "hstay_kyc",
            "objectKey": "uploads/sample.png",
        },
    )

    assert response.status_code == 200
    assert response.json()["object_key"] == "uploads/sample.png"
    assert gcs_downloader.calls == [("hstay_kyc", "uploads/sample.png")]
    assert not downloaded_path.exists()


@pytest.mark.parametrize(
    ("url_downloader_exc", "gcs_downloader_exc", "service_exc", "expected_status", "payload"),
    [
        (
            InvalidDocumentURLError("bad url"),
            None,
            None,
            400,
            {
                "document_id": "doc1",
                "organization_id": "org1",
                "property_id": "prop1",
                "document_url": "https://example.com/sample.png",
            },
        ),
        (
            DocumentDownloadError("network failure"),
            None,
            None,
            502,
            {
                "document_id": "doc1",
                "organization_id": "org1",
                "property_id": "prop1",
                "document_url": "https://example.com/sample.png",
            },
        ),
        (
            None,
            GCSDownloadError("gcs network failure"),
            None,
            502,
            {
                "document_id": "doc1",
                "organization_id": "org1",
                "property_id": "prop1",
                "bucket": "hstay_kyc",
                "object_key": "uploads/sample.png",
            },
        ),
        (
            None,
            None,
            EmptyOCRTextError("empty"),
            422,
            {
                "document_id": "doc1",
                "organization_id": "org1",
                "property_id": "prop1",
                "document_url": "https://example.com/sample.png",
            },
        ),
        (
            None,
            None,
            LangExtractServiceError("llm failure"),
            502,
            {
                "document_id": "doc1",
                "organization_id": "org1",
                "property_id": "prop1",
                "document_url": "https://example.com/sample.png",
            },
        ),
        (
            None,
            None,
            LangExtractServiceError("llm failure"),
            502,
            {
                "document_id": "doc1",
                "organization_id": "org1",
                "property_id": "prop1",
                "bucket": "hstay_kyc",
                "object_key": "uploads/sample.png",
            },
        ),
    ],
)
def test_extract_v2_error_mapping(
    client: TestClient,
    tmp_path: Path,
    url_downloader_exc: Exception | None,
    gcs_downloader_exc: Exception | None,
    service_exc: Exception | None,
    expected_status: int,
    payload: dict[str, str],
) -> None:
    downloaded_path: Path | None = None
    if url_downloader_exc is None and gcs_downloader_exc is None:
        downloaded_path = tmp_path / "downloaded.png"
        downloaded_path.write_bytes(b"fake")

    service = FakeService(result=_sample_result(), exc=service_exc)
    url_path = downloaded_path if payload.get("document_url") else tmp_path / "unused.png"
    gcs_path = downloaded_path if payload.get("object_key") else tmp_path / "unused2.png"
    downloader = FakeURLDownloader(path=url_path, exc=url_downloader_exc)
    gcs_downloader = FakeGCSDownloader(path=gcs_path, exc=gcs_downloader_exc)
    app.dependency_overrides[get_extraction_service] = lambda: service
    app.dependency_overrides[get_document_downloader] = lambda: downloader
    app.dependency_overrides[get_gcs_downloader] = lambda: gcs_downloader

    response = client.post("/v2/extract", json=payload)

    assert response.status_code == expected_status
    detail = response.json()["detail"]
    assert "code" in detail
    assert "message" in detail
    if downloaded_path is not None:
        assert not downloaded_path.exists()
