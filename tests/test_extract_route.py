from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes.extract import get_extraction_service
from app.core.errors import (
    DoclingServiceError,
    EmptyOCRTextError,
    InvalidFileExtensionError,
    LangExtractServiceError,
    PathTraversalError,
    SourceFileNotFoundError,
)
from app.models.schemas import (
    DocumentType,
    ExtractionRequest,
    ExtractionResponse,
    OcrPayload,
    OtherFields,
    TimingsMs,
)


class FakeService:
    def __init__(self, response: ExtractionResponse | None = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc

    def process(self, request: ExtractionRequest) -> ExtractionResponse:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _sample_response() -> ExtractionResponse:
    return ExtractionResponse(
        filename="sample.png",
        document_type_requested=None,
        document_type_detected=DocumentType.OTHER,
        ocr=OcrPayload(text="sample text", text_preview="sample text", char_count=11),
        fields=OtherFields(),
        extractions=[],
        issues=[],
        timings_ms=TimingsMs(validation=1, ocr=1, detection=1, extraction=1, total=4),
    )


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_extract_success(client: TestClient) -> None:
    app.dependency_overrides[get_extraction_service] = lambda: FakeService(response=_sample_response())

    response = client.post("/v1/extract", json={"filename": "sample.png"})

    assert response.status_code == 200
    assert response.json()["filename"] == "sample.png"


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (PathTraversalError("bad path"), 400),
        (InvalidFileExtensionError("bad extension"), 400),
        (SourceFileNotFoundError("missing"), 404),
        (EmptyOCRTextError("empty"), 422),
        (LangExtractServiceError("llm failure"), 502),
        (DoclingServiceError("ocr failure"), 502),
    ],
)
def test_extract_error_mapping(client: TestClient, exc: Exception, expected_status: int) -> None:
    app.dependency_overrides[get_extraction_service] = lambda: FakeService(exc=exc)

    response = client.post("/v1/extract", json={"filename": "sample.png"})

    assert response.status_code == expected_status
    detail = response.json()["detail"]
    assert "code" in detail
    assert "message" in detail
