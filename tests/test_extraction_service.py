from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import (
    EmptyOCRTextError,
    InvalidFileExtensionError,
    LangExtractServiceError,
    PathTraversalError,
    SourceFileNotFoundError,
)
from app.models.schemas import DocumentType, ExtractionRequest, ExtractionSpan
from app.services.extraction_service import ExtractionService


class FakeDoclingAdapter:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self, image_path: Path) -> str:
        return self._text


class FakeLangExtractAdapter:
    def __init__(self, spans: list[ExtractionSpan] | None = None, exc: Exception | None = None) -> None:
        self._spans = spans or []
        self._exc = exc

    def extract(self, *, ocr_text: str, document_type: DocumentType) -> list[ExtractionSpan]:
        if self._exc:
            raise self._exc
        return list(self._spans)


def _build_service(
    tmp_path: Path,
    *,
    ocr_text: str,
    spans: list[ExtractionSpan] | None = None,
    langextract_exc: Exception | None = None,
) -> ExtractionService:
    settings = Settings(
        OPENAI_API_KEY="test-key",
        IMAGE_DIRECTORY=tmp_path,
        ALLOWED_EXTENSIONS=(".png", ".jpg"),
        OCR_PREVIEW_CHARS=32,
    )
    return ExtractionService(
        settings=settings,
        docling_adapter=FakeDoclingAdapter(ocr_text),
        langextract_adapter=FakeLangExtractAdapter(spans=spans, exc=langextract_exc),
    )


def test_detect_document_type_pan() -> None:
    service = _build_service(Path("."), ocr_text="PAN ABCDE1234F")
    assert service.detect_document_type("INCOME TAX DEPARTMENT ABCDE1234F") == DocumentType.PAN


def test_detect_document_type_aadhaar() -> None:
    service = _build_service(Path("."), ocr_text="Aadhaar 1234 5678 9012")
    assert service.detect_document_type("Government of India 1234 5678 9012") == DocumentType.AADHAAR


def test_detect_document_type_passport() -> None:
    service = _build_service(Path("."), ocr_text="passport")
    text = "REPUBLIC OF INDIA\nPassport No: N1234567\nNationality: INDIAN"
    assert service.detect_document_type(text) == DocumentType.PASSPORT


def test_path_traversal_rejected(tmp_path: Path) -> None:
    service = _build_service(tmp_path, ocr_text="ABCDE1234F")
    request = ExtractionRequest(filename="../secret.png")

    with pytest.raises(PathTraversalError):
        service.process(request)


def test_invalid_extension_rejected(tmp_path: Path) -> None:
    service = _build_service(tmp_path, ocr_text="ABCDE1234F")
    request = ExtractionRequest(filename="sample.pdf")

    with pytest.raises(InvalidFileExtensionError):
        service.process(request)


def test_missing_file_returns_not_found(tmp_path: Path) -> None:
    service = _build_service(tmp_path, ocr_text="ABCDE1234F")
    request = ExtractionRequest(filename="missing.png")

    with pytest.raises(SourceFileNotFoundError):
        service.process(request)


def test_empty_ocr_returns_422(tmp_path: Path) -> None:
    (tmp_path / "sample.png").write_bytes(b"fake")
    service = _build_service(tmp_path, ocr_text="   ")
    request = ExtractionRequest(filename="sample.png")

    with pytest.raises(EmptyOCRTextError):
        service.process(request)


def test_langextract_failure_bubbles_as_domain_error(tmp_path: Path) -> None:
    (tmp_path / "sample.png").write_bytes(b"fake")
    service = _build_service(
        tmp_path,
        ocr_text="ABCDE1234F",
        langextract_exc=LangExtractServiceError("LLM downstream failed"),
    )

    with pytest.raises(LangExtractServiceError):
        service.process(ExtractionRequest(filename="sample.png"))


def test_mismatch_adds_issue_and_uses_detected_type(tmp_path: Path) -> None:
    (tmp_path / "sample.png").write_bytes(b"fake")
    service = _build_service(tmp_path, ocr_text="INCOME TAX DEPARTMENT\nPAN: ABCDE1234F")

    response = service.process(
        ExtractionRequest(filename="sample.png", document_type=DocumentType.AADHAAR)
    )

    assert response.document_type_detected == DocumentType.PAN
    assert any(issue.code == "DOCUMENT_TYPE_MISMATCH" for issue in response.issues)


def test_detection_inconclusive_uses_requested_type(tmp_path: Path) -> None:
    (tmp_path / "sample.png").write_bytes(b"fake")
    service = _build_service(tmp_path, ocr_text="generic identity card")

    response = service.process(
        ExtractionRequest(filename="sample.png", document_type=DocumentType.PASSPORT)
    )

    assert response.document_type_detected == DocumentType.PASSPORT
    assert any(issue.code == "DETECTION_INCONCLUSIVE" for issue in response.issues)


def test_include_flags_and_regex_fallback(tmp_path: Path) -> None:
    (tmp_path / "sample.png").write_bytes(b"fake")
    service = _build_service(tmp_path, ocr_text="INCOME TAX DEPARTMENT\nABCDE1234F")

    response = service.process(
        ExtractionRequest(
            filename="sample.png",
            include_ocr_text=False,
            include_extractions=False,
        )
    )

    assert response.ocr.text is None
    assert response.ocr.text_preview
    assert response.extractions is None
    assert response.fields.pan_number is not None
    assert response.fields.pan_number.source_extraction_class == "regex_fallback"


def test_process_from_path_success(tmp_path: Path) -> None:
    image_path = tmp_path / "downloaded.png"
    image_path.write_bytes(b"fake")

    service = _build_service(tmp_path, ocr_text="INCOME TAX DEPARTMENT\nABCDE1234F")

    result = service.process_from_path(
        image_path,
        document_type=None,
        include_ocr_text=True,
        include_extractions=True,
    )

    assert result.document_type_detected == DocumentType.PAN
    assert result.ocr.text is not None
    assert result.extractions is not None
    assert result.timings_ms.validation is None
    assert result.timings_ms.download is None
