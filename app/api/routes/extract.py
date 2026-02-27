"""Extraction routes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import time

from fastapi import APIRouter, Depends

from app import __version__
from app.core.config import Settings, get_settings
from app.core.errors import (
    DomainError,
    GCSDownloadError,
    InvalidDocumentSourceError,
    domain_error_to_http_exception,
)
from app.models.schemas import (
    ExtractionRequest,
    ExtractionRequestV2,
    ExtractionResponse,
    ExtractionResponseV2,
    TimingsMs,
)
from app.services.download_service import DocumentDownloader
from app.services.docling_service import DoclingAdapter
from app.services.extraction_service import ExtractionService
from app.services.gcs_download_service import GCSDownloader
from app.services.langextract_service import LangExtractAdapter

router = APIRouter()


@lru_cache(maxsize=1)
def get_extraction_service() -> ExtractionService:
    settings = get_settings()
    return ExtractionService(
        settings=settings,
        docling_adapter=DoclingAdapter(),
        langextract_adapter=LangExtractAdapter(settings),
    )


@lru_cache(maxsize=1)
def get_document_downloader() -> DocumentDownloader:
    return DocumentDownloader(settings=get_settings())


@lru_cache(maxsize=1)
def get_gcs_downloader() -> GCSDownloader | None:
    settings = get_settings()
    if not settings.gcs_credentials:
        return None
    return GCSDownloader(settings=settings)


@router.post("/v1/extract", response_model=ExtractionResponse)
def extract_document(
    request: ExtractionRequest,
    service: ExtractionService = Depends(get_extraction_service),
) -> ExtractionResponse:
    try:
        return service.process(request)
    except DomainError as exc:
        raise domain_error_to_http_exception(exc) from exc


@router.post("/v2/extract", response_model=ExtractionResponseV2)
def extract_document_v2(
    request: ExtractionRequestV2,
    service: ExtractionService = Depends(get_extraction_service),
    downloader: DocumentDownloader = Depends(get_document_downloader),
    gcs_downloader: GCSDownloader | None = Depends(get_gcs_downloader),
    settings: Settings = Depends(get_settings),
) -> ExtractionResponseV2:
    download_ms: int | None = None
    temp_path: Path | None = None
    resolved_bucket: str | None = None

    try:
        t0 = time.perf_counter()
        if request.object_key:
            resolved_bucket = request.bucket or settings.gcs_default_bucket
            if not resolved_bucket:
                raise InvalidDocumentSourceError(
                    "bucket is required when object_key is provided and GCS_DEFAULT_BUCKET is unset"
                )
            if gcs_downloader is None:
                raise GCSDownloadError("GCS downloader is not configured")
            temp_path = gcs_downloader.download(resolved_bucket, request.object_key)
        else:
            if not request.document_url:
                raise InvalidDocumentSourceError(
                    "document_url is required when object_key is not provided"
                )
            temp_path = downloader.download(request.document_url)
        download_ms = _to_ms(t0)

        result = service.process_from_path(
            temp_path,
            document_type=request.document_type,
            include_ocr_text=request.include_ocr_text,
            include_extractions=request.include_extractions,
        )

        return ExtractionResponseV2(
            document_id=request.document_id,
            organization_id=request.organization_id,
            property_id=request.property_id,
            document_url=request.document_url,
            bucket=resolved_bucket,
            object_key=request.object_key,
            document_type_requested=result.document_type_requested,
            document_type_detected=result.document_type_detected,
            ocr=result.ocr,
            fields=result.fields,
            extractions=result.extractions,
            issues=result.issues,
            timings_ms=TimingsMs(
                validation=None,
                download=download_ms,
                ocr=result.timings_ms.ocr,
                detection=result.timings_ms.detection,
                extraction=result.timings_ms.extraction,
                total=result.timings_ms.total,
            ),
        )
    except DomainError as exc:
        raise domain_error_to_http_exception(exc) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {
        "status": "healthy",
        "version": __version__,
    }


def _to_ms(start_time: float) -> int:
    return int(round((time.perf_counter() - start_time) * 1000))
