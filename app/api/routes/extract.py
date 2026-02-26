"""Extraction routes."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends

from app import __version__
from app.core.config import get_settings
from app.core.errors import DomainError, domain_error_to_http_exception
from app.models.schemas import ExtractionRequest, ExtractionResponse
from app.services.docling_service import DoclingAdapter
from app.services.extraction_service import ExtractionService
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


@router.post("/v1/extract", response_model=ExtractionResponse)
def extract_document(
    request: ExtractionRequest,
    service: ExtractionService = Depends(get_extraction_service),
) -> ExtractionResponse:
    try:
        return service.process(request)
    except DomainError as exc:
        raise domain_error_to_http_exception(exc) from exc


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {
        "status": "healthy",
        "version": __version__,
    }
