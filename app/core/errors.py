"""Domain errors and HTTP mapping."""

from __future__ import annotations

from fastapi import HTTPException, status


class DomainError(Exception):
    """Base class for domain-level failures."""

    error_code = "DOMAIN_ERROR"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class SourceFileNotFoundError(DomainError):
    error_code = "SOURCE_FILE_NOT_FOUND"
    status_code = status.HTTP_404_NOT_FOUND


class PathTraversalError(DomainError):
    error_code = "PATH_TRAVERSAL"
    status_code = status.HTTP_400_BAD_REQUEST


class InvalidFileExtensionError(DomainError):
    error_code = "INVALID_FILE_EXTENSION"
    status_code = status.HTTP_400_BAD_REQUEST


class EmptyOCRTextError(DomainError):
    error_code = "EMPTY_OCR_TEXT"
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class DoclingServiceError(DomainError):
    error_code = "DOCLING_ERROR"
    status_code = status.HTTP_502_BAD_GATEWAY


class LangExtractServiceError(DomainError):
    error_code = "LANGEXTRACT_ERROR"
    status_code = status.HTTP_502_BAD_GATEWAY


def domain_error_to_http_exception(error: DomainError) -> HTTPException:
    """Map domain exceptions to an HTTPException payload."""

    return HTTPException(
        status_code=error.status_code,
        detail={
            "code": error.error_code,
            "message": str(error),
        },
    )
