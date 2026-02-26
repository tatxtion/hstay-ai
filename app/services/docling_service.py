"""Docling OCR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.errors import DoclingServiceError


class DoclingAdapter:
    """Convert identity document images to OCR text using Docling + RapidOCR."""

    def __init__(self) -> None:
        self._converter = self._build_converter()

    def _build_converter(self) -> Any:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            try:
                from docling.document_converter import DocumentConverter, ImageFormatOption
            except ImportError:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.pipeline_options import ImageFormatOption  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise DoclingServiceError(f"Unable to initialize Docling: {exc}") from exc

        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = RapidOcrOptions(force_full_page_ocr=True)

            return DocumentConverter(
                format_options={
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
        except Exception as exc:
            raise DoclingServiceError(f"Unable to configure Docling pipeline: {exc}") from exc

    def extract_text(self, image_path: Path) -> str:
        """Run OCR and return plain text."""

        try:
            result = self._converter.convert(source=str(image_path))
            return result.document.export_to_text()
        except Exception as exc:
            raise DoclingServiceError(f"Docling OCR failed: {exc}") from exc
