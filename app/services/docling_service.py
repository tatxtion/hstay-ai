"""Docling OCR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.errors import DoclingServiceError


class DoclingAdapter:
    """Convert identity document images/PDFs to OCR text using Docling + RapidOCR."""

    def __init__(self) -> None:
        self._converter = self._build_converter()

    def _build_converter(self) -> Any:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            PdfFormatOption = None
            try:
                from docling.document_converter import (
                    DocumentConverter,
                    ImageFormatOption,
                    PdfFormatOption,
                )
            except ImportError:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.pipeline_options import ImageFormatOption  # type: ignore
                try:
                    from docling.datamodel.pipeline_options import PdfFormatOption  # type: ignore
                except ImportError:
                    PdfFormatOption = None
        except Exception as exc:  # pragma: no cover - import guard
            raise DoclingServiceError(f"Unable to initialize Docling: {exc}") from exc

        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            # ID extraction only needs OCR text; table reconstruction is expensive and unnecessary.
            pipeline_options.do_table_structure = False
            pipeline_options.ocr_options = RapidOcrOptions(
                backend="onnxruntime",
                lang=["english"],
                force_full_page_ocr=True,
            )

            format_options: dict[Any, Any] = {
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
            if PdfFormatOption is not None:
                format_options[InputFormat.PDF] = PdfFormatOption(
                    pipeline_options=pipeline_options,
                )

            return DocumentConverter(
                format_options=format_options
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
