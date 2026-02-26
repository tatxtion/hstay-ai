"""Request and response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PAN = "PAN"
    AADHAAR = "AADHAAR"
    PASSPORT = "PASSPORT"
    OTHER = "OTHER"


class FieldEvidence(BaseModel):
    value: str | None = None
    evidence: str | None = None
    start_pos: int | None = None
    end_pos: int | None = None
    source_extraction_class: str | None = None


class PanFields(BaseModel):
    pan_number: FieldEvidence | None = None
    full_name: FieldEvidence | None = None
    father_name: FieldEvidence | None = None
    date_of_birth: FieldEvidence | None = None


class AadhaarFields(BaseModel):
    aadhaar_number: FieldEvidence | None = None
    full_name: FieldEvidence | None = None
    date_of_birth: FieldEvidence | None = None
    year_of_birth: FieldEvidence | None = None
    gender: FieldEvidence | None = None
    address: FieldEvidence | None = None
    care_of: FieldEvidence | None = None
    pin_code: FieldEvidence | None = None


class PassportFields(BaseModel):
    passport_number: FieldEvidence | None = None
    surname: FieldEvidence | None = None
    given_names: FieldEvidence | None = None
    nationality: FieldEvidence | None = None
    date_of_birth: FieldEvidence | None = None
    sex: FieldEvidence | None = None
    place_of_birth: FieldEvidence | None = None
    place_of_issue: FieldEvidence | None = None
    date_of_issue: FieldEvidence | None = None
    date_of_expiry: FieldEvidence | None = None
    file_number: FieldEvidence | None = None
    mrz_line_1: FieldEvidence | None = None
    mrz_line_2: FieldEvidence | None = None


class OtherFields(BaseModel):
    id_number: FieldEvidence | None = None
    full_name: FieldEvidence | None = None
    date_of_birth: FieldEvidence | None = None
    address: FieldEvidence | None = None


class ExtractionSpan(BaseModel):
    extraction_class: str
    extraction_text: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    start_pos: int | None = None
    end_pos: int | None = None
    group_index: int | None = None
    extraction_index: int | None = None


class Issue(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


class OcrPayload(BaseModel):
    text: str | None = None
    text_preview: str
    char_count: int


class TimingsMs(BaseModel):
    validation: int
    ocr: int
    detection: int
    extraction: int
    total: int


class ExtractionRequest(BaseModel):
    filename: str
    document_type: DocumentType | None = None
    include_ocr_text: bool = True
    include_extractions: bool = True


class ExtractionResponse(BaseModel):
    filename: str
    document_type_requested: DocumentType | None = None
    document_type_detected: DocumentType
    ocr: OcrPayload
    fields: PanFields | AadhaarFields | PassportFields | OtherFields
    extractions: list[ExtractionSpan] | None = None
    issues: list[Issue] = Field(default_factory=list)
    timings_ms: TimingsMs
