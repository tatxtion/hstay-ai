"""Request and response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


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
    validation: int | None = None
    download: int | None = None
    ocr: int
    detection: int
    extraction: int
    total: int


class ExtractionRequest(BaseModel):
    filename: str
    document_type: DocumentType | None = None
    include_ocr_text: bool = True
    include_extractions: bool = True


class ExtractionRequestV2(BaseModel):
    document_id: str
    organization_id: str
    property_id: str
    document_url: str | None = None
    bucket: str | None = None
    object_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("object_key", "objectKey"),
    )
    document_type: DocumentType | None = None
    include_ocr_text: bool = True
    include_extractions: bool = True

    @model_validator(mode="after")
    def validate_document_source(self) -> "ExtractionRequestV2":
        self.document_url = (self.document_url or "").strip() or None
        self.bucket = (self.bucket or "").strip() or None
        self.object_key = (self.object_key or "").strip() or None

        has_url = self.document_url is not None
        has_object_key = self.object_key is not None
        if not has_url and not has_object_key:
            raise ValueError("Either document_url or object_key must be provided")
        return self


class ExtractionResponse(BaseModel):
    filename: str
    document_type_requested: DocumentType | None = None
    document_type_detected: DocumentType
    ocr: OcrPayload
    fields: PanFields | AadhaarFields | PassportFields | OtherFields
    extractions: list[ExtractionSpan] | None = None
    issues: list[Issue] = Field(default_factory=list)
    timings_ms: TimingsMs


class ExtractionResponseV2(BaseModel):
    document_id: str
    organization_id: str
    property_id: str
    document_url: str | None = None
    bucket: str | None = None
    object_key: str | None = None
    document_type_requested: DocumentType | None = None
    document_type_detected: DocumentType
    ocr: OcrPayload
    fields: PanFields | AadhaarFields | PassportFields | OtherFields
    extractions: list[ExtractionSpan] | None = None
    issues: list[Issue] = Field(default_factory=list)
    timings_ms: TimingsMs
