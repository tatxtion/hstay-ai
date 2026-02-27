"""Extraction orchestration service."""

from __future__ import annotations

import html
import re
import time
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.errors import (
    EmptyOCRTextError,
    InvalidFileExtensionError,
    PathTraversalError,
    SourceFileNotFoundError,
)
from app.models.schemas import (
    AadhaarFields,
    DocumentType,
    ExtractionRequest,
    ExtractionResponse,
    ExtractionSpan,
    FieldEvidence,
    Issue,
    OcrPayload,
    OtherFields,
    PanFields,
    PassportFields,
    TimingsMs,
)
from app.services.docling_service import DoclingAdapter
from app.services.langextract_service import LangExtractAdapter

PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PASSPORT_PATTERN = re.compile(r"\b[A-PR-WYa-pr-wy][1-9]\d{6}\b")
MRZ_PASSPORT_PATTERN = re.compile(r"^P<IND", re.MULTILINE)
PIN_CODE_PATTERN = re.compile(r"\b\d{6}\b")
MRZ_TD3_BLOCK_PATTERN = re.compile(r"(?P<line1>[A-Z0-9<]{44})\s+(?P<line2>[A-Z0-9<]{44})")
MRZ_TD3_LINE2_PATTERN = re.compile(
    r"(?P<line2>[A-Z0-9<]{9}[0-9<][A-Z]{3}[0-9]{6}[0-9<][MF<X][0-9]{6}[0-9<][A-Z0-9<]{14}[0-9<]{2})"
)


class ExtractionService:
    """Orchestrates file validation, OCR, detection, and structured extraction."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        docling_adapter: DoclingAdapter | None = None,
        langextract_adapter: LangExtractAdapter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.docling_adapter = docling_adapter or DoclingAdapter()
        self.langextract_adapter = langextract_adapter or LangExtractAdapter(self.settings)

    def process(self, request: ExtractionRequest) -> ExtractionResponse:
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        image_path = self._validate_and_resolve_path(request.filename)
        validation_ms = _to_ms(t0)

        t0 = time.perf_counter()
        ocr_text = self.docling_adapter.extract_text(image_path).strip()
        if not ocr_text:
            raise EmptyOCRTextError("OCR output is empty for the provided image")
        ocr_ms = _to_ms(t0)

        t0 = time.perf_counter()
        detected_type = self.detect_document_type(ocr_text)
        issues: list[Issue] = []

        target_type = detected_type
        if request.document_type and request.document_type != DocumentType.OTHER:
            if detected_type == DocumentType.OTHER:
                target_type = request.document_type
                issues.append(
                    Issue(
                        code="DETECTION_INCONCLUSIVE",
                        message=(
                            "Document type detection was inconclusive; "
                            f"using requested type {request.document_type.value}."
                        ),
                        severity="warning",
                    )
                )
            elif request.document_type != detected_type:
                issues.append(
                    Issue(
                        code="DOCUMENT_TYPE_MISMATCH",
                        message=(
                            f"Requested type {request.document_type.value} does not match "
                            f"detected type {detected_type.value}; proceeding with detected type."
                        ),
                        severity="warning",
                    )
                )

        detection_ms = _to_ms(t0)

        t0 = time.perf_counter()
        spans = self.langextract_adapter.extract(ocr_text=ocr_text, document_type=target_type)
        fields = self._map_fields(target_type, spans, ocr_text)
        extraction_ms = _to_ms(t0)

        total_ms = _to_ms(t_total_start)

        return ExtractionResponse(
            filename=request.filename,
            document_type_requested=request.document_type,
            document_type_detected=target_type,
            ocr=OcrPayload(
                text=ocr_text if request.include_ocr_text else None,
                text_preview=self._text_preview(ocr_text),
                char_count=len(ocr_text),
            ),
            fields=fields,
            extractions=spans if request.include_extractions else None,
            issues=issues,
            timings_ms=TimingsMs(
                validation=validation_ms,
                ocr=ocr_ms,
                detection=detection_ms,
                extraction=extraction_ms,
                total=total_ms,
            ),
        )

    def detect_document_type(self, ocr_text: str) -> DocumentType:
        if PAN_PATTERN.search(ocr_text):
            return DocumentType.PAN

        if AADHAAR_PATTERN.search(ocr_text):
            return DocumentType.AADHAAR

        passport_score = 0
        lower = ocr_text.lower()
        if PASSPORT_PATTERN.search(ocr_text):
            passport_score += 2
        if MRZ_PASSPORT_PATTERN.search(ocr_text):
            passport_score += 2
        if _extract_mrz_td3_lines(ocr_text)[1] is not None:
            passport_score += 2

        passport_keywords = (
            "passport",
            "republic of india",
            "nationality",
            "date of issue",
            "date of expiry",
            "place of issue",
        )
        passport_score += sum(1 for keyword in passport_keywords if keyword in lower)

        if passport_score >= 2:
            return DocumentType.PASSPORT

        return DocumentType.OTHER

    def _validate_and_resolve_path(self, filename: str) -> Path:
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise PathTraversalError("Filename must be a basename without directories")

        extension = Path(filename).suffix.lower()
        if extension not in self.settings.allowed_extensions:
            allowed = ", ".join(self.settings.allowed_extensions)
            raise InvalidFileExtensionError(
                f"Unsupported extension '{extension}'. Allowed extensions: {allowed}"
            )

        image_root = self.settings.image_directory.resolve()
        candidate = (image_root / filename).resolve()

        if image_root != candidate and image_root not in candidate.parents:
            raise PathTraversalError("Resolved file path escapes configured image directory")

        if not candidate.exists() or not candidate.is_file():
            raise SourceFileNotFoundError(f"Source file not found: {filename}")

        return candidate

    def _map_fields(
        self,
        document_type: DocumentType,
        spans: list[ExtractionSpan],
        ocr_text: str,
    ) -> PanFields | AadhaarFields | PassportFields | OtherFields:
        if document_type == DocumentType.PAN:
            return self._map_pan_fields(spans, ocr_text)

        if document_type == DocumentType.AADHAAR:
            return self._map_aadhaar_fields(spans, ocr_text)

        if document_type == DocumentType.PASSPORT:
            return self._map_passport_fields(spans, ocr_text)

        return self._map_other_fields(spans, ocr_text)

    def _map_pan_fields(self, spans: list[ExtractionSpan], ocr_text: str) -> PanFields:
        pan_number = self._pick_field(spans, ocr_text, ["pan_number", "pan", "id_number", "document_number"])
        if pan_number is None:
            pan_number = self._regex_evidence(PAN_PATTERN, ocr_text)

        return PanFields(
            pan_number=pan_number,
            full_name=self._pick_field(spans, ocr_text, ["full_name", "name", "cardholder_name"]),
            father_name=self._pick_field(spans, ocr_text, ["father_name", "parent_name"]),
            date_of_birth=self._pick_field(spans, ocr_text, ["date_of_birth", "dob", "birth_date"]),
        )

    def _map_aadhaar_fields(self, spans: list[ExtractionSpan], ocr_text: str) -> AadhaarFields:
        aadhaar_number = self._pick_field(spans, ocr_text, ["aadhaar_number", "aadhaar", "uid", "id_number"])
        if aadhaar_number is None:
            aadhaar_number = self._regex_evidence(AADHAAR_PATTERN, ocr_text)

        return AadhaarFields(
            aadhaar_number=aadhaar_number,
            full_name=self._pick_field(spans, ocr_text, ["full_name", "name"]),
            date_of_birth=self._pick_field(spans, ocr_text, ["date_of_birth", "dob", "birth_date"]),
            year_of_birth=self._pick_field(spans, ocr_text, ["year_of_birth", "yob"]),
            gender=self._pick_field(spans, ocr_text, ["gender", "sex"]),
            address=self._pick_field(spans, ocr_text, ["address", "residential_address"]),
            care_of=self._pick_field(spans, ocr_text, ["care_of", "c_o", "co"]),
            pin_code=self._pick_field(spans, ocr_text, ["pin_code", "postal_code"]) or self._regex_evidence(PIN_CODE_PATTERN, ocr_text),
        )

    def _map_passport_fields(self, spans: list[ExtractionSpan], ocr_text: str) -> PassportFields:
        passport_number = self._pick_field(spans, ocr_text, ["passport_number", "passport_no", "id_number"])
        if passport_number is None:
            passport_number = self._regex_evidence(PASSPORT_PATTERN, ocr_text)

        fields = PassportFields(
            passport_number=passport_number,
            surname=self._pick_field(spans, ocr_text, ["surname", "last_name", "family_name"]),
            given_names=self._pick_field(spans, ocr_text, ["given_names", "first_name", "name"]),
            nationality=self._pick_field(spans, ocr_text, ["nationality"]),
            date_of_birth=self._pick_field(spans, ocr_text, ["date_of_birth", "dob", "birth_date"]),
            sex=self._pick_field(spans, ocr_text, ["sex", "gender"]),
            place_of_birth=self._pick_field(spans, ocr_text, ["place_of_birth"]),
            place_of_issue=self._pick_field(spans, ocr_text, ["place_of_issue"]),
            date_of_issue=self._pick_field(spans, ocr_text, ["date_of_issue", "issue_date"]),
            date_of_expiry=self._pick_field(spans, ocr_text, ["date_of_expiry", "expiry_date"]),
            file_number=self._pick_field(spans, ocr_text, ["file_number"]),
            mrz_line_1=self._pick_field(spans, ocr_text, ["mrz_line_1"]),
            mrz_line_2=self._pick_field(spans, ocr_text, ["mrz_line_2"]),
        )

        if (
            fields.sex is None
            or fields.nationality is None
            or fields.mrz_line_1 is None
            or fields.mrz_line_2 is None
        ):
            mrz_line_1, mrz_line_2 = _extract_mrz_td3_lines(ocr_text)
            if mrz_line_1 is not None and fields.mrz_line_1 is None:
                fields.mrz_line_1 = FieldEvidence(
                    value=mrz_line_1,
                    evidence=mrz_line_1,
                    source_extraction_class="mrz_fallback",
                )
            if mrz_line_2 is not None and fields.mrz_line_2 is None:
                fields.mrz_line_2 = FieldEvidence(
                    value=mrz_line_2,
                    evidence=mrz_line_2,
                    source_extraction_class="mrz_fallback",
                )

            if mrz_line_2 is not None:
                if fields.nationality is None:
                    nationality = mrz_line_2[10:13].replace("<", "").strip() or None
                    if nationality is not None:
                        fields.nationality = FieldEvidence(
                            value=nationality,
                            evidence=nationality,
                            source_extraction_class="mrz_fallback",
                        )

                if fields.sex is None:
                    sex = mrz_line_2[20]
                    if sex in {"M", "F", "X"}:
                        fields.sex = FieldEvidence(
                            value=sex,
                            evidence=sex,
                            source_extraction_class="mrz_fallback",
                        )

        return fields

    def _map_other_fields(self, spans: list[ExtractionSpan], ocr_text: str) -> OtherFields:
        id_number = self._pick_field(spans, ocr_text, ["id_number", "document_number", "identifier"])
        if id_number is None:
            id_number = (
                self._regex_evidence(PAN_PATTERN, ocr_text)
                or self._regex_evidence(AADHAAR_PATTERN, ocr_text)
                or self._regex_evidence(PASSPORT_PATTERN, ocr_text)
            )

        return OtherFields(
            id_number=id_number,
            full_name=self._pick_field(spans, ocr_text, ["full_name", "name"]),
            date_of_birth=self._pick_field(spans, ocr_text, ["date_of_birth", "dob", "birth_date"]),
            address=self._pick_field(spans, ocr_text, ["address"]),
        )

    def _pick_field(
        self,
        spans: list[ExtractionSpan],
        ocr_text: str,
        aliases: list[str],
    ) -> FieldEvidence | None:
        normalized_aliases = {_normalize_key(alias) for alias in aliases}
        for span in spans:
            if _normalize_key(span.extraction_class) in normalized_aliases:
                return self._build_evidence(span, ocr_text)
        return None

    def _build_evidence(self, span: ExtractionSpan, ocr_text: str) -> FieldEvidence:
        evidence = span.extraction_text
        if span.start_pos is not None and span.end_pos is not None:
            if 0 <= span.start_pos <= span.end_pos <= len(ocr_text):
                evidence = ocr_text[span.start_pos : span.end_pos]

        return FieldEvidence(
            value=span.extraction_text,
            evidence=evidence,
            start_pos=span.start_pos,
            end_pos=span.end_pos,
            source_extraction_class=span.extraction_class,
        )

    def _regex_evidence(self, pattern: re.Pattern[str], ocr_text: str) -> FieldEvidence | None:
        match = pattern.search(ocr_text)
        if not match:
            return None

        return FieldEvidence(
            value=match.group(0),
            evidence=match.group(0),
            start_pos=match.start(),
            end_pos=match.end(),
            source_extraction_class="regex_fallback",
        )

    def _text_preview(self, text: str) -> str:
        preview_limit = max(self.settings.ocr_preview_chars, 0)
        if len(text) <= preview_limit:
            return text
        return f"{text[:preview_limit]}..."


def _to_ms(start_time: float) -> int:
    return int(round((time.perf_counter() - start_time) * 1000))


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _extract_mrz_td3_lines(ocr_text: str) -> tuple[str | None, str | None]:
    """Return (mrz_line_1, mrz_line_2) for TD3 passports if present in OCR text.

    Docling exports Markdown-ish text and may HTML-escape `<` as `&lt;`, so unescape first.
    """

    normalized = html.unescape(ocr_text).upper()
    for match in MRZ_TD3_BLOCK_PATTERN.finditer(normalized):
        line2 = match.group("line2")
        if MRZ_TD3_LINE2_PATTERN.fullmatch(line2):
            return match.group("line1"), line2

    match = MRZ_TD3_LINE2_PATTERN.search(normalized)
    if match:
        return None, match.group("line2")

    return None, None
