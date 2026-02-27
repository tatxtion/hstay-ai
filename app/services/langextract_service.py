"""LangExtract adapter."""

from __future__ import annotations

from typing import Any, Iterable

from app.core.config import Settings
from app.core.errors import LangExtractServiceError
from app.models.schemas import DocumentType, ExtractionSpan


PROMPT_DESCRIPTION = """Extract structured fields from OCR text of identity documents (PAN, Aadhaar, Passport, ID Card, Voter ID).
Return grounded extractions for identifiers, names, dates, nationality, sex/gender, and address fields.
For passports, MRZ (machine readable zone) lines may be present; you may extract them as `mrz_line_1`/`mrz_line_2`.
Use the smallest exact text span possible from the source OCR text."""


class LangExtractAdapter:
    """Extract grounded entities from OCR text using LangExtract + OpenAI."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def extract(self, *, ocr_text: str, document_type: DocumentType) -> list[ExtractionSpan]:
        if not self._settings.openai_api_key:
            raise LangExtractServiceError("OPENAI_API_KEY is required for extraction requests")

        try:
            import langextract as lx
        except Exception as exc:  # pragma: no cover - import guard
            raise LangExtractServiceError(f"Unable to import langextract: {exc}") from exc

        examples = self._build_examples(lx, document_type)

        try:
            result = lx.extract(
                text_or_documents=ocr_text,
                prompt_description=PROMPT_DESCRIPTION,
                examples=examples,
                model_id=self._settings.openai_model,
                api_key=self._settings.openai_api_key,
                fence_output=True,
                use_schema_constraints=False,
            )
        except Exception as exc:
            raise LangExtractServiceError(f"LangExtract call failed: {exc}") from exc

        return self._normalize_output(result)

    def _build_examples(self, lx: Any, document_type: DocumentType) -> list[Any]:
        """Build few-shot examples if the installed LangExtract version exposes typed helpers."""

        data_mod = getattr(lx, "data", None)
        example_data_cls = getattr(data_mod, "ExampleData", None)
        extraction_cls = getattr(data_mod, "Extraction", None)

        if example_data_cls is None or extraction_cls is None:
            return []

        payloads = _example_payloads(document_type)
        built_examples: list[Any] = []

        for payload in payloads:
            extraction_objects: list[Any] = []
            for item in payload["extractions"]:
                kwargs: dict[str, Any] = {
                    "extraction_class": item["extraction_class"],
                    "extraction_text": item["extraction_text"],
                }

                char_interval_cls = getattr(data_mod, "CharInterval", None)
                if char_interval_cls is not None:
                    try:
                        kwargs["char_interval"] = char_interval_cls(
                            start_pos=item["start_pos"],
                            end_pos=item["end_pos"],
                        )
                    except Exception:
                        pass

                try:
                    extraction_objects.append(extraction_cls(**kwargs))
                except Exception:
                    continue

            if not extraction_objects:
                continue

            try:
                built_examples.append(
                    example_data_cls(
                        text=payload["text"],
                        extractions=extraction_objects,
                    )
                )
            except Exception:
                continue

        return built_examples

    def _normalize_output(self, result: Any) -> list[ExtractionSpan]:
        spans: list[ExtractionSpan] = []
        extractions = list(self._iter_extractions(result))

        for idx, extraction in enumerate(extractions):
            extraction_class = _read_attr(extraction, "extraction_class") or "unknown"
            extraction_text = _read_attr(extraction, "extraction_text") or _read_attr(extraction, "text") or ""
            attributes = _read_attr(extraction, "attributes")
            if not isinstance(attributes, dict):
                attributes = {}

            interval = _read_attr(extraction, "char_interval")
            start_pos, end_pos = _coerce_interval(interval)

            spans.append(
                ExtractionSpan(
                    extraction_class=str(extraction_class),
                    extraction_text=str(extraction_text),
                    attributes=attributes,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    group_index=_coerce_int(_read_attr(extraction, "group_index")),
                    extraction_index=_coerce_int(_read_attr(extraction, "extraction_index"))
                    or idx,
                )
            )

        return spans

    def _iter_extractions(self, result: Any) -> Iterable[Any]:
        if result is None:
            return []

        if _looks_like_extraction(result):
            return [result]

        if isinstance(result, list):
            if result and _looks_like_extraction(result[0]):
                return result

            collected: list[Any] = []
            for item in result:
                collected.extend(_extract_extractions_from_document(item))
            return collected

        return _extract_extractions_from_document(result)


def _example_payloads(document_type: DocumentType) -> list[dict[str, Any]]:
    base_examples: dict[DocumentType, list[dict[str, Any]]] = {
        DocumentType.PAN: [
            {
                "text": "INCOME TAX DEPARTMENT\nName: RAVI KUMAR\nFather Name: MAHESH KUMAR\nDOB: 12/07/1989\nPAN: ABCDE1234F",
                "extractions": [
                    {
                        "extraction_class": "full_name",
                        "extraction_text": "RAVI KUMAR",
                        "start_pos": 29,
                        "end_pos": 39,
                    },
                    {
                        "extraction_class": "father_name",
                        "extraction_text": "MAHESH KUMAR",
                        "start_pos": 53,
                        "end_pos": 65,
                    },
                    {
                        "extraction_class": "date_of_birth",
                        "extraction_text": "12/07/1989",
                        "start_pos": 71,
                        "end_pos": 81,
                    },
                    {
                        "extraction_class": "pan_number",
                        "extraction_text": "ABCDE1234F",
                        "start_pos": 87,
                        "end_pos": 97,
                    },
                ],
            }
        ],
        DocumentType.AADHAAR: [
            {
                "text": "Government of India\nName: SITA DEVI\nDOB: 02/11/1994\nFemale\n1234 5678 9012",
                "extractions": [
                    {
                        "extraction_class": "full_name",
                        "extraction_text": "SITA DEVI",
                        "start_pos": 28,
                        "end_pos": 37,
                    },
                    {
                        "extraction_class": "date_of_birth",
                        "extraction_text": "02/11/1994",
                        "start_pos": 43,
                        "end_pos": 53,
                    },
                    {
                        "extraction_class": "gender",
                        "extraction_text": "Female",
                        "start_pos": 54,
                        "end_pos": 60,
                    },
                    {
                        "extraction_class": "aadhaar_number",
                        "extraction_text": "1234 5678 9012",
                        "start_pos": 61,
                        "end_pos": 75,
                    },
                ],
            }
        ],
        DocumentType.PASSPORT: [
            {
                "text": (
                    "REPUBLIC OF INDIA\n"
                    "Passport No: N1234567\n"
                    "Surname: SHARMA\n"
                    "Given Names: AMIT\n"
                    "Nationality: INDIAN\n"
                    "Sex: M\n"
                    "Date of Birth: 10/01/1990"
                ),
                "extractions": [
                    {
                        "extraction_class": "passport_number",
                        "extraction_text": "N1234567",
                        "start_pos": 31,
                        "end_pos": 39,
                    },
                    {
                        "extraction_class": "surname",
                        "extraction_text": "SHARMA",
                        "start_pos": 49,
                        "end_pos": 55,
                    },
                    {
                        "extraction_class": "given_names",
                        "extraction_text": "AMIT",
                        "start_pos": 69,
                        "end_pos": 73,
                    },
                    {
                        "extraction_class": "nationality",
                        "extraction_text": "INDIAN",
                        "start_pos": 87,
                        "end_pos": 93,
                    },
                    {
                        "extraction_class": "sex",
                        "extraction_text": "M",
                        "start_pos": 99,
                        "end_pos": 100,
                    },
                    {
                        "extraction_class": "date_of_birth",
                        "extraction_text": "10/01/1990",
                        "start_pos": 116,
                        "end_pos": 126,
                    },
                ],
            }
        ],
        DocumentType.OTHER: [
            {
                "text": "ID CARD\nName: SAMPLE USER\nID: XYZ12345",
                "extractions": [
                    {
                        "extraction_class": "full_name",
                        "extraction_text": "SAMPLE USER",
                        "start_pos": 14,
                        "end_pos": 25,
                    },
                    {
                        "extraction_class": "id_number",
                        "extraction_text": "XYZ12345",
                        "start_pos": 31,
                        "end_pos": 39,
                    },
                ],
            }
        ],
    }

    return base_examples.get(document_type, base_examples[DocumentType.OTHER])


def _extract_extractions_from_document(item: Any) -> list[Any]:
    if item is None:
        return []

    if isinstance(item, dict):
        values = item.get("extractions")
        return values if isinstance(values, list) else []

    values = getattr(item, "extractions", None)
    return values if isinstance(values, list) else []


def _looks_like_extraction(item: Any) -> bool:
    if isinstance(item, dict):
        return "extraction_class" in item or "extraction_text" in item
    return hasattr(item, "extraction_class") or hasattr(item, "extraction_text")


def _read_attr(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _coerce_interval(interval: Any) -> tuple[int | None, int | None]:
    if interval is None:
        return None, None

    if isinstance(interval, dict):
        return _coerce_int(interval.get("start_pos")), _coerce_int(interval.get("end_pos"))

    start = getattr(interval, "start_pos", None)
    end = getattr(interval, "end_pos", None)
    return _coerce_int(start), _coerce_int(end)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None
