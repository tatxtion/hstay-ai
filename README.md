# hstay-ai Document Extraction PoC

FastAPI service for extracting structured data from Indian identity documents (PAN, Aadhaar, Passport) using Docling OCR (RapidOCR) and LangExtract with OpenAI.

## Prerequisites

- Python `3.12.x`
- `uv` package manager

## Setup

```bash
uv sync --locked
cp .env.example .env
```

## Run

```bash
uv run fastapi dev app/main.py
```

Or run with the convenience entrypoint:

```bash
uv run python main.py
```

## Endpoints

- `GET /healthz`
- `POST /v1/extract`
- `POST /v2/extract`

### Health check

```bash
curl http://localhost:8000/healthz
```

### v1 extraction request (filesystem)

Place an input file (image or PDF) inside `./img`, then call:

```bash
curl -X POST http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "sample.png",
    "include_ocr_text": true,
    "include_extractions": true
  }'
```

### v2 extraction request (URL)

`/v2/extract` downloads the document from a URL and returns the same extraction payload plus caller metadata.

```bash
curl -X POST http://localhost:8000/v2/extract \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc1",
    "organization_id": "org1",
    "property_id": "prop1",
    "document_url": "https://example.com/sample.png",
    "include_ocr_text": true,
    "include_extractions": true
  }'
```

### v2 extraction request (GCS)

`/v2/extract` also supports GCS object downloads. Provide `object_key` and optionally `bucket`.
If both `document_url` and `object_key` are provided, `object_key` (GCS) takes precedence.

```bash
curl -X POST http://localhost:8000/v2/extract \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc1",
    "organization_id": "org1",
    "property_id": "prop1",
    "bucket": "hstay_kyc",
    "object_key": "uploads/sample.png",
    "include_ocr_text": true,
    "include_extractions": true
  }'
```

GCS configuration env vars:
- `GCS_CREDENTIALS` (required for GCS mode; base64-encoded service account JSON)
- `GCS_DEFAULT_BUCKET` (optional; used when request omits `bucket`)

## Error mapping

- `400`: path traversal, invalid extension, or invalid v2 source input
- `404`: source file not found
- `422`: empty OCR text
- `502`: Docling/LangExtract/download upstream failures (HTTP or GCS)

## Security guards

- Basename-only filename validation
- Resolved path constrained to `IMAGE_DIRECTORY`
- Extension allowlist for supported image/PDF formats

## Notes on dependency footprint

`docling[rapidocr]` + `langextract[openai]` pull a large transitive dependency graph (including `torch`, `onnxruntime`, and platform-specific acceleration packages). First `uv sync` can take significant time and bandwidth.
This project pins `torch`/`torchvision`/`torchaudio` to the PyTorch CPU wheel index via `tool.uv.sources` to avoid installing CUDA runtime wheels.

## Testing

```bash
uv run pytest
```
