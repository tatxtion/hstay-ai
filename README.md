# hstay-ai Document Extraction PoC

FastAPI service for extracting structured data from Indian identity documents (PAN, Aadhaar, Passport) using:

- Docling OCR (RapidOCR) + LangExtract with configurable LLM provider (`openai` or `openrouter`) for `v1` and `v2`
- LlamaParse + LangExtract with configurable LLM provider (`openai` or `openrouter`) for `v3`

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
- `POST /v3/extract`

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

### v3 extraction request (URL, LlamaParse-backed)

`/v3/extract` has the same request shape as `/v2/extract`, but OCR parsing is done by LlamaParse instead of Docling.

```bash
curl -X POST http://localhost:8000/v3/extract \
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

LlamaParse configuration env vars:

- `LLAMA_CLOUD_API_KEY` (required for `/v3/extract`)
- `LLAMA_PARSE_TIER` (default: `agentic`)
- `LLAMA_PARSE_VERSION` (default: `latest`; pin for reproducible production behavior)
- `LLAMA_PARSE_RESULT_TYPE` (default: `markdown`; supported: `markdown`, `text`)

## LLM provider configuration

All extraction routes (`/v1/extract`, `/v2/extract`, `/v3/extract`) use the same LangExtract-backed LLM provider configured via environment variables.

- `LLM_PROVIDER` (default: `openai`; supported: `openai`, `openrouter`)
- `OPENAI_API_KEY` / `OPENAI_MODEL` for `LLM_PROVIDER=openai`
- `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` / `OPENROUTER_BASE_URL` for `LLM_PROVIDER=openrouter`

OpenRouter model slugs should include provider prefixes (for example: `openai/gpt-4o`, `anthropic/claude-sonnet-4.5`, `openrouter/auto`).
For production stability, prefer fixed model slugs over rolling aliases or `openrouter/auto`.
Keep OpenRouter keys in environment variables only, set credit limits where appropriate, and never commit keys to git.

## Error mapping

- `400`: path traversal, invalid extension, or invalid v2/v3 source input
- `404`: source file not found
- `422`: empty OCR text
- `502`: Docling/LlamaParse/LangExtract/download upstream failures (HTTP or GCS)

OCR provider-specific codes:

- `DOCLING_ERROR` (`v1`, `v2`)
- `LLAMAPARSE_ERROR` (`v3`)

## Security guards

- Basename-only filename validation
- Resolved path constrained to `IMAGE_DIRECTORY`
- Extension allowlist for supported image/PDF formats

## Notes on dependency footprint

`docling[rapidocr]` + `langextract[openai]` pull a large transitive dependency graph (including `torch`, `onnxruntime`, and platform-specific acceleration packages). First `uv sync` can take significant time and bandwidth.
This project pins `torch`/`torchvision`/`torchaudio` to the PyTorch CPU wheel index via `tool.uv.sources` to avoid installing CUDA runtime wheels.

With dual parser support, the service now also depends on `llama-cloud` for `/v3/extract`. This shifts OCR compute for v3 to the external LlamaParse API and may reduce local compute at the cost of API latency/network dependency.

## Testing

```bash
uv run pytest
```

## Deployment

```bash
docker build . -t hstay-ai
docker tag hstay-ai gcr.io/hstay-486519/hstay-ai
docker push asia-south1-docker.pkg.dev/hstay-486519/hstay/hstay-ai
```
