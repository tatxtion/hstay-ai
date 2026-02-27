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

### Health check

```bash
curl http://localhost:8000/healthz
```

### Extraction request

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

## Error mapping

- `400`: path traversal or invalid extension
- `404`: source file not found
- `422`: empty OCR text
- `502`: Docling or LangExtract upstream failures

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
