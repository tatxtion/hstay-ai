FROM python:3.12-slim

WORKDIR /app

# System deps for onnxruntime (libgomp1) and opencv (libgl, libglib, libxcb)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install runtime deps from lockfile (CPU-only torch source pinned in pyproject.toml)
RUN uv sync --locked --no-dev --no-install-project

# Copy application code
COPY app/ app/
COPY main.py .

# Create image directory (ephemeral on Cloud Run)
RUN mkdir -p /app/img

EXPOSE 8080

CMD uv run --no-dev uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
