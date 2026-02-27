"""Convenience runner for local development."""

from __future__ import annotations

import os

# Force CPU execution by hiding CUDA devices from PyTorch and related libs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import uvicorn


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
