from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.errors import LlamaParseServiceError
from app.services.llamaparse_service import LlamaParseAdapter


class FakeFilesApi:
    def __init__(self, file_id: str = "file-1", exc: Exception | None = None) -> None:
        self.file_id = file_id
        self.exc = exc
        self.calls: list[tuple[Path, str]] = []

    def create(self, *, file: Path, purpose: str) -> SimpleNamespace:
        self.calls.append((file, purpose))
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(id=self.file_id)


class FakeParsingApi:
    def __init__(self, result: object | None = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[dict[str, object]] = []

    def parse(
        self,
        *,
        file_id: str,
        tier: str,
        version: str,
        expand: list[str],
    ) -> object:
        self.calls.append(
            {
                "file_id": file_id,
                "tier": tier,
                "version": version,
                "expand": expand,
            }
        )
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result


class FakeClient:
    def __init__(self, files_api: FakeFilesApi, parsing_api: FakeParsingApi) -> None:
        self.files = files_api
        self.parsing = parsing_api


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "LLAMA_CLOUD_API_KEY": "llx-test",
        "LLAMA_PARSE_TIER": "agentic",
        "LLAMA_PARSE_VERSION": "latest",
        "LLAMA_PARSE_RESULT_TYPE": "markdown",
    }
    base.update(overrides)
    return Settings(**base)


def test_extract_text_success_markdown(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"fake")
    parse_result = SimpleNamespace(
        markdown=SimpleNamespace(
            pages=[SimpleNamespace(markdown="page1"), SimpleNamespace(markdown="page2")]
        )
    )
    files_api = FakeFilesApi()
    parsing_api = FakeParsingApi(result=parse_result)
    adapter = LlamaParseAdapter(settings=_settings(), client=FakeClient(files_api, parsing_api))

    output = adapter.extract_text(file_path)

    assert output == "page1\n\npage2"
    assert files_api.calls == [(file_path, "parse")]
    assert parsing_api.calls[0]["expand"] == ["markdown"]


def test_extract_text_success_text_result_type(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"fake")
    parse_result = SimpleNamespace(
        text=SimpleNamespace(pages=[SimpleNamespace(text="line1"), SimpleNamespace(text="line2")])
    )
    files_api = FakeFilesApi()
    parsing_api = FakeParsingApi(result=parse_result)
    adapter = LlamaParseAdapter(
        settings=_settings(LLAMA_PARSE_RESULT_TYPE="text"),
        client=FakeClient(files_api, parsing_api),
    )

    output = adapter.extract_text(file_path)

    assert output == "line1\n\nline2"
    assert parsing_api.calls[0]["expand"] == ["text"]


def test_extract_text_missing_api_key_raises() -> None:
    adapter = LlamaParseAdapter(settings=Settings())

    with pytest.raises(LlamaParseServiceError, match="LLAMA_CLOUD_API_KEY"):
        adapter.extract_text(Path("dummy.pdf"))


def test_extract_text_upload_failure_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"fake")
    files_api = FakeFilesApi(exc=RuntimeError("upload failed"))
    parsing_api = FakeParsingApi(result=SimpleNamespace())
    adapter = LlamaParseAdapter(settings=_settings(), client=FakeClient(files_api, parsing_api))

    with pytest.raises(LlamaParseServiceError, match="LlamaParse OCR failed"):
        adapter.extract_text(file_path)


def test_extract_text_parse_failure_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"fake")
    files_api = FakeFilesApi()
    parsing_api = FakeParsingApi(exc=RuntimeError("parse failed"))
    adapter = LlamaParseAdapter(settings=_settings(), client=FakeClient(files_api, parsing_api))

    with pytest.raises(LlamaParseServiceError, match="LlamaParse OCR failed"):
        adapter.extract_text(file_path)


def test_extract_text_unsupported_result_type_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"fake")
    files_api = FakeFilesApi()
    parsing_api = FakeParsingApi(result=SimpleNamespace())
    adapter = LlamaParseAdapter(
        settings=_settings(LLAMA_PARSE_RESULT_TYPE="items"),
        client=FakeClient(files_api, parsing_api),
    )

    with pytest.raises(LlamaParseServiceError, match="Unsupported LLAMA_PARSE_RESULT_TYPE"):
        adapter.extract_text(file_path)


def test_extract_text_empty_pages_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"fake")
    parse_result = SimpleNamespace(markdown=SimpleNamespace(pages=[]))
    files_api = FakeFilesApi()
    parsing_api = FakeParsingApi(result=parse_result)
    adapter = LlamaParseAdapter(settings=_settings(), client=FakeClient(files_api, parsing_api))

    with pytest.raises(LlamaParseServiceError, match="missing pages"):
        adapter.extract_text(file_path)
