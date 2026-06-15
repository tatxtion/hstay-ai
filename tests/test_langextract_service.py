from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.errors import LangExtractServiceError
from app.models.schemas import DocumentType
from app.services.langextract_service import LangExtractAdapter


def test_extract_uses_openai_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_extract(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    fake_lx = SimpleNamespace(
        extract=fake_extract,
        factory=SimpleNamespace(ModelConfig=lambda **kwargs: kwargs),
    )
    monkeypatch.setitem(sys.modules, "langextract", fake_lx)

    adapter = LangExtractAdapter(
        Settings(
            OPENAI_API_KEY="openai-test-key",
            OPENAI_MODEL="gpt-4o",
        )
    )

    spans = adapter.extract(ocr_text="sample text", document_type=DocumentType.OTHER)

    assert spans == []
    assert captured["model_id"] == "gpt-4o"
    assert captured["api_key"] == "openai-test-key"
    assert captured["fence_output"] is True
    assert captured["use_schema_constraints"] is False
    assert "config" not in captured


def test_extract_openrouter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_lx = SimpleNamespace(
        extract=lambda **_: [],
        factory=SimpleNamespace(ModelConfig=lambda **kwargs: kwargs),
    )
    monkeypatch.setitem(sys.modules, "langextract", fake_lx)

    adapter = LangExtractAdapter(
        Settings(
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY=None,
            OPENROUTER_MODEL="openai/gpt-4o",
        )
    )

    with pytest.raises(LangExtractServiceError, match="OPENROUTER_API_KEY"):
        adapter.extract(ocr_text="sample text", document_type=DocumentType.OTHER)


def test_extract_openrouter_builds_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeModelConfig:
        def __init__(
            self,
            model_id: str | None = None,
            provider: str | None = None,
            provider_kwargs: dict[str, object] | None = None,
        ) -> None:
            self.model_id = model_id
            self.provider = provider
            self.provider_kwargs = provider_kwargs or {}

    def fake_extract(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    fake_lx = SimpleNamespace(
        extract=fake_extract,
        factory=SimpleNamespace(ModelConfig=FakeModelConfig),
    )
    monkeypatch.setitem(sys.modules, "langextract", fake_lx)

    adapter = LangExtractAdapter(
        Settings(
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="openrouter-test-key",
            OPENROUTER_MODEL="anthropic/claude-sonnet-4.5",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
        )
    )

    spans = adapter.extract(ocr_text="sample text", document_type=DocumentType.OTHER)

    assert spans == []
    assert "model_id" not in captured
    assert "api_key" not in captured
    assert isinstance(captured["config"], FakeModelConfig)
    config = captured["config"]
    assert config.model_id == "anthropic/claude-sonnet-4.5"
    assert config.provider == "openai"
    assert config.provider_kwargs == {
        "api_key": "openrouter-test-key",
        "base_url": "https://openrouter.ai/api/v1",
    }


def test_extract_unsupported_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_lx = SimpleNamespace(
        extract=lambda **_: [],
        factory=SimpleNamespace(ModelConfig=lambda **kwargs: kwargs),
    )
    monkeypatch.setitem(sys.modules, "langextract", fake_lx)

    settings = Settings(OPENAI_API_KEY="openai-test-key")
    object.__setattr__(settings, "llm_provider", "custom")
    adapter = LangExtractAdapter(settings)

    with pytest.raises(LangExtractServiceError, match="Unsupported LLM_PROVIDER 'custom'"):
        adapter.extract(ocr_text="sample text", document_type=DocumentType.OTHER)
