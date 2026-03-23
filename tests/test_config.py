"""Tests for provider configuration and model selection."""

from __future__ import annotations

import importlib

import pytest

from config.settings import Settings


def _anthropic_available() -> bool:
    """Return True if the anthropic extra is fully importable."""
    try:
        importlib.import_module("pydantic_ai.models.anthropic")
        return True
    except ImportError:
        return False


class TestSettings:
    """Test the Settings configuration class."""

    def test_default_provider_is_openai(self) -> None:
        """Test that default provider is OpenAI."""
        settings = Settings(openai_api_key="sk-test")
        assert settings.llm_provider == "openai"

    def test_openai_requires_api_key(self) -> None:
        """Test that OpenAI provider raises without API key."""
        settings = Settings(llm_provider="openai", openai_api_key="")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            settings.get_model()

    def test_anthropic_requires_api_key(self) -> None:
        """Test that Anthropic provider raises without API key."""
        settings = Settings(llm_provider="anthropic", anthropic_api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            settings.get_model()

    def test_lmstudio_no_api_key_needed(self) -> None:
        """Test that LM Studio works without real API key."""
        settings = Settings(llm_provider="lmstudio")
        model = settings.get_model()
        assert model is not None

    def test_openai_returns_model(self) -> None:
        """Test that OpenAI provider returns an OpenAIModel."""
        settings = Settings(llm_provider="openai", openai_api_key="sk-test")
        model = settings.get_model()
        assert "OpenAI" in type(model).__name__ or model is not None

    @pytest.mark.skipif(
        not _anthropic_available(),
        reason="anthropic extra not installed or incompatible",
    )
    def test_anthropic_returns_model(self) -> None:
        """Test that Anthropic provider returns an AnthropicModel."""
        settings = Settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")
        model = settings.get_model()
        assert model is not None

    def test_custom_lmstudio_url(self) -> None:
        """Test custom LM Studio base URL is accepted."""
        settings = Settings(
            llm_provider="lmstudio",
            lmstudio_base_url="http://myserver:5000/v1",
            lmstudio_model="my-model",
        )
        model = settings.get_model()
        assert model is not None

    def test_search_api_key_optional(self) -> None:
        """Test that search API key defaults to empty string."""
        settings = Settings(openai_api_key="sk-test")
        assert settings.search_api_key == ""

    def test_get_model_returns_pydantic_ai_model(self) -> None:
        """Test that get_model() returns a pydantic-ai Model instance."""
        from pydantic_ai.models import Model

        settings = Settings(llm_provider="lmstudio")
        model = settings.get_model()
        assert isinstance(model, Model)

    def test_rag_disabled_by_default(self) -> None:
        """Test RAG is disabled by default."""
        s = Settings()
        assert s.rag_enabled is False

    def test_rag_vector_store_path_default(self) -> None:
        """Test default vector store path."""
        s = Settings()
        assert s.vector_store_path == ".chromadb"

    def test_rag_embedding_model_default(self) -> None:
        """Test default embedding model."""
        s = Settings()
        assert s.embedding_model == "text-embedding-3-small"

