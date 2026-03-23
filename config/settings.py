"""Application settings with multi-provider LLM support."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_ai.models import Model
from pydantic_settings import BaseSettings

# Placeholder key used by LM Studio, which requires a non-empty value
# but does not perform actual authentication.
_LMSTUDIO_PLACEHOLDER_KEY = "lmstudio"


class Settings(BaseSettings):
    """Configuration for the reasoning agent application.

    Reads from environment variables and .env file.
    Supports OpenAI, Anthropic (Claude), and LM Studio providers.
    """

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Provider selection
    llm_provider: Literal["openai", "anthropic", "lmstudio"] = Field(
        default="openai",
        description="Which LLM provider to use",
    )

    # OpenAI settings
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")

    # Anthropic settings
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model name",
    )

    # LM Studio settings
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="LM Studio API base URL",
    )
    lmstudio_model: str = Field(
        default="local-model",
        description="LM Studio model name",
    )

    # Search tool settings
    search_api_key: str = Field(default="", description="Search API key (optional)")

    # RAG settings
    rag_enabled: bool = Field(
        default=False,
        description="Enable retrieval-augmented generation",
    )
    vector_store_path: str = Field(
        default=".chromadb",
        description="Path to persist the ChromaDB vector store",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name (cloud providers)",
    )

    def get_model(self) -> Model:
        """Return the pydantic-ai model instance for the configured provider.

        Returns:
            A pydantic-ai Model instance corresponding to the active provider.

        Raises:
            ValueError: If the selected provider requires an API key that is not set,
                or if an unknown provider value is encountered.
        """
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIChatModel(
                self.openai_model,
                provider=OpenAIProvider(api_key=self.openai_api_key),
            )

        if self.llm_provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when using Anthropic provider"
                )
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel(
                self.anthropic_model,
                provider=AnthropicProvider(api_key=self.anthropic_api_key),
            )

        if self.llm_provider == "lmstudio":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIChatModel(
                self.lmstudio_model,
                provider=OpenAIProvider(
                    base_url=self.lmstudio_base_url,
                    api_key=_LMSTUDIO_PLACEHOLDER_KEY,
                ),
            )

        raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
