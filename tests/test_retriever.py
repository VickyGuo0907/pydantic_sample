"""Tests for the RAG vector store retriever."""

from __future__ import annotations

import pytest

from schemas.retrieval import RetrievalResult
from tools.retriever import VectorStore, chunk_text


class TestChunkText:
    """Test the text chunking helper."""

    def test_short_text_is_single_chunk(self) -> None:
        """Text shorter than chunk_size returns one chunk."""
        chunks = chunk_text("Hello world.", chunk_size=200, overlap=20)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_is_split(self) -> None:
        """Text longer than chunk_size is split into multiple chunks."""
        text = "word " * 100  # 500 chars
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_overlap_creates_shared_content(self) -> None:
        """Consecutive chunks share content equal to overlap size."""
        text = "abcdefghij" * 20  # 200 chars
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # End of chunk[0] and start of chunk[1] should share 10 chars
        assert chunks[0][-10:] == chunks[1][:10]

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty input returns empty list."""
        assert chunk_text("", chunk_size=200, overlap=20) == []


class TestVectorStore:
    """Integration tests for VectorStore using an in-memory ChromaDB."""

    @pytest.fixture
    def store(self) -> VectorStore:
        """Return an in-memory VectorStore (no disk writes)."""
        return VectorStore(persist_path=None)

    def test_add_and_retrieve(self, store: VectorStore) -> None:
        """Documents added to the store can be retrieved by query."""
        store.add_documents(
            texts=["The speed of light is 299,792,458 m/s."],
            source="physics.txt",
        )
        result = store.retrieve("speed of light", top_k=1)
        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) == 1
        assert "light" in result.chunks[0].text.lower()

    def test_retrieve_returns_top_k(self, store: VectorStore) -> None:
        """retrieve() respects the top_k limit."""
        store.add_documents(
            texts=["Fact A.", "Fact B.", "Fact C.", "Fact D."],
            source="facts.txt",
        )
        result = store.retrieve("fact", top_k=2)
        assert len(result.chunks) <= 2

    def test_retrieve_empty_store_returns_empty(self, store: VectorStore) -> None:
        """Querying an empty store returns an empty RetrievalResult."""
        result = store.retrieve("anything", top_k=3)
        assert result.chunks == []

    def test_scores_are_in_valid_range(self, store: VectorStore) -> None:
        """All returned chunk scores are between 0.0 and 1.0."""
        store.add_documents(texts=["Some content here."], source="doc.txt")
        result = store.retrieve("content", top_k=1)
        for chunk in result.chunks:
            assert 0.0 <= chunk.score <= 1.0
