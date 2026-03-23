"""Pydantic models for RAG retrieval results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single retrieved document chunk with relevance score."""

    text: str = Field(description="The chunk text content")
    source: str = Field(description="Source filename or identifier")
    chunk_index: int = Field(ge=0, description="Position of chunk within source document")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score (0.0 to 1.0)")


class RetrievalResult(BaseModel):
    """Result of a vector store retrieval query."""

    query: str = Field(description="The query that produced these results")
    chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved chunks ranked by relevance",
    )

    def as_context_string(self) -> str:
        """Format retrieved chunks as a context block for LLM injection.

        Returns:
            A formatted string listing each chunk with its source.
        """
        if not self.chunks:
            return "No relevant documents found."
        lines: list[str] = ["Retrieved context:"]
        for i, chunk in enumerate(self.chunks, 1):
            lines.append(f"[{i}] ({chunk.source}) {chunk.text}")
        return "\n".join(lines)
