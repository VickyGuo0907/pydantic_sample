"""RAG retriever: ChromaDB vector store with OpenAI embeddings."""

from __future__ import annotations

import hashlib
import logging
import uuid

import chromadb
from chromadb.utils import embedding_functions

from schemas.retrieval import RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)

# Default chunk parameters
_DEFAULT_CHUNK_SIZE = 400   # characters
_DEFAULT_OVERLAP = 50       # characters of overlap between chunks
_COLLECTION_NAME = "documents"


def resolve_openai_key(raw_key: str | None) -> str | None:
    """Return the API key if it looks real, otherwise None.

    Placeholder values like ``sk-...`` (ending with ``...``) are treated as
    absent so the offline DefaultEmbeddingFunction is used instead.

    Args:
        raw_key: The raw key string from settings or environment.

    Returns:
        The key unchanged if it appears valid, otherwise None.
    """
    if not raw_key or raw_key.endswith("..."):
        return None
    return raw_key


def chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[str]:
    """Split text into overlapping fixed-size character chunks.

    Args:
        text: The raw text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to repeat across consecutive chunks.

    Returns:
        List of text chunks. Empty list if text is empty.
    """
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


class VectorStore:
    """Thin wrapper around ChromaDB for document storage and retrieval.

    Supports both in-memory (persist_path=None) and persistent modes.
    Uses ChromaDB's built-in default embedding function when no OpenAI key
    is supplied, allowing the store to work fully offline.
    """

    def __init__(
        self,
        persist_path: str | None = ".chromadb",
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Initialise the ChromaDB client and collection.

        Args:
            persist_path: Directory for persistent storage. None = in-memory.
            openai_api_key: OpenAI key for embedding. None = use default embeddings.
            embedding_model: OpenAI embedding model name.
        """
        if persist_path:
            self._client = chromadb.PersistentClient(path=persist_path)
            collection_name = _COLLECTION_NAME
        else:
            self._client = chromadb.EphemeralClient()
            # Unique name prevents cross-test contamination when multiple
            # EphemeralClient instances share the same in-process state.
            collection_name = f"documents_{uuid.uuid4().hex}"

        if openai_api_key:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=embedding_model,
            )
        else:
            ef = embedding_functions.DefaultEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
        )
        logger.debug(
            "VectorStore initialised (persist=%s, embedding=%s)",
            persist_path,
            embedding_model,
        )

    def add_documents(self, texts: list[str], source: str) -> None:
        """Embed and store a list of text chunks from a named source.

        Args:
            texts: Pre-chunked text strings to index.
            source: Identifier for the origin document (e.g. filename).
        """
        if not texts:
            return
        ids = [
            hashlib.sha256(f"{source}:{i}:{t[:40]}".encode()).hexdigest()
            for i, t in enumerate(texts)
        ]
        metadatas = [{"source": source, "chunk_index": i} for i, _ in enumerate(texts)]
        self._collection.upsert(documents=texts, ids=ids, metadatas=metadatas)
        logger.debug("Indexed %d chunks from %s", len(texts), source)

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Query the vector store for the most relevant chunks.

        Args:
            query: The search query.
            top_k: Maximum number of chunks to return.

        Returns:
            A RetrievalResult with ranked RetrievedChunk objects.
        """
        count = self._collection.count()
        if count == 0:
            return RetrievalResult(query=query, chunks=[])

        effective_k = min(top_k, count)
        response = self._collection.query(
            query_texts=[query],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        docs = response["documents"][0]
        metas = response["metadatas"][0]
        distances = response["distances"][0]

        for text, meta, distance in zip(docs, metas, distances):
            # ChromaDB returns L2 distance; convert to a [0, 1] relevance score.
            score = max(0.0, min(1.0, 1.0 / (1.0 + distance)))
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=meta.get("source", "unknown"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=score,
                )
            )
        return RetrievalResult(query=query, chunks=chunks)
