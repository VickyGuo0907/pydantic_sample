#!/usr/bin/env python
"""CLI script to chunk, embed, and index documents into the ChromaDB vector store.

Usage:
    python scripts/ingest.py                          # index data/sample_docs/
    python scripts/ingest.py --docs-dir my/docs       # index a custom directory
    python scripts/ingest.py --reset                  # wipe and re-index
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from tools.retriever import VectorStore, chunk_text, resolve_openai_key

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse ingest CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Index documents into ChromaDB for RAG")
    parser.add_argument(
        "--docs-dir",
        default="data/sample_docs",
        help="Directory of .txt documents to index (default: data/sample_docs)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the vector store before indexing",
    )
    return parser.parse_args()


def ingest(docs_dir: Path, settings: Settings, *, reset: bool = False) -> int:
    """Chunk and index all .txt files from docs_dir into ChromaDB.

    Args:
        docs_dir: Directory containing .txt source documents.
        settings: Application settings providing vector store config.
        reset: If True, delete the existing store before indexing.

    Returns:
        Total number of chunks indexed.
    """
    if reset:
        import shutil

        store_path = Path(settings.vector_store_path)
        if store_path.exists():
            shutil.rmtree(store_path)
            logger.info("Deleted existing vector store at %s", store_path)

    store = VectorStore(
        persist_path=settings.vector_store_path,
        openai_api_key=resolve_openai_key(settings.openai_api_key),
        embedding_model=settings.embedding_model,
    )

    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", docs_dir)
        return 0

    total = 0
    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        store.add_documents(texts=chunks, source=txt_file.name)
        logger.info("Indexed %s → %d chunks", txt_file.name, len(chunks))
        total += len(chunks)

    logger.info("Done. Total chunks indexed: %d", total)
    return total


def main() -> None:
    """Entry point for the ingest CLI."""
    args = parse_args()
    settings = Settings()
    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        logger.error("docs-dir does not exist: %s", docs_dir)
        sys.exit(1)
    ingest(docs_dir, settings, reset=args.reset)


if __name__ == "__main__":
    main()
