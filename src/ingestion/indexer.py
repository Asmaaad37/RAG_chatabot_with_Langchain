"""Index one file: load -> clean -> chunk -> skip duplicates -> embed -> store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config.settings import settings
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_pages, file_hash
from src.ingestion.loader import load_document
from src.vectorstore.chroma_store import VectorStore

logger = logging.getLogger(__name__)

# Called as on_progress(fraction_done_0_to_1, message) so the UI can show a progress bar.
ProgressCallback = Callable[[float, str], None]

EMBED_BATCH_SIZE = 32


@dataclass
class IndexResult:
    file_name: str
    status: str  # "indexed" or "already_indexed"
    pages: int = 0
    chunks_added: int = 0
    chunks_skipped: int = 0

    @property
    def message(self) -> str:
        if self.status == "already_indexed":
            return f"'{self.file_name}' is already indexed, so it was skipped."
        msg = f"Indexed '{self.file_name}': {self.pages} page(s), {self.chunks_added} new chunk(s)"
        if self.chunks_skipped:
            msg += f", {self.chunks_skipped} duplicate chunk(s) skipped"
        return msg + "."


def index_file(
    path: str | Path,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> IndexResult:
    """Add one PDF/TXT file to the vector store.

    Raises DocumentLoadError / EmptyDocumentError with a user-friendly message
    if the file can't be used.
    """
    path = Path(path)
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    report = on_progress or (lambda fraction, message: None)

    # 1. Whole-file duplicate check (cheap, so do it before anything else)
    report(0.0, f"Checking {path.name}...")
    fhash = file_hash(path)
    if store.has_file(fhash):
        logger.info("Skipping %s: already indexed", path.name)
        report(1.0, "Already indexed")
        return IndexResult(file_name=path.name, status="already_indexed")

    # 2. Load + clean, then chunk
    report(0.05, f"Reading {path.name}...")
    pages = load_document(path)
    chunks = chunk_pages(pages, fhash, chunk_size, chunk_overlap)

    # 3. Chunk-level duplicate check (same text already stored from another file)
    existing = store.existing_ids([c.id for c in chunks])
    new_chunks = [c for c in chunks if c.id not in existing]
    skipped = len(chunks) - len(new_chunks)

    # 4. Embed in batches so we can report progress
    vectors: list[list[float]] = []
    for start in range(0, len(new_chunks), EMBED_BATCH_SIZE):
        batch = new_chunks[start : start + EMBED_BATCH_SIZE]
        vectors.extend(embedder.embed_documents([c.text for c in batch]))
        done = min(start + EMBED_BATCH_SIZE, len(new_chunks))
        report(0.1 + 0.85 * done / len(new_chunks), f"Embedding chunks {done}/{len(new_chunks)}")

    # 5. Store
    if new_chunks:
        store.add_chunks(new_chunks, vectors)
    report(1.0, "Done")

    result = IndexResult(
        file_name=path.name,
        status="indexed",
        pages=len(pages),
        chunks_added=len(new_chunks),
        chunks_skipped=skipped,
    )
    logger.info(result.message)
    return result
