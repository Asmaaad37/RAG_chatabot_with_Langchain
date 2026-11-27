"""Split pages into chunks and give every chunk its metadata and hash."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def file_hash(path: str | Path) -> str:
    """SHA-256 of the file's bytes. Same content = same hash, whatever the file name."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text: str) -> str:
    """SHA-256 of a chunk's text, used to skip chunks we've already stored."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    text: str
    source: str        # file name
    page: int          # 1-based page number
    chunk_index: int   # position of the chunk within its file (0, 1, 2, ...)
    chunk_hash: str
    file_hash: str

    @property
    def id(self) -> str:
        # Using the text hash as the database id means the same text is stored once.
        return self.chunk_hash

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "chunk_hash": self.chunk_hash,
            "file_hash": self.file_hash,
        }


def chunk_pages(
    pages: list[Document], file_hash_value: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split each page separately (so every chunk knows its page) and drop repeats."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for page in pages:
        for piece in splitter.split_text(page.page_content):
            h = text_hash(piece)
            if h in seen:  # e.g. a header repeated on every page
                continue
            seen.add(h)
            chunks.append(
                Chunk(
                    text=piece,
                    source=page.metadata["source"],
                    page=page.metadata["page"],
                    chunk_index=len(chunks),
                    chunk_hash=h,
                    file_hash=file_hash_value,
                )
            )
    return chunks
