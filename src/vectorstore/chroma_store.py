"""A small wrapper around a persistent ChromaDB collection (cosine similarity)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

ADD_BATCH_SIZE = 500  # Chroma limits how many items one add() call may contain


@dataclass
class SearchHit:
    text: str
    source: str
    page: int
    chunk_index: int
    score: float  # cosine similarity: 1 = identical meaning, around 0 = unrelated
    id: str = ""
    rerank_score: float | None = None  # 0-1 relevance from the reranker, if it ran

    @property
    def relevance(self) -> float:
        """The score used for the threshold: the reranker's if available, else similarity."""
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass
class IndexedDocument:
    file_hash: str
    source: str
    chunks: int
    pages: int


class VectorStore:
    def __init__(
        self,
        path: str | Path = settings.chroma_dir,
        collection_name: str = settings.collection_name,
    ):
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # we pass our own bge vectors
        )
        logger.info("Vector store ready at %s (%d chunks)", path, self.count())

    # ----- reading -----

    def count(self) -> int:
        return self.collection.count()

    def has_file(self, file_hash: str) -> bool:
        found = self.collection.get(where={"file_hash": file_hash}, limit=1, include=[])
        return bool(found["ids"])

    def existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        return set(self.collection.get(ids=ids, include=[])["ids"])

    def list_documents(self) -> list[IndexedDocument]:
        """One entry per indexed file, sorted by file name."""
        metadatas = self.collection.get(include=["metadatas"])["metadatas"] or []
        by_file: dict[str, dict] = {}
        for m in metadatas:
            entry = by_file.setdefault(
                m["file_hash"], {"source": m["source"], "chunks": 0, "pages": set()}
            )
            entry["chunks"] += 1
            entry["pages"].add(m["page"])
        docs = [
            IndexedDocument(file_hash=h, source=e["source"], chunks=e["chunks"], pages=len(e["pages"]))
            for h, e in by_file.items()
        ]
        return sorted(docs, key=lambda d: d.source.lower())

    def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        """Return the k most similar chunks, best first."""
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                SearchHit(
                    text=text,
                    source=meta["source"],
                    page=meta["page"],
                    chunk_index=meta["chunk_index"],
                    score=round(1.0 - distance, 4),  # Chroma returns cosine *distance*
                    id=chunk_id,
                )
            )
        return hits

    def all_chunks(self) -> tuple[list[str], list[str], list[dict]]:
        """Every stored chunk as (ids, texts, metadatas). Used to build the keyword index."""
        data = self.collection.get(include=["documents", "metadatas"])
        return data["ids"], data["documents"] or [], data["metadatas"] or []

    def similarities(self, ids: list[str], query_vector: list[float]) -> dict[str, float]:
        """Cosine similarity between the query and specific chunks (vectors are normalized,
        so this is just a dot product). Lets keyword-search hits show a similarity too."""
        if not ids:
            return {}
        data = self.collection.get(ids=ids, include=["embeddings"])
        return {
            chunk_id: round(float(sum(a * b for a, b in zip(vector, query_vector))), 4)
            for chunk_id, vector in zip(data["ids"], data["embeddings"])
        }

    # ----- writing -----

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for start in range(0, len(chunks), ADD_BATCH_SIZE):
            batch = chunks[start : start + ADD_BATCH_SIZE]
            self.collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                embeddings=vectors[start : start + ADD_BATCH_SIZE],
            )

    def delete_document(self, file_hash: str) -> None:
        self.collection.delete(where={"file_hash": file_hash})
        logger.info("Deleted document %s", file_hash[:12])

    def clear(self) -> None:
        """Remove every chunk (keeps the collection itself)."""
        ids = self.collection.get(include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
