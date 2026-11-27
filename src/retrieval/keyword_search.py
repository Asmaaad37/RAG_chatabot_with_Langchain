"""Keyword search with BM25, the classic search-engine ranking.

Embeddings find chunks with similar *meaning*; BM25 finds chunks that share the
question's exact *words*. BM25 is better at names, numbers, codes and acronyms
("1.2M", "DINOv2", "FiLM"), so the two together ("hybrid search") miss less.
"""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from src.vectorstore.chroma_store import SearchHit, VectorStore

logger = logging.getLogger(__name__)

# Very common words carry no meaning for search.
STOPWORDS = set(
    "a an and are as at be but by can could did do does for from had has have how i if in into is it "
    "its me my of on or our so than that the their them then there these they this to us was we were "
    "what when where which who whom why will with would you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, e.g. "DINOv2 uses 1.2M" -> ["dinov2", "uses", "1.2m"]."""
    words = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOPWORDS]


class KeywordIndex:
    """A BM25 index over all stored chunks, rebuilt automatically when they change."""

    def __init__(self, store: VectorStore):
        self.store = store
        self._signature: int | None = None
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []

    def _refresh(self) -> None:
        ids, texts, metas = self.store.all_chunks()
        signature = hash(frozenset(ids))
        if signature == self._signature:
            return  # nothing was added or deleted
        self._ids, self._texts, self._metas = ids, texts, metas
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None
        self._signature = signature
        logger.info("Keyword index built over %d chunks", len(ids))

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Top-k chunks by BM25. `score` is filled with 0 here; the retriever adds similarity."""
        self._refresh()
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            SearchHit(
                text=self._texts[i],
                source=self._metas[i]["source"],
                page=self._metas[i]["page"],
                chunk_index=self._metas[i]["chunk_index"],
                score=0.0,
                id=self._ids[i],
            )
            for i in best
            if scores[i] > 0  # no shared words = not a keyword match
        ]
