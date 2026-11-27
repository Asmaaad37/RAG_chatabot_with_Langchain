"""Find the chunks that best answer a question, and decide if any are good enough.

Steps (each can be switched off in settings):
  1. Embedding search     - chunks with a similar meaning
  2. Keyword search (BM25) - chunks sharing the exact words      (hybrid_search)
  3. Merge the lists with reciprocal rank fusion
  4. Rerank the merged candidates with a cross-encoder, keep top-k (use_reranker)
  5. Compare the best score with the threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import settings
from src.embeddings.embedder import Embedder
from src.retrieval.keyword_search import KeywordIndex
from src.vectorstore.chroma_store import SearchHit, VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant for reciprocal rank fusion


@dataclass
class RetrievalResult:
    queries: list[str]
    hits: list[SearchHit]                 # the top-k chunks, best first
    threshold: float                      # similarity threshold
    rerank_threshold: float | None = None  # reranker threshold (None when not reranked)
    hybrid: bool = False
    reranked: bool = False

    def passes(self, hit: SearchHit) -> bool:
        """A chunk counts as relevant if EITHER score clears its threshold.

        Neither score is reliable alone: similarity can't tell a list of names
        from an unrelated question, and the reranker sometimes underrates lists.
        Together they rarely both miss.
        """
        if hit.score >= self.threshold:
            return True
        return hit.rerank_score is not None and self.rerank_threshold is not None \
            and hit.rerank_score >= self.rerank_threshold

    @property
    def best_score(self) -> float:
        return max((h.relevance for h in self.hits), default=0.0)

    @property
    def relevant_hits(self) -> list[SearchHit]:
        """Only the chunks that pass a threshold. These go to the LLM."""
        return [h for h in self.hits if self.passes(h)]

    @property
    def found(self) -> bool:
        return any(self.passes(h) for h in self.hits)


def fuse_rankings(rankings: list[list[SearchHit]], top_k: int) -> list[SearchHit]:
    """Merge several ranked lists with reciprocal rank fusion (RRF).

    A chunk earns 1 / (60 + rank) points from every list it appears in, so chunks
    that rank well in *any* list rise to the top. Each chunk keeps its best similarity.
    """
    points: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    for hits in rankings:
        for rank, hit in enumerate(hits):
            key = hit.id or f"{hit.source}#{hit.chunk_index}"
            points[key] = points.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in best or hit.score > best[key].score:
                best[key] = hit
    ordered = sorted(points, key=points.get, reverse=True)
    return [best[key] for key in ordered[:top_k]]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.keyword_index = KeywordIndex(store)

    def retrieve(
        self,
        queries: str | list[str],
        top_k: int | None = None,
        threshold: float | None = None,
        hybrid: bool | None = None,
        use_reranker: bool | None = None,
        reranker_model: str | None = None,
        rerank_threshold: float | None = None,
    ) -> RetrievalResult:
        """Search with one query, or several (e.g. a rewritten follow-up + the original wording).

        The first query is the "main" one: it's what the reranker judges chunks against.
        """
        queries = [queries] if isinstance(queries, str) else list(dict.fromkeys(queries))
        top_k = top_k or settings.top_k
        hybrid = settings.hybrid_search if hybrid is None else hybrid
        use_reranker = settings.use_reranker if use_reranker is None else use_reranker
        threshold = settings.score_threshold if threshold is None else threshold
        if use_reranker:
            rerank_threshold = settings.rerank_threshold if rerank_threshold is None else rerank_threshold
        else:
            rerank_threshold = None

        # The reranker needs a wider pool of candidates to choose from.
        pool = max(settings.rerank_candidates, top_k) if use_reranker else top_k

        rankings: list[list[SearchHit]] = []
        for query in queries:
            vector = self.embedder.embed_query(query)
            rankings.append(self.store.search(vector, k=pool))
            if hybrid:
                keyword_hits = self.keyword_index.search(query, k=pool)
                # Keyword hits don't have a similarity yet; compute it for display/threshold.
                sims = self.store.similarities([h.id for h in keyword_hits], vector)
                for h in keyword_hits:
                    h.score = sims.get(h.id, 0.0)
                rankings.append(keyword_hits)

        hits = fuse_rankings(rankings, pool) if len(rankings) > 1 else rankings[0][:pool]

        if use_reranker and hits:
            from src.retrieval.reranker import get_reranker  # loaded only when needed

            reranker = get_reranker(reranker_model or settings.reranker_model)
            hits = reranker.rerank(queries[0], hits, top_k)
        else:
            hits = hits[:top_k]

        result = RetrievalResult(
            queries=queries, hits=hits, threshold=threshold, rerank_threshold=rerank_threshold,
            hybrid=hybrid, reranked=use_reranker,
        )
        logger.info(
            "Retrieved %d chunk(s) for %s (hybrid=%s, reranked=%s), found=%s",
            len(hits), queries, hybrid, use_reranker, result.found,
        )
        return result
