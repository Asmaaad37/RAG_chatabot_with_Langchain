"""Rerank search results with a cross-encoder (runs locally, free).

Search compares a question vector with chunk vectors that were made separately.
A cross-encoder instead reads the question and a chunk *together*, so it judges
"does this chunk answer this question?" much more precisely. It's too slow to run
on every chunk, so we use it on the top ~20 search results and keep the best few.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from config.settings import settings  # noqa: E402
from src.vectorstore.chroma_store import SearchHit  # noqa: E402

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = settings.reranker_model):
        self.model_name = model_name
        logger.info("Loading reranker %s", model_name)
        try:
            # Sigmoid turns the raw output into a 0-1 relevance score.
            self.model = CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid())
        except OSError as e:
            raise RuntimeError(
                f"Couldn't load the reranker '{model_name}'. The first run needs an internet "
                "connection to download it."
            ) from e

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Score every hit against the query, then return the top_k best (sets hit.rerank_score)."""
        if not hits:
            return []
        scores = self.model.predict([(query, h.text) for h in hits], show_progress_bar=False)
        for hit, score in zip(hits, scores):
            hit.rerank_score = round(float(score), 4)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_k]


@lru_cache(maxsize=2)
def get_reranker(model_name: str = settings.reranker_model) -> Reranker:
    return Reranker(model_name)
