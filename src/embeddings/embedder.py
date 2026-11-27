"""Local embeddings with BAAI/bge-small-en-v1.5 (free, runs on your CPU)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

# Keep transformers away from TensorFlow (if installed elsewhere it can break imports)
# and hide a harmless Windows warning about symlinks in the model cache.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402

from config.settings import settings  # noqa: E402

logger = logging.getLogger(__name__)


class Embedder:
    """Turns text into normalized vectors (length 1), so cosine similarity = dot product."""

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        query_prefix: str = settings.embedding_query_prefix,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        logger.info("Loading embedding model %s", model_name)
        try:
            self.model = SentenceTransformer(model_name)
        except OSError as e:
            raise RuntimeError(
                f"Couldn't load the embedding model '{model_name}'. The first run needs an "
                "internet connection to download it (about 130 MB)."
            ) from e

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed document chunks. No prefix: bge only wants it on queries."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a user question, with bge's instruction prefix."""
        vector = self.model.encode(self.query_prefix + query, normalize_embeddings=True)
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Load the model once and reuse it (loading takes a few seconds)."""
    return Embedder()
