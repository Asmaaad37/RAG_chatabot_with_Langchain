"""All app settings live here.

Values come from the .env file (see .env.example). Anything not set in .env
falls back to the defaults below, so the app works with only GROQ_API_KEY set.

Usage elsewhere in the code:
    from config.settings import settings
    print(settings.chunk_size)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read .env from the project root (does nothing if the file doesn't exist).
load_dotenv(PROJECT_ROOT / ".env")

# The app's display name (browser tab, logo, headers).
APP_NAME = "My AI"

# bge models work best when *queries* (not documents) start with this instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} in .env must be a whole number, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} in .env must be a number, got {raw!r}") from None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} in .env must be true or false, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    # LLM
    groq_api_key: str = field(repr=False)  # never printed in logs
    groq_model: str
    temperature: float
    user_name: str               # shown in "Welcome back, <name>" (optional)
    whisper_model: str           # speech-to-text model for voice input

    # Embeddings
    embedding_model: str
    embedding_query_prefix: str

    # Chunking
    chunk_size: int
    chunk_overlap: int

    # Retrieval
    top_k: int
    score_threshold: float       # cosine similarity threshold (used when the reranker is off)
    hybrid_search: bool          # add BM25 keyword search to embedding search
    use_reranker: bool           # rescore candidates with a cross-encoder
    reranker_model: str
    rerank_candidates: int       # how many search results the reranker looks at
    rerank_threshold: float      # reranker relevance threshold (used when the reranker is on)

    # Chat
    history_turns: int

    # Paths
    data_dir: Path
    uploads_dir: Path
    chroma_dir: Path
    chats_dir: Path
    collection_name: str

    # Logging
    log_level: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.groq_api_key)

    def validate(self) -> None:
        """Raise a friendly ValueError if settings don't make sense together."""
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be 0 or more and smaller than CHUNK_SIZE.")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be at least 1.")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("SCORE_THRESHOLD must be between 0 and 1.")
        if not 0.0 <= self.rerank_threshold <= 1.0:
            raise ValueError("RERANK_THRESHOLD must be between 0 and 1.")
        if self.rerank_candidates < self.top_k:
            raise ValueError("RERANK_CANDIDATES must be at least TOP_K.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("TEMPERATURE must be between 0 and 2.")
        if self.history_turns < 0:
            raise ValueError("HISTORY_TURNS must be 0 or more.")


def load_settings() -> Settings:
    """Build a Settings object from the environment (.env) and defaults."""
    # DATA_DIR lets you run a separate copy (e.g. a demo) without touching your real data.
    data_dir = Path(_env_str("DATA_DIR", str(PROJECT_ROOT / "data")))
    s = Settings(
        groq_api_key=_env_str("GROQ_API_KEY", ""),
        groq_model=_env_str("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=_env_float("TEMPERATURE", 0.2),
        user_name=_env_str("USER_NAME", ""),
        whisper_model=_env_str("WHISPER_MODEL", "whisper-large-v3-turbo"),
        embedding_model=_env_str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_query_prefix=BGE_QUERY_PREFIX,
        chunk_size=_env_int("CHUNK_SIZE", 600),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 100),
        top_k=_env_int("TOP_K", 5),
        score_threshold=_env_float("SCORE_THRESHOLD", 0.55),
        hybrid_search=_env_bool("HYBRID_SEARCH", True),
        # On by default: it ranks answers higher and rejects off-topic questions far more
        # reliably than similarity alone (~1.3 s per question on CPU). See scripts/evaluate.py.
        use_reranker=_env_bool("USE_RERANKER", True),
        reranker_model=_env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_candidates=_env_int("RERANK_CANDIDATES", 20),
        rerank_threshold=_env_float("RERANK_THRESHOLD", 0.1),
        history_turns=_env_int("HISTORY_TURNS", 3),
        data_dir=data_dir,
        uploads_dir=data_dir / "uploads",
        chroma_dir=data_dir / "chroma_db",
        chats_dir=data_dir / "chats",
        collection_name="documents",
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )
    s.validate()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return s


def setup_logging(level: str | None = None) -> None:
    """Configure logging once for the whole app."""
    logging.basicConfig(
        level=level or settings.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries are very chatty at INFO level.
    for noisy in ("httpx", "httpx2", "huggingface_hub", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# A single shared instance, imported by the rest of the app.
settings = load_settings()
