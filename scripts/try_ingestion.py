"""Try Phase 2 by hand: index a file into the real store, then run a search.

Run from the project root (with the venv active):
    python scripts/try_ingestion.py
    python scripts/try_ingestion.py path\\to\\your.pdf --query "your question"

Run it twice to see the duplicate check skip the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings, setup_logging  # noqa: E402
from src.embeddings.embedder import get_embedder  # noqa: E402
from src.ingestion.indexer import index_file  # noqa: E402
from src.ingestion.loader import DocumentLoadError  # noqa: E402
from src.vectorstore.chroma_store import VectorStore  # noqa: E402


def show_hits(store: VectorStore, embedder, query: str) -> None:
    print(f"\nQuery: {query!r}")
    for i, hit in enumerate(store.search(embedder.embed_query(query), k=settings.top_k), 1):
        preview = hit.text[:90].replace("\n", " ")
        print(f"  [{i}] score={hit.score:.3f}  {hit.source} p.{hit.page}  {preview}...")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", default=[str(ROOT / "data/sample_docs/VirtualTryAll.pdf")])
    parser.add_argument("--query", default="What is Diffuse to Choose and what problem does it solve?")
    args = parser.parse_args()

    setup_logging()
    store = VectorStore()
    embedder = get_embedder()

    for f in args.files:
        try:
            result = index_file(f, store, embedder)
            print(result.message)
        except DocumentLoadError as e:
            print(f"Error: {e}")

    print(f"\nIndexed documents ({store.count()} chunks in total):")
    for d in store.list_documents():
        print(f"  - {d.source}: {d.pages} page(s), {d.chunks} chunk(s)")

    show_hits(store, embedder, args.query)
    # An unrelated question shows what a "low" score looks like (useful for SCORE_THRESHOLD).
    show_hits(store, embedder, "What is a good recipe for chocolate cake?")
    print(f"\nCurrent SCORE_THRESHOLD = {settings.score_threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
