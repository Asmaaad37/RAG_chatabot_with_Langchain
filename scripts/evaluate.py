"""Measure retrieval quality on a test set, for several search setups.

Run from the project root (with the venv active):
    python scripts/evaluate.py                                  # all test sets in data/eval/
    python scripts/evaluate.py data/eval/my_questions.json      # one test set
    python scripts/evaluate.py --chunk-size 800 --chunk-overlap 120

A test set is a JSON file like data/eval/sample_questions.json:
  - "document": the PDF/TXT to test against
  - "questions": each with "answer_contains" (text that proves the right chunk was found)
  - "off_topic": questions the app should answer with "I couldn't find this"

The document is indexed into a temporary database, so your real index isn't touched.
No LLM is called, so this is free and needs no API key.

Metrics:
  found@k   share of questions where a chunk containing the answer is in the top k
  MRR       how high the first correct chunk ranks (1.0 = always first)
  accuracy  share of questions handled correctly with the configured thresholds:
            real questions must have the answer among the chunks sent to the LLM,
            off-topic ones must be rejected ("I couldn't find this")
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import settings, setup_logging  # noqa: E402
from src.embeddings.embedder import get_embedder  # noqa: E402
from src.ingestion.indexer import index_file  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.vectorstore.chroma_store import VectorStore  # noqa: E402

SMALL_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LARGE_RERANKER = "BAAI/bge-reranker-base"

# (label, hybrid, reranker model or None)
SETUPS = [
    ("embeddings only", False, None),
    ("hybrid", True, None),
    ("embeddings + small reranker", False, SMALL_RERANKER),
    ("hybrid + small reranker", True, SMALL_RERANKER),
    ("hybrid + large reranker", True, LARGE_RERANKER),
]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())  # ignore line breaks and extra spaces


def contains_answer(text: str, needles: list[str]) -> bool:
    text = _normalize(text)
    return any(_normalize(n) in text for n in needles)


def evaluate(test_file: Path, chunk_size: int, chunk_overlap: int, top_k: int, setups) -> None:
    spec = json.loads(test_file.read_text(encoding="utf-8"))
    document = ROOT / spec["document"]
    embedder = get_embedder()

    store = VectorStore(path=tempfile.mkdtemp(prefix="eval_"), collection_name="eval_docs")
    result = index_file(document, store, embedder, chunk_size, chunk_overlap)
    print(f"\n{test_file.name}: {document.name}, {result.chunks_added} chunks "
          f"(size {chunk_size}, overlap {chunk_overlap}), top-k {top_k}")
    print(f"{len(spec['questions'])} questions, {len(spec['off_topic'])} off-topic\n")

    retriever = Retriever(store, embedder)
    print(f"thresholds: similarity {settings.score_threshold}, reranker {settings.rerank_threshold}\n")
    header = f"{'setup':<30} {'found@k':>8} {'MRR':>6} {'accuracy':>9} {'sec/q':>7}"
    print(header)
    print("-" * len(header))

    for label, hybrid, reranker in setups:
        # Thresholds come from settings, exactly as the app uses them.
        options = dict(top_k=top_k, hybrid=hybrid, use_reranker=reranker is not None,
                       reranker_model=reranker)
        retriever.retrieve("warm-up", **options)  # load models before timing

        started = time.perf_counter()
        found_count, correct, ranks, problems = 0, 0, [], []
        for q in spec["questions"]:
            result = retriever.retrieve(q["question"], **options)
            rank = next((i for i, h in enumerate(result.hits, 1)
                         if contains_answer(h.text, q["answer_contains"])), None)
            ranks.append(1.0 / rank if rank else 0.0)
            found_count += rank is not None
            # Correct = the chunk with the answer is among those actually sent to the LLM.
            if any(contains_answer(h.text, q["answer_contains"]) for h in result.relevant_hits):
                correct += 1
            else:
                problems.append(f"missed: {q['question']}" if rank is None
                                else f"found but below thresholds: {q['question']}")
        for q in spec["off_topic"]:
            if retriever.retrieve(q, **options).found:
                problems.append(f"off-topic not rejected: {q}")
            else:
                correct += 1
        total = len(spec["questions"]) + len(spec["off_topic"])
        per_question = (time.perf_counter() - started) / total

        print(f"{label:<30} {found_count / len(spec['questions']):>8.0%} "
              f"{sum(ranks) / len(ranks):>6.2f} {correct / total:>9.0%} {per_question:>7.2f}")
        for p in problems:
            print(f"{'':<4}{p}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--skip-large", action="store_true", help="skip the 1.1 GB bge reranker")
    args = parser.parse_args()

    setup_logging("WARNING")
    setups = [s for s in SETUPS if not (args.skip_large and s[2] == LARGE_RERANKER)]
    files = args.files or sorted((ROOT / "data" / "eval").glob("*.json"))
    for f in files:
        evaluate(f, args.chunk_size, args.chunk_overlap, args.top_k, setups)
    return 0


if __name__ == "__main__":
    sys.exit(main())
