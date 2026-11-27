"""Chat with your indexed documents from the command line (Phase 3 test).

Run from the project root (with the venv active):
    python scripts/ask.py                         # interactive chat, follow-ups work
    python scripts/ask.py "What is Diffuse to Choose?"   # one question

In the chat: type a question, "/clear" to forget the conversation, or "exit" to quit.
Index files first with:  python scripts/try_ingestion.py <file>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # answers can contain any character

from config.settings import setup_logging  # noqa: E402
from src.generation.generator import GenerationError  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402


def ask_once(pipeline: RAGPipeline, question: str, history: list[dict]) -> None:
    response = pipeline.ask(question, history=history)

    print("\nAssistant: ", end="", flush=True)
    try:
        for piece in response.stream():
            print(piece, end="", flush=True)
    except GenerationError as e:
        print(f"\n[Error] {e}")
        return
    print()

    if response.cited_sources:
        print("\nSources: " + "   ".join(s.label for s in response.cited_sources))

    # "Under the hood"
    print("\n  --- under the hood ---")
    if response.search_query != question:
        print(f"  search query (rewritten follow-up): {response.search_query!r}")
    if response.retrieval:
        for h in response.retrieval.hits:
            used = "used" if response.retrieval.passes(h) else "below threshold"
            rerank = f"relevance={h.rerank_score:.3f}  " if h.rerank_score is not None else ""
            print(f"  {rerank}similarity={h.score:.3f}  {h.source} p.{h.page}  ({used})")
    print(f"  model={response.model}  temperature={response.temperature}  "
          f"answered by LLM={response.used_llm}  time={response.response_time}s\n")

    history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response.answer},
    ]


def main() -> int:
    setup_logging("WARNING")  # keep the chat readable; use INFO to see every step
    pipeline = RAGPipeline()
    docs = pipeline.list_documents()
    print(f"Indexed documents: {', '.join(d.source for d in docs) or 'none'}")

    history: list[dict] = []
    if len(sys.argv) > 1:
        ask_once(pipeline, " ".join(sys.argv[1:]), history)
        return 0

    print('Ask a question ("/clear" to reset, "exit" to quit).')
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue
        if question:
            ask_once(pipeline, question, history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
