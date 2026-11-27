"""Phase 1 setup check.

Run from the project root:
    python scripts/check_setup.py

It checks, in order:
  1. Python version
  2. Settings load from .env
  3. The Groq API key is present
  4. The embedding model downloads/loads and produces normalized vectors
  5. Groq answers a tiny test prompt with the configured model
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Let "from config.settings import ..." work when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless warning on Windows


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    problems = 0
    print("\nRAG chatbot - setup check\n")

    # 1. Python version
    if sys.version_info >= (3, 11):
        ok(f"Python {sys.version.split()[0]}")
    else:
        fail(f"Python {sys.version.split()[0]} found, but 3.11+ is required.")
        return 1

    # Running outside a venv is the most common setup mistake.
    if sys.prefix == sys.base_prefix:
        fail(f"Not running inside a virtual environment (using {sys.executable}). "
             "Activate it first:  .\\.venv\\Scripts\\Activate.ps1")
        return 1
    ok(f"Virtual environment: {sys.prefix}")

    # 2. Settings
    try:
        from config.settings import settings
    except ValueError as e:
        fail(f"Your .env has a problem: {e}")
        return 1
    ok(f"Settings loaded (model={settings.groq_model}, embeddings={settings.embedding_model})")

    # 3. API key
    if settings.has_api_key:
        ok("GROQ_API_KEY is set")
    else:
        fail("GROQ_API_KEY is missing. Copy .env.example to .env and paste your key "
             "from https://console.groq.com/keys")
        problems += 1

    # 4. Embedding model (downloads ~130 MB the first time)
    try:
        from sentence_transformers import SentenceTransformer

        print("  ...    loading embedding model (first run downloads it)")
        model = SentenceTransformer(settings.embedding_model)
        vec = model.encode(settings.embedding_query_prefix + "hello", normalize_embeddings=True)
        length = float((vec**2).sum() ** 0.5)
        ok(f"Embedding model works (dimension={len(vec)}, vector length={length:.3f})")
    except Exception as e:  # noqa: BLE001 - we want any failure reported plainly
        fail(f"Could not load the embedding model: {e}")
        problems += 1

    # 5. Groq call
    if settings.has_api_key:
        try:
            import groq
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=0,
                max_tokens=256,
            )
            reply = llm.invoke("Reply with the single word: ready")
            ok(f"Groq replied: {reply.content.strip()[:60]!r}")
        except groq.AuthenticationError:
            fail("Groq rejected the API key. Check GROQ_API_KEY in .env.")
            problems += 1
        except groq.NotFoundError:
            fail(f"Groq doesn't know the model {settings.groq_model!r}. "
                 "Pick one from https://console.groq.com/docs/models")
            problems += 1
        except groq.RateLimitError:
            fail("Groq rate limit hit. Wait a minute and try again.")
            problems += 1
        except groq.APIConnectionError:
            fail("Could not reach Groq. Check your internet connection.")
            problems += 1
        except groq.BadRequestError as e:
            fail(f"Groq refused the request (the model may be retired): {e}")
            problems += 1

    print()
    if problems:
        print(f"{problems} problem(s) found - fix them and run this again.")
        return 1
    print("All good! Phase 1 setup is complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
