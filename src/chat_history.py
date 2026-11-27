"""Save chats to disk so they show up in the History list and can be reopened.

Each chat is one small JSON file in data/chats/:
    {"id": "...", "title": "first question", "created": 1764200000.0,
     "updated": 1764200100.0, "messages": [{"role": "user", "content": "..."}, ...]}
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

# Module-level so tests can point them at a temporary folder.
CHATS_DIR = settings.chats_dir
# Every 👍/👎 is also appended here (one JSON object per line). Answers rated 👎 make
# good test questions for scripts/evaluate.py.
FEEDBACK_FILE = settings.data_dir / "feedback.jsonl"


@dataclass
class Chat:
    id: str
    title: str
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    messages: list[dict] = field(default_factory=list)

    @property
    def question_count(self) -> int:
        return sum(m["role"] == "user" for m in self.messages)

    def matches(self, text: str) -> bool:
        """True if the search text appears in the title or any message."""
        text = text.lower().strip()
        return not text or text in self.title.lower() or any(
            text in m["content"].lower() for m in self.messages
        )


def new_chat(first_question: str) -> Chat:
    title = " ".join(first_question.split())
    return Chat(id=uuid.uuid4().hex[:12], title=title[:80] + ("…" if len(title) > 80 else ""))


class ChatStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else CHATS_DIR
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, chat_id: str) -> Path:
        # ids are hex we generated; never let a stray value point outside the folder
        return self.path / f"{Path(chat_id).name}.json"

    def save(self, chat: Chat) -> None:
        chat.updated = time.time()
        self._file(chat.id).write_text(json.dumps(asdict(chat), ensure_ascii=False), encoding="utf-8")

    def load(self, chat_id: str) -> Chat | None:
        try:
            data = json.loads(self._file(chat_id).read_text(encoding="utf-8"))
            return Chat(**data)
        except (OSError, ValueError, TypeError) as e:
            logger.warning("Couldn't open chat %s: %s", chat_id, e)
            return None

    def list(self) -> list[Chat]:
        """All saved chats, most recently updated first. Broken files are skipped."""
        chats = [c for f in self.path.glob("*.json") if (c := self.load(f.stem))]
        return sorted(chats, key=lambda c: c.updated, reverse=True)

    def delete(self, chat_id: str) -> None:
        self._file(chat_id).unlink(missing_ok=True)

    def delete_all(self) -> int:
        """Delete every saved chat. Returns how many were deleted."""
        files = list(self.path.glob("*.json"))
        for f in files:
            f.unlink(missing_ok=True)
        logger.info("Deleted all %d saved chats", len(files))
        return len(files)


def delete_feedback() -> None:
    """Delete the 👍/👎 log (data/feedback.jsonl)."""
    Path(FEEDBACK_FILE).unlink(missing_ok=True)


def log_feedback(chat: Chat, index: int, rating: str | None) -> None:
    """Record a 👍 ("up"), 👎 ("down") or cleared (None) rating for chat.messages[index]."""
    message = chat.messages[index]
    question = next((m["content"] for m in reversed(chat.messages[:index]) if m["role"] == "user"), "")
    entry = {
        "time": time.time(),
        "chat_id": chat.id,
        "rating": rating,
        "question": question,
        "answer": message["content"],
        "sources": (message.get("details") or {}).get("sources", []),
    }
    path = Path(FEEDBACK_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def time_ago(timestamp: float) -> str:
    """'just now', '5 min ago', '3 h ago', '2 days ago'."""
    seconds = max(0, time.time() - timestamp)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    days = int(seconds // 86400)
    return f"{days} day{'s' if days > 1 else ''} ago"
