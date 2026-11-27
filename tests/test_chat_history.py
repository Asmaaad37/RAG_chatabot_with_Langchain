"""Tests for saving and loading chats (src/chat_history.py)."""

import time

from src.chat_history import ChatStore, new_chat, time_ago


def test_save_load_list_delete(tmp_path):
    store = ChatStore(tmp_path)
    older = new_chat("First question?")
    older.messages.append({"role": "user", "content": "First question?"})
    store.save(older)
    time.sleep(0.01)
    newer = new_chat("Second question?")
    store.save(newer)

    assert [c.id for c in store.list()] == [newer.id, older.id]  # most recent first
    assert store.load(older.id).messages[0]["content"] == "First question?"

    store.delete(older.id)
    assert [c.id for c in store.list()] == [newer.id]


def test_title_is_tidied_and_shortened():
    chat = new_chat("  What   is\nthis?  " + "x" * 200)
    assert chat.title.startswith("What is this?") and chat.title.endswith("…")
    assert len(chat.title) <= 81


def test_search_matches_title_and_messages():
    chat = new_chat("Group 4 students")
    chat.messages.append({"role": "assistant", "content": "Sara Khan, Omar Ali"})
    assert chat.matches("group 4") and chat.matches("omar") and chat.matches("")
    assert not chat.matches("network layer")


def test_broken_files_are_skipped(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert ChatStore(tmp_path).list() == []


def test_time_ago():
    now = time.time()
    assert time_ago(now) == "just now"
    assert time_ago(now - 300) == "5 min ago"
    assert time_ago(now - 3 * 3600) == "3 h ago"
    assert time_ago(now - 2 * 86400) == "2 days ago"


def test_delete_all_and_feedback(tmp_path, monkeypatch):
    from src import chat_history

    store = ChatStore(tmp_path / "chats")
    for q in ["One?", "Two?", "Three?"]:
        store.save(new_chat(q))
    assert store.delete_all() == 3
    assert store.list() == []

    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text('{"rating": "up"}\n', encoding="utf-8")
    monkeypatch.setattr(chat_history, "FEEDBACK_FILE", feedback)
    chat_history.delete_feedback()
    assert not feedback.exists()
    chat_history.delete_feedback()  # deleting again is fine
