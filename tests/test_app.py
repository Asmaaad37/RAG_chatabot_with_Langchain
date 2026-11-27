"""Phase 4 tests: the Streamlit app runs end to end (no browser, no Groq).

Uses Streamlit's AppTest with a temporary vector store, so your real index
isn't touched, and a fake LLM, so no API calls are made.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import chat_history
from src import rag_pipeline as rp
from src.generation.prompts import NOT_FOUND_MESSAGE
from src.ingestion.indexer import index_file
from src.vectorstore.chroma_store import VectorStore

from tests.test_ingestion import BREAD, PYTHON, SOLAR, write

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app_store(tmp_path, monkeypatch):
    """Point the app at an empty temporary store and a fake LLM."""
    db = tmp_path / "chroma"
    monkeypatch.setattr(rp, "VectorStore", lambda: VectorStore(path=db, collection_name="app_docs"))
    monkeypatch.setattr(rp, "stream_answer",
                        lambda *a, **k: iter(["Sourdough tastes sour because it ferments slowly ", "[1]."]))
    monkeypatch.setattr(rp, "condense_question", lambda question, history: question)
    monkeypatch.setattr(chat_history, "CHATS_DIR", tmp_path / "chats")  # don't touch real chats
    st.cache_resource.clear()  # forget any pipeline cached by an earlier test
    yield db
    st.cache_resource.clear()


def run_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_empty_app_asks_for_documents(app_store):
    at = run_app()
    assert "No documents yet" in " ".join(m.value for m in at.markdown)
    assert at.chat_input[0].disabled  # nothing to chat with yet


def test_question_shows_answer_sources_and_details(app_store, tmp_path, embedder):
    store = VectorStore(path=app_store, collection_name="app_docs")
    index_file(write(tmp_path, "topics.txt", "\n\n".join([SOLAR, BREAD, PYTHON])),
               store, embedder, chunk_size=300, chunk_overlap=0)

    at = run_app()
    assert not at.button(key="card_0").disabled  # starter cards are clickable once docs exist

    at.chat_input[0].set_value("Why is sourdough bread sour?").run()
    assert not at.exception, at.exception

    assistant = at.chat_message[1]
    page_text = " ".join(m.value for m in assistant.markdown)
    assert "ferments slowly" in page_text
    assert "source-chip" in page_text and "topics.txt" in page_text  # source chip
    assert at.chat_message[1].expander[0].label == "🔍 Under the hood"
    assert "similarity" in " ".join(m.value for m in at.chat_message[1].expander[0].markdown)


def test_out_of_scope_question_in_the_app(app_store, tmp_path, embedder, monkeypatch):
    store = VectorStore(path=app_store, collection_name="app_docs")
    index_file(write(tmp_path, "topics.txt", "\n\n".join([SOLAR, BREAD, PYTHON])),
               store, embedder, chunk_size=300, chunk_overlap=0)

    def no_llm(*args, **kwargs):
        raise AssertionError("The LLM should not have been called")

    monkeypatch.setattr(rp, "stream_answer", no_llm)

    at = run_app()
    at.chat_input[0].set_value("Who won the football World Cup in 1998?").run()
    assert not at.exception, at.exception
    assert NOT_FOUND_MESSAGE in " ".join(m.value for m in at.chat_message[1].markdown)
    assert "LLM not called" in " ".join(m.value for m in at.chat_message[1].expander[0].markdown)


def test_answer_without_citations_still_shows_sources(app_store, tmp_path, embedder, monkeypatch):
    store = VectorStore(path=app_store, collection_name="app_docs")
    index_file(write(tmp_path, "topics.txt", "\n\n".join([SOLAR, BREAD, PYTHON])),
               store, embedder, chunk_size=300, chunk_overlap=0)
    monkeypatch.setattr(rp, "stream_answer", lambda *a, **k: iter(["It ferments slowly."]))  # no [n]

    at = run_app()
    at.chat_input[0].set_value("Why is sourdough bread sour?").run()
    page_text = " ".join(m.value for m in at.chat_message[1].markdown)
    assert "Retrieved" in page_text and "source-chip" in page_text


def test_chats_are_saved_listed_reopened_and_deleted(app_store, tmp_path, embedder):
    store = VectorStore(path=app_store, collection_name="app_docs")
    index_file(write(tmp_path, "topics.txt", "\n\n".join([SOLAR, BREAD, PYTHON])),
               store, embedder, chunk_size=300, chunk_overlap=0)

    # Ask a question via a starter card: a chat is created and saved.
    at = run_app()
    at.button(key="card_0").click().run()
    assert not at.exception, at.exception
    saved = chat_history.ChatStore().list()
    assert len(saved) == 1 and len(saved[0].messages) == 2  # question + answer

    # Back to home: the chat is in History.
    at.button(key="back").click().run()
    assert saved[0].title in " ".join(m.value for m in at.markdown)

    # Reopen it from History.
    at.button(key=f"open_{saved[0].id}").click().run()
    assert len(at.chat_message) == 2

    # Delete it from the ⋮ menu.
    at.button(key="menu_delete").click().run()
    assert chat_history.ChatStore().list() == []


def test_feedback_is_saved_and_regenerate_replaces_the_last_answer(app_store, tmp_path, embedder, monkeypatch):
    import json

    store = VectorStore(path=app_store, collection_name="app_docs")
    index_file(write(tmp_path, "topics.txt", "\n\n".join([SOLAR, BREAD, PYTHON])),
               store, embedder, chunk_size=300, chunk_overlap=0)
    monkeypatch.setattr(chat_history, "FEEDBACK_FILE", tmp_path / "feedback.jsonl")

    at = run_app()
    at.chat_input[0].set_value("Why is sourdough bread sour?").run()
    chat_id = chat_history.ChatStore().list()[0].id

    # 👍 on the answer: stored in the chat and in the feedback log.
    at.feedback(key=f"feedback_{chat_id}_1").set_value(1).run()
    assert not at.exception, at.exception
    assert chat_history.ChatStore().load(chat_id).messages[1]["feedback"] == "up"
    logged = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert logged["rating"] == "up" and logged["question"] == "Why is sourdough bread sour?"

    # Regenerate: the answer is replaced (not added), with a new reply from the fake LLM.
    monkeypatch.setattr(rp, "stream_answer", lambda *a, **k: iter(["A second answer [1]."]))
    at.button(key="regen_1").click().run()
    assert not at.exception, at.exception
    messages = chat_history.ChatStore().load(chat_id).messages
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "A second answer [1]."
    assert messages[1].get("feedback") is None  # the old 👍 doesn't carry over
    assert at.feedback(key=f"feedback_{chat_id}_1").value is None


def test_voice_mode_opens_and_closes(app_store, tmp_path, embedder):
    store = VectorStore(path=app_store, collection_name="app_docs")
    index_file(write(tmp_path, "topics.txt", "\n\n".join([SOLAR, BREAD, PYTHON])),
               store, embedder, chunk_size=300, chunk_overlap=0)

    at = run_app()
    at.button(key="card_voice").click().run()
    assert not at.exception, at.exception
    page = " ".join(m.value for m in at.markdown)
    assert 'class="orb' in page and "Ask me anything" in page
    assert len(at.chat_input) == 0  # the voice screen has the mic instead of the text box

    at.button(key="voice_exit").click().run()
    assert not at.exception, at.exception
    assert "home-title" in " ".join(m.value for m in at.markdown)


def test_welcome_splash_shows_once_with_the_users_name(app_store, monkeypatch):
    import dataclasses

    from src.ui import components

    monkeypatch.setattr(components, "settings", dataclasses.replace(components.settings, user_name="Sam"))
    at = run_app()
    first = " ".join(m.value for m in at.markdown)
    assert "welcome-splash" in first and "Welcome back," in first and "Sam" in first
    assert "Welcome back, Sam 👋" in first            # the home greeting too
    assert "My AI" in first                               # renamed app (sidebar logo)

    at.run()  # any later interaction: the splash doesn't play again
    assert "welcome-splash" not in " ".join(m.value for m in at.markdown)


def test_greeting_without_a_name(app_store, monkeypatch):
    import dataclasses

    from src.ui import components

    monkeypatch.setattr(components, "settings", dataclasses.replace(components.settings, user_name=""))
    at = run_app()
    page = " ".join(m.value for m in at.markdown)
    assert "Welcome back 👋" in page and "Welcome back," not in page


def _app_with_two_saved_chats(tmp_path, monkeypatch):
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text('{"rating": "up"}\n', encoding="utf-8")
    monkeypatch.setattr(chat_history, "FEEDBACK_FILE", feedback)
    store = chat_history.ChatStore()
    first, second = chat_history.new_chat("First chat?"), chat_history.new_chat("Second chat?")
    store.save(first)
    store.save(second)
    return run_app(), first, second, feedback


def test_delete_one_chat_from_history(app_store, tmp_path, monkeypatch):
    at, first, second, _ = _app_with_two_saved_chats(tmp_path, monkeypatch)
    at.button(key=f"del_chat_{first.id}").click().run()
    assert not at.exception, at.exception
    assert [c.id for c in chat_history.ChatStore().list()] == [second.id]
    assert "First chat?" not in " ".join(m.value for m in at.markdown)


def test_clear_all_history_asks_first(app_store, tmp_path, monkeypatch):
    at, _, _, feedback = _app_with_two_saved_chats(tmp_path, monkeypatch)

    # Cancel keeps everything.
    at.button(key="clear_history").click().run()
    assert "all 2 saved chats" in " ".join(m.value for m in at.markdown)
    at.button(key="clear_cancel").click().run()
    assert len(chat_history.ChatStore().list()) == 2

    # Confirm without the box: chats go, ratings stay.
    at.button(key="clear_history").click().run()
    at.button(key="clear_confirm").click().run()
    assert not at.exception, at.exception
    assert chat_history.ChatStore().list() == []
    assert feedback.exists()
    assert "Your chats will appear here." in " ".join(c.value for c in at.caption)


def test_clear_all_history_can_also_delete_ratings(app_store, tmp_path, monkeypatch):
    at, _, _, feedback = _app_with_two_saved_chats(tmp_path, monkeypatch)
    at.button(key="clear_history").click().run()
    at.checkbox(key="clear_feedback").check()      # tick the box and click
    at.button(key="clear_confirm").click().run()   # "Delete all" in the same run
    assert not at.exception, at.exception
    assert chat_history.ChatStore().list() == []
    assert not feedback.exists()
