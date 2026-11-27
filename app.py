"""My AI: chat with your documents (Streamlit UI).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from config.settings import APP_NAME, settings, setup_logging
from src.generation.generator import GenerationError
from src.generation.prompts import VOICE_STYLE
from src.ingestion.loader import DocumentLoadError
from src.chat_history import Chat, ChatStore, delete_feedback, log_feedback, new_chat, time_ago
from src.rag_pipeline import RAGPipeline
from src.ui import components as ui
from src.ui.styles import inject_css
from src.voice.transcriber import TranscriptionError, transcribe

st.set_page_config(page_title=f"{APP_NAME} · Chat with your documents", page_icon="✦",
                   initial_sidebar_state="expanded")
setup_logging()
inject_css()
logger = logging.getLogger("app")

# Starter cards on the home screen: (card text, question it asks)
STARTERS = [
    ("Summarise my documents", "Summarise the main ideas of my documents."),
    ("Key findings", "What are the key findings or main points?"),
    ("Explain simply", "Explain the main topic in simple terms, as if I'm new to it."),
]
HISTORY_PREVIEW = 5  # chats shown before "See all"

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/auto_awesome:"


@st.cache_resource(show_spinner="Loading the embedding model (only the first time)...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def init_state() -> None:
    st.session_state.setdefault("chat", None)            # the open Chat, or None = home screen
    st.session_state.setdefault("welcomed", False)       # the welcome animation has played
    st.session_state.setdefault("uploader_key", 0)       # changing it empties the uploader
    st.session_state.setdefault("index_notices", [])     # results to show after indexing
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("show_all_history", False)
    st.session_state.setdefault("regenerate", False)     # True = re-answer the last question
    st.session_state.setdefault("voice_mode", False)     # True = show the voice screen
    st.session_state.setdefault("voice_turn", 0)         # bumped each turn to reset the recorder
    st.session_state.setdefault("voice_autoplay", False)  # speak the newest answer once


# ---------------------------------------------------------------- navigation (button callbacks)

def go_home() -> None:
    st.session_state.chat = None


def open_chat(chat_id: str) -> None:
    st.session_state.chat = ChatStore().load(chat_id)


def ask_starter(question: str) -> None:
    st.session_state.pending_question = question


def delete_current_chat() -> None:
    chat = st.session_state.chat
    if chat:
        ChatStore().delete(chat.id)
    st.session_state.chat = None
    st.toast("Chat deleted")


def enter_voice_mode() -> None:
    st.session_state.voice_mode = True


def exit_voice_mode() -> None:
    st.session_state.voice_mode = False  # the speaker button disappears, which stops its speech


def delete_chat(chat_id: str) -> None:
    """🗑 on a History row."""
    ChatStore().delete(chat_id)
    if st.session_state.chat and st.session_state.chat.id == chat_id:
        st.session_state.chat = None
    st.toast("Chat deleted")


def clear_all_history() -> None:
    """'Delete all' in the confirmation dialog. Runs as a button callback, i.e. before the
    page redraws, so it works the same whether or not the dialog is drawn again."""
    also_feedback = st.session_state.get("clear_feedback", False)
    deleted = ChatStore().delete_all()
    if also_feedback:
        delete_feedback()
    st.session_state.chat = None
    st.session_state.show_all_history = False
    st.toast(f"Deleted {deleted} chat{'s' if deleted != 1 else ''}"
             + (" and your ratings" if also_feedback else ""))


@st.dialog("Clear all history?", icon=":material/delete_sweep:")
def confirm_clear_history(count: int) -> None:
    st.write(f"This deletes **all {count} saved chat{'s' if count != 1 else ''}**. It can't be undone.")
    st.checkbox("Also delete my 👍/👎 ratings", key="clear_feedback")
    cancel, confirm = st.columns(2)
    with cancel:
        if st.button("Cancel", key="clear_cancel", width="stretch"):
            st.rerun()  # closes the dialog
    with confirm:
        if st.button("Delete all", key="clear_confirm", type="primary", width="stretch",
                     on_click=clear_all_history):
            st.rerun()  # closes the dialog


def toggle_history() -> None:
    st.session_state.show_all_history = not st.session_state.show_all_history


def rate_answer(index: int, key: str) -> None:
    """Save a 👍/👎 click on chat.messages[index] (st.feedback gives 1 = up, 0 = down)."""
    chat = st.session_state.chat
    value = st.session_state.get(key)
    rating = {1: "up", 0: "down"}.get(value)
    chat.messages[index]["feedback"] = rating
    ChatStore().save(chat)
    log_feedback(chat, index, rating)
    if rating:
        st.toast("Thanks for the feedback!")


def regenerate_last_answer() -> None:
    """Drop the last answer; the next run answers the same question again."""
    chat = st.session_state.chat
    if chat and chat.messages and chat.messages[-1]["role"] == "assistant":
        chat.messages.pop()
        # The new answer gets the same position, so clear the old answer's 👍/👎.
        st.session_state.pop(f"feedback_{chat.id}_{len(chat.messages)}", None)
        st.session_state.regenerate = True


# ---------------------------------------------------------------- sidebar

def index_uploads(pipeline: RAGPipeline, files, chunk_size: int, chunk_overlap: int) -> None:
    notices = []
    progress = st.progress(0.0, text="Starting...")
    for i, upload in enumerate(files):
        name = Path(upload.name).name  # never trust folders in an uploaded name
        dest = settings.uploads_dir / name
        dest.write_bytes(upload.getvalue())

        def on_progress(fraction: float, message: str, i=i, name=name) -> None:
            progress.progress(min((i + fraction) / len(files), 1.0), text=f"{name}: {message}")

        try:
            result = pipeline.index_file(dest, chunk_size, chunk_overlap, on_progress)
            notices.append(("success" if result.status == "indexed" else "info", result.message))
        except DocumentLoadError as e:  # empty, scanned, damaged or unsupported file
            notices.append(("error", str(e)))
            dest.unlink(missing_ok=True)
        except Exception:
            logger.exception("Indexing failed for %s", name)
            notices.append(("error", f"Something went wrong while indexing '{name}'. See the terminal log."))

    st.session_state.index_notices = notices
    st.session_state.uploader_key += 1
    st.rerun()


def sidebar(pipeline: RAGPipeline, docs) -> dict:
    """Draw the sidebar and return the current settings."""
    with st.sidebar:
        ui.brand()
        st.caption("Upload files, then chat with them.")

        # --- upload + index (indexing runs at the end, once the chunk settings are read) ---
        st.subheader("📤 Add documents")
        files = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}", label_visibility="collapsed",
        )
        index_clicked = bool(files) and st.button(
            f"Index {len(files)} file{'s' if len(files) > 1 else ''}", type="primary", width="stretch"
        )
        for kind, message in st.session_state.index_notices:
            getattr(st, kind)(message)
        st.session_state.index_notices = []

        # --- indexed documents ---
        st.subheader(f"📚 Indexed documents ({len(docs)})")
        if not docs:
            st.caption("Nothing indexed yet.")
        for doc in docs:
            card, button = st.columns([5, 1], vertical_alignment="center")
            with card:
                ui.doc_card(doc.source, doc.pages, doc.chunks)
            with button:
                if st.button("", icon=":material/delete:", key=f"delete_{doc.file_hash}",
                             help=f"Remove {doc.source}"):
                    pipeline.delete_document(doc.file_hash)
                    (settings.uploads_dir / doc.source).unlink(missing_ok=True)
                    st.toast(f"Removed {doc.source}")
                    st.rerun()

        # --- settings ---
        with st.expander("⚙️ Settings"):
            top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, settings.top_k)
            hybrid = st.toggle("Hybrid search (meaning + keywords)", settings.hybrid_search,
                               help="Also match exact words, names and numbers. Recommended.")
            use_reranker = st.toggle(
                "Reranker", settings.use_reranker,
                help="Rescore the top candidates with a cross-encoder. Can be more precise, "
                     "but adds 1–2 seconds per question. The first use downloads ~90 MB.",
            )
            threshold = st.slider(
                "Similarity threshold", 0.0, 1.0, settings.score_threshold, 0.01,
                help="A chunk is used if its similarity reaches this. If no chunk qualifies, the app "
                     "replies \"I couldn't find this in your documents\" without calling the LLM.",
            )
            rerank_threshold = None
            if use_reranker:
                rerank_threshold = st.slider(
                    "Reranker threshold", 0.0, 1.0, settings.rerank_threshold, 0.01,
                    help="A chunk is also used if the reranker scores it at least this, "
                         "even when its similarity is lower.",
                )
            temperature = st.slider("Temperature", 0.0, 1.0, settings.temperature, 0.05,
                                    help="Lower = more focused answers.")
            chunk_size = st.slider("Chunk size (characters)", 200, 2000, settings.chunk_size, 50,
                                   help="Applies to files you index from now on.")
            chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, settings.chunk_overlap, 10,
                                      help="Applies to files you index from now on.")
            if chunk_overlap >= chunk_size:
                chunk_overlap = chunk_size // 5
                st.warning(f"Overlap must be smaller than the chunk size, so {chunk_overlap} will be used.")

        st.button("＋ New chat", width="stretch", on_click=go_home,
                  help="Start a fresh chat. The current one stays in History.")

        if index_clicked:
            index_uploads(pipeline, files, chunk_size, chunk_overlap)

    return {"top_k": top_k, "threshold": threshold, "temperature": temperature,
            "hybrid": hybrid, "use_reranker": use_reranker, "rerank_threshold": rerank_threshold}


# ---------------------------------------------------------------- chat

# ---------------------------------------------------------------- home screen

def home_screen(docs, store: ChatStore) -> None:
    ui.home_top(settings.groq_model, len(docs))
    ui.home_heading()

    search = st.text_input("Search chats", placeholder="Search your chats…", key="history_search",
                           label_visibility="collapsed", icon=":material/search:")

    if not docs:
        ui.empty_state()
    cards = st.columns(len(STARTERS) + 1)
    for i, (column, (label, question)) in enumerate(zip(cards, STARTERS)):
        with column:
            st.button(label, key=f"card_{i}", on_click=ask_starter, args=(question,),
                      disabled=not docs, width="stretch")
    with cards[-1]:
        st.button("Talk to your docs 🎙", key="card_voice", on_click=enter_voice_mode,
                  disabled=not docs, width="stretch")

    # --- History ---
    all_chats = store.list()
    chats = [c for c in all_chats if c.matches(search)]
    title_col, link_col, clear_col = st.columns([6, 1, 1], vertical_alignment="center")
    with title_col:
        ui.section_title("History")
    with link_col:
        if len(chats) > HISTORY_PREVIEW:
            label = "Show less" if st.session_state.show_all_history else "See all"
            st.button(label, key="see_all", on_click=toggle_history)
    with clear_col:
        if all_chats and st.button("Clear all", key="clear_history", help="Delete all saved chats"):
            confirm_clear_history(len(all_chats))

    if not chats:
        st.caption("No chats match your search." if search else "Your chats will appear here.")
    shown = chats if st.session_state.show_all_history else chats[:HISTORY_PREVIEW]
    for chat in shown:
        row, bin_col, arrow = st.columns([12, 1, 1], vertical_alignment="center")
        with row:
            questions = f"{chat.question_count} question{'s' if chat.question_count != 1 else ''}"
            ui.history_item(chat.title, f"{time_ago(chat.updated)} · {questions}")
        with bin_col:
            st.button("", icon=":material/delete:", key=f"del_chat_{chat.id}", on_click=delete_chat,
                      args=(chat.id,), help="Delete this chat")
        with arrow:
            st.button("›", key=f"open_{chat.id}", on_click=open_chat, args=(chat.id,),
                      help="Open this chat")


# ---------------------------------------------------------------- chat screen

def chat_header(chat: Chat) -> None:
    with st.container(key="chat_header"):
        back, title, voice, menu = st.columns([1, 11, 1, 1], vertical_alignment="center")
        with back:
            st.button("‹", key="back", on_click=go_home, help="Back to home")
        with title:
            ui.chat_head(chat.title)
        with voice:
            st.button("", icon=":material/graphic_eq:", key="voice_btn", on_click=enter_voice_mode,
                      help="Voice mode: talk to your documents")
        with menu:
            with st.popover("", icon=":material/more_vert:", key="chat_menu", help="More"):
                st.button("＋ New chat", key="menu_new", on_click=go_home, width="stretch")
                st.button("🗑 Delete this chat", key="menu_delete", on_click=delete_current_chat,
                          width="stretch")


def action_row(chat: Chat, index: int) -> None:
    """Copy · 👍 👎 · Read aloud · Regenerate, under an answer (like the mockup)."""
    message = chat.messages[index]
    is_last = index == len(chat.messages) - 1
    with st.container(key=f"actions_{index}", horizontal=True, vertical_alignment="center", gap="small"):
        if not message.get("error"):
            ui.copy_button(ui.text_to_copy(message))
            key = f"feedback_{chat.id}_{index}"
            default = {"up": 1, "down": 0}.get(message.get("feedback"))
            st.feedback("thumbs", key=key, default=default, on_change=rate_answer, args=(index, key))
            ui.speaker(message["content"])  # 🔊 read aloud
        if is_last:  # only the latest answer can be regenerated
            st.button("Regenerate" if not message.get("error") else "Try again",
                      key=f"regen_{index}", icon=":material/refresh:", type="tertiary",
                      on_click=regenerate_last_answer, help="Answer this question again")


def show_message(chat: Chat, index: int) -> None:
    message = chat.messages[index]
    role = message["role"]
    with st.container(key=f"msg_{role}_{index}"):
        with st.chat_message(role, avatar=USER_AVATAR if role == "user" else ASSISTANT_AVATAR):
            if message.get("error"):
                st.error(message["content"])
            elif message.get("voice"):  # spoken question: show a small mic icon
                st.markdown(f":material/mic: {message['content']}")
            else:
                st.markdown(message["content"])
                if message.get("details"):
                    ui.source_chips(message["details"])
                    ui.under_the_hood(message["details"])
            if role == "assistant":
                action_row(chat, index)


def answer(pipeline: RAGPipeline, chat: Chat, question: str, options: dict) -> None:
    # History = earlier turns only, without error messages.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in chat.messages[:-1]
        if not m.get("error")
    ]
    with st.container(key=f"msg_assistant_{len(chat.messages)}"):
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            typing = st.empty()
            typing.markdown(ui.TYPING, unsafe_allow_html=True)  # "..." while searching
            try:
                response = pipeline.ask(question, history=history, **ask_options(options))
                stream = response.stream()
                first = next(stream, "")  # wait for the first words before showing the answer
                typing.empty()
                text = st.write_stream(_chain(first, stream))
            except GenerationError as e:  # missing key, rate limit, connection, ...
                typing.empty()
                st.error(str(e))
                chat.messages.append({"role": "assistant", "content": str(e), "error": True})
                action_row(chat, len(chat.messages) - 1)
                return

            if not str(text).strip():
                message = "The model returned an empty answer. Please try again."
                st.warning(message)
                chat.messages.append({"role": "assistant", "content": message, "error": True})
                action_row(chat, len(chat.messages) - 1)
                return

            details = ui.response_details(response)
            ui.source_chips(details)
            ui.under_the_hood(details)
            chat.messages.append({"role": "assistant", "content": response.answer, "details": details})
            action_row(chat, len(chat.messages) - 1)


def _chain(first: str, rest):
    if first:
        yield first
    yield from rest


def read_chat_input(submitted) -> tuple[str | None, bool]:
    """Turn the chat box's value into (question, spoken?).

    With accept_audio=True, st.chat_input returns an object with .text and .audio.
    Typed text wins; otherwise a recording is transcribed with Groq Whisper.
    """
    if not submitted:
        return None, False
    text = (submitted.text or "").strip()
    if text:
        return text, False
    if submitted.audio is None:
        return None, False
    with st.spinner("Listening…"):
        try:
            return transcribe(submitted.audio.getvalue()), True
        except TranscriptionError as e:
            st.toast(str(e), icon=":material/mic_off:")
            return None, False


# ---------------------------------------------------------------- voice mode

def last_turn(chat: Chat | None) -> tuple[str | None, dict | None]:
    """The latest question and its answer message (None if not answered yet)."""
    if not chat or not chat.messages:
        return None, None
    reply = chat.messages[-1] if chat.messages[-1]["role"] == "assistant" else None
    question = next((m["content"] for m in reversed(chat.messages) if m["role"] == "user"), None)
    return question, reply


def voice_screen(pipeline: RAGPipeline, docs, store: ChatStore, options: dict) -> None:
    """Right-hand phone of the mockup: orb, the last exchange, and replay · mic · close."""
    chat: Chat | None = st.session_state.chat
    turn = st.session_state.voice_turn

    with st.container(key="chat_header"):
        back, title = st.columns([1, 13], vertical_alignment="center")
        with back:
            st.button("‹", key="back", on_click=exit_voice_mode, help="Leave voice mode")
        with title:
            ui.chat_head(chat.title if chat else "Voice mode · talk to your documents")

    orb_slot, text_slot = st.empty(), st.empty()
    question, reply = last_turn(chat)
    with orb_slot:
        ui.orb("idle")
    with text_slot:
        if reply and reply.get("error"):
            ui.voice_text(question, None, reply["content"])
        else:
            ui.voice_text(question, reply["content"] if reply else None,
                          "Tap the mic, ask your question, then submit." if docs else "Upload a document first.")

    with st.container(key="voice_controls"):
        left, middle, right = st.columns([1, 5, 1], vertical_alignment="center")
        with middle:
            with st.container(key=f"voice_mic_{turn}"):
                audio = st.audio_input("Ask with your voice", key=f"voice_rec_{turn}",
                                       label_visibility="collapsed", disabled=not docs)
        with right:
            st.button("", icon=":material/close:", key="voice_exit", on_click=exit_voice_mode,
                      help="Leave voice mode")
        with left:
            if reply and not reply.get("error"):
                # Speaks the new answer automatically once; afterwards it's a replay button.
                ui.speaker(reply["content"], autoplay=st.session_state.voice_autoplay, big=True, nonce=turn)
    st.session_state.voice_autoplay = False

    if audio is None:
        return

    # ---- a new recording: transcribe, answer, save, then redraw ----
    with orb_slot:
        ui.orb("thinking")
    with text_slot:
        ui.voice_text(None, None, "Listening…")
    try:
        heard = transcribe(audio.getvalue())
    except TranscriptionError as e:
        st.toast(str(e), icon=":material/mic_off:")
        st.session_state.voice_turn += 1  # fresh recorder
        st.rerun()

    if chat is None:
        chat = st.session_state.chat = new_chat(heard)
    chat.messages.append({"role": "user", "content": heard, "voice": True})
    with text_slot:
        ui.voice_text(heard, None, "Thinking…")

    history = [{"role": m["role"], "content": m["content"]}
               for m in chat.messages[:-1] if not m.get("error")]
    try:
        response = pipeline.ask(heard, history=history, style=VOICE_STYLE, **ask_options(options))
        text = response.text()
        if text.strip():
            message = {"role": "assistant", "content": response.answer,
                       "details": ui.response_details(response)}
        else:
            message = {"role": "assistant", "content": "The model returned an empty answer. Please try again.",
                       "error": True}
    except GenerationError as e:
        message = {"role": "assistant", "content": str(e), "error": True}
    chat.messages.append(message)
    store.save(chat)

    st.session_state.voice_turn += 1
    st.session_state.voice_autoplay = not message.get("error")
    st.rerun()


def ask_options(options: dict) -> dict:
    """Sidebar settings as keyword arguments for RAGPipeline.ask()."""
    return dict(top_k=options["top_k"], score_threshold=options["threshold"],
                temperature=options["temperature"], hybrid=options["hybrid"],
                use_reranker=options["use_reranker"], rerank_threshold=options["rerank_threshold"])


# ---------------------------------------------------------------- page

def main() -> None:
    init_state()
    if not st.session_state.welcomed:  # first run of this visit: "Welcome back, <name>"
        ui.welcome_splash()
        st.session_state.welcomed = True
    try:
        pipeline = get_pipeline()
    except Exception as e:
        logger.exception("Startup failed")
        st.error(f"The app couldn't start: {e}")
        st.stop()

    store = ChatStore()
    docs = pipeline.list_documents()
    options = sidebar(pipeline, docs)

    if not settings.has_api_key:
        st.warning("No Groq API key found. Copy `.env.example` to `.env`, add your free key from "
                   "https://console.groq.com/keys, then restart the app.")

    if st.session_state.voice_mode:
        voice_screen(pipeline, docs, store, options)
        return

    # A question can come from the chat box (typed or spoken) or from a starter card.
    submitted = st.chat_input("Send a message…", disabled=not docs, accept_audio=True)
    typed, spoken = read_chat_input(submitted)
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None
    if question:
        if st.session_state.chat is None:  # first question starts a new saved chat
            st.session_state.chat = new_chat(question)
        message = {"role": "user", "content": question}
        if spoken:
            message["voice"] = True
        st.session_state.chat.messages.append(message)

    chat: Chat | None = st.session_state.chat
    if chat is None:
        home_screen(docs, store)
        return

    # "Regenerate" removed the last answer: answer the last question again.
    if st.session_state.regenerate and chat.messages and chat.messages[-1]["role"] == "user":
        question = chat.messages[-1]["content"]
    st.session_state.regenerate = False

    chat_header(chat)
    for i in range(len(chat.messages)):
        show_message(chat, i)
    if question:
        try:
            answer(pipeline, chat, question, options)
        finally:
            store.save(chat)  # save even if something failed, so the question isn't lost


main()
