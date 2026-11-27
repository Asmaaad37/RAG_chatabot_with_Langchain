"""Small HTML building blocks for the Streamlit UI.

Everything user- or document-provided goes through html.escape() before it is
put into HTML, so file names or chunk text can't break the page.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from config.settings import APP_NAME, settings
from src.generation.prompts import NOT_FOUND_MESSAGE
from src.rag_pipeline import RAGResponse


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


LOGO = '<div class="logo-dot">✦</div>'
TYPING = '<div class="typing"><span></span><span></span><span></span></div>'


# ---------- home screen ----------

def home_top(model: str, doc_count: int) -> None:
    """Logo on the left, a black pill with the model on the right (like the mockup)."""
    docs = f"{doc_count} doc{'s' if doc_count != 1 else ''}"
    _html(f'<div class="top-row">{LOGO}<span class="model-pill">⚡ {escape(model)} · {docs}</span></div>')


def welcome_text() -> str:
    """'Welcome back, Sam' (name from USER_NAME in .env), or just 'Welcome back'."""
    name = settings.user_name.strip()
    return f"Welcome back, {name}" if name else "Welcome back"


def welcome_splash() -> None:
    """Full-screen "Welcome back, <name>" that rises in, then fades away (CSS only).

    It never blocks clicks (pointer-events: none), even while visible.
    """
    name = settings.user_name.strip()
    line1 = "Welcome back," if name else "Welcome back"
    name_html = f'<div class="welcome-name">{escape(name)}</div>' if name else ""
    with st.container(key="welcome_splash"):
        _html(
            '<div class="welcome-splash"><div class="welcome-inner">'
            f'<div class="welcome-logo">{LOGO}</div>'
            f'<div class="welcome-line">{escape(line1)}</div>{name_html}'
            "</div></div>"
        )


def home_heading() -> None:
    _html(
        f'<div class="home-eyebrow">{escape(welcome_text())} 👋</div>'
        '<div class="home-title">Ask, explore,<br><span>understand your docs</span></div>'
    )


def section_title(text: str) -> None:
    _html(f'<div class="section-title">{escape(text)}</div>')


def history_item(title: str, subtitle: str) -> None:
    _html(
        '<div class="hist-item"><div class="hist-icon">✦</div><div class="hist-text">'
        f'<div class="hist-title">{escape(title)}</div>'
        f'<div class="hist-sub">{escape(subtitle)}</div>'
        "</div></div>"
    )


def empty_state() -> None:
    _html(
        '<div class="empty-card">'
        "<b>No documents yet.</b><br>"
        "Upload a PDF or TXT file in the sidebar, click <b>Index</b>, then ask your first question."
        "</div>"
    )


COPY_ICON = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" '
             'width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>')
CHECK_ICON = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#E0459B" '
              'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>')


def js_string(text: str) -> str:
    """Text as a JavaScript string literal that is safe inside a <script> tag.

    json.dumps escapes quotes and backslashes; < > & are escaped too, so the text
    can't close the script tag or be read as HTML.
    """
    import json

    return (json.dumps(text).replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))


def copy_button(text: str) -> None:
    """A small copy-to-clipboard button.

    Streamlit has no built-in one, so this is a tiny HTML page in its own iframe
    (st.iframe), which Streamlit allows to write to the clipboard.
    The answer text is inserted only as an escaped JavaScript string, never as HTML.
    """
    payload = js_string(text)
    st.iframe(
        f"""
<style>
  html, body {{ margin: 0; background: transparent; overflow: hidden; }}
  button {{ background: transparent; border: none; color: #8E8E9A; cursor: pointer;
           width: 32px; height: 32px; border-radius: 8px; display: grid; place-items: center; }}
  button:hover {{ color: #F5F5F7; background: rgba(255,255,255,.06); }}
</style>
<button id="copy" title="Copy answer">{COPY_ICON}</button>
<script>
  const text = {payload};
  const button = document.getElementById("copy");
  button.onclick = async () => {{
    try {{
      await navigator.clipboard.writeText(text);
    }} catch (e) {{  // older browsers: copy via a hidden text box
      const box = document.createElement("textarea");
      box.value = text; document.body.appendChild(box); box.select();
      document.execCommand("copy"); box.remove();
    }}
    button.innerHTML = '{CHECK_ICON}'; button.title = "Copied!";
    setTimeout(() => {{ button.innerHTML = '{COPY_ICON}'; button.title = "Copy answer"; }}, 1500);
  }};
</script>
""",
        height=32,
        width=34,
    )


SPEAKER_ICON = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 5 6 9H2v6h4l5 4V5z"/>'
                '<path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M19 5a10 10 0 0 1 0 14"/></svg>')
STOP_ICON = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="#E0459B"><rect x="6" y="6" '
             'width="12" height="12" rx="2"/></svg>')
REPLAY_ICON = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
               'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7"/>'
               '<path d="M3 4v5h5"/></svg>')


def speaker(text: str, autoplay: bool = False, big: bool = False, nonce: int = 0) -> None:
    """A button that reads `text` aloud with the browser's built-in voice (free, offline).

    Browsers only allow speech after the user has clicked on the page. This button
    lives in its own iframe, so it uses the *main page's* speech engine (same origin),
    which has seen the user's click (e.g. on the mic). That's what lets voice mode
    speak answers automatically. Click again to stop; speech also stops when the
    button is removed from the page (e.g. when leaving voice mode).
    """
    from src.voice.speech import clean_for_speech

    size = 52 if big else 32
    icon, title = (REPLAY_ICON, "Play the answer again") if big else (SPEAKER_ICON, "Read aloud")
    st.iframe(
        f"""
<style>
  html, body {{ margin: 0; background: transparent; overflow: hidden; }}
  button {{ background: {'#17171F' if big else 'transparent'}; border: {'1px solid rgba(255,255,255,.12)' if big else 'none'};
           color: {'#F5F5F7' if big else '#8E8E9A'}; cursor: pointer; width: {size}px; height: {size}px;
           border-radius: {'50%' if big else '8px'}; display: grid; place-items: center; }}
  button:hover {{ color: #F5F5F7; border-color: #E0459B; }}
</style>
<button id="speak" title="{title}">{icon}</button>
<!-- {nonce} -->
<script>
  const text = {js_string(clean_for_speech(text))};
  const button = document.getElementById("speak");
  // Prefer the main page's speech engine (it has the user's click); fall back to our own.
  let host = window;
  try {{ if (window.parent.speechSynthesis) host = window.parent; }} catch (e) {{}}
  const synth = host.speechSynthesis;
  let speaking = false;  // shown as playing (updates when the browser starts/stops)
  let requested = false; // we asked to speak and it hasn't finished yet
  const show = (on) => {{ speaking = on; button.innerHTML = on ? '{STOP_ICON}' : '{icon}'; }};
  function speak() {{
    synth.cancel();
    const u = new host.SpeechSynthesisUtterance(text);
    const voices = synth.getVoices().filter(v => v.lang && v.lang.startsWith("en"));
    u.voice = voices.find(v => /natural|google|online/i.test(v.name)) || voices[0] || null;
    u.onstart = () => show(true);
    u.onend = () => {{ requested = false; show(false); }};
    u.onerror = (e) => {{ requested = false; show(false); document.body.dataset.error = e.error; }};
    requested = true;
    synth.speak(u);
    document.body.dataset.started = "1";
  }}
  button.onclick = () => speaking ? (synth.cancel(), show(false)) : speak();
  // When this button disappears (e.g. leaving voice mode), stop its speech.
  window.addEventListener("pagehide", () => {{ if (requested) synth.cancel(); }});
  {"setTimeout(speak, 150);" if autoplay else ""}
</script>
""",
        height=size,
        width=size + 2,
    )


def orb(state: str = "idle") -> None:
    """The glowing animated orb from the mockup. state: "idle" or "thinking"."""
    _html(f'<div class="orb-wrap"><div class="orb {escape(state)}"><span></span><span></span></div></div>')


def voice_text(question: str | None, answer: str | None, status: str) -> None:
    """Centered text under the orb: the last question (white) and answer (dimmed)."""
    from src.voice.speech import clean_for_speech

    parts = ['<div class="voice-text">']
    if question:
        parts.append(f'<div class="voice-q">{escape(question)}</div>')
    if answer:
        parts.append(f'<div class="voice-a">{escape(clean_for_speech(answer))}</div>')
    if not question and not answer:
        parts.append('<div class="voice-q">Ask me anything about your documents</div>')
    parts.append(f'<div class="voice-status">{escape(status)}</div></div>')
    _html("".join(parts))


def text_to_copy(message: dict) -> str:
    """The answer plus its sources, ready to paste elsewhere."""
    sources = (message.get("details") or {}).get("sources") or []
    return message["content"] + ("\n\nSources:\n" + "\n".join(sources) if sources else "")


def chat_head(title: str) -> None:
    _html(
        f'<div class="chat-head">{LOGO}<div style="min-width:0">'
        f'<div class="chat-head-title">{escape(APP_NAME)}</div>'
        f'<div class="chat-head-sub">{escape(title)}</div>'
        "</div></div>"
    )


def doc_card(name: str, pages: int, chunks: int) -> None:
    _html(
        f'<div class="doc-card" title="{escape(name)}">'
        f'<div class="doc-name">📄 {escape(name)}</div>'
        f'<div class="doc-meta">{pages} page{"s" if pages != 1 else ""} · {chunks} chunks</div>'
        "</div>"
    )


def brand() -> None:
    _html(f'<div class="brand">{LOGO}<span>{escape(APP_NAME)}</span></div>')


# ---------- answers ----------

def response_details(response: RAGResponse) -> dict:
    """Everything the UI needs to redraw an answer later, as plain data."""
    hits = []
    if response.retrieval:
        number = 0
        for h in response.retrieval.hits:
            used = response.used_llm and response.retrieval.passes(h)
            number += used
            hits.append({
                "number": number if used else None, "source": h.source, "page": h.page,
                "score": h.score, "rerank_score": h.rerank_score, "text": h.text, "used": used,
            })
    # Chips show the sources the answer cites. If the model forgot to cite,
    # fall back to the chunks it was given (labelled "Retrieved" so it's honest).
    sources, sources_label = [s.label for s in response.cited_sources], "Sources"
    if not sources and response.used_llm and not response.answer.startswith(NOT_FOUND_MESSAGE):
        sources, sources_label = [s.label for s in response.sources], "Retrieved"
    return {
        "sources": sources,
        "sources_label": sources_label,
        "hits": hits,
        "threshold": response.retrieval.threshold if response.retrieval else None,
        "rerank_threshold": response.retrieval.rerank_threshold if response.retrieval else None,
        "hybrid": bool(response.retrieval and response.retrieval.hybrid),
        "reranked": bool(response.retrieval and response.retrieval.reranked),
        "question": response.question,
        "search_query": response.search_query,
        "model": response.model,
        "temperature": response.temperature,
        "response_time": response.response_time,
        "used_llm": response.used_llm,
    }


def source_chips(details: dict) -> None:
    if not details["sources"]:
        return
    chips = []
    for label in details["sources"]:  # "[1] file.pdf, p. 3"
        number, _, rest = label.partition(" ")
        chips.append(f'<span class="source-chip"><b>{escape(number)}</b> {escape(rest)}</span>')
    label = escape(details.get("sources_label", "Sources"))
    _html(f'<div class="source-row"><span class="source-label">{label}</span>{"".join(chips)}</div>')


def under_the_hood(details: dict) -> None:
    with st.expander("🔍 Under the hood"):
        time_text = f"{details['response_time']:.2f}s" if details["response_time"] is not None else "–"
        used = sum(h["used"] for h in details["hits"])
        pills = [
            f"⚡ {escape(details['model'])}",
            f"🌡️ temperature {details['temperature']:.2f}",
            f"⏱️ {time_text}",
            f"🧩 {used}/{len(details['hits'])} chunks used",
        ]
        if details["threshold"] is not None:
            pills.append(f"🔎 {'hybrid search' if details.get('hybrid') else 'embedding search'}")
            if details.get("reranked"):
                pills.append("🏅 reranked")
            target = f"🎯 similarity ≥ {details['threshold']:.2f}"
            if details.get("rerank_threshold") is not None:
                target += f" or relevance ≥ {details['rerank_threshold']:.2f}"
            pills.append(target)
        if not details["used_llm"]:
            pills.append("🚫 LLM not called")
        _html('<div class="pill-row">' + "".join(f'<span class="meta-pill">{p}</span>' for p in pills) + "</div>")

        if details["search_query"] != details["question"]:
            _html(f'<p class="query-note">Follow-up rewritten for search: '
                  f'<i>“{escape(details["search_query"])}”</i></p>')

        for h in details["hits"]:
            tag = f"[{h['number']}]" if h["used"] else "not used"
            rerank = h.get("rerank_score")
            # The bar shows the score the threshold is checked against.
            width = max(0.0, min(1.0, rerank if rerank is not None else h["score"])) * 100
            scores = f'similarity <b>{h["score"]:.3f}</b>'
            if rerank is not None:
                scores = f'relevance <b>{rerank:.3f}</b> · {scores}'
            _html(
                f'<div class="chunk-card{"" if h["used"] else " unused"}">'
                f'<div class="chunk-head"><span><b>{tag}</b> {escape(h["source"])} · page {h["page"]}</span>'
                f'<span>{scores}</span></div>'
                f'<div class="score-bar"><span style="width:{width:.0f}%"></span></div>'
                # <br> instead of newlines: a blank line would end the HTML block in Markdown
                f'<div class="chunk-text">{escape(h["text"]).replace(chr(10), "<br>")}</div>'
                "</div>"
            )
        if not details["hits"]:
            st.caption("No chunks were retrieved.")
