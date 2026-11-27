"""Custom CSS: near-black theme with a magenta/purple glow (based on the user's mockup).

Base colours and the Kanit font are set in .streamlit/config.toml. Elements made with
st.container(key="x") or st.button(key="x") get the CSS class "st-key-x", which is how
this file targets specific parts of the page.
"""

import streamlit as st

CSS = """
<style>
:root {
  --bg: #0A0A0F;
  --card: #17171F;
  --card-2: #1F1F29;
  --pink: #E0459B;
  --violet: #8B3DFF;
  --text: #F5F5F7;
  --muted: #8E8E9A;
  --border: rgba(255, 255, 255, 0.08);
  --gradient: linear-gradient(135deg, #E0459B 0%, #8B3DFF 100%);
}

/* ---------- page: black with a magenta/purple glow at the top ---------- */
.stApp {
  background:
    radial-gradient(ellipse 55% 32% at 30% -4%, rgba(224, 69, 155, 0.30), transparent 70%),
    radial-gradient(ellipse 45% 28% at 75% -6%, rgba(139, 61, 255, 0.26), transparent 70%),
    var(--bg);
  background-attachment: fixed;
}
[data-testid="stHeader"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
  background: transparent;
}
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem; }

/* ---------- welcome splash: "Welcome back, (name)" when the app opens ---------- */
/* its Streamlit wrapper is taken out of the layout so it leaves no gap */
[data-testid="stLayoutWrapper"]:has(> .st-key-welcome_splash), .st-key-welcome_splash {
  position: absolute; width: 0; height: 0;
}
.welcome-splash {
  position: fixed; inset: 0; z-index: 1000000; display: grid; place-items: center;
  pointer-events: none;  /* never blocks clicks */
  background:
    radial-gradient(ellipse 45% 35% at 50% 45%, rgba(224, 69, 155, 0.22), transparent 70%),
    radial-gradient(ellipse 35% 30% at 55% 55%, rgba(139, 61, 255, 0.20), transparent 70%),
    var(--bg);
  animation: splash-out .7s ease 2.3s forwards;
}
.welcome-inner { text-align: center; }
.welcome-logo { display: inline-block; opacity: 0; animation: pop-in .6s cubic-bezier(.2,.9,.3,1.3) .05s forwards; }
.welcome-logo .logo-dot { width: 64px; height: 64px; font-size: 1.6rem; box-shadow: 0 0 50px rgba(224, 69, 155, .6); }
.welcome-line, .welcome-name { opacity: 0; animation: rise-in .8s ease forwards; }
.welcome-line { font-size: 1.6rem; font-weight: 300; color: var(--muted); margin-top: 1.1rem; animation-delay: .35s; }
.welcome-name {
  font-size: 3.2rem; font-weight: 600; line-height: 1.15; animation-delay: .65s;
  background: var(--gradient); -webkit-background-clip: text; background-clip: text; color: transparent;
}
@keyframes rise-in { from { opacity: 0; transform: translateY(16px); filter: blur(8px); }
                     to   { opacity: 1; transform: none; filter: blur(0); } }
@keyframes pop-in  { from { opacity: 0; transform: scale(.6); } to { opacity: 1; transform: none; } }
@keyframes splash-out { to { opacity: 0; visibility: hidden; } }
@media (prefers-reduced-motion: reduce) {
  .welcome-logo, .welcome-line, .welcome-name { animation: none; opacity: 1; }
  .welcome-splash { animation: splash-out .3s ease 1.2s forwards; }
}

/* ---------- home: top row, heading ---------- */
.top-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.4rem; }
.logo-dot {
  width: 42px; height: 42px; border-radius: 50%; background: var(--gradient);
  display: grid; place-items: center; font-size: 1.15rem; color: #fff;
  box-shadow: 0 0 24px rgba(224, 69, 155, .45);
}
.model-pill {
  background: #000; color: var(--text); border: 1px solid var(--border);
  border-radius: 999px; padding: .45rem 1rem; font-size: .82rem; font-weight: 500;
}
.home-eyebrow { color: var(--muted); font-size: .95rem; margin-bottom: .2rem; }
.home-title { font-size: 2.5rem; font-weight: 600; line-height: 1.12; margin: 0 0 1.2rem; color: var(--text); }
.home-title span { background: var(--gradient); -webkit-background-clip: text; background-clip: text; color: transparent; }
.section-row { display: flex; justify-content: space-between; align-items: baseline; margin: 1.6rem 0 .4rem; }
.section-title { font-size: 1.15rem; font-weight: 500; line-height: 1.4; }
/* markdown adds a bottom margin; drop it so the title lines up with the buttons beside it */
[data-testid="stMarkdownContainer"]:has(> div > .section-title) { margin-bottom: 0; }

/* search box: rounded pill */
.st-key-history_search input { border-radius: 999px; padding-left: 1rem; }
.st-key-history_search [data-baseweb="input"], .st-key-history_search [data-baseweb="base-input"] {
  border-radius: 999px; background: transparent;
}

/* ---------- starter cards (square, with an arrow) ---------- */
[class*="st-key-card_"] button {
  height: 128px; width: 100%;
  display: flex; align-items: flex-start; justify-content: flex-start; text-align: left;
  padding: 1rem; border-radius: 20px;
  background: rgba(23, 23, 31, .75); border: 1px solid rgba(255, 255, 255, .14);
  position: relative; transition: all .15s ease;
}
[class*="st-key-card_"] button > div { justify-content: flex-start; align-items: flex-start; width: 100%; }
[class*="st-key-card_"] button p {
  font-size: 1rem; font-weight: 500; line-height: 1.25; text-align: left;
  white-space: normal; overflow: visible; text-overflow: clip; padding-right: 1rem;
}
[class*="st-key-card_"] button::after {
  content: "↗"; position: absolute; right: 1rem; bottom: .8rem; font-size: 1.1rem; color: var(--text);
}
[class*="st-key-card_"] button:hover {
  border-color: var(--pink); box-shadow: 0 0 0 3px rgba(224, 69, 155, .15); color: var(--text);
}
[class*="st-key-card_"] button:disabled { opacity: .45; }

/* ---------- history rows ---------- */
.hist-item { display: flex; align-items: center; gap: .8rem; padding: .35rem 0; min-width: 0; }
.hist-icon {
  flex: 0 0 34px; height: 34px; border-radius: 50%; display: grid; place-items: center;
  background: rgba(224, 69, 155, .16); color: var(--pink); font-size: .95rem;
}
.hist-text { min-width: 0; }
.hist-title { font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-sub { color: var(--muted); font-size: .8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
[class*="st-key-open_"] button, .st-key-see_all button, [class*="st-key-back"] button {
  background: transparent; border: none; color: var(--text); font-size: 1.3rem; padding: 0 .4rem;
}
.st-key-see_all button { font-size: .85rem; color: var(--muted); }
/* history: 🗑 per row and "Clear all" are quiet until hovered */
[class*="st-key-del_chat_"] button, .st-key-clear_history button {
  background: transparent; border: none; color: var(--muted); padding: 0 .4rem;
}
.st-key-clear_history button p { font-size: .85rem; white-space: nowrap; }
.st-key-clear_history { margin-top: .5rem; }  /* line up with the History title */
[class*="st-key-del_chat_"] button:hover, .st-key-clear_history button:hover { color: #FF6B8B; background: transparent; }
[class*="st-key-open_"] button:hover, .st-key-see_all button:hover, [class*="st-key-back"] button:hover {
  color: var(--pink); background: transparent;
}
.empty-card {
  background: var(--card); border: 1px dashed rgba(255,255,255,.18); border-radius: 18px;
  padding: 1.2rem; text-align: center; color: var(--muted); margin: .6rem 0 1rem;
}
.empty-card b { color: var(--text); }

/* ---------- chat header ---------- */
.chat-head { display: flex; align-items: center; gap: .7rem; min-width: 0; }
.chat-head .logo-dot { width: 36px; height: 36px; font-size: 1rem; }
.chat-head-title { font-weight: 600; line-height: 1.1; }
.chat-head-sub { color: var(--muted); font-size: .8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.st-key-chat_header { border-bottom: 1px solid var(--border); padding-bottom: .6rem; margin-bottom: .8rem; }
.st-key-chat_menu button { background: transparent; border: none; color: var(--text); }
.st-key-chat_menu button div[aria-hidden="true"] { display: none; }  /* hide the ˅ arrow and its space */

/* ---------- chat bubbles ----------
   Each message sits in st.container(key="msg_user_N" / "msg_assistant_N"). */
[class*="st-key-msg_"] [data-testid="stChatMessage"] { background: transparent; padding: .25rem 0; gap: .6rem; }
[class*="st-key-msg_"] [data-testid*="stChatMessageAvatar"] {
  width: 28px; height: 28px; border-radius: 50%; background: var(--gradient); color: #fff;
}

/* user: gradient bubble on the right */
[class*="st-key-msg_user"] [data-testid="stChatMessage"] { flex-direction: row-reverse; }
[class*="st-key-msg_user"] [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; max-width: 78%; margin: 0 0 0 auto;
  background: var(--gradient); color: #fff;
  border-radius: 22px 22px 6px 22px; padding: .75rem 1.05rem;
  box-shadow: 0 10px 30px rgba(178, 61, 230, .28);
}
/* Streamlit pulls markdown up 1rem to hide the last paragraph's margin; undo both here */
[class*="st-key-msg_user"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
[class*="st-key-msg_user"] [data-testid="stMarkdownContainer"] p { color: #fff; margin: 0; }

/* assistant: dark grey bubble on the left */
[class*="st-key-msg_assistant"] [data-testid="stChatMessageContent"] {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 22px 22px 22px 6px; padding: .9rem 1.1rem; max-width: 88%;
}

/* action row under answers: copy · 👍 👎 · regenerate */
[class*="st-key-actions_"] { margin-top: .35rem; gap: .15rem; }
[class*="st-key-actions_"] iframe { border: none; }
[class*="st-key-actions_"] button { color: var(--muted); }
[class*="st-key-actions_"] button:hover { color: var(--text); }
[class*="st-key-actions_"] [data-testid="stBaseButton-tertiary"] p { font-size: .82rem; }

/* typing indicator */
.typing { display: inline-flex; gap: 5px; padding: .35rem .1rem; }
.typing span { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); animation: blink 1.2s infinite both; }
.typing span:nth-child(2) { animation-delay: .2s; }
.typing span:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%, 80%, 100% { opacity: .25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }

/* ---------- voice mode: orb, text, controls ---------- */
.orb-wrap { display: grid; place-items: center; padding: 2.2rem 0 1.4rem; }
.orb {
  position: relative; width: 210px; height: 210px; border-radius: 50%;
  filter: drop-shadow(0 0 40px rgba(224, 69, 155, .45)) drop-shadow(0 0 80px rgba(139, 61, 255, .35));
  animation: orb-float 6s ease-in-out infinite;
}
.orb span {
  position: absolute; inset: 0; mix-blend-mode: screen;
  border-radius: 42% 58% 63% 37% / 41% 44% 56% 59%;
  animation: orb-morph 8s ease-in-out infinite, orb-spin 14s linear infinite;
}
.orb span:nth-child(1) {
  background:
    radial-gradient(circle at 30% 30%, rgba(255,255,255,.55), transparent 35%),
    conic-gradient(from 20deg, #E0459B, #8B3DFF, #3B1C8C, #E0459B);
  opacity: .9;
}
.orb span:nth-child(2) {
  inset: 14px; background: conic-gradient(from 200deg, #8B3DFF, #E0459B, #F7A8D8, #8B3DFF);
  opacity: .55; animation-duration: 6s, 10s; animation-direction: alternate, reverse;
}
.orb.thinking span { animation-duration: 2.5s, 3s; }
.orb.thinking { animation-duration: 1.6s; }
@keyframes orb-morph {
  0%, 100% { border-radius: 42% 58% 63% 37% / 41% 44% 56% 59%; }
  33%      { border-radius: 63% 37% 45% 55% / 55% 61% 39% 45%; }
  66%      { border-radius: 38% 62% 56% 44% / 62% 38% 62% 38%; }
}
@keyframes orb-spin { to { transform: rotate(360deg); } }
@keyframes orb-float { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.05); } }

.voice-text { text-align: center; max-width: 560px; margin: 0 auto 1.2rem; }
.voice-q { font-size: 1.35rem; font-weight: 500; line-height: 1.35; color: var(--text); }
.voice-a { font-size: 1.05rem; line-height: 1.5; color: var(--muted); margin-top: .6rem; }
.voice-status { color: var(--pink); font-size: .85rem; margin-top: .8rem; letter-spacing: .04em; }

/* bottom controls: replay · mic · close */
.st-key-voice_controls { margin-top: 1rem; }
.st-key-voice_controls iframe { border: none; }
.st-key-voice_exit button {
  width: 52px; height: 52px; border-radius: 50%; background: var(--card); border: 1px solid rgba(255,255,255,.12);
}
[class*="st-key-voice_mic"] [data-testid="stAudioInput"] {
  border-radius: 999px; border: 1px solid rgba(224, 69, 155, .55); background: var(--card);
  box-shadow: 0 0 30px rgba(224, 69, 155, .25);
}
.st-key-voice_btn button { background: transparent; border: none; font-size: 1.1rem; }

/* ---------- source chips ---------- */
.source-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0 .2rem; align-items: center; }
.source-label { font-size: .75rem; color: var(--muted); margin-right: .1rem; }
.source-chip {
  font-size: .76rem; padding: .2rem .7rem; border-radius: 999px;
  background: rgba(224, 69, 155, .12); border: 1px solid rgba(224, 69, 155, .38); color: #F7D4E8;
}
.source-chip b { color: #fff; font-weight: 600; }

/* ---------- "under the hood" ---------- */
[data-testid="stExpander"] details { background: rgba(0, 0, 0, .35); border: 1px solid var(--border); border-radius: 14px; }
[data-testid="stExpander"] summary { color: var(--muted); font-size: .88rem; }
.pill-row { display: flex; flex-wrap: wrap; gap: .4rem; }
.meta-pill {
  font-size: .74rem; padding: .18rem .65rem; border-radius: 999px;
  background: var(--card-2); border: 1px solid var(--border); color: var(--muted);
}
.chunk-card { background: rgba(255,255,255,.02); border: 1px solid var(--border); border-radius: 12px; padding: .6rem .8rem; margin: .5rem 0; }
.chunk-card.unused { opacity: .5; }
.chunk-head { display: flex; justify-content: space-between; gap: .5rem; font-size: .78rem; color: var(--muted); }
.chunk-head b { color: var(--text); font-weight: 500; }
.score-bar { height: 4px; border-radius: 4px; background: rgba(255,255,255,.07); margin: .35rem 0 .45rem; }
.score-bar > span { display: block; height: 100%; border-radius: 4px; background: var(--gradient); }
.chunk-text { font-size: .8rem; line-height: 1.5; color: #C9C9D3; }
.query-note { font-size: .8rem; color: var(--muted); margin: .4rem 0 0; }

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"] { background: var(--gradient); border: none; color: #fff; }
[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.1); border: none; color: #fff; }
[data-testid="stBaseButton-secondary"] { background: var(--card); border: 1px solid var(--border); }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--pink); color: var(--text); }

/* ---------- sidebar ---------- */
.brand { font-size: 1.35rem; font-weight: 600; margin-bottom: .2rem; display: flex; align-items: center; gap: .55rem; }
.brand .logo-dot { width: 32px; height: 32px; font-size: .9rem; }
.doc-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: .45rem .75rem; }
.doc-name { font-size: .88rem; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-meta { font-size: .74rem; color: var(--muted); }

/* ---------- chat input: rounded pill ---------- */
[data-testid="stChatInput"] { border-radius: 999px; background: var(--card); border: 1px solid rgba(255,255,255,.12); }
[data-testid="stChatInput"]:focus-within { box-shadow: 0 0 0 2px rgba(180, 76, 232, .45); }
[data-testid="stChatInputSubmitButton"] { border-radius: 50%; background: var(--gradient); color: #fff; }
</style>
"""


def inject_css() -> None:
    st.html(CSS)
