"""All prompt text in one place, so it's easy to read and tweak."""

from __future__ import annotations

from src.vectorstore.chroma_store import SearchHit

NOT_FOUND_MESSAGE = "I couldn't find this in your documents."
NO_DOCUMENTS_MESSAGE = "No documents are indexed yet. Upload a PDF or TXT file first."

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions using ONLY the numbered document excerpts in the user's message.

Rules:
- Use only information from the excerpts. Never use outside knowledge, even if you know the answer.
- Cite the excerpt number for every fact, in square brackets: [1], or [1][3] for several. Only cite numbers that exist.
- If the excerpts don't contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}" You may add one short sentence about what the excerpts do cover.
- If the excerpts answer only part of the question, answer that part and say clearly what is missing.
- Stay on the exact subject of the question. If an excerpt describes something else (for example another method or product), don't present it as being about the subject.
- Be clear and concise. Use short paragraphs or bullet points.
- Earlier messages are the chat so far. Use them to understand follow-up questions, but they are not sources."""

# Added to the system prompt in voice mode, where the answer is read aloud.
VOICE_STYLE = """This answer will be read aloud. Reply in at most three short, natural spoken sentences.
Don't use lists, headings, tables or Markdown. Still cite sources like [1] after the facts they support."""

ANSWER_TEMPLATE = """Document excerpts:

{context}

Question: {question}"""

CONDENSE_PROMPT = """Rewrite the follow-up question so it can be understood without the conversation, \
by filling in any names or topics it refers to. If it is already clear on its own, return it unchanged. \
Reply with only the rewritten question.

Conversation:
{conversation}

Follow-up question: {question}"""


def format_context(hits: list[SearchHit]) -> str:
    """Number the chunks [1], [2], ... with their file and page."""
    blocks = [f"[{i}] {h.source}, page {h.page}\n{h.text}" for i, h in enumerate(hits, 1)]
    return "\n\n".join(blocks)
