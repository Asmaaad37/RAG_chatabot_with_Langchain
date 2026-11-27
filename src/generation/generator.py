"""Talk to Groq: rewrite follow-up questions and stream grounded answers."""

from __future__ import annotations

import logging
import re
from typing import Iterator

import groq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config.settings import settings
from src.generation.prompts import ANSWER_TEMPLATE, CONDENSE_PROMPT, SYSTEM_PROMPT, format_context
from src.vectorstore.chroma_store import SearchHit

logger = logging.getLogger(__name__)

# A chat history is a list of {"role": "user" | "assistant", "content": "..."}
History = list[dict[str, str]]


class GenerationError(Exception):
    """Something went wrong talking to Groq. The message is safe to show to users."""


class MissingAPIKeyError(GenerationError):
    pass


def friendly_error(e: Exception) -> GenerationError:
    """Turn a Groq exception into a short message a user can act on."""
    if isinstance(e, groq.AuthenticationError):
        return GenerationError("Groq rejected your API key. Check GROQ_API_KEY in your .env file.")
    if isinstance(e, groq.RateLimitError):
        return GenerationError("Groq's free-tier rate limit was reached. Wait about a minute and try again.")
    if isinstance(e, groq.NotFoundError):
        return GenerationError(
            f"Groq doesn't recognise the model '{settings.groq_model}'. "
            "Pick another one from https://console.groq.com/docs/models and set GROQ_MODEL in .env."
        )
    if isinstance(e, (groq.APIConnectionError, groq.APITimeoutError)):
        return GenerationError("Couldn't reach Groq. Check your internet connection and try again.")
    if isinstance(e, groq.APIStatusError):
        return GenerationError(f"Groq returned an error ({e.status_code}). Please try again.")
    return GenerationError(f"Something went wrong while generating the answer: {e}")


def make_llm(temperature: float | None = None, max_tokens: int | None = None) -> ChatGroq:
    if not settings.has_api_key:
        raise MissingAPIKeyError(
            "No Groq API key found. Copy .env.example to .env and add your free key "
            "from https://console.groq.com/keys"
        )
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=max_tokens,
        max_retries=2,
        # gpt-oss models "think" before answering. Keep it short for faster replies.
        # The thinking text is returned separately and never shown as the answer.
        reasoning_effort="low" if settings.groq_model.startswith("openai/gpt-oss") else None,
    )


def _recent_history(history: History, turns: int) -> History:
    """The last `turns` question/answer pairs."""
    return history[-2 * turns :] if turns > 0 else []


def _strip_citations(text: str) -> str:
    # Old answers' [1], [2] point to *old* excerpts. Remove them so the model isn't confused.
    return re.sub(r"\s?\[\d+\]", "", text)


def _history_messages(history: History) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(msg["content"]))
        else:
            messages.append(AIMessage(_strip_citations(msg["content"])))
    return messages


def condense_question(question: str, history: History, turns: int | None = None) -> str:
    """Turn a follow-up like "what about its limits?" into a standalone search query.

    Without this, the search would look for "its limits" and find nothing useful.
    Falls back to the original question if Groq fails.
    """
    recent = _recent_history(history, settings.history_turns if turns is None else turns)
    if not recent:
        return question

    conversation = "\n".join(
        f"{m['role'].title()}: {_strip_citations(m['content'])[:800]}" for m in recent
    )
    try:
        llm = make_llm(temperature=0)
        reply = llm.invoke(CONDENSE_PROMPT.format(conversation=conversation, question=question))
        rewritten = reply.content.strip().strip('"')
        if rewritten:
            logger.info("Rewrote follow-up %r -> %r", question, rewritten)
            return rewritten
    except (groq.GroqError, MissingAPIKeyError) as e:
        logger.warning("Couldn't rewrite the follow-up question (%s); using it as-is", e)
    return question


def stream_answer(
    question: str,
    hits: list[SearchHit],
    history: History | None = None,
    temperature: float | None = None,
    turns: int | None = None,
    style: str | None = None,
) -> Iterator[str]:
    """Yield the answer piece by piece as Groq generates it.

    Raises GenerationError (with a friendly message) if Groq fails.
    """
    llm = make_llm(temperature)
    recent = _recent_history(history or [], settings.history_turns if turns is None else turns)

    system = SYSTEM_PROMPT + ("\n\n" + style if style else "")  # e.g. shorter answers for voice
    messages: list[BaseMessage] = [SystemMessage(system)]
    messages += _history_messages(recent)
    messages.append(
        HumanMessage(ANSWER_TEMPLATE.format(context=format_context(hits), question=question))
    )

    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except groq.GroqError as e:
        logger.error("Groq error: %s", e)
        raise friendly_error(e) from e
