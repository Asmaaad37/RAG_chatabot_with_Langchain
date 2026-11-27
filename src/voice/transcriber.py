"""Speech to text with Groq's Whisper model (free tier).

The recording from the chat box's mic is sent to Groq, which returns the text.
"""

from __future__ import annotations

import logging

import groq

from config.settings import settings

logger = logging.getLogger(__name__)

MAX_AUDIO_MB = 25  # Groq's limit for one file on the free tier


class TranscriptionError(Exception):
    """Speech couldn't be turned into text. The message is safe to show to users."""


# Peak loudness below this (out of 32767 for 16-bit audio) counts as silence.
SILENCE_PEAK = 400


def is_silent(wav: bytes) -> bool:
    """True if a 16-bit WAV recording has no sound louder than background hiss.

    Whisper tends to invent text (like "Thank you.") for silent audio, so we
    check before sending. Non-WAV or other formats are assumed not silent.
    """
    import io
    import wave
    from array import array

    try:
        with wave.open(io.BytesIO(wav)) as w:
            if w.getsampwidth() != 2:
                return False
            samples = array("h", w.readframes(w.getnframes()))
    except (wave.Error, EOFError):
        return False
    return not samples or max(abs(s) for s in samples) < SILENCE_PEAK


def transcribe(audio: bytes, filename: str = "question.wav") -> str:
    """Return the spoken text in a WAV (or other audio) recording."""
    if not settings.has_api_key:
        raise TranscriptionError(
            "Voice input needs a Groq API key. Add GROQ_API_KEY to your .env file."
        )
    if not audio:
        raise TranscriptionError("The recording was empty. Please try again.")
    if is_silent(audio):
        raise TranscriptionError("I couldn't hear anything. Check your microphone and try again.")
    if len(audio) > MAX_AUDIO_MB * 1024 * 1024:
        raise TranscriptionError(f"That recording is too long (max {MAX_AUDIO_MB} MB). Please keep it short.")

    client = groq.Groq(api_key=settings.groq_api_key, max_retries=2)
    try:
        result = client.audio.transcriptions.create(
            file=(filename, audio),
            model=settings.whisper_model,
            response_format="json",
            temperature=0.0,
        )
    except groq.AuthenticationError:
        raise TranscriptionError("Groq rejected your API key. Check GROQ_API_KEY in your .env file.") from None
    except groq.RateLimitError:
        raise TranscriptionError("Groq's free-tier limit for voice was reached. Wait a minute and try again.") from None
    except groq.NotFoundError:
        raise TranscriptionError(
            f"Groq doesn't recognise the speech model '{settings.whisper_model}'. Set WHISPER_MODEL in .env."
        ) from None
    except (groq.APIConnectionError, groq.APITimeoutError):
        raise TranscriptionError("Couldn't reach Groq. Check your internet connection.") from None
    except groq.APIStatusError as e:
        logger.error("Transcription failed: %s", e)
        raise TranscriptionError("Groq couldn't process that recording. Please try again.") from None

    text = (result.text or "").strip()
    logger.info("Transcribed %d bytes of audio -> %r", len(audio), text[:80])
    if not text:
        raise TranscriptionError("I couldn't hear any words. Please try again a bit closer to the mic.")
    return text
