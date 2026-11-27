"""Tests for voice input (src/voice/transcriber.py). No real Groq calls."""

import io
import math
import wave
from array import array
from types import SimpleNamespace

import pytest

from src.voice import transcriber
from src.voice.transcriber import TranscriptionError, is_silent, transcribe


def make_wav(peak: int, seconds: float = 0.5, rate: int = 16000) -> bytes:
    """A 16-bit mono WAV: a 440 Hz tone with the given peak (0 = silence)."""
    samples = array("h", (int(peak * math.sin(2 * math.pi * 440 * i / rate)) for i in range(int(rate * seconds))))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buffer.getvalue()


class FakeGroq:
    """Stands in for groq.Groq and returns a fixed transcript."""
    text = "Which group is Nora Silva in?"
    calls = 0

    def __init__(self, **kwargs):
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        FakeGroq.calls += 1
        assert kwargs["model"] and kwargs["file"][1]
        return SimpleNamespace(text=FakeGroq.text)


@pytest.fixture
def fake_groq(monkeypatch):
    FakeGroq.calls, FakeGroq.text = 0, "Which group is Nora Silva in?"
    monkeypatch.setattr(transcriber.groq, "Groq", FakeGroq)
    return FakeGroq


def test_silence_detection():
    assert is_silent(make_wav(peak=0))
    assert is_silent(make_wav(peak=100))          # faint hiss
    assert not is_silent(make_wav(peak=8000))     # speech-level sound
    assert not is_silent(b"not a wav file")       # unknown format: let Whisper decide


def test_transcribe_returns_text(fake_groq):
    assert transcribe(make_wav(peak=8000)) == "Which group is Nora Silva in?"
    assert fake_groq.calls == 1


def test_silent_recording_is_rejected_without_calling_groq(fake_groq):
    with pytest.raises(TranscriptionError, match="couldn't hear"):
        transcribe(make_wav(peak=0))
    assert fake_groq.calls == 0


def test_empty_transcript_is_an_error(fake_groq):
    fake_groq.text = "   "
    with pytest.raises(TranscriptionError, match="couldn't hear any words"):
        transcribe(make_wav(peak=8000))


def test_empty_recording_is_an_error(fake_groq):
    with pytest.raises(TranscriptionError, match="empty"):
        transcribe(b"")


# ---------- reading answers aloud ----------

from src.voice.speech import clean_for_speech  # noqa: E402


def test_clean_for_speech_removes_citations_and_markdown():
    answer = "**Group 5** is the *Network Layer* group [1].\n\n- Nora Silva\n- Leo Park【2】\n\n## Notes\nSee [docs](https://x.y)."
    spoken = clean_for_speech(answer)
    assert spoken == "Group 5 is the Network Layer group. Nora Silva Leo Park. Notes See docs."
    for leftover in ["[", "]", "*", "#", "http", "【"]:
        assert leftover not in spoken
