"""Prepare answers for reading aloud.

The speaking itself happens in the browser (its built-in speechSynthesis voice):
it's free, works offline and needs no API. See src/ui/components.py: speaker().
"""

from __future__ import annotations

import re


def clean_for_speech(text: str) -> str:
    """Remove things that sound wrong when read aloud: [1] citations, Markdown
    symbols, link URLs, and list bullets."""
    text = re.sub(r"\s?[\[【]\d+[\]】]", "", text)                 # [1], 【2】
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)            # [label](url) -> label
    text = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]*", "", text, flags=re.M)  # # headings
    text = re.sub(r"^[ \t]*(?:[-*•]|\d+[.)])[ \t]+", "", text, flags=re.M)  # list bullets / numbers
    text = re.sub(r"[*_`>|]+", "", text)                            # bold, italics, code, quotes, tables
    text = re.sub(r"\n{2,}", ". ", text)                             # paragraph breaks -> pause
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.\s*\.", ".", text)                             # no double full stops
    return text.strip()
