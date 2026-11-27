"""Tests for small UI helpers that don't need a running app."""

import json

from src.ui.components import js_string, text_to_copy


def test_js_string_cannot_break_out_of_a_script_tag():
    nasty = 'Answer </script><script>alert("x")</script> & "quotes" \ done'
    literal = js_string(nasty)
    assert "<" not in literal and ">" not in literal and "&" not in literal
    assert json.loads(literal) == nasty  # still the exact same text when decoded


def test_text_to_copy_adds_sources():
    message = {"content": "Group 5 [1].", "details": {"sources": ["[1] groups.pdf, p. 2"]}}
    assert text_to_copy(message) == "Group 5 [1].\n\nSources:\n[1] groups.pdf, p. 2"
    assert text_to_copy({"content": "No sources here."}) == "No sources here."


def test_css_has_nothing_that_looks_like_an_html_tag():
    """Streamlit sanitises st.html with DOMPurify, which silently drops a <style> block
    whose text contains tag-like text (e.g. a comment mentioning "<name>"). That once
    removed all custom styling, so keep the CSS free of "<" + letter or "/"."""
    import re

    from src.ui.styles import CSS

    inner = CSS.strip().removeprefix("<style>").removesuffix("</style>")
    assert not re.search(r"<[A-Za-z/!]", inner), re.search(r".{0,40}<[A-Za-z/!].{0,40}", inner).group()
