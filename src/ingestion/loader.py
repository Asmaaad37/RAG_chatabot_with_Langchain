"""Load PDF and TXT files into clean text, one LangChain Document per page."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentLoadError(Exception):
    """A file couldn't be read. The message is safe to show to users."""


class EmptyDocumentError(DocumentLoadError):
    """A file was read but contained no usable text."""


# Lines at least this long are prose that the PDF wrapped. Shorter lines are usually
# table rows, list items or headings (e.g. one student name per row).
LONG_LINE = 40


def _join_wrapped_lines(text: str) -> str:
    """Undo PDF line wrapping in paragraphs, but keep tables and lists one item per line.

    A line break becomes a space when the line before is long (a wrapped sentence)
    or the next line starts lowercase (a sentence continuing). Blank lines
    (paragraph breaks) are always kept.
    """
    lines = text.split("\n")
    out = lines[0]
    for previous, line in zip(lines, lines[1:]):
        wrapped = previous and line and (len(previous) >= LONG_LINE or line[0].islower())
        out += (" " if wrapped else "\n") + line
    return out


def clean_text(text: str) -> str:
    """Tidy up text extracted from PDFs/TXT so chunks embed well."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join words hyphenated across a line break: "inform-\nation" -> "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Trim spaces around newlines
    text = re.sub(r" *\n *", "\n", text)
    text = _join_wrapped_lines(text)
    # Keep at most one blank line between paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path) -> list[Document]:
    try:
        return PyPDFLoader(str(path)).load()
    except Exception as e:  # pypdf raises many different error types
        raise DocumentLoadError(
            f"Couldn't read '{path.name}'. The PDF may be damaged or password-protected."
        ) from e


def _load_txt(path: Path) -> list[Document]:
    # Try UTF-8 first, then fall back to the common Windows encoding.
    for encoding in ("utf-8", "cp1252"):
        try:
            return TextLoader(str(path), encoding=encoding).load()
        except RuntimeError:  # TextLoader wraps decode errors in RuntimeError
            continue
    raise DocumentLoadError(f"Couldn't read '{path.name}'. Please save it as UTF-8 text.")


def load_document(path: str | Path) -> list[Document]:
    """Read a file and return cleaned pages.

    Each Document has metadata {"source": <file name>, "page": <1-based page>}.
    TXT files count as a single page 1. Pages with no text are dropped.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"'{path.name}' isn't supported. Please upload a PDF or TXT file."
        )
    if not path.exists():
        raise DocumentLoadError(f"File not found: {path}")

    raw_docs = _load_pdf(path) if ext == ".pdf" else _load_txt(path)

    pages: list[Document] = []
    for i, doc in enumerate(raw_docs):
        text = clean_text(doc.page_content)
        if not text:
            continue
        # PyPDF counts pages from 0; people count from 1.
        page = int(doc.metadata.get("page", i)) + 1 if ext == ".pdf" else 1
        pages.append(Document(page_content=text, metadata={"source": path.name, "page": page}))

    if not pages:
        hint = " It may be a scanned PDF (images only), which isn't supported." if ext == ".pdf" else ""
        raise EmptyDocumentError(f"No text found in '{path.name}'.{hint}")

    logger.info("Loaded %s: %d page(s) with text (of %d)", path.name, len(pages), len(raw_docs))
    return pages
