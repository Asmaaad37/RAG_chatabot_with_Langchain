"""Phase 2 tests: loading, cleaning, chunking, deduplication and storage.

Run with:  pytest
Each test uses its own temporary Chroma folder, so your real index isn't touched.
"""

from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_pages, file_hash
from src.ingestion.indexer import index_file
from src.ingestion.loader import DocumentLoadError, EmptyDocumentError, clean_text, load_document
from src.vectorstore.chroma_store import VectorStore

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "VirtualTryAll.pdf"

SOLAR = (
    "Solar panels convert sunlight into electricity using photovoltaic cells. "
    "Most home systems are installed on the roof and face south to catch the most light. "
    "An inverter changes the direct current from the panels into alternating current for the house."
)
BREAD = (
    "Sourdough bread is leavened with a starter of wild yeast and lactic acid bacteria. "
    "The dough is left to ferment slowly, often overnight, which gives the bread its sour taste. "
    "Bakers score the loaf with a blade before baking so it can expand in the oven."
)
PYTHON = (
    "Python is a programming language known for readable code. "
    "A virtual environment keeps each project's packages separate from the rest of the system. "
    "Packages are installed with pip from the Python Package Index."
)


# ---------- fixtures ----------

@pytest.fixture
def store(tmp_path):
    return VectorStore(path=tmp_path / "chroma", collection_name="test_docs")


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------- loading and cleaning ----------

def test_clean_text_fixes_pdf_artifacts():
    # hyphenated word rejoined, spaces/tabs collapsed, blank lines reduced to one
    assert clean_text("Retrieval is use-\nful.   Really\tuseful.\n\n\n\nNew paragraph.") == (
        "Retrieval is useful. Really useful.\n\nNew paragraph."
    )
    # a long line followed by a capital letter is a wrapped sentence -> joined
    assert clean_text("This is a long line of prose that the PDF wrapped right before\nGroq.") == (
        "This is a long line of prose that the PDF wrapped right before Groq."
    )
    # a short line followed by lowercase is a sentence continuing -> joined
    assert clean_text("Short line\ncontinues here.") == "Short line continues here."
    # a short line followed by a capital letter is a separate item -> kept
    assert clean_text("Adam Reed\nMaya Chen") == "Adam Reed\nMaya Chen"


def test_clean_text_keeps_table_rows_and_list_items_apart():
    raw = "Group 4: Transport Layer \nSara Khan \nOmar Ali \nLina haddad \n \nGroup 5"
    assert clean_text(raw) == (
        "Group 4: Transport Layer\nSara Khan\nOmar Ali\nLina haddad\n\nGroup 5"
    )


def test_pdf_pages_are_numbered_from_one():
    pages = load_document(SAMPLE_PDF)
    assert pages[0].metadata == {"source": "VirtualTryAll.pdf", "page": 1}
    assert pages[-1].metadata["page"] == 18


def test_txt_is_a_single_page(tmp_path):
    pages = load_document(write(tmp_path, "notes.txt", SOLAR))
    assert len(pages) == 1 and pages[0].metadata["page"] == 1


def test_empty_file_gives_friendly_error(tmp_path):
    with pytest.raises(EmptyDocumentError, match="No text found"):
        load_document(write(tmp_path, "empty.txt", "   \n\n  "))


def test_unsupported_file_type(tmp_path):
    with pytest.raises(DocumentLoadError, match="isn't supported"):
        load_document(write(tmp_path, "data.csv", "a,b,c"))


# ---------- chunking ----------

def test_chunks_carry_metadata_and_respect_size(tmp_path):
    path = write(tmp_path, "mix.txt", "\n\n".join([SOLAR, BREAD, PYTHON]))
    chunks = chunk_pages(load_document(path), file_hash(path), chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    first = chunks[0].metadata()
    assert first["source"] == "mix.txt" and first["page"] == 1
    assert len(first["chunk_hash"]) == 64 and len(first["file_hash"]) == 64


def test_repeated_text_in_one_file_is_kept_once(tmp_path):
    path = write(tmp_path, "repeat.txt", "\n\n".join([SOLAR, SOLAR, SOLAR]))
    chunks = chunk_pages(load_document(path), file_hash(path), chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 1


# ---------- indexing, duplicates and search ----------

def test_index_then_skip_same_file(tmp_path, store, embedder):
    path = write(tmp_path, "mix.txt", "\n\n".join([SOLAR, BREAD, PYTHON]))

    first = index_file(path, store, embedder, chunk_size=200, chunk_overlap=20)
    assert first.status == "indexed" and first.chunks_added > 0
    count = store.count()

    # Same content under a different name is still the same file.
    copy = write(tmp_path, "copy_of_mix.txt", path.read_text(encoding="utf-8"))
    second = index_file(copy, store, embedder, chunk_size=200, chunk_overlap=20)
    assert second.status == "already_indexed"
    assert store.count() == count


def test_duplicate_chunks_across_files_are_skipped(tmp_path, store, embedder):
    a = write(tmp_path, "a.txt", "\n\n".join([SOLAR, BREAD]))
    b = write(tmp_path, "b.txt", "\n\n".join([SOLAR, BREAD, PYTHON]))  # shares two paragraphs

    # chunk_size=300 puts each paragraph (~250 chars) in its own chunk
    index_file(a, store, embedder, chunk_size=300, chunk_overlap=0)
    result = index_file(b, store, embedder, chunk_size=300, chunk_overlap=0)

    assert result.chunks_skipped == 2  # SOLAR and BREAD are already stored
    assert result.chunks_added == 1    # only PYTHON is new
    assert store.count() == 3


def test_search_finds_the_relevant_chunk(tmp_path, store, embedder):
    path = write(tmp_path, "mix.txt", "\n\n".join([SOLAR, BREAD, PYTHON]))
    index_file(path, store, embedder, chunk_size=300, chunk_overlap=0)

    hits = store.search(embedder.embed_query("How does sourdough get its sour flavour?"), k=3)
    assert "Sourdough" in hits[0].text or "ferment" in hits[0].text
    assert hits[0].score > hits[-1].score
    assert 0.0 <= hits[-1].score <= 1.0
    assert hits[0].source == "mix.txt" and hits[0].page == 1


def test_persistence_and_delete(tmp_path, embedder):
    db = tmp_path / "chroma"
    path = write(tmp_path, "solar.txt", SOLAR)
    index_file(path, VectorStore(path=db, collection_name="test_docs"), embedder)

    reopened = VectorStore(path=db, collection_name="test_docs")  # like restarting the app
    docs = reopened.list_documents()
    assert [d.source for d in docs] == ["solar.txt"]

    reopened.delete_document(docs[0].file_hash)
    assert reopened.count() == 0
    assert reopened.list_documents() == []
