"""Phase 3 tests: retrieval, the score threshold, prompts and citations.

These never call Groq (no API key or internet needed).
"""

import pytest

from src import rag_pipeline as rp
from src.generation.prompts import NO_DOCUMENTS_MESSAGE, NOT_FOUND_MESSAGE, format_context
from src.ingestion.indexer import index_file
from src.retrieval.retriever import Retriever, fuse_rankings
from src.vectorstore.chroma_store import SearchHit, VectorStore

from tests.test_ingestion import BREAD, PYTHON, SOLAR, write


@pytest.fixture
def pipeline(tmp_path, embedder, monkeypatch):
    """A pipeline over a temporary store holding three short topics."""
    # Fail loudly if anything tries to call Groq.
    def no_llm(*args, **kwargs):
        raise AssertionError("The LLM should not have been called")

    monkeypatch.setattr(rp, "stream_answer", no_llm)
    monkeypatch.setattr(rp, "condense_question", no_llm)

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_docs")
    path = write(tmp_path, "topics.txt", "\n\n".join([SOLAR, BREAD, PYTHON]))
    index_file(path, store, embedder, chunk_size=300, chunk_overlap=0)
    return rp.RAGPipeline(store=store, embedder=embedder)


def hit(chunk_index: int, score: float) -> SearchHit:
    return SearchHit(text=f"chunk {chunk_index}", source="f.txt", page=1,
                     chunk_index=chunk_index, score=score)


# ---------- retrieval ----------

def test_relevant_question_passes_threshold(pipeline):
    result = Retriever(pipeline.store, pipeline.embedder).retrieve(
        "How do solar panels make electricity?", top_k=3, threshold=0.55
    )
    assert result.found
    assert "Solar" in result.relevant_hits[0].text
    assert all(h.score >= 0.55 for h in result.relevant_hits)


def test_out_of_scope_question_skips_the_llm(pipeline):
    # no_llm in the fixture makes this test fail if Groq is called.
    response = pipeline.ask("Who won the football World Cup in 1998?", score_threshold=0.55)
    assert response.text() == NOT_FOUND_MESSAGE
    assert response.used_llm is False
    assert response.sources == []
    assert response.retrieval.best_score < 0.55


def test_empty_index_says_so(tmp_path, embedder):
    empty = rp.RAGPipeline(store=VectorStore(path=tmp_path / "db", collection_name="empty_docs"),
                           embedder=embedder)
    assert empty.ask("Anything?").text() == NO_DOCUMENTS_MESSAGE


def test_blank_question_is_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_fusion_lifts_a_chunk_found_by_both_queries():
    first = [hit(1, 0.80), hit(2, 0.70), hit(3, 0.60)]
    second = [hit(3, 0.65), hit(4, 0.50)]
    fused = fuse_rankings([first, second], top_k=3)
    assert fused[0].chunk_index == 3           # in both lists, so it wins
    assert fused[0].score == 0.65              # keeps its best score
    assert len(fused) == 3


# ---------- prompts and citations ----------

def test_context_is_numbered_with_file_and_page():
    text = format_context([hit(0, 0.9), hit(1, 0.8)])
    assert text.startswith("[1] f.txt, page 1\nchunk 0")
    assert "[2] f.txt, page 1\nchunk 1" in text


def test_cited_sources_match_the_answer(pipeline, monkeypatch):
    monkeypatch.setattr(rp, "stream_answer", lambda *a, **k: iter(["Sourdough ", "is sour [2]."]))
    response = pipeline.ask("Why is sourdough bread sour?", top_k=3, score_threshold=0.3)

    assert response.text() == "Sourdough is sour [2]."
    assert response.used_llm
    assert [s.number for s in response.cited_sources] == [2]
    assert response.cited_sources[0].label == "[2] topics.txt, p. 1"
    assert response.response_time is not None


def test_full_width_citation_brackets_are_normalised(pipeline, monkeypatch):
    monkeypatch.setattr(rp, "stream_answer", lambda *a, **k: iter(["Sourdough is sour【", "1】."]))
    response = pipeline.ask("Why is sourdough bread sour?", top_k=3, score_threshold=0.3)
    assert response.text() == "Sourdough is sour[1]."
    assert [s.number for s in response.cited_sources] == [1]
