"""Tests for keyword (BM25) search, hybrid search and the reranker. No Groq calls."""

import pytest

from src.ingestion.indexer import index_file
from src.retrieval.keyword_search import KeywordIndex, tokenize
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import VectorStore

from tests.test_ingestion import BREAD, PYTHON, SOLAR, write

# A chunk whose only distinctive content is an exact code: keyword search's strength.
PART = "Maintenance note. Replace filter part ZX-4471 every six months to keep the pump efficient."


@pytest.fixture
def store(tmp_path, embedder):
    s = VectorStore(path=tmp_path / "chroma", collection_name="hybrid_docs")
    path = write(tmp_path, "notes.txt", "\n\n".join([SOLAR, BREAD, PYTHON, PART]))
    index_file(path, s, embedder, chunk_size=300, chunk_overlap=0)
    return s


def test_tokenize_keeps_numbers_and_codes_and_drops_stopwords():
    assert tokenize("What is the DINOv2 model trained on 1.2M images?") == [
        "dinov2", "model", "trained", "1.2m", "images"
    ]
    assert tokenize("part ZX-4471") == ["part", "zx-4471"]


def test_keyword_search_finds_exact_code(store):
    hits = KeywordIndex(store).search("Which part is ZX-4471?", k=3)
    assert "ZX-4471" in hits[0].text
    assert hits[0].id  # ids are needed to merge with embedding results


def test_keyword_search_ignores_questions_with_no_shared_words(store):
    assert KeywordIndex(store).search("football world cup", k=3) == []


def test_keyword_index_updates_after_new_file(tmp_path, store, embedder):
    index = KeywordIndex(store)
    assert index.search("telescope", k=3) == []
    extra = write(tmp_path, "space.txt", "A telescope collects light from distant galaxies.")
    index_file(extra, store, embedder)
    assert "telescope" in index.search("telescope", k=3)[0].text


def test_hybrid_retrieval_puts_exact_match_first(store, embedder):
    result = Retriever(store, embedder).retrieve(
        "ZX-4471", top_k=3, threshold=0.0, hybrid=True, use_reranker=False
    )
    assert result.hybrid
    assert "ZX-4471" in result.hits[0].text
    assert result.hits[0].score > 0  # similarity filled in for keyword hits too


def test_reranker_scores_the_right_chunk_highest(store, embedder):
    # Uses the small local reranker (~90 MB download on first run).
    result = Retriever(store, embedder).retrieve(
        "Why does sourdough bread taste sour?", top_k=3, threshold=0.99, use_reranker=True,
        rerank_threshold=0.1,
    )
    assert result.reranked
    assert "Sourdough" in result.hits[0].text
    assert result.hits[0].rerank_score > result.hits[-1].rerank_score
    assert result.hits[0].relevance == result.hits[0].rerank_score
    # similarity can't reach 0.99, so this passes only via the reranker threshold
    assert result.found and result.relevant_hits[0] is result.hits[0]


def test_chunk_passes_if_either_score_clears_its_threshold():
    from src.retrieval.retriever import RetrievalResult
    from src.vectorstore.chroma_store import SearchHit

    def hit(similarity, rerank):
        return SearchHit(text="t", source="f", page=1, chunk_index=0, score=similarity, rerank_score=rerank)

    result = RetrievalResult(queries=["q"], hits=[], threshold=0.55, rerank_threshold=0.1)
    assert result.passes(hit(0.62, 0.009))      # list of names: similarity saves it
    assert result.passes(hit(0.51, 0.965))      # name lookup: reranker saves it
    assert not result.passes(hit(0.53, 0.0))    # off-topic: both low
