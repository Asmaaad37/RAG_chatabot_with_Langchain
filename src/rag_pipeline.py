"""The whole RAG app behind one class: index files, then ask questions.

    pipeline = RAGPipeline()
    pipeline.index_file("data/sample_docs/VirtualTryAll.pdf")
    response = pipeline.ask("What is Diffuse to Choose?")
    for piece in response.stream():      # streaming answer
        print(piece, end="")
    print(response.cited_sources)        # [Source(number=1, source='...', page=1, ...)]
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from config.settings import settings
from src.embeddings.embedder import Embedder, get_embedder
from src.generation.generator import History, condense_question, stream_answer
from src.generation.prompts import NO_DOCUMENTS_MESSAGE, NOT_FOUND_MESSAGE
from src.ingestion.indexer import IndexResult, ProgressCallback, index_file
from src.retrieval.retriever import RetrievalResult, Retriever
from src.vectorstore.chroma_store import IndexedDocument, VectorStore

logger = logging.getLogger(__name__)

CITATION_BRACKETS = str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]"})


@dataclass
class Source:
    number: int   # the [n] used in the answer
    source: str   # file name
    page: int
    score: float

    @property
    def label(self) -> str:
        return f"[{self.number}] {self.source}, p. {self.page}"


@dataclass
class RAGResponse:
    question: str
    search_query: str                  # differs from question when a follow-up was rewritten
    retrieval: RetrievalResult | None  # None when nothing is indexed
    used_llm: bool
    model: str
    temperature: float
    answer: str = ""
    response_time: float | None = None  # seconds, set once the answer has finished streaming
    _tokens: Iterator[str] = field(default_factory=lambda: iter(()), repr=False)
    _started: float = field(default_factory=time.perf_counter, repr=False)

    def stream(self) -> Iterator[str]:
        """Yield answer pieces as they arrive (use with st.write_stream or print)."""
        for piece in self._tokens:
            # Some models cite as 【1】 (full-width brackets); make it [1] like the prompt asks.
            piece = piece.translate(CITATION_BRACKETS)
            self.answer += piece
            yield piece
        self.response_time = round(time.perf_counter() - self._started, 2)

    def text(self) -> str:
        """Wait for the full answer and return it."""
        for _ in self.stream():
            pass
        return self.answer

    @property
    def sources(self) -> list[Source]:
        """Every chunk that was sent to the LLM, numbered like in the prompt."""
        if not self.used_llm or self.retrieval is None:
            return []
        return [
            Source(number=i, source=h.source, page=h.page, score=h.score)
            for i, h in enumerate(self.retrieval.relevant_hits, 1)
        ]

    @property
    def cited_sources(self) -> list[Source]:
        """Only the sources the answer actually cites, e.g. [1] and [3]."""
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", self.answer)}
        return [s for s in self.sources if s.number in cited]


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self.store = store or VectorStore()
        self.retriever = Retriever(self.store, self.embedder)

    # ----- documents -----

    def index_file(
        self,
        path: str | Path,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IndexResult:
        return index_file(path, self.store, self.embedder, chunk_size, chunk_overlap, on_progress)

    def list_documents(self) -> list[IndexedDocument]:
        return self.store.list_documents()

    def delete_document(self, file_hash: str) -> None:
        self.store.delete_document(file_hash)

    # ----- questions -----

    def ask(
        self,
        question: str,
        history: History | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        temperature: float | None = None,
        hybrid: bool | None = None,
        use_reranker: bool | None = None,
        rerank_threshold: float | None = None,
        style: str | None = None,
    ) -> RAGResponse:
        """Retrieve, then (only if something relevant was found) stream an answer from Groq.

        Call .stream() or .text() on the result to get the answer. Groq problems
        surface there as GenerationError with a friendly message.
        """
        started = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("Please type a question.")
        history = history or []
        temperature = settings.temperature if temperature is None else temperature

        def response(**kwargs) -> RAGResponse:
            return RAGResponse(
                question=question, model=settings.groq_model,
                temperature=temperature, _started=started, **kwargs,
            )

        if self.store.count() == 0:
            return response(search_query=question, retrieval=None, used_llm=False,
                            _tokens=iter([NO_DOCUMENTS_MESSAGE]))

        # For follow-ups, search with the rewritten question AND the user's own words,
        # then merge: the rewrite adds context, the original avoids over-matching on names.
        search_query = condense_question(question, history) if history else question
        retrieval = self.retriever.retrieve(
            [search_query, question], top_k, score_threshold, hybrid=hybrid,
            use_reranker=use_reranker, rerank_threshold=rerank_threshold,
        )

        if not retrieval.found:
            # Nothing relevant enough: answer straight away, without calling the LLM.
            logger.info("Best score below threshold - not calling the LLM")
            return response(search_query=search_query, retrieval=retrieval, used_llm=False,
                            _tokens=iter([NOT_FOUND_MESSAGE]))

        tokens = stream_answer(question, retrieval.relevant_hits, history, temperature, style=style)
        return response(search_query=search_query, retrieval=retrieval, used_llm=True, _tokens=tokens)
