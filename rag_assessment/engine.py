from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from langchain_core.documents import Document

from typing import Protocol

import numpy as np

from rag_assessment.mock_vertex import MockGenerativeModel, MockTextEmbeddingModel
from rag_assessment.models import BenchmarkReport, Chunk, QueryComparison, SearchResult
from rag_assessment.vector_store import FaissVectorStore


class EmbeddingModel(Protocol):
    dimensions: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class QueryExpander(Protocol):
    def expand_query(self, query: str) -> str: ...


class ContextAwareRetrievalEngine:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        query_expander: QueryExpander | None = None,
    ) -> None:
        self.embedding_model = embedding_model or MockTextEmbeddingModel.from_pretrained("textembedding-gecko@mock")
        self.query_expander = query_expander or MockGenerativeModel()
        self._chunks: list[Chunk] = []
        self._documents: list[Document] = []
        self._vector_store = FaissVectorStore(self.embedding_model.dimensions)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def vector_backend(self) -> str:
        return self._vector_store.backend_name

    def ingest(self, chunks: Sequence[Chunk | str]) -> None:
        self._chunks = [
            chunk if isinstance(chunk, Chunk) else Chunk(chunk_id=f"chunk_{index:03d}", text=chunk)
            for index, chunk in enumerate(chunks)
        ]
        self._documents = [
            Document(page_content=chunk.text, metadata={"chunk_id": chunk.chunk_id, **chunk.metadata})
            for chunk in self._chunks
        ]
        embeddings = self.embedding_model.embed_documents([document.page_content for document in self._documents])
        self._vector_store.add(embeddings)

    def search_raw(self, query: str, top_k: int = 3) -> list[SearchResult]:
        return self._search(query, top_k)

    def search_expanded(self, query: str, top_k: int = 3) -> tuple[str, list[SearchResult]]:
        expanded_query = self.query_expander.expand_query(query)
        return expanded_query, self._search(expanded_query, top_k)

    def compare_query(self, query: str, top_k: int = 3) -> QueryComparison:
        expanded_query, expanded_results = self.search_expanded(query, top_k)
        return QueryComparison(
            input_query=query,
            expanded_query=expanded_query,
            strategy_a=self.search_raw(query, top_k),
            strategy_b=expanded_results,
        )

    def run_benchmark(self, queries: Sequence[str], top_k: int = 3) -> BenchmarkReport:
        return BenchmarkReport(
            metadata={
                "created_at": datetime.now(UTC).isoformat(),
                "embedding_model": self.embedding_model.__class__.__name__,
                "query_expander": self.query_expander.__class__.__name__,
                "vector_store": self.vector_backend,
                "strategy_a": "Raw Vector Search",
                "strategy_b": "AI-Enhanced Retrieval",
                "top_k": top_k,
                "chunk_count": self.chunk_count,
            },
            comparisons=[self.compare_query(query, top_k) for query in queries],
        )

    def _search(self, query: str, top_k: int) -> list[SearchResult]:
        query_vector = self.embedding_model.embed_query(query)
        matches = self._vector_store.search(query_vector, top_k)
        results: list[SearchResult] = []
        for rank, (index, score) in enumerate(matches, start=1):
            chunk = self._chunks[index]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )
        return results
