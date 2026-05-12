from __future__ import annotations

import numpy as np


class FaissVectorStore:
    """Small vector store wrapper using FAISS when available."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self._vectors: np.ndarray | None = None
        try:
            import faiss  # type: ignore
        except ImportError:
            faiss = None
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dimensions) if faiss is not None else None

    @property
    def backend_name(self) -> str:
        return "faiss.IndexFlatIP" if self._index is not None else "numpy.inner"

    def add(self, vectors: np.ndarray) -> None:
        normalized = self._normalize(vectors.astype("float32"))
        self._vectors = normalized
        if self._index is not None:
            self._index = self._faiss.IndexFlatIP(self.dimensions)
            self._index.add(normalized)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._vectors is None:
            raise ValueError("Vector store is empty. Call ingest() before search().")

        query = self._normalize(query_vector.reshape(1, -1).astype("float32"))
        if self._index is not None:
            scores, indices = self._index.search(query, top_k)
            return [(int(index), float(score)) for index, score in zip(indices[0], scores[0]) if index >= 0]

        scores = np.inner(self._vectors, query[0])
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(index), float(scores[index])) for index in ranked]

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
