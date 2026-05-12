from __future__ import annotations

import numpy as np


class SentenceTransformerTextEmbeddingModel:
    """Vertex-like wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        dimensions = self._model.get_sentence_embedding_dimension()
        if dimensions is None:
            raise ValueError(f"Could not determine embedding dimension for {model_name}")
        self.dimensions = dimensions

    @classmethod
    def from_pretrained(cls, model_name: str) -> "SentenceTransformerTextEmbeddingModel":
        return cls(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
