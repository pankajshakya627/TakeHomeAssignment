from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class MockEmbedding:
    values: list[float]


@dataclass(frozen=True)
class MockGenerationResponse:
    text: str


class MockTextEmbeddingModel:
    """Offline stand-in for vertexai.language_models.TextEmbeddingModel.

    It uses a deterministic hashed bag-of-terms vector so benchmark output is
    stable without network access or cloud credentials.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    @classmethod
    def from_pretrained(cls, _: str) -> "MockTextEmbeddingModel":
        return cls()

    def get_embeddings(self, texts: list[str]) -> list[MockEmbedding]:
        return [MockEmbedding(self._embed(text).tolist()) for text in texts]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([embedding.values for embedding in self.get_embeddings(texts)], dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype="float32")
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * self._token_weight(token)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    @staticmethod
    def _token_weight(token: str) -> float:
        if len(token) <= 2:
            return 0.2
        return 1.0 + math.log1p(len(token)) / 8.0


class MockGenerativeModel:
    """Offline stand-in for vertexai.generative_models.GenerativeModel."""

    _expansions: tuple[tuple[tuple[str, ...], str], ...] = (
        (
            ("peak", "load", "traffic", "scale", "busy"),
            "horizontal autoscaling queue depth request queueing backpressure Kubernetes pods Kafka workers latency throughput",
        ),
        (
            ("credential", "credentials", "leak", "customer", "data"),
            "envelope encryption key rotation short-lived credentials scoped IAM secret revocation audit logging",
        ),
        (
            ("bad", "model", "outputs", "hallucination", "users"),
            "model guardrails grounded citations toxicity policy checks hallucinated tool outputs confidence fallback safe response",
        ),
    )

    def generate_content(self, prompt: str) -> MockGenerationResponse:
        source_query = prompt.lower()
        for keywords, expansion in self._expansions:
            if any(keyword in source_query for keyword in keywords):
                return MockGenerationResponse(f"{prompt.strip()} {expansion}")
        return MockGenerationResponse(
            f"{prompt.strip()} relevant architecture reliability security operations implementation details"
        )

    def expand_query(self, query: str) -> str:
        return self.generate_content(query).text
