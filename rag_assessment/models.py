from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryComparison(BaseModel):
    input_query: str
    expanded_query: str
    strategy_a: list[SearchResult]
    strategy_b: list[SearchResult]


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    metadata: dict[str, Any]
    comparisons: list[QueryComparison]
