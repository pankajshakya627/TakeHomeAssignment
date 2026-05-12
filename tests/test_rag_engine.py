import json

from rag_assessment.engine import ContextAwareRetrievalEngine
from rag_assessment.mock_vertex import MockGenerativeModel, MockTextEmbeddingModel
from rag_assessment.sample_data import TECHNICAL_PARAGRAPHS


def build_engine() -> ContextAwareRetrievalEngine:
    engine = ContextAwareRetrievalEngine(
        embedding_model=MockTextEmbeddingModel(),
        query_expander=MockGenerativeModel(),
    )
    engine.ingest(TECHNICAL_PARAGRAPHS)
    return engine


def test_ingest_indexes_all_chunks() -> None:
    engine = build_engine()

    assert engine.chunk_count == len(TECHNICAL_PARAGRAPHS)


def test_mock_text_embedding_model_exposes_vertex_like_get_embeddings() -> None:
    model = MockTextEmbeddingModel(dimensions=16)

    embeddings = model.get_embeddings(["autoscaling queue depth", "key rotation"])

    assert len(embeddings) == 2
    assert len(embeddings[0].values) == 16
    assert embeddings[0].values != embeddings[1].values


def test_mock_generative_model_expands_query_with_retrieval_terms() -> None:
    model = MockGenerativeModel()

    expanded_query = model.generate_content("How does the system handle peak load?")

    assert "horizontal autoscaling" in expanded_query
    assert "backpressure" in expanded_query


def test_raw_search_returns_ranked_chunks() -> None:
    engine = build_engine()

    results = engine.search_raw("How does the system handle peak load?", top_k=3)

    assert len(results) == 3
    assert results[0].score >= results[1].score >= results[2].score
    assert all(result.chunk_id for result in results)


def test_expanded_search_rewrites_query_and_changes_ranking() -> None:
    engine = build_engine()

    comparison = engine.compare_query("How does the system handle peak load?", top_k=3)

    assert "autoscaling" in comparison.expanded_query.lower()
    assert "backpressure" in comparison.expanded_query.lower()
    assert comparison.strategy_a[0].chunk_id != comparison.strategy_b[0].chunk_id
    assert comparison.strategy_b[0].chunk_id == "scaling_peak_load"


def test_benchmark_report_is_json_serializable_for_three_queries() -> None:
    engine = build_engine()

    report = engine.run_benchmark(
        [
            "How does the system handle peak load?",
            "What protects customer data if credentials leak?",
            "How are bad model outputs detected before users see them?",
        ],
        top_k=3,
    )
    payload = report.model_dump()

    assert len(report.comparisons) == 3
    assert payload["metadata"]["strategy_a"] == "Raw Vector Search"
    assert payload["metadata"]["strategy_b"] == "AI-Enhanced Retrieval"
    assert json.loads(report.model_dump_json())["comparisons"][0]["input_query"]
