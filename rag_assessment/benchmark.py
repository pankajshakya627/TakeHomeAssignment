from __future__ import annotations

import argparse
import json

from rag_assessment.engine import ContextAwareRetrievalEngine
from rag_assessment.sample_data import TECHNICAL_PARAGRAPHS
from rag_assessment.sentence_transformer_embedding import SentenceTransformerTextEmbeddingModel


DEFAULT_QUERIES = [
    "How does the system handle peak load?",
    "What protects customer data if credentials leak?",
    "How are bad model outputs detected before users see them?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the semantic RAG retrieval benchmark.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument(
        "--embedding-backend",
        choices=["mock", "sentence-transformers"],
        default="sentence-transformers",
        help="Use deterministic offline mock embeddings or a local sentence-transformers model.",
    )
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embedding_model = None
    if args.embedding_backend == "sentence-transformers":
        embedding_model = SentenceTransformerTextEmbeddingModel(args.model_name)

    engine = ContextAwareRetrievalEngine(embedding_model=embedding_model)
    engine.ingest(TECHNICAL_PARAGRAPHS)
    report = engine.run_benchmark(DEFAULT_QUERIES, top_k=args.top_k)

    if args.json:
        print(report.model_dump_json(indent=2))
        return

    print(json.dumps(report.metadata, indent=2))
    print()
    for comparison in report.comparisons:
        print(f"Query: {comparison.input_query}")
        print(f"Expanded: {comparison.expanded_query}")
        print("Strategy A | Raw Vector Search")
        for result in comparison.strategy_a:
            print(f"  {result.rank}. {result.chunk_id} score={result.score:.4f}")
        print("Strategy B | AI-Enhanced Retrieval")
        for result in comparison.strategy_b:
            print(f"  {result.rank}. {result.chunk_id} score={result.score:.4f}")
        print()


if __name__ == "__main__":
    main()
