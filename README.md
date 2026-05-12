# Context-Aware Retrieval Engine

Local implementation of the Senior Gen AI semantic RAG assessment. The project
ingests a small technical text corpus, creates embeddings, stores vectors in a
local similarity index, and benchmarks two retrieval strategies.

## What This Demonstrates

- Local document ingestion and chunk management.
- Embedding generation through a Vertex AI-like interface.
- FAISS vector search with cosine-style inner product scoring.
- Mocked query expansion through a Vertex AI-like generative model.
- Structured benchmark output using Pydantic models.
- A comparison between raw vector retrieval and AI-enhanced retrieval.

## Retrieval Strategies

Strategy A, Raw Vector Search, embeds the user query exactly as submitted and
retrieves the top matching chunks.

Strategy B, AI-Enhanced Retrieval, first expands the user query with relevant
technical vocabulary, then embeds the rewritten query and performs the same
vector search.

Example:

```text
Input:    How does the system handle peak load?
Expanded: How does the system handle peak load? horizontal autoscaling queue depth
          request queueing backpressure Kubernetes pods Kafka workers latency throughput
```

## Project Layout

```text
.circleci/
  config.yml                           CI workflow for pytest
rag_assessment/
  benchmark.py                         CLI benchmark runner
  engine.py                            RAG orchestration class
  mock_vertex.py                       Mock TextEmbeddingModel and GenerativeModel
  models.py                            Pydantic response schemas
  sample_data.py                       8 technical assessment paragraphs
  sentence_transformer_embedding.py    Optional sentence-transformers wrapper
  vector_store.py                      FAISS vector store with NumPy fallback
tests/
  test_rag_engine.py                   Retrieval and benchmark tests
run_benchmark.py                       Convenience entrypoint
retrieval_benchmark.md                 Captured Strategy A vs Strategy B evidence
requirements.txt                       Python dependencies
```

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If you do not want to create a virtual environment, install dependencies into
your active Python environment instead:

```bash
python3 -m pip install -r requirements.txt
```

## Run The Benchmark

Readable table output:

```bash
python3 run_benchmark.py
```

Structured JSON output:

```bash
python3 -m rag_assessment.benchmark --json
```

Change the number of retrieved chunks:

```bash
python3 -m rag_assessment.benchmark --top-k 5
```

## FAISS Notes

When `faiss-cpu` is installed, the vector store uses `faiss.IndexFlatIP`.
If FAISS is unavailable, the same API falls back to NumPy inner product search
so the project can still run in minimal offline environments.

Vectors are L2-normalized before indexing. With normalized vectors, FAISS inner
product is equivalent to cosine similarity, which is a good fit for semantic
embedding retrieval because ranking depends on direction rather than raw vector
magnitude. Euclidean distance is less useful here unless the embedding model was
trained and calibrated for L2 distance specifically.

This workspace also supports the project-local `.deps` install created during
development:

```bash
PYTHONPATH=.deps:. python3 -m rag_assessment.benchmark --json
```

The benchmark metadata reports the active backend:

```json
{
  "vector_store": "faiss.IndexFlatIP"
}
```

or:

```json
{
  "vector_store": "numpy.inner"
}
```

## Embedding Backends

The default benchmark backend is `SentenceTransformerTextEmbeddingModel`, which
uses the local `sentence-transformers` library to simulate Vertex AI
`textembedding-gecko` style semantic embeddings.

The default model is `all-MiniLM-L6-v2`:

```bash
python3 -m rag_assessment.benchmark \
  --embedding-backend sentence-transformers \
  --model-name all-MiniLM-L6-v2
```

The sentence-transformers path may download model weights on first use unless
the model is already cached locally.

For deterministic offline tests or demos, use the mocked embedding backend:

```bash
python3 -m rag_assessment.benchmark --embedding-backend mock
```

`MockTextEmbeddingModel` behaves like
`vertexai.language_models.TextEmbeddingModel` without cloud credentials or
network calls. `MockGenerativeModel` behaves like a Vertex generative model and
returns a response object with a `.text` field.

## Run Tests

```bash
python3 -m pytest
```

Expected result:

```text
8 passed
```

## Expected Benchmark Behavior

The benchmark runs three complex queries:

- `How does the system handle peak load?`
- `What protects customer data if credentials leak?`
- `How are bad model outputs detected before users see them?`

For the peak-load query, raw search intentionally starts with ambiguous phrasing.
The expanded query adds terms like `autoscaling`, `queue depth`, and
`backpressure`, causing Strategy B to rank the scalability chunk first.

## Assessment Requirement Mapping

- Embedding model: `SentenceTransformerTextEmbeddingModel` by default, with
  `MockTextEmbeddingModel` available for deterministic offline tests.
- Vector database: `FaissVectorStore` using FAISS when installed.
- Mocking: `MockTextEmbeddingModel` and `MockGenerativeModel` mirror the Vertex
  AI model interfaces used in the assessment.
- Orchestration: `ContextAwareRetrievalEngine` manages ingestion, search, query
  expansion, and benchmarking.
- Benchmarking: `run_benchmark.py` and `rag_assessment.benchmark` output
  structured comparisons for three queries.
- Dev evidence: `retrieval_benchmark.md` contains captured Strategy A vs
  Strategy B output using `sentence-transformers` and FAISS.

## Production Migration To Vertex AI Vector Search

To migrate this local implementation to GCP:

1. Replace `MockTextEmbeddingModel` with
   `vertexai.language_models.TextEmbeddingModel.from_pretrained(...)` or the
   current Vertex AI embeddings API.
2. Generate embeddings during ingestion and store the vectors plus metadata in a
   durable document store such as Cloud Storage, Firestore, BigQuery, or
   AlloyDB, depending on the serving needs.
3. Create a Vertex AI Vector Search index configured for cosine-compatible
   retrieval. For normalized vectors, use dot-product distance if available, or
   use the cosine distance option supported by the target index type.
4. Upsert chunk IDs, embeddings, and metadata into the deployed Vector Search
   index endpoint.
5. Replace `FaissVectorStore.search()` with a client call to the Vector Search
   index endpoint, then hydrate returned chunk IDs from the document store.
6. Replace `MockGenerativeModel` with Gemini on Vertex AI for query rewriting,
   keeping the same `expand_query(query) -> str` boundary so retrieval logic
   remains unchanged.
7. Add production evaluation using held-out queries, expected chunk IDs,
   Recall@K, MRR, latency percentiles, and cost metrics.
