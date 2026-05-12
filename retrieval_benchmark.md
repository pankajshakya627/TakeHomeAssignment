# Retrieval Benchmark Evidence

Command:

```bash
PYTHONPATH=.deps:. python3 -m rag_assessment.benchmark
```

Metadata:

```json
{
  "embedding_model": "MockTextEmbeddingModel",
  "query_expander": "MockGenerativeModel",
  "vector_store": "faiss.IndexFlatIP",
  "strategy_a": "Raw Vector Search",
  "strategy_b": "AI-Enhanced Retrieval",
  "top_k": 3,
  "chunk_count": 8
}
```

## Query 1

Input query:

```text
How does the system handle peak load?
```

Expanded query:

```text
How does the system handle peak load? horizontal autoscaling queue depth request queueing backpressure Kubernetes pods Kafka workers latency throughput
```

| Rank | Strategy A: Raw Vector Search | Score | Strategy B: AI-Enhanced Retrieval | Score |
|---:|---|---:|---|---:|
| 1 | `deployment_rollbacks` | 0.1827 | `scaling_peak_load` | 0.4672 |
| 2 | `rag_indexing` | 0.1767 | `deployment_rollbacks` | 0.1432 |
| 3 | `scaling_peak_load` | 0.1126 | `rag_indexing` | 0.1370 |

Observation: the raw query is ambiguous around "handle" and "system", while
query expansion adds operational terms that move the scalability chunk to rank 1.

## Query 2

Input query:

```text
What protects customer data if credentials leak?
```

Expanded query:

```text
What protects customer data if credentials leak? envelope encryption key rotation short-lived credentials scoped IAM secret revocation audit logging
```

| Rank | Strategy A: Raw Vector Search | Score | Strategy B: AI-Enhanced Retrieval | Score |
|---:|---|---:|---|---:|
| 1 | `security_credentials` | 0.1998 | `security_credentials` | 0.5758 |
| 2 | `cache_rate_limit` | 0.0648 | `cache_rate_limit` | 0.1039 |
| 3 | `scaling_peak_load` | 0.0000 | `rag_indexing` | 0.1014 |

Observation: both strategies retrieve the correct security chunk first, but the
expanded query creates a much stronger similarity score by adding specific
security controls.

## Query 3

Input query:

```text
How are bad model outputs detected before users see them?
```

Expanded query:

```text
How are bad model outputs detected before users see them? model guardrails grounded citations toxicity policy checks hallucinated tool outputs confidence fallback safe response
```

| Rank | Strategy A: Raw Vector Search | Score | Strategy B: AI-Enhanced Retrieval | Score |
|---:|---|---:|---|---:|
| 1 | `model_guardrails` | 0.1311 | `model_guardrails` | 0.5557 |
| 2 | `security_credentials` | 0.0645 | `data_pipeline` | 0.1045 |
| 3 | `cache_rate_limit` | 0.0598 | `observability_slo` | 0.0784 |

Observation: the expanded query strengthens the intended AI quality match by
adding terms for guardrails, citations, policy checks, hallucinations, and safe
fallbacks.
