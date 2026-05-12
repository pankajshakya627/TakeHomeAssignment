# Retrieval Benchmark Evidence

Command:

```bash
PYTHONPATH=.deps:. python3 -m rag_assessment.benchmark
```

Metadata:

```json
{
  "embedding_model": "SentenceTransformerTextEmbeddingModel",
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
| 1 | `scaling_peak_load` | 0.3229 | `scaling_peak_load` | 0.7285 |
| 2 | `cache_rate_limit` | 0.2684 | `data_pipeline` | 0.3669 |
| 3 | `observability_slo` | 0.2600 | `cache_rate_limit` | 0.3455 |

Observation: both strategies identify the scalability chunk, but query expansion
more than doubles its similarity score and changes the rest of the top-3 set by
adding operational terms such as autoscaling, queue depth, backpressure, Kafka,
latency, and throughput.

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
| 1 | `security_credentials` | 0.7531 | `security_credentials` | 0.8604 |
| 2 | `cache_rate_limit` | 0.2101 | `observability_slo` | 0.2623 |
| 3 | `model_guardrails` | 0.1911 | `cache_rate_limit` | 0.2431 |

Observation: both strategies retrieve the security chunk first, while the
expanded query produces a stronger top score by adding explicit controls such as
envelope encryption, key rotation, scoped IAM, revocation, and audit logging.

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
| 1 | `model_guardrails` | 0.4094 | `model_guardrails` | 0.7019 |
| 2 | `observability_slo` | 0.3695 | `observability_slo` | 0.3794 |
| 3 | `deployment_rollbacks` | 0.2876 | `deployment_rollbacks` | 0.3364 |

Observation: both strategies retrieve the AI quality chunk first. The expanded
query strengthens the intended match by adding guardrail, citation, toxicity,
policy, hallucination, confidence, fallback, and safe-response terms.
