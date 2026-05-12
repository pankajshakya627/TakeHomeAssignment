from __future__ import annotations

from rag_assessment.models import Chunk


TECHNICAL_PARAGRAPHS: list[Chunk] = [
    Chunk(
        chunk_id="scaling_peak_load",
        text=(
            "During seasonal traffic spikes the platform keeps checkout responsive "
            "through horizontal autoscaling, request queueing, and explicit backpressure. Kubernetes "
            "adds stateless API pods when CPU, latency, and queue depth rise, while "
            "workers drain Kafka partitions independently so checkout traffic does "
            "not block reporting jobs."
        ),
        metadata={"topic": "scalability"},
    ),
    Chunk(
        chunk_id="cache_rate_limit",
        text=(
            "The edge tier uses CDN caching, token bucket rate limiting, and hot-key "
            "protection to absorb bursty reads. Frequently requested catalog data is "
            "served from Redis with short TTLs, reducing database pressure during "
            "promotions and product launches."
        ),
        metadata={"topic": "performance"},
    ),
    Chunk(
        chunk_id="security_credentials",
        text=(
            "Customer data is protected with envelope encryption, short-lived service "
            "credentials, and automatic key rotation. If a credential leaks, scoped "
            "IAM permissions, secret revocation, and audit logging limit lateral "
            "movement and provide an incident trail."
        ),
        metadata={"topic": "security"},
    ),
    Chunk(
        chunk_id="model_guardrails",
        text=(
            "Generated answers pass through model guardrails before release. The "
            "pipeline validates grounded citations, checks toxicity and policy "
            "violations, rejects hallucinated tool outputs, and falls back to a safe "
            "response when confidence is low."
        ),
        metadata={"topic": "ai_quality"},
    ),
    Chunk(
        chunk_id="observability_slo",
        text=(
            "Observability combines distributed tracing, RED metrics, structured logs, "
            "and SLO burn-rate alerts. On-call engineers inspect latency percentiles, "
            "error budgets, and dependency spans to identify regressions quickly."
        ),
        metadata={"topic": "operations"},
    ),
    Chunk(
        chunk_id="data_pipeline",
        text=(
            "The analytics pipeline ingests events through Kafka, validates schemas in "
            "a registry, and writes curated tables with idempotent Spark jobs. Late "
            "arriving records are reconciled through watermarking and replayable "
            "object storage."
        ),
        metadata={"topic": "data_engineering"},
    ),
    Chunk(
        chunk_id="rag_indexing",
        text=(
            "The retrieval system chunks documents by semantic section, embeds each "
            "chunk, stores vectors in a nearest-neighbor index, and returns the top "
            "matches with source metadata. Re-ranking can be added after vector search "
            "for higher precision."
        ),
        metadata={"topic": "retrieval"},
    ),
    Chunk(
        chunk_id="deployment_rollbacks",
        text=(
            "Deployments use canary releases, automated smoke tests, and progressive "
            "traffic shifting. If health checks fail, the release controller rolls "
            "traffic back and records the failed version for diagnosis."
        ),
        metadata={"topic": "deployment"},
    ),
]
