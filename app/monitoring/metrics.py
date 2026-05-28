"""
All Prometheus metric definitions live here.
Import these objects anywhere in the app — they are module-level singletons.
"""
from prometheus_client import Counter, Histogram, Gauge

# --- Request metrics ---
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# --- Guardrail metrics ---
guardrail_blocks_total = Counter(
    "guardrail_blocks_total",
    "Total requests blocked by guardrails",
    ["stage", "reason"],  # stage: input|output, reason: pii_in_input|unsafe_content|pii_in_output
)

# --- RAG metrics ---
rag_chunks_retrieved = Histogram(
    "rag_chunks_retrieved",
    "Number of document chunks retrieved per query",
    buckets=[0, 1, 2, 3, 4, 5],
)

rag_query_duration_seconds = Histogram(
    "rag_query_duration_seconds",
    "Time spent on RAG retrieval + LLM call",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# --- LLM metrics ---
llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "Time spent waiting for LLM response",
    ["backend"],  # ollama | groq
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# --- Auth metrics ---
auth_attempts_total = Counter(
    "auth_attempts_total",
    "Login attempts",
    ["result"],  # success | failure
)
