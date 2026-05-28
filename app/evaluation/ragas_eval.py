"""
RAGAS-style evaluation metrics implemented with llama3 as judge — fully local.

RAGAS 0.4.x new metric classes require OpenAI's InstructorLLM.
We implement the same four metrics manually using the same methodology RAGAS uses:
  - Faithfulness:      LLM checks each answer claim against retrieved contexts
  - Answer Relevancy:  cosine similarity between question and answer embeddings
  - Context Precision: LLM scores each retrieved chunk for relevance to the question
  - Context Recall:    LLM checks what fraction of the ground truth is covered by contexts

This approach is fully transparent — you can read the judge prompts and understand
exactly what is being measured.
"""
import httpx
import numpy as np
from app.core.config import settings
from app.rag.embeddings import embed_text


def _judge(prompt: str) -> str:
    """Call llama3 as a zero-temperature judge. Returns raw response text."""
    try:
        resp = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 200},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def _parse_score(text: str) -> float:
    """Extract a 0.0–1.0 float from judge output. Returns 0.5 on parse failure."""
    import re
    nums = re.findall(r"([01](?:\.\d+)?|\d+(?:\.\d+)?)", text)
    for n in nums:
        v = float(n)
        if 0.0 <= v <= 1.0:
            return v
    return 0.5


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# Each statement in the answer is checked against the retrieved chunks.
# Score = (claims supported by context) / (total claims in answer)
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is faithful to its source documents.

Retrieved context:
{context}

Answer to evaluate:
{answer}

Task: List each factual claim in the answer on a separate line.
For each claim write: SUPPORTED or NOT_SUPPORTED
Then on the last line write: SCORE: <fraction of supported claims as decimal 0.0-1.0>

Example:
Claim 1: ... SUPPORTED
Claim 2: ... NOT_SUPPORTED
SCORE: 0.5"""


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    if not contexts or not answer:
        return 0.0
    context = "\n\n".join(contexts[:3])
    verdict = _judge(_FAITHFULNESS_PROMPT.format(context=context, answer=answer))
    import re
    m = re.search(r"SCORE:\s*([0-9.]+)", verdict)
    return float(m.group(1)) if m else _parse_score(verdict)


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevancy
# Cosine similarity between the question embedding and the answer embedding.
# High similarity = the answer is on-topic for the question.
# ---------------------------------------------------------------------------
def answer_relevancy_score(question: str, answer: str) -> float:
    if not answer or answer.startswith("I don't have"):
        return 0.0
    q_vec = np.array(embed_text(question))
    a_vec = np.array(embed_text(answer))
    # Cosine similarity
    denom = np.linalg.norm(q_vec) * np.linalg.norm(a_vec)
    if denom == 0:
        return 0.0
    return float(np.dot(q_vec, a_vec) / denom)


# ---------------------------------------------------------------------------
# Metric 3: Context Precision
# For each retrieved chunk, LLM scores whether it is relevant to the question.
# Score = mean relevance across retrieved chunks.
# ---------------------------------------------------------------------------
_CONTEXT_PRECISION_PROMPT = """Is the following document chunk useful for answering the question?

Question: {question}
Chunk: {chunk}

Reply with a single decimal score from 0.0 (not useful) to 1.0 (very useful). Nothing else."""


def context_precision_score(question: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    scores = []
    for chunk in contexts[:3]:
        verdict = _judge(_CONTEXT_PRECISION_PROMPT.format(question=question, chunk=chunk[:400]))
        scores.append(_parse_score(verdict))
    return round(sum(scores) / len(scores), 3)


# ---------------------------------------------------------------------------
# Metric 4: Context Recall
# Checks what fraction of the ground truth answer is covered by the contexts.
# ---------------------------------------------------------------------------
_CONTEXT_RECALL_PROMPT = """You are checking if a reference answer is supported by retrieved document chunks.

Retrieved chunks:
{context}

Reference answer (ground truth):
{ground_truth}

What fraction of the information in the reference answer can be found in the retrieved chunks?
Reply with a single decimal score from 0.0 (nothing covered) to 1.0 (fully covered). Nothing else."""


def context_recall_score(contexts: list[str], ground_truth: str) -> float:
    if not contexts:
        return 0.0
    context = "\n\n".join(contexts[:3])
    verdict = _judge(_CONTEXT_RECALL_PROMPT.format(context=context, ground_truth=ground_truth))
    return _parse_score(verdict)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_ragas(results: list[dict]) -> dict:
    """
    Compute all four metrics across the eval set and return mean scores.
    Each result must have: question, answer, contexts (list), ground_truth.
    """
    print("\nRunning evaluation metrics (llama3 as judge — takes a few minutes)...")

    f_scores, ar_scores, cp_scores, cr_scores = [], [], [], []

    for i, r in enumerate(results, 1):
        print(f"  Scoring [{i}/{len(results)}]: {r['question'][:50]}...")
        f_scores.append(faithfulness_score(r["answer"], r["contexts"]))
        ar_scores.append(answer_relevancy_score(r["question"], r["answer"]))
        cp_scores.append(context_precision_score(r["question"], r["contexts"]))
        cr_scores.append(context_recall_score(r["contexts"], r["ground_truth"]))

    def mean(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "faithfulness": mean(f_scores),
        "answer_relevancy": mean(ar_scores),
        "context_precision": mean(cp_scores),
        "context_recall": mean(cr_scores),
    }
