"""
Runs each eval test case through the live RAG pipeline and collects:
  - the generated answer
  - the retrieved context chunks
  - latency

This produces the raw data that the metrics layer scores.
"""
import time
from app.rag.embeddings import embed_text
from app.rag.vectorstore import search
from app.llm.router import ask_llm
from app.rag.retriever import SYSTEM_PROMPT, RAG_PROMPT


def run_single(question: str, user_role: str) -> dict:
    """
    Run one question through RAG and return the answer + retrieved chunks.
    Returns dict with keys: question, answer, contexts, latency_seconds.
    """
    start = time.perf_counter()

    query_vector = embed_text(question)
    results = search(query_vector, user_role, n_results=3)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not chunks:
        return {
            "question": question,
            "answer": "I don't have that information in the company knowledge base.",
            "contexts": [],
            "sources": [],
            "latency_seconds": round(time.perf_counter() - start, 2),
        }

    context_parts = [f"[Source: {meta['source']}]\n{chunk}" for chunk, meta in zip(chunks, metadatas)]
    context = "\n\n".join(context_parts)
    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = ask_llm(prompt=prompt, system_prompt=SYSTEM_PROMPT)

    return {
        "question": question,
        "answer": answer.strip(),
        "contexts": chunks,          # raw chunk texts — what RAGAS needs
        "sources": [m["source"] for m in metadatas],
        "latency_seconds": round(time.perf_counter() - start, 2),
    }


def run_all(test_cases: list[dict]) -> list[dict]:
    """Run all test cases and attach ground_truth to each result."""
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {case['question'][:60]}...")
        result = run_single(case["question"], case["user_role"])
        result["ground_truth"] = case["ground_truth"]
        result["user_role"] = case["user_role"]
        results.append(result)
    return results
