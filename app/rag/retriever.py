import time
from app.rag.embeddings import embed_text
from app.rag.vectorstore import search
from app.llm.router import ask_llm
from app.monitoring.metrics import rag_chunks_retrieved, rag_query_duration_seconds

SYSTEM_PROMPT = """You are a helpful internal company assistant for Nexora Technologies.
Answer questions using ONLY the company documents provided to you.
If the answer is not in the documents, say exactly: "I don't have that information in the company knowledge base."
Never make up information. Be concise and professional.
Always mention which document your answer came from."""

RAG_PROMPT = """Company documents:
{context}

---
Question: {question}

Answer based only on the documents above."""


def query(question: str, user_role: str) -> dict:
    start = time.perf_counter()

    query_vector = embed_text(question)
    results = search(query_vector, user_role, n_results=3)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    rag_chunks_retrieved.observe(len(chunks))

    if not chunks:
        rag_query_duration_seconds.observe(time.perf_counter() - start)
        return {
            "answer": "I don't have that information in the company knowledge base.",
            "sources": [],
            "chunks_used": 0,
        }

    context_parts = [f"[Source: {meta['source']}]\n{chunk}" for chunk, meta in zip(chunks, metadatas)]
    context = "\n\n".join(context_parts)

    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = ask_llm(prompt=prompt, system_prompt=SYSTEM_PROMPT)

    rag_query_duration_seconds.observe(time.perf_counter() - start)

    sources = list({m["source"] for m in metadatas})
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
    }
