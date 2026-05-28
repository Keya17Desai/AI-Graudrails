import time
from groq import Groq
from app.core.config import settings
from app.monitoring.metrics import llm_request_duration_seconds


def ask_groq(prompt: str, system_prompt: str = "") -> str:
    client = Groq(api_key=settings.groq_api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    llm_request_duration_seconds.labels(backend="groq").observe(time.perf_counter() - start)
    return response.choices[0].message.content
