import time
import requests
from app.core.config import settings
from app.monitoring.metrics import llm_request_duration_seconds


def ask_ollama(prompt: str, system_prompt: str = "") -> str:
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    start = time.perf_counter()
    response = requests.post(
        f"{settings.ollama_host}/api/generate",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    llm_request_duration_seconds.labels(backend="ollama").observe(time.perf_counter() - start)
    return response.json()["response"]


def is_ollama_available() -> bool:
    try:
        r = requests.get(f"{settings.ollama_host}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
