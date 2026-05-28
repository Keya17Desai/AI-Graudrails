from app.llm.ollama_client import ask_ollama, is_ollama_available
from app.llm.groq_client import ask_groq
from app.core.config import settings


def ask_llm(prompt: str, system_prompt: str = "") -> str:
    if is_ollama_available():
        return ask_ollama(prompt, system_prompt)
    if settings.groq_api_key:
        return ask_groq(prompt, system_prompt)
    raise RuntimeError("No LLM available. Start Ollama or set GROQ_API_KEY.")
