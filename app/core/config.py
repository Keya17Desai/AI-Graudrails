from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "company-chatbot"
    app_env: str = "development"

    jwt_secret_key: str = "change-me"
    jwt_expire_minutes: int = 480

    postgres_user: str = "chatbot"
    postgres_password: str = "chatbot123"
    postgres_db: str = "chatbotdb"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # "http" = external ChromaDB container (local dev)
    # "embedded" = in-process PersistentClient (HuggingFace Spaces)
    chroma_mode: str = "http"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_dir: str = "./chroma_data"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    groq_api_key: str = ""

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    class Config:
        env_file = ".env"


settings = Settings()
