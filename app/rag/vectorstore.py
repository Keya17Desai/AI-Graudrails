import chromadb
from app.core.config import settings

_client = None
_collection = None

COLLECTION_NAME = "company_documents"


def get_client():
    global _client
    if _client is None:
        if settings.chroma_mode == "embedded":
            # Embedded mode: ChromaDB runs in-process, data stored in a local directory.
            # Used on HuggingFace Spaces where a separate ChromaDB container can't run.
            _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        else:
            _client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def search(query_embedding: list, user_role: str, n_results: int = 5) -> dict:
    where = {f"can_{user_role}": True}
    return get_collection().query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
