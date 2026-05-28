import os
import uuid
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.embeddings import embed_batch
from app.rag.vectorstore import get_collection

ALL_ROLES = ["employee", "hr", "finance", "marketing", "c_level"]

# which roles can access each folder
ROLE_MAP = {
    "hr":        ["hr", "c_level"],
    "finance":   ["finance", "c_level"],
    "marketing": ["marketing", "finance", "c_level"],
    "general":   ["employee", "hr", "finance", "marketing", "c_level"],
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    loaders = {
        ".txt":  TextLoader,
        ".pdf":  PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".csv":  CSVLoader,
    }
    cls = loaders.get(ext)
    if not cls:
        raise ValueError(f"Unsupported file type: {ext}")
    return cls(file_path)


def _allowed_roles_for_path(file_path: str) -> list:
    for folder, roles in ROLE_MAP.items():
        if f"/{folder}/" in file_path.replace("\\", "/"):
            return roles
    return ["c_level"]


def ingest_file(file_path: str) -> int:
    docs = _get_loader(file_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    texts = [c.page_content for c in chunks]
    embeddings = embed_batch(texts)
    allowed_roles = _allowed_roles_for_path(file_path)
    filename = Path(file_path).name

    collection = get_collection()
    # Store one boolean per role — clean RBAC filtering in ChromaDB
    role_flags = {f"can_{r}": (r in allowed_roles) for r in ALL_ROLES}

    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"source": filename, "chunk_index": i, **role_flags}
            for i, _ in enumerate(chunks)
        ],
    )
    return len(chunks)


def ingest_folder(folder_path: str) -> dict:
    results = {}
    supported = {".txt", ".pdf", ".docx", ".csv"}
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if Path(filename).suffix.lower() not in supported:
                continue
            file_path = os.path.join(root, filename)
            count = ingest_file(file_path)
            results[filename] = count
            print(f"  Ingested: {filename} → {count} chunks")
    return results
