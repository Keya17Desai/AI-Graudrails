"""Run this once to ingest all documents into ChromaDB."""
import sys
from app.rag.ingestion import ingest_folder

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/documents"
    print(f"Ingesting documents from: {folder}\n")
    results = ingest_folder(folder)
    total = sum(results.values())
    print(f"\nDone. {len(results)} files ingested, {total} total chunks stored.")
