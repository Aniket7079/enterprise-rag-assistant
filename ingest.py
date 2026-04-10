from pathlib import Path

from src.config import CHROMA_DIR, COLLECTION_NAME 
from src.embeddings import get_embeddings 
from src.loader import load_documents
from src.retriever import build_vectorstore
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

print("API KEY:", os.getenv("OPENAI_API_KEY"))  # debug line

def main() -> None:
    docs = load_documents(Path("data/docs"))
    if not docs:
        raise SystemExit("No documents found in data/docs. Add .txt or .md files first.")

    vectorstore = build_vectorstore(
        documents=docs,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embeddings=get_embeddings(),
    )

    count = vectorstore._collection.count()  # type: ignore[attr-defined]
    print(f"Indexed {count} chunks into Chroma at '{CHROMA_DIR}'.")


if __name__ == "__main__":
    main()
