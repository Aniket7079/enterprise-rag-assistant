from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from .config import CHROMA_DIR, COLLECTION_NAME, TOP_K, DOCS_DIR
from .loader import load_documents
from .embeddings import get_embeddings


def build_vectorstore(documents, persist_directory: str, collection_name: str, embeddings):
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def ensure_vectorstore(force_rebuild: bool = False) -> Chroma:
    embeddings = get_embeddings()
    persist_path = Path(CHROMA_DIR)
    docs = load_documents(DOCS_DIR)

    if not docs:
        raise ValueError("No docs found in data/docs.")

    if force_rebuild and persist_path.exists():
        import shutil
        shutil.rmtree(persist_path)

    persist_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )

    # Populate the collection the first time, or rebuild from scratch on refresh.
    if force_rebuild or vectorstore._collection.count() == 0:  # type: ignore[attr-defined]
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(persist_path),
            collection_name=COLLECTION_NAME,
        )

    return vectorstore


def get_retriever() -> Any:
    vectorstore = ensure_vectorstore(force_rebuild=False)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
