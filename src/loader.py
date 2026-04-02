from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DOCS_DIR


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_documents(docs_dir: Optional[Path] = None) -> List[Document]:
    base_dir = docs_dir or DOCS_DIR
    if not base_dir.exists():
        return []

    raw_docs: List[Document] = []

    for path in sorted(base_dir.glob("**/*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        content = _read_text_file(path).strip()
        if not content:
            continue

        raw_docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(path.relative_to(base_dir.parent)),
                    "filename": path.name,
                    "file_type": path.suffix.lower(),
                },
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    for idx, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = idx + 1
    return chunks
