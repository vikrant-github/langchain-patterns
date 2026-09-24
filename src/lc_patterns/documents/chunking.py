from hashlib import sha256

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents while preserving their metadata."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )

        return splitter.split_documents(documents)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("Unable to split the document into chunks.") from exc


def add_chunk_metadata(chunks: list[Document]) -> list[Document]:
    """Add deterministic application metadata to each document chunk."""
    try:
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = sha256(
                f"{chunk.metadata.get('source', '')}:{chunk.page_content}".encode()
            ).hexdigest()

            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                }
            )

        return chunks
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Unable to add metadata to document chunks.") from exc