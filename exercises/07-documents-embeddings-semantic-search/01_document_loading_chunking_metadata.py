from hashlib import sha256
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path("data/raw/httpd-docs.pdf")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_document(path: Path) -> list[Document]:
    """Load a PDF into LangChain Document objects."""
    try:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        loader = PyPDFLoader(str(path))
        return loader.load()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"Unable to load the PDF document from {path}: {exc}") from exc


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


def main() -> None:
    try:
        documents = load_document(PDF_PATH)

        print(f"Loaded pages: {len(documents)}")

        if documents:
            print("\nFirst document metadata:")
            print(documents[0].metadata)

            print("\nFirst document content:")
            print(documents[0].page_content[:500])

        chunks = split_documents(documents)
        chunks = add_chunk_metadata(chunks)

        print(f"\nCreated chunks: {len(chunks)}")

        if chunks:
            print("\nFirst chunk metadata:")
            print(chunks[0].metadata)

            print("\nFirst chunk content:")
            print(chunks[0].page_content)

            print(f"\nFirst chunk length: {len(chunks[0].page_content)} characters")
    except RuntimeError as exc:
        print(f"Error: {exc}")
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()