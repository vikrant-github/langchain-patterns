from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path("data/raw/httpd-docs.pdf")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_document(path: Path):
    """Load a PDF into LangChain Document objects."""
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    loader = PyPDFLoader(str(path))
    return loader.load()


def split_documents(documents):
    """Split documents while preserving their metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    return splitter.split_documents(documents)


def main() -> None:
    documents = load_document(PDF_PATH)

    print(f"Loaded pages: {len(documents)}")

    if documents:
        print("\nFirst document metadata:")
        print(documents[0].metadata)

        print("\nFirst document content:")
        print(documents[0].page_content[:500])

    chunks = split_documents(documents)

    print(f"\nCreated chunks: {len(chunks)}")

    if chunks:
        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)

        print("\nFirst chunk content:")
        print(chunks[0].page_content)

        print(f"\nFirst chunk length: {len(chunks[0].page_content)} characters")


if __name__ == "__main__":
    main()