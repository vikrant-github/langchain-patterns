from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from lc_patterns.documents.chunking import (
    add_chunk_metadata,
    split_documents,
)
from lc_patterns.embeddings.aws_bedrock_embeddings import (
    create_titan_text_embeddings,
)

PDF_PATH = Path("data/raw/httpd-docs.pdf")


def load_document(path: Path):
    """Load a PDF into LangChain Document objects."""
    try:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        loader = PyPDFLoader(str(path))
        return loader.load()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to load the PDF document from {path}: {exc}"
        ) from exc


def main() -> None:
    try:
        documents = load_document(PDF_PATH)
        chunks = split_documents(documents)
        chunks = add_chunk_metadata(chunks)

        print(f"Loaded pages: {len(documents)}")
        print(f"Created chunks: {len(chunks)}")

        embeddings = create_titan_text_embeddings()

        texts = [chunk.page_content for chunk in chunks]
        vectors = embeddings.embed_documents(texts)

        print(f"Generated embeddings: {len(vectors)}")
        print(f"Embedding dimensions: {len(vectors[0])}")

        print("\nFirst chunk:")
        print(chunks[0].page_content[:300])

        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)

        print("\nFirst embedding:")
        print(vectors[0][:10])
    except RuntimeError as exc:
        print(f"Error: {exc}")
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()