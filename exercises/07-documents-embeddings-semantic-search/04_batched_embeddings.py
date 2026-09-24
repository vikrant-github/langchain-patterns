from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from lc_patterns.documents.chunking import (
    add_chunk_metadata,
    split_documents,
)
from lc_patterns.embeddings.aws_bedrock_embeddings import (
    create_titan_text_embeddings,
)
from lc_patterns.embeddings.batching import embed_documents_in_batches

PDF_PATH = Path("data/raw/httpd-docs.pdf")
BATCH_SIZE = 50


def load_document(path: Path):
    """Load a PDF into LangChain Document objects."""
    try:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        return PyPDFLoader(str(path)).load()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to load the PDF document from {path}: {exc}"
        ) from exc


def main() -> None:
    try:
        documents = load_document(PDF_PATH)
        chunks = add_chunk_metadata(split_documents(documents))

        texts = [chunk.page_content for chunk in chunks]

        print(f"Loaded pages: {len(documents)}")
        print(f"Created chunks: {len(chunks)}")
        print(f"Embedding batch size: {BATCH_SIZE}")

        embeddings = create_titan_text_embeddings()

        vectors = embed_documents_in_batches(
            embeddings=embeddings,
            texts=texts,
            batch_size=BATCH_SIZE,
        )

        print(f"Generated embeddings: {len(vectors)}")
        print(f"Embedding dimensions: {len(vectors[0])}")
    except RuntimeError as exc:
        print(f"Error: {exc}")
    except (AttributeError, IndexError, OSError, TypeError, ValueError) as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()