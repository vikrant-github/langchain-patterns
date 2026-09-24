"""
Semantic search over document embeddings.

Run:
    uv run python exercises/07-documents-embeddings-semantic-search/03_similarity_search.py

This exercise demonstrates:
- Loading the Apache HTTP Server documentation PDF.
- Chunking documents with preserved metadata.
- Generating embeddings with Amazon Titan Text Embeddings V2.
- Storing chunks and embeddings in an in-memory vector store.
- Performing semantic similarity search with relevance scores.

The vector store is intentionally in-memory; persistence is addressed separately.
"""

from pathlib import Path
from time import perf_counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore

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
        print(f"Starting PDF load: {PDF_PATH}")
        start_time = perf_counter()
        documents = load_document(PDF_PATH)
        print(f"PDF load complete in {perf_counter() - start_time:.2f}s")

        print("Chunking documents...")
        start_time = perf_counter()
        chunks = split_documents(documents)
        chunks = add_chunk_metadata(chunks)
        print(f"Chunking complete: {len(chunks)} chunks in {perf_counter() - start_time:.2f}s")

        print("Initializing Bedrock embeddings...")
        start_time = perf_counter()
        embeddings = create_titan_text_embeddings()
        print(f"Embedding client ready in {perf_counter() - start_time:.2f}s")

        texts = [chunk.page_content for chunk in chunks]
        vectors: list[list[float]] = []
        print(f"Generating embeddings for {len(texts)} chunks...")
        for index, text in enumerate(texts, start=1):
            print(f"  Embedding chunk {index}/{len(texts)} ({len(text)} chars)")
            vectors.extend(embeddings.embed_documents([text]))

        print(f"Generated embeddings: {len(vectors)} vectors")
        print(f"Embedding dimensions: {len(vectors[0])}")

        print("Building in-memory vector store...")
        start_time = perf_counter()
        vector_store = InMemoryVectorStore.from_documents(
            chunks,
            embedding=embeddings,
        )
        print(f"Vector store created in {perf_counter() - start_time:.2f}s")

        queries = [
            "How does Apache HTTP Server handle authentication?",
            "How do I configure virtual hosts?",
            "How does Apache handle access control?",
            "How do I configure SSL and HTTPS?",
        ]

        for query_index, query in enumerate(queries, start=1):
            print(f'\n[{query_index}/{len(queries)}] Query: "{query}"')
            start_time = perf_counter()
            results = vector_store.similarity_search_with_score(
                query,
                k=3,
            )
            print(f"Search completed in {perf_counter() - start_time:.2f}s")

            for rank, (document, score) in enumerate(results, start=1):
                print(f"\n{rank}. Score: {score:.4f}")
                print(f"   Source: {document.metadata.get('source')}")
                print(f"   Page: {document.metadata.get('page')}")
                print(f"   Chunk ID: {document.metadata.get('chunk_id')}")
                print(f"   Content: {document.page_content[:300]}...")
    except RuntimeError as exc:
        print(f"Error: {exc}")
    except (AttributeError, IndexError, OSError, TypeError, ValueError) as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()