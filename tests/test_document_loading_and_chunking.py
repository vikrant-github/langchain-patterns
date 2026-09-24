from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from langchain_core.documents import Document

from lc_patterns.documents.chunking import (
    add_chunk_metadata,
    split_documents,
)

EXERCISE_PATH = (
    Path(__file__).parents[1]
    / "exercises"
    / "07-documents-embeddings-semantic-search"
    / "01_document_loading_chunking_metadata.py"
)

spec = spec_from_file_location("document_loading_chunking_metadata", EXERCISE_PATH)
assert spec is not None
assert spec.loader is not None

module = module_from_spec(spec)
spec.loader.exec_module(module)

load_document = module.load_document


def test_load_document_raises_for_missing_file() -> None:
    with pytest.raises(RuntimeError, match="Unable to load the PDF document"):
        load_document(Path("does-not-exist.pdf"))


def test_split_documents_creates_chunks_and_preserves_metadata() -> None:
    documents = [
        Document(
            page_content=" ".join(["This is test content."] * 100),
            metadata={"source": "test.pdf", "page": 1},
        )
    ]

    chunks = split_documents(documents)

    assert len(chunks) > 1
    assert all(chunk.page_content for chunk in chunks)
    assert all(chunk.metadata["source"] == "test.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)


def test_add_chunk_metadata_assigns_chunk_identity_and_index() -> None:
    chunks = [
        Document(page_content="First chunk", metadata={"source": "test.pdf"}),
        Document(page_content="Second chunk", metadata={"source": "test.pdf"}),
    ]

    result = add_chunk_metadata(chunks)

    assert result is chunks
    assert result[0].metadata["chunk_index"] == 0
    assert result[1].metadata["chunk_index"] == 1
    assert result[0].metadata["chunk_id"] != result[1].metadata["chunk_id"]
    assert len(result[0].metadata["chunk_id"]) == 64
    assert len(result[1].metadata["chunk_id"]) == 64