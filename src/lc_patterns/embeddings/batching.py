from langchain_core.embeddings import Embeddings


def embed_documents_in_batches(
    embeddings: Embeddings,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    """Embed document texts in configurable batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embeddings.embed_documents(batch))

    return vectors