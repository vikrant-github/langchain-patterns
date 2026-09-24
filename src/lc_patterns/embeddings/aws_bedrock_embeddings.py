from langchain_aws import BedrockEmbeddings


def create_titan_text_embeddings() -> BedrockEmbeddings:
    """Create Amazon Titan Text Embeddings V2."""
    try:
        return BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
            model_kwargs={
                "dimensions": 1024,
                "normalize": True,
            },
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("Failed to initialize Bedrock embeddings") from error