from unittest.mock import patch

import pytest

from lc_patterns.embeddings.aws_bedrock_embeddings import (
    create_titan_text_embeddings,
)


@patch("lc_patterns.embeddings.aws_bedrock_embeddings.BedrockEmbeddings")
def test_create_titan_text_embeddings_configuration(mock_embeddings) -> None:
    embeddings = create_titan_text_embeddings()

    assert embeddings is mock_embeddings.return_value
    mock_embeddings.assert_called_once_with(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
        model_kwargs={
            "dimensions": 1024,
            "normalize": True,
        },
    )


@patch("lc_patterns.embeddings.aws_bedrock_embeddings.BedrockEmbeddings")
def test_create_titan_text_embeddings_wraps_initialization_error(
    mock_embeddings,
) -> None:
    original_error = ValueError("invalid Bedrock configuration")
    mock_embeddings.side_effect = original_error

    with pytest.raises(
        RuntimeError, match="Failed to initialize Bedrock embeddings"
    ) as raised_error:
        create_titan_text_embeddings()

    assert raised_error.value.__cause__ is original_error
