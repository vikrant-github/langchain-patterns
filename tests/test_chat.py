from unittest.mock import patch

import pytest

from lc_patterns.models.chat import get_nova_2_lite, get_nova_pro


@patch("lc_patterns.models.chat.ChatBedrockConverse")
def test_get_nova_2_lite_configuration(mock_chat) -> None:
    model = get_nova_2_lite()

    assert model is mock_chat.return_value
    mock_chat.assert_called_once_with(
        model="us.amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
    )


@patch("lc_patterns.models.chat.ChatBedrockConverse")
def test_get_nova_pro_configuration(mock_chat) -> None:
    model = get_nova_pro()

    assert model is mock_chat.return_value
    mock_chat.assert_called_once_with(
        model="amazon.nova-pro-v1:0",
        region_name="us-east-1",
    )


@patch("lc_patterns.models.chat.ChatBedrockConverse")
def test_get_nova_2_lite_wraps_initialization_error(mock_chat) -> None:
    original_error = ValueError("invalid Bedrock configuration")
    mock_chat.side_effect = original_error

    with pytest.raises(
        RuntimeError, match="Failed to initialize the Nova 2 Lite chat model"
    ) as raised_error:
        get_nova_2_lite()

    assert raised_error.value.__cause__ is original_error


@patch("lc_patterns.models.chat.ChatBedrockConverse")
def test_get_nova_pro_wraps_initialization_error(mock_chat) -> None:
    original_error = ValueError("invalid Bedrock configuration")
    mock_chat.side_effect = original_error

    with pytest.raises(
        RuntimeError, match="Failed to initialize the Nova Pro chat model"
    ) as raised_error:
        get_nova_pro()

    assert raised_error.value.__cause__ is original_error
