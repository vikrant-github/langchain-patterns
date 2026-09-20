from langchain_aws import ChatBedrockConverse

from lc_patterns.models.chat import get_nova_2_lite, get_nova_pro


def test_get_nova_2_lite_configuration() -> None:
    model = get_nova_2_lite()

    assert isinstance(model, ChatBedrockConverse)
    assert model.model == "us.amazon.nova-2-lite-v1:0"
    assert model.region_name == "us-east-1"


def test_get_nova_pro_configuration() -> None:
    model = get_nova_pro()

    assert isinstance(model, ChatBedrockConverse)
    assert model.model == "amazon.nova-pro-v1:0"
    assert model.region_name == "us-east-1"
