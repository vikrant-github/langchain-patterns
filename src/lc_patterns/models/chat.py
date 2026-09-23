from langchain_aws import ChatBedrockConverse


def get_nova_2_lite() -> ChatBedrockConverse:
    """Create the project's default Nova 2 Lite chat model."""
    try:
        return ChatBedrockConverse(
            model="us.amazon.nova-2-lite-v1:0",
            region_name="us-east-1",
        )
    except Exception as error:
        raise RuntimeError("Failed to initialize the Nova 2 Lite chat model") from error


def get_nova_pro() -> ChatBedrockConverse:
    """Create the project's Nova Pro chat model."""
    try:
        return ChatBedrockConverse(
            model="amazon.nova-pro-v1:0",
            region_name="us-east-1",
        )
    except Exception as error:
        raise RuntimeError("Failed to initialize the Nova Pro chat model") from error