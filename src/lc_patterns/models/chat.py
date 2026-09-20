from langchain_aws import ChatBedrockConverse


def get_chat_model() -> ChatBedrockConverse:
    """Create the project's default chat model."""
    return ChatBedrockConverse(
        model="us.amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
    )