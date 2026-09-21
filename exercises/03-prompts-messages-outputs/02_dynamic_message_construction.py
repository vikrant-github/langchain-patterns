"""Demonstrate few-shot prompting through dynamic message construction.

The message array contains system instructions, human/AI example pairs,
and a new user question. The examples provide the few-shot context the
model uses to infer the expected response pattern.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from lc_patterns.models.chat import get_nova_2_lite


def create_conversation(
    role: str, examples: list[dict[str, str]], new_question: str
) -> list[BaseMessage]:
    """Build message arrays programmatically."""
    messages: list[BaseMessage] = [SystemMessage(content=f"You are a {role}.")]

    # Add few-shot examples
    for example in examples:
        messages.append(HumanMessage(content=example["question"]))
        messages.append(AIMessage(content=example["answer"]))

    messages.append(HumanMessage(content=new_question))
    return messages


def print_token_usage(response):
    """Print token counts returned by the model response."""
    usage = response.usage_metadata
    if usage:
        print(
            f"Tokens: input={usage.get('input_tokens', 'N/A')}, "
            f"output={usage.get('output_tokens', 'N/A')}, "
            f"total={usage.get('total_tokens', 'N/A')}"
        )
    else:
        print("Token usage information unavailable.")


def main():
    nova_2_lite = get_nova_2_lite()

    # Few-Shot Learning with Messages
    emoji_messages = create_conversation(
        "emoji translator",
        [
            {"question": "happy", "answer": "😊"},
            {"question": "sad", "answer": "😢"},
            {"question": "excited", "answer": "🎉"},
        ],
        "surprised",
    )

    print(f"Messages constructed: {len(emoji_messages)}")

    try:
        response = nova_2_lite.invoke(emoji_messages)
        print(f"AI Response: {response.content}")  # Expected: 😮
        print_token_usage(response)
    except Exception as error:  # noqa: BLE001
        print(f"Message invocation failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")


if __name__ == "__main__":
    main()