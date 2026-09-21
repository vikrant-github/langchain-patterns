"""Demonstrate reusable LangChain prompt templates.

Use ChatPromptTemplate for structured system and human messages, and
PromptTemplate for a single text prompt with reusable input variables.
"""

from lc_patterns.models.chat import get_nova_2_lite
from lc_patterns.prompts.templates import chat_template, simple_template


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

    try:
        chat_chain = chat_template | nova_2_lite
        chat_response = chat_chain.invoke(
            {
                "role": "Databricks Unity Catalog ML architect",
                "question": (
                    "Explain how Unity Catalog can be used to govern ML models "
                    "across development and production in two sentences."
                ),
            }
        )
        print("ChatPromptTemplate:", chat_response.content)
        print_token_usage(chat_response)

        simple_chain = simple_template | nova_2_lite
        simple_response = simple_chain.invoke(
            {
                "topic": "Databricks Unity Catalog ML",
                "question": (
                    "What is the purpose of registering an ML model in Unity Catalog?"
                ),
            }
        )
        print("PromptTemplate:", simple_response.content)
        print_token_usage(simple_response)
    except Exception as error:  # noqa: BLE001
        print(f"Prompt invocation failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")


if __name__ == "__main__":
    main()