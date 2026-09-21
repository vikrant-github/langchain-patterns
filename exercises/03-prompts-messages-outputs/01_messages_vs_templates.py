"""Compare direct message invocation with reusable prompt templates.

Message arrays are constructed explicitly and passed directly to the model.
A ChatPromptTemplate separates prompt structure from input values, allowing
the same prompt definition to be reused with different inputs.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from lc_patterns.models.chat import get_nova_2_lite


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

    # APPROACH 1: Messages (dynamic - great for agents)
    print("APPROACH 1: Message Arrays\n")

    messages = [
        SystemMessage(content="You are a helpful translator."),
        HumanMessage(content="Translate 'Hello, world!' to French"),
    ]

    message_response = nova_2_lite.invoke(messages)
    print("Response:", message_response.content)
    print_token_usage(message_response)

    # APPROACH 2: Templates (reusable - great for RAG)
    print("\nAPPROACH 2: Templates\n")

    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful translator."),
            ("human", "Translate '{text}' to {language}"),
        ]
    )

    template_chain = template | nova_2_lite
    template_response = template_chain.invoke(
        {
            "text": "Hello, world!",
            "language": "French",
        }
    )

    print("Response:", template_response.content)
    print_token_usage(template_response)


if __name__ == "__main__":
    main()