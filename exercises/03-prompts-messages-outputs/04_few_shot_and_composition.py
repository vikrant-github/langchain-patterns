"""Demonstrate few-shot prompting and prompt template composition."""

from botocore import model
from langchain_core.prompts import ChatPromptTemplate

from lc_patterns.models.chat import get_nova_2_lite
from lc_patterns.prompts.templates import create_few_shot_template


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


def few_shot_example(model):
    """Demonstrate few-shot prompting with reusable examples."""
    examples = [
        {
            "input": "What is a managed table?",
            "output": (
                "A table whose data is managed by the catalog and stored "
                "in a managed storage location."
            ),
        },
        {
            "input": "What is a view?",
            "output": "A view is a virtual table defined by a SQL query.",
        },
        {
            "input": "What is a model?",
            "output": (
                "A model is a registered machine learning artifact "
                "that can be governed and discovered."
            ),
        },
    ]

    few_shot_prompt = create_few_shot_template(examples)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Answer Databricks Unity Catalog questions using the style "
                    "demonstrated by the examples."
                ),
            ),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    try:
        response = (prompt | model).invoke(
            {"input": "Why would an ML team register a model in Unity Catalog?"}
        )
        print("Few-Shot Response:")
        print(response.content)
        print_token_usage(response)
    except Exception as error:  # noqa: BLE001
        print(f"Few-shot invocation failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")


def composition_example(model):
    """Demonstrate composing common and specialized prompt instructions."""
    base_instructions = """You are a Databricks ML architect.
Provide technically precise answers.
Keep responses concise."""

    architecture_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", base_instructions),
            (
                "system",
                "Focus on architecture, governance, and production concerns.",
            ),
            ("human", "{question}"),
        ]
    )

    troubleshooting_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", base_instructions),
            (
                "system",
                "Focus on diagnosing technical problems and identifying likely causes.",
            ),
            ("human", "{question}"),
        ]
    )

    try:
        architecture_response = (architecture_prompt | model).invoke(
            {
                "question": (
                    "Why should ML models be governed through Unity Catalog "
                    "in an enterprise environment?"
                )
            }
        )
        print("\nArchitecture Response:")
        print(architecture_response.content)
        print_token_usage(architecture_response)
    except Exception as error:  # noqa: BLE001
        print(f"Architecture invocation failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")

    try:
        troubleshooting_response = (troubleshooting_prompt | model).invoke(
            {
                "question": (
                    "A production model cannot be loaded from Unity Catalog. "
                    "What should be checked first?"
                )
            }
        )
        print("\nTroubleshooting Response:")
        print(troubleshooting_response.content)
        print_token_usage(troubleshooting_response)
    except Exception as error:  # noqa: BLE001
        print(f"Troubleshooting invocation failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")


def main():
    nova_2_lite = get_nova_2_lite()

    few_shot_example(nova_2_lite)
    composition_example(nova_2_lite)


if __name__ == "__main__":
    main()