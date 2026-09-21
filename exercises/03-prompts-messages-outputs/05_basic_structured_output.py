"""Demonstrate structured LLM output using a Pydantic schema."""

from lc_patterns.models.chat import get_nova_2_lite
from lc_patterns.schemas.structured_outputs import ModelMetadata


def main():
    nova_2_lite = get_nova_2_lite()
    structured_model = nova_2_lite.with_structured_output(
        ModelMetadata,
        include_raw=True,
    )

    text = """
    Churn Prediction is a gradient boosting model, version 3.2,
    used to predict the likelihood that a customer will leave a service.
    """

    try:
        result = structured_model.invoke(
            f"Extract ML model metadata from: {text}"
        )
        parsed = result["parsed"]
        raw = result["raw"]
        usage = raw.usage_metadata

        print(f"Name: {parsed.name}")
        print(f"Version: {parsed.version}")
        print(f"Framework: {parsed.framework}")
        print(f"Purpose: {parsed.purpose}")
        print(
            "Tokens: "
            f"input={usage.get('input_tokens', 'N/A')}, "
            f"output={usage.get('output_tokens', 'N/A')}, "
            f"total={usage.get('total_tokens', 'N/A')}"
        )
    except Exception as error:  # noqa: BLE001
        print(f"Structured-output invocation failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")


if __name__ == "__main__":
    main()