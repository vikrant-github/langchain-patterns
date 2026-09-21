"""Demonstrate nested Pydantic schemas with structured LLM output."""

from lc_patterns.models.chat import get_nova_2_lite
from lc_patterns.schemas.structured_outputs import GovernedModel


def main():
    model = get_nova_2_lite()

    structured_model = model.with_structured_output(
        GovernedModel,
        include_raw=True,
    )

    text = """
    The Customer Churn Prediction model, version 3.2, is built with XGBoost
    to predict customer attrition. It is currently in production and is owned
    by the Customer Analytics team. Tony Stark is the primary contact.
    Production deployment requires approval before changes are released.
    """

    try:
        result = structured_model.invoke(
            f"Extract the ML model and governance information from: {text}"
        )

        structured_result = result["parsed"]
        raw_response = result["raw"]

        print(f"Name: {structured_result.model.name}")
        print(f"Version: {structured_result.model.version}")
        print(f"Framework: {structured_result.model.framework}")
        print(f"Purpose: {structured_result.model.purpose}")

        print(f"Lifecycle: {structured_result.governance.lifecycle_stage}")
        print(f"Owner Team: {structured_result.governance.owner.team}")
        print(f"Contact: {structured_result.governance.owner.contact}")
        print(
            f"Approval Required: "
            f"{structured_result.governance.approval_required}"
        )

        usage = raw_response.usage_metadata
        if usage:
            print(
                f"Tokens: input={usage.get('input_tokens', 'N/A')}, "
                f"output={usage.get('output_tokens', 'N/A')}, "
                f"total={usage.get('total_tokens', 'N/A')}"
            )
        else:
            print("Token usage information unavailable.")
    except Exception as error:  # noqa: BLE001
        print(f"Structured-output invocation failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")


if __name__ == "__main__":
    main()