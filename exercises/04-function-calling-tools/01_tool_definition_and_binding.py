"""Define a LangChain tool and expose it to a chat model."""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from lc_patterns.models.chat import get_nova_2_lite


class DeploymentStatusInput(BaseModel):
    """Input schema for the deployment status tool."""

    service_name: str = Field(description="Name of the service.")
    environment: Literal["dev", "staging", "prod"] = Field(
        description="Deployment environment to inspect."
    )


@tool(args_schema=DeploymentStatusInput)
def get_deployment_status(service_name: str, environment: str) -> str:
    """Return the deployment status of a service in an environment."""
    status = {
        ("document-service", "prod"): "healthy",
        ("document-service", "staging"): "degraded",
    }

    return status.get(
        (service_name, environment),
        "service or environment not found",
    )


def main() -> None:
    """Demonstrate direct tool invocation and model-generated tool calls."""

    # Tool exists independently of the LLM.
    print("Tool name:", get_deployment_status.name)
    print("Tool description:", get_deployment_status.description)

    # Application can execute the tool directly.
    result = get_deployment_status.invoke(
        {
            "service_name": "document-service",
            "environment": "prod",
        }
    )
    print("Direct result:", result)

    # Bind the tool to the model so the LLM can call it.
    nova_2_lite = get_nova_2_lite()
    model_with_tools = nova_2_lite.bind_tools([get_deployment_status])

    try:
        # LLM decides whether to request the tool.
        response = model_with_tools.invoke(
            "Check the production status of the document-service."
        )

        print("Tool calls:", response.tool_calls)

        usage = response.usage_metadata
        if usage:
            print(
                f"Tokens: input={usage.get('input_tokens', 'N/A')}, "
                f"output={usage.get('output_tokens', 'N/A')}, "
                f"total={usage.get('total_tokens', 'N/A')}"
            )
        else:
            print("Token usage information unavailable.")
    except Exception as error:  # noqa: BLE001
        print(f"Tool-calling invocation failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")


if __name__ == "__main__":
    main()