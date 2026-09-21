from langchain_core.messages import HumanMessage, ToolMessage

from lc_patterns.models.chat import get_nova_2_lite
from lc_patterns.tools.customer import get_customer
from lc_patterns.tools.weather import get_weather


def print_token_usage(response) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage:
        print(
            "  Tokens: "
            f"input={usage.get('input_tokens', 'N/A')}, "
            f"output={usage.get('output_tokens', 'N/A')}, "
            f"total={usage.get('total_tokens', 'N/A')}"
        )
    else:
        print("  ⚠️ Token usage information unavailable.")


def run_react_loop(query: str, tools: list, max_iterations: int = 5) -> str:
    model = get_nova_2_lite()
    model_with_tools = model.bind_tools(tools)

    tools_by_name = {tool.name: tool for tool in tools}
    messages = [HumanMessage(content=query)]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        try:
            response = model_with_tools.invoke(messages)
            messages.append(response)
            print_token_usage(response)
        except Exception as error:  # noqa: BLE001
            print(f"⚠️ Model invocation failed: {error}")
            return f"Model invocation failed: {error}"

        if not response.tool_calls:
            print("No more tool calls.")
            return response.content

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"Action: {tool_name}({tool_args})")

            try:
                tool_result = tools_by_name[tool_name].invoke(tool_args)
            except Exception as error:  # noqa: BLE001
                tool_result = f"Tool '{tool_name}' failed: {error}"
                print(f"Observation: {tool_result}")
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            print(f"Observation: {tool_result}")

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

    return "Maximum iterations reached."


def main() -> None:
    tools = [get_customer, get_weather]

    query = (
        "Check customer C001 and tell me their account status. "
        "Also check the weather in Delhi."
    )

    result = run_react_loop(query, tools)

    print(f"\nFinal Answer: {result}")


if __name__ == "__main__":
    main()