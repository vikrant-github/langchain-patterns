"""Execute a complete LangChain tool-calling loop."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lc_patterns.models.chat import get_nova_2_lite
from lc_patterns.prompts.templates import simple_template
from lc_patterns.tools.weather import get_weather


def main() -> None:
    """Demonstrate LLM planning, tool execution, and final response."""

    model = get_nova_2_lite()
    model_with_tools = model.bind_tools([get_weather])

    prompt = simple_template.invoke(
        {
            "topic": "weather",
            "question": "What is the current temperature in Seattle?",
        }
    )

    try:
        # Step 1: LLM generates the tool call.
        response = model_with_tools.invoke(
            [HumanMessage(content=prompt.text)]
        )

        usage = response.usage_metadata
        if usage:
            print(
                f"Planning tokens: input={usage.get('input_tokens', 'N/A')}, "
                f"output={usage.get('output_tokens', 'N/A')}, "
                f"total={usage.get('total_tokens', 'N/A')}"
            )
        else:
            print("Planning token usage information unavailable.")

        tool_call = response.tool_calls[0]

        # Step 2: Application executes the requested tool.
        tool_result = get_weather.invoke(tool_call["args"])

        # Step 3: Send the tool result back to the LLM.
        messages = [
            HumanMessage(content=prompt.text),
            AIMessage(content="", tool_calls=response.tool_calls),
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"],
            ),
        ]

        final_response = model_with_tools.invoke(messages)

        usage = final_response.usage_metadata
        if usage:
            print(
                f"Final response tokens: "
                f"input={usage.get('input_tokens', 'N/A')}, "
                f"output={usage.get('output_tokens', 'N/A')}, "
                f"total={usage.get('total_tokens', 'N/A')}"
            )
        else:
            print("Final response token usage information unavailable.")

        print("Tool call:", tool_call)
        print("Tool result:", tool_result)
        print("Final response:", final_response.content)
    except Exception as error:  # noqa: BLE001
        print(f"Tool execution loop failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error):
            print("Authentication failed.")


if __name__ == "__main__":
    main()