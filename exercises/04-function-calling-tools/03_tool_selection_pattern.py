"""Demonstrate application-controlled and LLM-controlled tool selection."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from lc_patterns.models.chat import get_nova_2_lite
from lc_patterns.prompts.templates import simple_template
from lc_patterns.tools.weather import get_weather


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""

    expression: str = Field(description="Arithmetic expression to evaluate.")


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except (NameError, SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        return f"Calculation error: {error}"


def main() -> None:
    """Demonstrate two different tool-selection models."""

    # Application-controlled tool execution.
    direct_result = calculator.invoke({"expression": "125 * 8"})
    print("Application-selected tool result:", direct_result)

    model = get_nova_2_lite()
    model_with_tools = model.bind_tools([calculator, get_weather])

    prompt = simple_template.invoke(
        {
            "topic": "tool selection",
            "question": "What is the weather in Seattle?",
        }
    )

    # LLM-controlled tool selection.
    response = model_with_tools.invoke(
        [HumanMessage(content=prompt.text)]
    )

    print("LLM-selected tool:", response.tool_calls)

    tool_call = response.tool_calls[0]

    if tool_call["name"] == get_weather.name:
        tool_result = get_weather.invoke(tool_call["args"])
    elif tool_call["name"] == calculator.name:
        tool_result = calculator.invoke(tool_call["args"])
    else:
        raise ValueError(f"Unknown tool: {tool_call['name']}")

    messages = [
        HumanMessage(content=prompt.text),
        AIMessage(content="", tool_calls=response.tool_calls),
        ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"],
        ),
    ]

    final_response = model_with_tools.invoke(messages)

    print("Tool result:", tool_result)
    print("Final response:", final_response.content)


if __name__ == "__main__":
    main()