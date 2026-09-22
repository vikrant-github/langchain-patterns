"""Middleware for logging agent model and tool activity."""

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import ToolMessage


class AgentLoggingMiddleware(AgentMiddleware):
    """Log model and tool activity during agent execution."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable,
    ):
        print(
            f"[Model] messages={len(request.state['messages'])}, "
            f"tools={len(request.tools)}"
        )

        response = handler(request)

        print(f"[Model] tool_calls={len(response.result[-1].tool_calls)}")

        return response

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable,
    ) -> ToolMessage:
        tool_name = request.tool_call["name"]

        print(f"[Tool] calling={tool_name}")

        try:
            result = handler(request)
            print(f"[Tool] completed={tool_name}")
            return result
        except Exception as exc:  # noqa: BLE001
            print(f"[Tool] failed={tool_name}: {exc}")
            return ToolMessage(
                content=f"Tool '{tool_name}' failed: {exc}",
                tool_call_id=request.tool_call["id"],
            )