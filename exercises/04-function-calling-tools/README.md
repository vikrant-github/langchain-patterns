# Chapter 4: Function Calling and Tool Execution

## Engineering Takeaways

A tool is an independently executable capability. In this chapter, the Python function is first exposed as a LangChain tool with `@tool`, which gives it a name, description, and callable interface. The tool can be invoked directly as a Python function-like object, but its runtime contract is still defined by the function signature and, where needed, a Pydantic schema. That schema is not a convenience wrapper; it is the validation boundary for tool inputs.

`bind_tools()` is a model-level capability, not a property of the tool itself. It makes the tool available to the chat model so the model can decide whether to request it during a conversation. The distinction is architectural: the tool exists as a reusable capability, while the model binding determines whether that capability is visible to the LLM for selection during inference.

### Tool definition vs. tool binding

- A tool is an independently executable capability.
- `@tool` exposes the Python function as a LangChain tool.
- Pydantic defines and validates the tool input contract.
- `bind_tools()` is a chat-model capability that makes tools available to the LLM for tool calling.

The implementation demonstrates that tool definition and tool binding are separate concerns. A tool can exist and be invoked directly without any model binding; model binding is only required when the application expects the LLM to decide whether and when the tool is appropriate.

### Who controls tool execution

Execution control is determined by the orchestration model, not by whether a tool is deterministic or non-deterministic.

- If the application decides when to execute the tool, invoke it directly with `tool.invoke(...)`; no model binding is required.
- If the LLM is expected to decide whether and when a tool should be used, the tool must be bound to the chat model with `bind_tools(...)`.
- The need for binding is therefore a question of control flow, not a question of complexity, side effects, or determinism.

This is an important engineering boundary: binding does not make the function run; it only makes the tool discoverable and callable by the model.

### Tool selection is influenced by description, not enforced by it

A bound tool is selected by the model using the metadata exposed to it: name, description, and input schema. A clearer or more specific description can improve the likelihood that the model chooses the right tool for a given query, but it is a ranking signal, not a dispatch guarantee. In production, tool descriptions should be precise and scoped; when selection priority matters, the application should enforce routing or validation explicitly rather than assuming the model will always prefer one tool over another.

### Tool-calling execution model

The tool-calling loop is explicit and deliberately separated:

1. The LLM does not execute the Python function.
2. The LLM produces a structured tool call.
3. Application code executes the requested tool.
4. The tool result is returned to the LLM as part of the message history.
5. The LLM then produces the final response.

This makes tool use predictable and auditable. The model is responsible for planning and selecting a tool, while the application remains the execution layer. The result is then fed back into the conversation so the LLM can reason with the outcome before answering.

### Separation of responsibilities

- LLM: planning and tool selection
- Application: tool execution and orchestration
- Tool: domain capability
- Pydantic: input contract and validation

This separation matters in production: the model decides intent, the application decides execution, and the tool is a narrow, testable implementation of a domain operation. Validated inputs and explicit execution boundaries keep the system understandable and debuggable without collapsing all responsibility into the model layer.
