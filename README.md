# langchain-patterns

Engineering-focused LangChain patterns and Bedrock experiments for practical implementation, library extraction, and repeatable local validation.

This repository evaluates current LangChain APIs, turns concepts into reusable patterns, and keeps experimentation grounded in small, reviewable implementations.

## Engineering Approach

The working loop is:

```text
Concept
	↓
Understand
	↓
Implement independently
	↓
Extract reusable engineering pattern
	↓
Exercise
	↓
Test
	↓
Ruff + pytest
	↓
Commit and push
	↓
GitHub Actions
```

Exercises are intentionally small engineering surfaces. Reusable code belongs in `src/lc_patterns`; hands-on implementations belong in `exercises`. Tests are added after a chapter or module is complete. CI validates the repository without requiring AWS access. Deployment is intentionally deferred.

Chapter 4 introduces the core distinction between a tool definition and a bound model capability: `@tool` defines the executable capability and its validated input contract, while `bind_tools()` is what makes that tool selectable by the chat model for tool calling.

## Current Repository

The implemented repository currently contains:

```text
langchain-patterns/
├── src/lc_patterns/
│   ├── __init__.py
│   ├── config/
│   │   └── __init__.py
│   ├── mcp/
│   │   └── client.py
│   ├── middleware/
│   │   └── logging.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── chat.py
│   ├── prompts/
│   │   └── templates.py
│   ├── schemas/
│   │   └── structured_outputs.py
│   ├── tools/
│   │   ├── customer.py
│   │   └── weather.py
│   └── py.typed
├── exercises/
│   ├── 01_models/
│   │   ├── basic_invocation.py
│   │   ├── message_interaction.py
│   │   └── model_comparison.py
│   ├── 02-chat-models/
│   │   ├── 01_multi_turn.py
│   │   └── 02_parameters.py
│   ├── 03-prompts-messages-outputs/
│   │   ├── 01_messages_vs_templates.py
│   │   ├── 02_dynamic_message_construction.py
│   │   ├── 03_prompt_templates.py
│   │   ├── 04_few_shot_and_composition.py
│   │   ├── 05_basic_structured_output.py
│   │   └── 06_complex_pydantic_schema.py
│   ├── 04-function-calling-tools/
│   │   ├── 01_tool_definition_and_binding.py
│   │   ├── 02_tool_execution_loop.py
│   │   ├── 03_tool_selection_pattern.py
│   │   └── README.md
│   ├── 05-agents/
│   │   ├── 01_create_agent.py
│   │   ├── 02_manual_react.py
│   │   └── 03_agent_middleware.py
│   └── 06-mcp/
│       └── 01_context7_agent.py
├── tests/
│   ├── __init__.py
│   ├── test_chat.py
│   ├── test_customer_tool.py
│   └── test_prompts_messages_outputs.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## Technology Stack

| Technology | Purpose |
| --- | --- |
| Python 3.12 | Runtime and CI interpreter, pinned by `.python-version` |
| uv | Project creation, Python selection, dependency resolution, and environment management |
| LangChain | Model abstractions and invocation APIs |
| `langchain-aws` | Amazon Bedrock integration through `ChatBedrockConverse` |
| Pydantic 2.x | Validation and typed configuration used by the dependency stack |
| pytest | Unit test runner |
| Ruff | Linting and repository quality checks |
| GitHub Actions | CI execution on pushes and pull requests |
| AWS CLI | AWS authentication and Bedrock account validation |
| Amazon Bedrock | Managed model runtime |
| Amazon Nova 2 Lite | Default model used by the basic invocation exercise |
| Amazon Nova Pro | Structured-message and comparison exercise model |

## Python Setup

The project was initialized as a library:

```bash
uv init --lib --name lc-patterns
```

The generated Python requirement was adjusted to Python 3.12. The current project configuration is:

```toml
requires-python = ">=3.12"
```

Create or select the interpreter and synchronize the environment:

```bash
uv python pin 3.12
uv sync
```

uv manages the project environment in `.venv` and records resolved dependencies in `uv.lock`. The `.venv` directory is ignored by Git.

Verify the environment:

```bash
uv run python --version
uv run python -c "import langchain; import langchain_aws; import pydantic; print('core imports: OK')"
uv run ruff --version
uv run pytest --version
```

## Dependencies

Runtime dependencies declared in `pyproject.toml` are:

- `langchain`
- `langchain-aws`
- `pydantic`
- `botocore[crt]`

`botocore[crt]` was added for the AWS login credential provider and brings the AWS CRT support required by that provider.

Development dependencies are:

- `pytest`
- `ruff`

The explicit addition command was:

```bash
uv add "botocore[crt]"
uv sync
```

## AWS CLI Setup

AWS CLI was installed in the development environment for account authentication and validation. The exact installer command was not recorded in Git history. The installer and extracted files were initially created inside the repository because installation was run from the repository directory. They were removed with:

```bash
rm -rf aws awscliv2.zip
```

That cleanup removed repository-local installer artifacts. It did not uninstall AWS CLI.

Verify the CLI independently:

```bash
aws --version
```

## AWS Authentication

The configured region is `us-east-1`. Authenticate through the AWS CLI:

```bash
aws login --remote
aws sts get-caller-identity
```

Authentication uses the AWS CLI credential mechanism. `boto3` and `langchain-aws` use the standard AWS credential chain. Access keys and secrets are not stored in source code. Credentials must remain outside Git. `.env` and other `.env.*` files are ignored by `.gitignore`, while `.env.example` remains allowed for future non-secret configuration documentation.

## Troubleshooting AWS Login

For this container environment, use the cross-device login flow:

```bash
aws login --remote
```

Open the URL printed by the command in a browser outside the container. Enter the authorization code shown after completing the browser login, then verify the session:

```bash
aws sts get-caller-identity
```

A successful identity check confirms that the AWS session is ready for the Bedrock exercises.

## Amazon Bedrock Setup

Bedrock access is configured for `us-east-1`. Model availability was checked with:

```bash
aws bedrock list-foundation-models \
	--region us-east-1 \
	--query "modelSummaries[?modelId=='amazon.nova-2-lite-v1:0'].[modelId,modelName]" \
	--output table
```

The first model ID attempt used `amazon.nova-2-lite-v1:0`. A LangChain invocation returned an AWS account verification `AccessDeniedException`. The Bedrock console showed GLOBAL and US Nova 2 Lite options. Selecting the US option produced the working model ID:

```text
us.amazon.nova-2-lite-v1:0
```

Direct `boto3` invocation succeeded. LangChain validation also succeeded with:

```python
ChatBedrockConverse(
		model="us.amazon.nova-2-lite-v1:0",
		region_name="us-east-1",
)
```

Nova Pro was subsequently added and is used by the model comparison exercise.

## Reusable Model Layer

The current reusable model layer is [src/lc_patterns/models/chat.py](src/lc_patterns/models/chat.py). It exposes:

- `get_nova_2_lite()`, using `us.amazon.nova-2-lite-v1:0`
- `get_nova_pro()`, using `amazon.nova-pro-v1:0`

Both use region `us-east-1` and return `ChatBedrockConverse` instances. Model construction is centralized so exercises do not repeatedly construct Bedrock clients. The provider integration remains visible and understandable. No provider factory, model registry, or base model abstraction was introduced because the current surface does not justify one. Additional model concerns will be introduced only when they are justified by the implementation.

## Chapter 1: Models

The current Chapter 1 implementation is in `exercises/01_models/`:

| Concept | Exercise | Provider | Implementation |
| --- | --- | --- | --- |
| Models | `basic_invocation.py` | Amazon Bedrock | Invokes Nova 2 Lite through the reusable model layer. |
| Prompts and messages | `message_interaction.py` | Amazon Bedrock | Uses `SystemMessage` and `HumanMessage` to control response context and style. |
| Model comparison | `model_comparison.py` | Amazon Bedrock | Compares Nova 2 Lite and Nova Pro response times and output. |

These are live Bedrock exercises. They are not executed by CI.

## Chapter 2: Chat Models

The implemented concepts in `exercises/02-chat-models/` are:

| Concept | Exercise | Implementation |
| --- | --- | --- |
| Multi-turn conversation | `01_multi_turn.py` | Maintains message history across three model responses. |
| Streaming | `01_multi_turn.py` | Streams the third response and handles string or list-based Bedrock content chunks. |
| Retry handling | `01_multi_turn.py` | Retries model execution up to three times with `with_retry()`. |
| Error handling | `01_multi_turn.py` | Reports general failures and identifies rate-limit and authentication errors. |
| Parameters | `02_parameters.py` | Compares supported Nova 2 Lite temperature values. |
| Token tracking | `02_parameters.py` | Reports input, output, and total tokens from response metadata. |

The implementations use the centralized Bedrock model layer in `src/lc_patterns/models/chat.py`.

### `01_multi_turn.py`

Demonstrates a three-turn Databricks MLOps conversation using message history, streaming for the final response, retry handling, and concise error handling for rate-limit and authentication failures. The streaming loop handles Bedrock chunks whose `content` is either a string or a list of content blocks.

### `02_parameters.py`

Compares supported Nova 2 Lite temperature values, invokes each configuration twice, and reports input, output, and total token usage from LangChain response metadata.

## Chapter 3: Prompts, Messages, and Structured Output

Chapter 3 introduces the prompt and message abstractions that support more advanced LangChain patterns. The conceptual progression is:

```text
Prompts & Messages
├── Messages
│   ├── SystemMessage
│   ├── HumanMessage
│   ├── AIMessage
│   └── Dynamic construction → Agents
├── Templates
│   ├── ChatPromptTemplate
│   ├── PromptTemplate
│   ├── FewShotChatMessagePromptTemplate
│   └── Template composition → RAG systems
└── Structured Output
	├── Pydantic BaseModel
	├── Field descriptions
	├── with_structured_output
	└── Nested schemas → Data extraction
```

The chapter connects three layers of the LangChain API:

| Layer | Concepts | Prepares for |
| --- | --- | --- |
| Messages | `SystemMessage`, `HumanMessage`, `AIMessage`, and dynamic message construction | Agents |
| Templates | `ChatPromptTemplate`, `PromptTemplate`, `FewShotChatMessagePromptTemplate`, and template composition | RAG systems |
| Structured output | Pydantic `BaseModel`, field descriptions, `with_structured_output`, and nested schemas | Data extraction |

The corresponding implementations are in `exercises/03-prompts-messages-outputs/` and the reusable prompt/schema layer is in `src/lc_patterns/prompts/templates.py` and `src/lc_patterns/schemas/structured_outputs.py`.

## Chapter 4: Function Calling and Tool Execution

Chapter 4 is complete. The repository includes the implemented tool-calling work in `exercises/04-function-calling-tools/` and the reusable tool implementations in `src/lc_patterns/tools/`, with the focus on the boundary between tool definition, model binding, execution, and tool selection.

### Manual ReAct loop

The repository also includes a manual ReAct-style agent exercise in `exercises/05-agents/02_manual_react.py`. It demonstrates:

- model/tool binding for a tool-calling loop
- explicit tool execution by application code
- observation collection and conversation history updates
- exception handling around model and tool failures
- token usage reporting from `usage_metadata`

This keeps the exercise readable and grounded in the underlying LangChain pattern: the model decides when a tool is needed, and the application executes it and feeds the result back into the conversation.

## Chapter 5: Agents and Middleware

The agent exercise in `exercises/05-agents/03_agent_middleware.py` demonstrates the runtime boundary created by `create_agent()`. `create_agent()` constructs the LangChain agent runtime and wires the model, tools, system prompt, and middleware together. The application does not replace the runtime loop; it configures it.

The key responsibility split is straightforward:

- The LLM decides whether a tool is needed and which tool to request.
- The LangChain agent runtime orchestrates the actual loop: model calls, tool execution, observations, iteration, and termination.
- Middleware is registered by the application and invoked by the LangChain runtime at the lifecycle hooks that matter.

This is not "middleware between tools." Middleware wraps or intercepts specific agent-runtime operations. It is a runtime extension point for the agent loop, not a second tool layer.

The reusable implementation is in `src/lc_patterns/middleware/logging.py` and is named `AgentLoggingMiddleware`. It is intentionally narrow and observability-focused:

- `wrap_model_call()` adds logging around model invocations, with visibility into the current message count and the available tool set before the call and the resulting tool-call structure after the call.
- `wrap_tool_call()` adds logging around tool invocations, tracking when a tool is called, when it succeeds, and when it fails by returning a `ToolMessage` error payload.

This is a good example of the general middleware model: the runtime owns orchestration, while middleware adds policy or infrastructure concerns at specific hooks.

```text
User → Agent Runtime → [Middleware] Model → [Middleware] Tool → Model → Final Answer
```

The important point is that middleware sits around the runtime operations, not around the tool definitions themselves. Observability is only one use case. The same mechanism can support error handling, retries, authorization, metrics, routing, and guardrails without changing the core agent control flow.

The Chapter 5 exercise demonstrates the pattern in practice and should be read alongside the reusable middleware implementation in `src/lc_patterns/middleware/logging.py`.

## Chapter 6: Model Context Protocol

Chapter 6 adds a live MCP integration using Context7. MCP, or Model Context Protocol, is a standardized protocol for connecting AI applications to external capabilities. It does not replace the LLM or the LangChain agent.

The implementation follows this architecture:

```text
Context7 MCP Server
	|
	v
Streamable HTTP
	|
	v
MCP Client
	|
	v
MCP Tools
	|
	v
LangChain Agent
	|
	v
Amazon Bedrock
```

Traditional LangChain tools are normally implemented and owned by the application. MCP allows capabilities exposed by an MCP server to be discovered and consumed by the application. The existing `create_agent()` pattern remains unchanged; MCP changes the source of the tools.

The client connects to an MCP server, while the server exposes capabilities such as tools. The client discovers those capabilities and makes them available to the application or agent.

The transport describes how communication occurs:

- `stdio`: process-to-process communication through standard input and output. This is commonly used when an MCP server is launched as a local subprocess.
- Streamable HTTP: network communication with an MCP endpoint. This is commonly used by remote or shared MCP servers.

These are common deployment patterns, not fixed location rules. A local server is not always `stdio`, and a remote server is not always HTTP.

Context7 is a software, library, and framework documentation service. In this example, its MCP server exposes documentation-related tools; it is not a generic document or Word-document service.

The reusable and chapter-specific code is separated as follows:

- `src/lc_patterns/mcp/client.py` provides reusable MCP client construction through `get_mcp_client(server_config)`.
- `exercises/06-mcp/01_context7_agent.py` contains the Context7 configuration and composes the discovered tools with the existing LangChain agent pattern.

The dependencies are `langchain-mcp-adapters==0.3.2` and `mcp==1.30.0`. The Context7 exercise is a live external integration and is not covered by normal pytest because it depends on external MCP/network infrastructure and Bedrock. Ruff validation and successful live execution are used for this example. Deterministic tests should be added when meaningful reusable behavior exists.
## Testing

The repository keeps tests focused on deterministic, local Python behavior. We do not add tests that call Bedrock or any other external LLM service.

### Existing tests

[tests/test_chat.py](tests/test_chat.py) validates the reusable Bedrock model configuration without invoking AWS. It verifies the `ChatBedrockConverse` type, the expected model IDs, and the `us-east-1` region.

### Chapter 3 tests

[tests/test_prompts_messages_outputs.py](tests/test_prompts_messages_outputs.py) adds local validation for the reusable prompt and schema code in Chapter 3:

- `ModelMetadata` valid construction
- `GovernedModel` valid nested construction
- invalid `lifecycle_stage` rejection
- nested field access and attribute traversal
- template `input_variables`
- successful template rendering
- few-shot template construction

These tests intentionally check logic that is pure Python and fully local. They verify the contract of the abstraction without depending on model output, token counts, or Bedrock availability.

Run the checks locally:

```bash
uv run pytest
```

Expected result at the current revision:

```text
8 passed
```

Run Ruff:

```bash
uv run ruff check .
```

Expected result:

```text
All checks passed!
```

## AI-assisted programming

AI is useful in this repository when it helps narrow the gap between intent and executable code. The goal is not to delegate design; it is to accelerate the mechanical parts of the work while preserving engineering discipline.

The pattern is simple and consistent:

1. Define the contract precisely.
2. Keep the test local and deterministic.
3. Ask for the smallest change that proves the behavior.
4. Review the output for correctness before accepting it.
5. Keep the boundary of the work explicit.

That is how the Chapter 3 pytest coverage was structured: validate schema constraints, prompt rendering, and example composition without calling Bedrock, without mocking the model layer, and without introducing a fake test framework.

The prompt is not a free-form request for a complete solution. It is a specification for a narrow runtime check on real Python behavior. In practice, that means validating:

- `ModelMetadata` accepts valid values and rejects invalid lifecycle metadata
- `GovernedModel` preserves nested ownership and governance fields
- `ChatPromptTemplate` and `PromptTemplate` render the expected strings
- `FewShotChatMessagePromptTemplate` contains the example sequence in the correct order

This keeps the feedback loop short and the tests meaningful. The engineering judgment still sits with the developer: decide the contract, keep the scope tight, and make sure the code is still readable, testable, and grounded in the actual repository architecture.

## Git Conventions

Use Conventional Commit prefixes consistently:

| Prefix | Use for |
| --- | --- |
| `feat:` | A new capability or module |
| `fix:` | A correction to existing behavior |
| `docs:` | Documentation-only changes |
| `chore:` | Tooling, environment, or repository maintenance |
| `test:` | Test additions or changes |
| `refactor:` | Behavior-preserving structural improvement |

Make focused commits that describe the change being introduced. Do not rewrite existing history.

## GitHub Actions CI

The current workflow is [.github/workflows/ci.yml](.github/workflows/ci.yml). It runs one `quality` job on every push and pull request:

```text
GitHub push / pull request
				↓
CI workflow
				↓
quality job
				↓
Checkout
				↓
Python 3.12
				↓
uv 0.12.17
				↓
uv sync
				↓
Ruff
				↓
pytest
```

The workflow uses `actions/checkout@v4`, `actions/setup-python@v5`, and `astral-sh/setup-uv@v6`. CI does not authenticate to AWS, call Bedrock, or run the live exercises. Tests remain independent of AWS credentials, keeping CI inexpensive and deterministic.

Inspect results through:

```text
GitHub repository → Actions → CI → workflow run → quality job
```

## Current Status

| Area | Status |
| --- | --- |
| Python / uv foundation | Complete |
| AWS CLI | Complete |
| AWS authentication | Complete |
| Bedrock | Complete |
| Reusable chat model layer | Complete |
| Chapter 1 | Complete |
| Model configuration tests | Complete |
| Chapter 2 chat models | Complete |
| Chapter 3 prompts and structured output | Complete |
| Chapter 4 function calling and tool execution | Complete |
| Chapter 5 manual agent loop | In progress |
| Ruff | Complete |
| GitHub Actions CI | Complete |
| Next | Higher-order LangChain orchestration and reuse patterns |

## Development Workflow

1. Study the concept.
2. Understand the API.
3. Implement the exercise independently.
4. Identify reusable engineering logic.
5. Add reusable code only when justified.
6. Finish the chapter.
7. Add tests.
8. Run Ruff.
9. Run pytest.
10. Commit.
11. Push.
12. Verify GitHub Actions.
13. Move to the next chapter.

## Next

The core Bedrock, prompt, and structured-output workflow is in place, and Chapter 3 is complete. Chapter 4 is complete and covers the tool layer, tool binding, execution flow, and selection boundary between model decision-making and Python execution. The manual ReAct exercise in Chapter 5 is also underway and adds the conversation loop, execution safeguards, and token reporting needed to reason through agent behavior in a controlled way. Deployment remains intentionally deferred.

## Revisit Notes

- Python: `3.12`
- Project environment: `.venv`
- Dependency manager: `uv`
- AWS region: `us-east-1`
- AWS authentication: AWS CLI login and the standard credential chain
- Working Nova 2 Lite model ID: `us.amazon.nova-2-lite-v1:0`
- LangChain Bedrock integration: `ChatBedrockConverse`
- AWS login credential provider requirement: `botocore[crt]`
- Live Bedrock exercises are not run by CI
- Unit tests must not require AWS credentials
- AWS credentials must never be committed
