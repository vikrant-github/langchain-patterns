"""
MCP: Remote MCP Server Integration.

Demonstrates how a LangChain agent can consume tools exposed by a
remote MCP server using Streamable HTTP transport.

Context7 is used as the MCP server. It exposes documentation tools
that the MCP client discovers and makes available to the LangChain
agent.

The agent uses Amazon Bedrock as the chat model and the same
create_agent() pattern established in Chapter 5.

Key flow:

    Context7 MCP Server
            ↓
    Streamable HTTP
            ↓
    MultiServerMCPClient
            ↓
    MCP tools
            ↓
    LangChain agent
            ↓
    Amazon Bedrock
"""

import asyncio

from langchain.agents import create_agent

from lc_patterns.mcp.client import get_mcp_client
from lc_patterns.models.chat import get_nova_2_lite


async def main() -> None:
    """Connect to Context7, retrieve MCP tools, and run the agent."""
    try:
        # Configure the remote MCP server.
        # Streamable HTTP is used because Context7 exposes its MCP
        # endpoint over HTTP.
        client = get_mcp_client(
            {
                "context7": {
                    "transport": "streamable_http",
                    "url": "https://mcp.context7.com/mcp",
                }
            }
        )

        # Ask the MCP server which tools it exposes.
        # The returned tools are converted into LangChain-compatible
        # tools by langchain-mcp-adapters.
        tools = await client.get_tools()

        # Create the chat model using the project's existing
        # Amazon Bedrock model configuration.
        model = get_nova_2_lite()

        # Create the LangChain agent using the MCP-provided tools.
        # This is the same create_agent() pattern established in Chapter 5;
        # only the source of the tools has changed.
        agent = create_agent(model, tools)

        # Send a request to the agent.
        # ainvoke() is asynchronous because the agent performs external
        # I/O, including communication with the MCP server and model.
        response = await agent.ainvoke(
            {
                "messages": [
                    (
                        "human",
                        "How do I create a REST API using Spring Boot? Get the latest documentation.",
                    )
                ]
            }
        )

        # The final message contains the agent's response after it has
        # used the MCP-provided documentation tools as needed.
        print(response["messages"][-1].content)
    except Exception as error:  # noqa: BLE001
        print(f"\n❌ Context7 agent failed: {error}")
        if "429" in str(error):
            print("Rate limit reached.")
        elif "401" in str(error) or "403" in str(error):
            print("Authentication or authorization failed.")


if __name__ == "__main__":
    # Start Python's asyncio event loop and execute the asynchronous
    # main() function.
    asyncio.run(main())