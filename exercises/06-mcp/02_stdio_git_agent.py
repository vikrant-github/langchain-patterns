"""
Chapter 6 Example 2: Local MCP Server with stdio Transport.

Demonstrates how a LangChain agent consumes tools exposed by a locally
launched MCP server through stdio.

The Git MCP server provides read-only repository information from its
local dataset.

Run:
    uv run python exercises/06-mcp/02_stdio_git_agent.py
"""

import asyncio
from pathlib import Path

from langchain.agents import create_agent

from lc_patterns.mcp.client import get_mcp_client
from lc_patterns.models.chat import get_nova_2_lite


async def main() -> None:
    """Connect to the local Git MCP server and run the agent."""

    print("🔌 MCP Integration Demo - Local Git Server\n")
    print("=" * 80 + "\n")

    server_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "lc_patterns"
        / "mcp"
        / "stdio_git_mcp_server.py"
    )

    print(f"📡 Connecting to local MCP server: {server_path}\n")

    try:
        # The client launches the server as a child process and communicates
        # with it over stdin/stdout; the server writes diagnostics to stderr.
        client = get_mcp_client(
            {
                "git": {
                    "transport": "stdio",
                    "command": "uv",
                    "args": ["run", "python", str(server_path)],
                }
            }
        )

        # Tool discovery also verifies that the subprocess started and speaks
        # the MCP protocol before the agent is created.
        print("🔧 Fetching tools from local Git MCP server...")
        tools = await client.get_tools()

        print(f"✅ Retrieved {len(tools)} tools from local Git MCP server:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description}")
        print()

        model = get_nova_2_lite()

        print("🤖 Creating agent with MCP tools...\n")
        agent = create_agent(model, tools)

        query = "What is the current CI status of the langchain-patterns repository?"
        print(f"👤 User: {query}\n")

        # Agent execution may invoke the discovered MCP tools through the same
        # stdio connection before returning the final response.
        response = await agent.ainvoke(
            {"messages": [("human", query)]}
        )
        last_message = response["messages"][-1]

        print(f"🤖 Agent: {last_message.content}\n")

        print("=" * 80 + "\n")
        print("💡 Key Concepts:")
        print("   • stdio connects the client to a locally launched MCP server")
        print("   • The MCP server exposes Git-related capabilities as tools")
        print("   • MultiServerMCPClient discovers and adapts those tools")
        print("   • MCP tools work seamlessly with create_agent()")
        print("   • Same create_agent() pattern, different tool source!")
        print("   • No manual loop needed - create_agent() handles the agent loop")

    except (OSError, RuntimeError, ValueError) as error:
        # Handle expected transport/configuration failures while allowing
        # unexpected programming or model errors to retain their traceback.
        print(f"❌ Error connecting to local Git MCP server: {error}")

    finally:
        # The MCP client manages the subprocess lifecycle after the operation.
        # This message is informational and also runs when setup fails early.
        print("\n✅ MCP client connection closed")


if __name__ == "__main__":
    asyncio.run(main())