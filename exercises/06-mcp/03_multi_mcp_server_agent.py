"""
Chapter 6 Example 3: Multi-Server MCP Agent.

Demonstrates how one LangChain agent can consume tools from multiple
MCP servers using different transports.

Servers:
- Context7: Streamable HTTP
- Local Git: stdio

Run:
    uv run python exercises/06-mcp/03_multi_mcp_server_agent.py
"""

import asyncio
from pathlib import Path

from langchain.agents import create_agent

from lc_patterns.mcp.mcp_client import get_mcp_client
from lc_patterns.models.chat import get_nova_2_lite


async def main() -> None:
    """Connect to multiple MCP servers and run a single agent."""

    print("🌐 MCP Integration Demo - Multiple MCP Servers\n")
    print("=" * 80 + "\n")

    try:
        server_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "lc_patterns"
            / "mcp"
            / "stdio_git_mcp_server.py"
        )

        client = get_mcp_client(
            {
                "context7": {
                    "transport": "streamable_http",
                    "url": "https://mcp.context7.com/mcp",
                },
                "git": {
                    "transport": "stdio",
                    "command": "uv",
                    "args": ["run", "python", str(server_path)],
                },
            }
        )

        print("🔧 Fetching tools from all MCP servers...")
        tools = await client.get_tools()

        print(f"✅ Retrieved {len(tools)} tools from 2 MCP servers:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description}")
        print()

        model = get_nova_2_lite()

        print("🤖 Creating agent with tools from both MCP servers...\n")
        agent = create_agent(model, tools)

        queries = [
            "What is the current CI status of the langchain-patterns repository?",
            "How do I create a REST API using Spring Boot? Get the latest documentation.",
        ]

        for query in queries:
            print(f"👤 User: {query}\n")

            try:
                response = await agent.ainvoke({"messages": [("human", query)]})
                last_message = response["messages"][-1]
                print(f"🤖 Agent: {last_message.content}\n")
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
                print(f"⚠️ Failed to answer query: {type(error).__name__}: {error}\n")

        print("=" * 80 + "\n")
        print("💡 Key Concepts:")
        print("   • One agent can consume tools from multiple MCP servers")
        print("   • MCP servers can use different transports")
        print("   • MultiServerMCPClient combines tools from all configured servers")
        print("   • The agent receives one combined tool set")
        print("   • The same create_agent() pattern works across all MCP examples")

    except (ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        print(f"❌ Error connecting to MCP servers: {type(error).__name__}: {error}")
    finally:
        print("\n✅ MCP client connection closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Script interrupted by user.")
    except (ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        print(f"\n❌ Fatal error: {type(error).__name__}: {error}")