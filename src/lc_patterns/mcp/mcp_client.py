from langchain_mcp_adapters.client import MultiServerMCPClient


def get_mcp_client(server_config: dict) -> MultiServerMCPClient:
    """Create an MCP client from the provided server configuration."""
    return MultiServerMCPClient(server_config)