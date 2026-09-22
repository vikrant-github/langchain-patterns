"""
Local Git MCP Server for stdio Transport.

This server runs as a subprocess and exposes read-only Git repository
capabilities through MCP tools.

The server communicates with the MCP client through stdio.
"""

import json
import sys
from json import JSONDecodeError
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stdio-git")


# Keep the dataset beside the server module so the server can be launched from
# any working directory without depending on the caller's current directory.
DATA_FILE = Path(__file__).with_name("repositories.json")


def load_repositories() -> list[dict]:
    """Load repository data from the local JSON dataset."""
    try:
        with DATA_FILE.open(encoding="utf-8") as file:
            repositories = json.load(file)
    except FileNotFoundError as error:
        raise RuntimeError(f"Repository data file not found: {DATA_FILE}") from error
    except JSONDecodeError as error:
        raise RuntimeError(f"Repository data file is not valid JSON: {DATA_FILE}") from error

    if not isinstance(repositories, list):
        raise TypeError("Repository data must contain a JSON array")

    return repositories


@mcp.tool()
def get_repository(repository: str) -> str:
    """
    Get information about a Git repository.

    Args:
        repository: Repository name.

    Returns:
        Repository information as a string.
    """
    # Exceptions from load_repositories() become MCP tool errors. They are not
    # caught here so configuration and data problems remain visible to clients.
    repositories = load_repositories()

    for repo in repositories:
        if repo["name"].lower() == repository.lower():
            return json.dumps(repo, indent=2)

    # A missing repository is an expected lookup failure, not a server failure.
    raise ValueError(f"Repository not found: {repository}")


@mcp.tool()
def get_pull_request(repository: str) -> str:
    """
    Get the latest pull request for a Git repository.

    Args:
        repository: Repository name.

    Returns:
        Pull request information as a string.
    """
    repositories = load_repositories()

    for repo in repositories:
        if repo["name"].lower() == repository.lower():
            return json.dumps(repo["latest_pull_request"], indent=2)

    raise ValueError(f"Repository not found: {repository}")


@mcp.tool()
def get_workflow_status(repository: str) -> str:
    """
    Get the latest CI workflow status for a Git repository.

    Args:
        repository: Repository name.

    Returns:
        Workflow status as a string.
    """
    repositories = load_repositories()

    for repo in repositories:
        if repo["name"].lower() == repository.lower():
            workflow = repo["workflow"]
            return (
                f"Workflow: {workflow['name']}\n"
                f"Status: {workflow['status']}\n"
                f"Branch: {workflow['branch']}\n"
            )

    raise ValueError(f"Repository not found: {repository}")


if __name__ == "__main__":
    # stdout belongs to the MCP protocol; diagnostics must go to stderr.
    print("📟 stdio MCP Git Server running...", file=sys.stderr)
    print(
        "🔧 Tools: get_repository, get_pull_request, get_workflow_status",
        file=sys.stderr,
    )

    # Keep transport startup outside a broad exception handler so protocol and
    # infrastructure failures retain their original traceback and exit status.
    mcp.run(transport="stdio")