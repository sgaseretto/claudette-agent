"""
MCP (Model Context Protocol) module - Integration with MCP servers.
"""
from typing import Any, Dict, List, Optional, Callable

try:
    from claude_agent_sdk import (
        create_sdk_mcp_server,
        tool as sdk_tool,
        ClaudeAgentOptions,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


class MCPServerConfig:
    """
    Configuration for an external MCP server.

    Example:
        >>> config = MCPServerConfig(
        ...     type="command",
        ...     command="python",
        ...     args=["-m", "my_mcp_server"]
        ... )
    """

    def __init__(
        self,
        type: str = "command",
        command: str = None,
        args: List[str] = None,
        url: str = None,
        env: Dict[str, str] = None
    ):
        """
        Initialize MCP server configuration.

        Args:
            type: Server type ("command" or "sse")
            command: Command to run (for "command" type)
            args: Command arguments (for "command" type)
            url: Server URL (for "sse" type)
            env: Environment variables
        """
        self.type = type
        self.command = command
        self.args = args or []
        self.url = url
        self.env = env or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SDK."""
        config = {"type": self.type}

        if self.type == "command":
            if self.command:
                config["command"] = self.command
            if self.args:
                config["args"] = self.args
        elif self.type == "sse":
            if self.url:
                config["url"] = self.url

        if self.env:
            config["env"] = self.env

        return config


class MCPToolkit:
    """
    Toolkit for managing MCP servers and tools.

    Example:
        >>> toolkit = MCPToolkit()
        >>> toolkit.add_command_server("sqlite", "sqlite3", [":memory:"])
        >>> toolkit.add_sdk_server("calculator", [add_tool, subtract_tool])
        >>> # Use with Client
        >>> client = Client('claude-sonnet-4-5-20250929')
        >>> for name, server in toolkit.servers.items():
        ...     client.add_mcp_server(name, server)
    """

    def __init__(self):
        """Initialize the toolkit."""
        self._servers: Dict[str, Any] = {}
        self._sdk_servers: Dict[str, Any] = {}

    @property
    def servers(self) -> Dict[str, Any]:
        """Get all servers (both external and SDK)."""
        return {**self._servers, **self._sdk_servers}

    def add_command_server(
        self,
        name: str,
        command: str,
        args: List[str] = None,
        env: Dict[str, str] = None
    ) -> None:
        """
        Add an external command-based MCP server.

        Args:
            name: Server name
            command: Command to run
            args: Command arguments
            env: Environment variables
        """
        self._servers[name] = MCPServerConfig(
            type="command",
            command=command,
            args=args,
            env=env
        ).to_dict()

    def add_sse_server(
        self,
        name: str,
        url: str,
        env: Dict[str, str] = None
    ) -> None:
        """
        Add an SSE (Server-Sent Events) MCP server.

        Args:
            name: Server name
            url: Server URL
            env: Environment variables
        """
        self._servers[name] = MCPServerConfig(
            type="sse",
            url=url,
            env=env
        ).to_dict()

    def add_sdk_server(
        self,
        name: str,
        tools: List[Callable],
        version: str = "1.0.0"
    ) -> None:
        """
        Add an in-process SDK MCP server.

        Args:
            name: Server name
            tools: List of tool functions
            version: Server version
        """
        if not SDK_AVAILABLE:
            raise ImportError("claude-agent-sdk is required for SDK servers")

        self._sdk_servers[name] = create_sdk_mcp_server(
            name=name,
            version=version,
            tools=tools
        )

    def remove_server(self, name: str) -> None:
        """Remove a server by name."""
        if name in self._servers:
            del self._servers[name]
        if name in self._sdk_servers:
            del self._sdk_servers[name]

    def get_allowed_tools(self, servers: List[str] = None) -> List[str]:
        """
        Get list of allowed tool names for specified servers.

        Args:
            servers: List of server names (None for all)

        Returns:
            List of tool names in format "mcp__{server}__{tool}"
        """
        if servers is None:
            servers = list(self.servers.keys())

        allowed = []
        for server_name in servers:
            # For SDK servers, we can introspect tools
            if server_name in self._sdk_servers:
                server = self._sdk_servers[server_name]
                if hasattr(server, 'tools'):
                    for tool in server.tools:
                        tool_name = getattr(tool, '__name__', str(tool))
                        allowed.append(f"mcp__{server_name}__{tool_name}")
                else:
                    # Generic pattern
                    allowed.append(f"mcp__{server_name}__*")

        return allowed


def create_mcp_server(
    name: str,
    version: str = "1.0.0",
    tools: List[Callable] = None
) -> Any:
    """
    Create an SDK MCP server with the given tools.

    This is a convenience function that wraps create_sdk_mcp_server.

    Args:
        name: Server name
        version: Version string
        tools: List of tool functions

    Returns:
        MCP server object

    Example:
        >>> @tool("add", "Add two numbers", {"a": float, "b": float})
        ... async def add(args):
        ...     return {"content": [{"type": "text", "text": str(args['a'] + args['b'])}]}
        >>> server = create_mcp_server("math", tools=[add])
    """
    if not SDK_AVAILABLE:
        raise ImportError("claude-agent-sdk is required")

    return create_sdk_mcp_server(
        name=name,
        version=version,
        tools=tools or []
    )


def mcp_tool(name: str, description: str, params: Dict[str, type]) -> Callable:
    """
    Create an MCP-compatible tool decorator.

    This wraps the SDK's tool decorator for creating tools that work with MCP servers.

    Args:
        name: Tool name
        description: Tool description
        params: Dict mapping parameter names to types

    Returns:
        Decorator function

    Example:
        >>> @mcp_tool("greet", "Greet a user", {"name": str})
        ... async def greet(args):
        ...     return {"content": [{"type": "text", "text": f"Hello, {args['name']}!"}]}
    """
    if not SDK_AVAILABLE:
        raise ImportError("claude-agent-sdk is required")

    return sdk_tool(name, description, params)


# Pre-configured server helpers
def sqlite_server(database: str = ":memory:") -> Dict[str, Any]:
    """
    Create a SQLite MCP server configuration.

    Args:
        database: Path to database or ":memory:"

    Returns:
        Server configuration dict
    """
    return MCPServerConfig(
        type="command",
        command="sqlite3",
        args=[database]
    ).to_dict()


def filesystem_server(root_path: str = ".") -> Dict[str, Any]:
    """
    Create a filesystem MCP server configuration.

    Args:
        root_path: Root path for file operations

    Returns:
        Server configuration dict
    """
    return MCPServerConfig(
        type="command",
        command="python",
        args=["-m", "mcp_filesystem", root_path]
    ).to_dict()
