"""
Tools module - Tool creation and MCP server integration.
"""
import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from functools import wraps

try:
    from claude_agent_sdk import (
        tool as sdk_tool,
        create_sdk_mcp_server,
        ClaudeAgentOptions,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


def get_schema(func: Callable) -> Dict[str, Any]:
    """
    Generate a Claude tool schema from a function.

    Args:
        func: The function to generate a schema for

    Returns:
        Dict containing the tool schema
    """
    if isinstance(func, dict):
        return func

    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    doc = inspect.getdoc(func) or ""

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ('self', 'cls'):
            continue

        param_type = hints.get(name, str)
        json_type = _python_to_json_type(param_type)

        prop = {"type": json_type}

        # Try to extract parameter description from docstring
        # Support both :param name: and Args: formats
        param_desc = _extract_param_description(doc, name)
        if param_desc:
            prop["description"] = param_desc

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": doc.split("\n")[0] if doc else f"Function {func.__name__}",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


def _extract_param_description(doc: str, param_name: str) -> Optional[str]:
    """Extract parameter description from docstring."""
    # Try :param name: format
    if f":param {param_name}:" in doc:
        desc_start = doc.index(f":param {param_name}:") + len(f":param {param_name}:")
        desc_end = doc.find("\n:", desc_start)
        if desc_end == -1:
            desc_end = doc.find("\n\n", desc_start)
        if desc_end == -1:
            desc_end = len(doc)
        return doc[desc_start:desc_end].strip()

    # Try Args: format (Google style)
    if "Args:" in doc:
        args_start = doc.index("Args:")
        args_section = doc[args_start:]

        # Find the parameter in the args section
        lines = args_section.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(f"{param_name}:") or stripped.startswith(f"{param_name} "):
                # Get description from same line after colon
                if ':' in stripped:
                    desc = stripped.split(':', 1)[1].strip()
                    # Continue to next lines if indented
                    j = i + 1
                    while j < len(lines) and lines[j].startswith('    ') and not lines[j].strip().endswith(':'):
                        desc += ' ' + lines[j].strip()
                        j += 1
                    return desc
    return None


def _python_to_json_type(py_type: type) -> str:
    """Convert Python type to JSON schema type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    origin = getattr(py_type, '__origin__', None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    return type_map.get(py_type, "string")


def tool(func: Callable = None, *, name: str = None, description: str = None, params: Dict = None):
    """
    Decorator to mark a function as a tool for Claude.

    Can be used in two ways:

    1. Simple decorator:
        @tool
        def my_func(x: int) -> str:
            '''Description of my function'''
            return str(x)

    2. With parameters:
        @tool(name="custom_name", description="Custom description")
        def my_func(x: int) -> str:
            return str(x)

    3. SDK-style with params dict:
        @tool("greet", "Greet a user", {"name": str})
        async def greet(args):
            return {"content": [{"type": "text", "text": f"Hello, {args['name']}!"}]}

    Args:
        func: The function to decorate (when used without arguments)
        name: Optional custom name for the tool
        description: Optional custom description
        params: Optional dict of parameter name -> type (SDK style)
    """
    # If called without parentheses: @tool
    if func is not None and callable(func):
        return _wrap_tool(func)

    # If called with SDK-style arguments: @tool("name", "desc", {params})
    if isinstance(func, str):
        tool_name = func
        tool_desc = name  # Second positional arg
        tool_params = description  # Third positional arg

        def decorator(f):
            # This is SDK-style tool
            if SDK_AVAILABLE:
                return sdk_tool(tool_name, tool_desc, tool_params)(f)
            else:
                # Wrap for non-SDK use
                f._tool_name = tool_name
                f._tool_description = tool_desc
                f._tool_params = tool_params
                return f

        return decorator

    # If called with keyword arguments: @tool(name="x", description="y")
    def decorator(f):
        wrapped = _wrap_tool(f)
        if name:
            wrapped._tool_name = name
        if description:
            wrapped._tool_description = description
        return wrapped

    return decorator


def _wrap_tool(func: Callable) -> Callable:
    """Wrap a function as a tool."""
    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        new_args = []
        for i, (param_name, arg) in enumerate(zip(sig.parameters, args)):
            if param_name in hints and isinstance(arg, dict):
                try:
                    new_args.append(hints[param_name](**arg))
                except (TypeError, ValueError):
                    new_args.append(arg)
            else:
                new_args.append(arg)

        new_kwargs = {}
        for k, v in kwargs.items():
            if k in hints and isinstance(v, dict):
                try:
                    new_kwargs[k] = hints[k](**v)
                except (TypeError, ValueError):
                    new_kwargs[k] = v
            else:
                new_kwargs[k] = v

        return func(*new_args, **new_kwargs)

    # Preserve tool metadata
    wrapper._is_tool = True
    wrapper._tool_schema = get_schema(func)

    return wrapper


def create_mcp_server(
    name: str,
    version: str = "1.0.0",
    tools: List[Callable] = None
) -> Any:
    """
    Create an MCP server with the given tools.

    Args:
        name: Name of the MCP server
        version: Version string
        tools: List of tool functions

    Returns:
        MCP server object compatible with claude-agent-sdk
    """
    if not SDK_AVAILABLE:
        raise ImportError(
            "claude-agent-sdk is not installed. "
            "Install it with: pip install claude-agent-sdk"
        )

    return create_sdk_mcp_server(
        name=name,
        version=version,
        tools=tools or []
    )


class MCPServer:
    """
    Wrapper for creating and managing MCP servers.

    Example:
        >>> server = MCPServer("calculator", "1.0.0")
        >>> @server.tool
        ... def add(a: int, b: int) -> int:
        ...     '''Add two numbers'''
        ...     return a + b
        >>> # Use with Client
        >>> client = Client('claude-sonnet-4-5-20250929')
        >>> client.add_mcp_server("calculator", server.server)
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize the MCP server.

        Args:
            name: Name of the server
            version: Version string
        """
        self.name = name
        self.version = version
        self._tools: List[Callable] = []
        self._server = None

    def tool(self, func: Callable = None, **kwargs) -> Callable:
        """
        Decorator to add a tool to this server.

        Args:
            func: The function to add as a tool

        Returns:
            The decorated function
        """
        def decorator(f):
            wrapped = _wrap_tool(f)
            self._tools.append(wrapped)
            self._server = None  # Invalidate cached server
            return wrapped

        if func is not None:
            return decorator(func)
        return decorator

    @property
    def server(self) -> Any:
        """Get or create the MCP server."""
        if self._server is None:
            self._server = create_mcp_server(
                name=self.name,
                version=self.version,
                tools=self._tools
            )
        return self._server

    def add_tool(self, func: Callable) -> None:
        """Add a tool function to the server."""
        self._tools.append(func)
        self._server = None  # Invalidate cached server


def search_conf(
    max_uses: int = None,
    allowed_domains: List[str] = None,
    blocked_domains: List[str] = None,
    user_location: Dict = None
) -> Dict[str, Any]:
    """
    Create a web search tool configuration.

    Args:
        max_uses: Maximum number of search uses
        allowed_domains: List of allowed domains
        blocked_domains: List of blocked domains
        user_location: User location dict

    Returns:
        Search configuration dict
    """
    conf = {'type': 'web_search_20250305', 'name': 'web_search'}

    if max_uses is not None:
        conf['max_uses'] = max_uses
    if allowed_domains is not None:
        conf['allowed_domains'] = allowed_domains
    if blocked_domains is not None:
        conf['blocked_domains'] = blocked_domains
    if user_location is not None:
        conf['user_location'] = user_location

    return conf
