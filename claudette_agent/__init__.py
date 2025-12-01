"""
claudette_agent - Claude Agent SDK wrapper with Claudette-compatible API.

This package provides a Claudette-compatible interface for the Claude Agent SDK,
allowing you to use the same familiar API but powered by the Claude Agent SDK
instead of the direct Anthropic API.

Example usage:
    >>> from claudette_agent import Chat, contents
    >>> chat = Chat(sp="You are a helpful assistant")
    >>> response = await chat("Hello!")
    >>> print(contents(response))

For structured outputs:
    >>> from claudette_agent import Chat, struct_sync
    >>> from pydantic import BaseModel
    >>> class Person(BaseModel):
    ...     '''A person's information'''
    ...     name: str
    ...     age: int
    >>> chat = Chat()
    >>> person = struct_sync(chat, "Extract: John is 25 years old", Person)
    >>> print(person.name, person.age)

For tools:
    >>> from claudette_agent import Chat, tool
    >>> @tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers'''
    ...     return a + b
    >>> chat = Chat(tools=[add])
    >>> response = await chat("What is 2 + 3?")
"""

__version__ = "0.1.0"

# Core types and utilities
from .core import (
    # Types
    Usage,
    Message,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ThinkingBlock,
    ToolResult,
    # Utilities
    usage,
    find_block,
    find_blocks,
    contents,
    mk_msg,
    mk_msgs,
    mk_toolres,
    mk_toolres_async,
    mk_funcres,
    mk_funcres_async,
    mk_tool_choice,
    get_schema,
    listify,
    mk_ns,
    call_func,
    call_func_async,
    think_md,
    get_costs,
    # Constants
    empty,
    model_types,
    all_models,
    models,
    pricing,
    DEFAULT_MODEL,
    OPUS_MODEL,
    HAIKU_MODEL,
)

# Client classes
from .client import (
    Client,
    AsyncClient,
)

# Chat classes
from .chat import (
    Chat,
    AsyncChat,
)

# Tool support
from .tools import (
    tool,
    get_schema,
    search_conf,
    create_mcp_server,
    MCPServer,
)

# Structured outputs
from .structured import (
    claude_schema,
    struct_sync,
    add_struct_to_client,
    add_struct_to_chat,
    StructuredMixin,
)

# Streaming
from .streaming import (
    StreamingResponse,
    StreamingMixin,
    stream_text,
    stream_text_sync,
    TextStream,
)

# MCP integration
from .mcp import (
    MCPServerConfig,
    MCPToolkit,
    mcp_tool,
    sqlite_server,
    filesystem_server,
)

# Apply mixins to classes
add_struct_to_client(Client)
add_struct_to_client(AsyncClient)
add_struct_to_chat(Chat)
add_struct_to_chat(AsyncChat)

# Public API
__all__ = [
    # Version
    '__version__',

    # Core types
    'Usage',
    'Message',
    'TextBlock',
    'ToolUseBlock',
    'ToolResultBlock',
    'ThinkingBlock',
    'ToolResult',

    # Utilities
    'usage',
    'find_block',
    'find_blocks',
    'contents',
    'mk_msg',
    'mk_msgs',
    'mk_toolres',
    'mk_toolres_async',
    'mk_funcres',
    'mk_funcres_async',
    'mk_tool_choice',
    'get_schema',
    'listify',
    'mk_ns',
    'call_func',
    'call_func_async',
    'think_md',
    'get_costs',

    # Constants
    'empty',
    'model_types',
    'all_models',
    'models',
    'pricing',
    'DEFAULT_MODEL',
    'OPUS_MODEL',
    'HAIKU_MODEL',

    # Client classes
    'Client',
    'AsyncClient',

    # Chat classes
    'Chat',
    'AsyncChat',

    # Tool support
    'tool',
    'search_conf',
    'create_mcp_server',
    'MCPServer',

    # Structured outputs
    'claude_schema',
    'struct_sync',
    'add_struct_to_client',
    'add_struct_to_chat',
    'StructuredMixin',

    # Streaming
    'StreamingResponse',
    'StreamingMixin',
    'stream_text',
    'stream_text_sync',
    'TextStream',

    # MCP integration
    'MCPServerConfig',
    'MCPToolkit',
    'mcp_tool',
    'sqlite_server',
    'filesystem_server',
]


# Convenience function for quick queries (similar to claudette's simple interface)
async def query(
    prompt: str,
    sp: str = "You are a helpful assistant.",
    model: str = DEFAULT_MODEL,
    **kwargs
) -> Message:
    """
    Quick query to Claude without maintaining conversation state.

    Args:
        prompt: The prompt to send
        sp: System prompt
        model: Model to use
        **kwargs: Additional options passed to Client

    Returns:
        Message with Claude's response

    Example:
        >>> response = await query("What is 2+2?")
        >>> print(contents(response))
    """
    client = Client(model=model, **kwargs)
    return await client(prompt, sp=sp)


def query_sync(
    prompt: str,
    sp: str = "You are a helpful assistant.",
    model: str = DEFAULT_MODEL,
    **kwargs
) -> Message:
    """
    Synchronous version of query.

    Args:
        prompt: The prompt to send
        sp: System prompt
        model: Model to use
        **kwargs: Additional options

    Returns:
        Message with Claude's response
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(
        query(prompt, sp=sp, model=model, **kwargs)
    )


# Add to exports
__all__.extend(['query', 'query_sync'])
