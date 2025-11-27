# claudette-agent

A Claude Agent SDK wrapper with Claudette-compatible API. This package lets you use the familiar [Claudette](https://github.com/AnswerDotAI/claudette) API while leveraging the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) for your Claude Code subscription.

## Installation

```bash
pip install claudette-agent
```

For structured outputs with Pydantic:

```bash
pip install claudette-agent[pydantic]
```

## Requirements

- Python 3.10+
- Claude Code subscription (for claude-agent-sdk)
- claude-agent-sdk installed and configured

## Quick Start

### Basic Chat

```python
import asyncio
from claudette_agent import Chat, contents

async def main():
    # Create a chat with a model and system prompt
    chat = Chat(model="claude-sonnet-4-5-20250929", sp="You are a helpful assistant")

    # Send a message
    response = await chat("Hello! What can you help me with?")
    print(contents(response))

    # Continue the conversation (maintains history)
    response = await chat("Can you explain that in more detail?")
    print(contents(response))

asyncio.run(main())
```

### Simple Query (No History)

```python
import asyncio
from claudette_agent import query, contents

async def main():
    # One-shot query
    response = await query("What is 2 + 2?")
    print(contents(response))

asyncio.run(main())
```

### Using Tools

```python
import asyncio
from claudette_agent import Chat, tool, contents

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 72F"

async def main():
    chat = Chat(
        model="claude-sonnet-4-5-20250929",
        sp="You are a helpful assistant with access to tools",
        tools=[calculate, get_weather]
    )

    response = await chat("What is 15 * 23?")
    print(contents(response))

asyncio.run(main())
```

### Tool Loop (Automatic Tool Following)

```python
import asyncio
from claudette_agent import Chat, tool, contents

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for '{query}': [relevant information]"

async def main():
    chat = Chat(model="claude-sonnet-4-5-20250929", tools=[search])

    # Automatically follow up tool calls
    results = await chat.toolloop(
        "Research Python async programming and summarize",
        max_steps=5
    )

    for result in results:
        print(contents(result))

asyncio.run(main())
```

### Structured Outputs with Pydantic

```python
import asyncio
from pydantic import BaseModel
from claudette_agent import Chat

class Person(BaseModel):
    """Information about a person."""
    name: str
    age: int
    occupation: str

async def main():
    chat = Chat(model="claude-sonnet-4-5-20250929")

    # Note: prompt comes first, then the model class
    person = await chat.struct(
        "Extract: John Smith is a 35-year-old software engineer",
        Person
    )

    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Occupation: {person.occupation}")

asyncio.run(main())
```

### Streaming Responses

```python
import asyncio
from claudette_agent import Chat

async def main():
    chat = Chat(model="claude-sonnet-4-5-20250929")

    async for chunk in chat.stream("Tell me a story about a brave knight"):
        print(chunk, end="", flush=True)

    print()  # Final newline

asyncio.run(main())
```

### MCP Server Integration

```python
import asyncio
from claudette_agent import (
    Chat, MCPToolkit, mcp_tool, create_mcp_server, contents
)

# Create tools for an MCP server
@mcp_tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args):
    result = args['a'] + args['b']
    return {"content": [{"type": "text", "text": str(result)}]}

@mcp_tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply(args):
    result = args['a'] * args['b']
    return {"content": [{"type": "text", "text": str(result)}]}

# Create MCP server
math_server = create_mcp_server("math", tools=[add, multiply])

async def main():
    # Use MCP toolkit for easier management
    toolkit = MCPToolkit()
    toolkit.add_sdk_server("math", [add, multiply])

    # Create chat with MCP servers
    chat = Chat(model="claude-sonnet-4-5-20250929", sp="You are a calculator assistant")
    chat.c.add_mcp_server("math", math_server)

    response = await chat("What is 15 + 27?")
    print(contents(response))

asyncio.run(main())
```

### Cost Tracking

```python
import asyncio
from claudette_agent import Chat, contents

async def main():
    chat = Chat(model="claude-sonnet-4-5-20250929")

    await chat("Explain quantum computing")
    await chat("What are its applications?")

    # Check usage
    print(f"Total tokens: {chat.use.total}")
    print(f"Input tokens: {chat.use.input_tokens}")
    print(f"Output tokens: {chat.use.output_tokens}")
    print(f"Cost: ${chat.cost:.6f}")

asyncio.run(main())
```

## API Reference

### Core Classes

- **`Client`**: Low-level client for direct API calls
- **`AsyncClient`**: Async version of Client
- **`Chat`**: High-level chat interface with conversation history
- **`AsyncChat`**: Async version of Chat

### Key Functions

- **`contents(response)`**: Extract text content from a response
- **`query(prompt)`**: Simple one-shot query (async)
- **`tool`**: Decorator to mark functions as tools
- **`mk_msg(content)`**: Create a message dict
- **`mk_msgs(msgs)`**: Convert messages to API format

### Structured Outputs

- **`claude_schema(model)`**: Generate Claude schema from Pydantic model
- **`chat.struct(prompt, ModelClass)`**: Get structured output as Pydantic model

### MCP Integration

- **`MCPServer`**: Create and manage MCP servers
- **`MCPToolkit`**: Manage multiple MCP servers
- **`mcp_tool`**: Decorator for MCP-compatible tools
- **`create_mcp_server`**: Create an MCP server

## Comparison with Claudette

This package provides the same API as Claudette, but uses the Claude Agent SDK instead of the Anthropic Python SDK. This means:

| Feature | Claudette | claudette-agent |
|---------|-----------|-----------------|
| API | Anthropic SDK | Claude Agent SDK |
| Auth | API Key | Claude Code subscription |
| Tools | Function calling | MCP + Function calling |
| Sessions | Manual | SDK-managed |

### Migration from Claudette

Simply change your import:

```python
# Before (Claudette)
from claudette import Chat, contents

# After (claudette-agent)
from claudette_agent import Chat, contents
```

Most code should work with minor changes:

1. **Async by default**: All `Chat` methods are async and need `await`
2. **Model parameter**: Pass `model="claude-sonnet-4-5-20250929"` to `Chat()`
3. **struct signature**: Use `chat.struct(prompt, ModelClass)` (prompt first)
4. **MCP support**: Additional MCP server integration features

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
