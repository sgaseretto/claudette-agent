# How claudette-agent Works

This document explains the internal architecture and design of claudette-agent, a Claudette-compatible wrapper for the Claude Agent SDK.

## Architecture Overview

```mermaid
graph TB
    subgraph "User Code"
        UC[User Application]
    end

    subgraph "claudette-agent"
        CA[Chat / AsyncChat]
        CL[Client / AsyncClient]
        CORE[core.py<br/>Types & Utilities]
        TOOLS[tools.py<br/>Tool Support]
        STRUCT[structured.py<br/>Pydantic Support]
        STREAM[streaming.py<br/>Streaming]
        MCP[mcp.py<br/>MCP Integration]
        TE[text_editor.py<br/>File Operations]
    end

    subgraph "Claude Agent SDK"
        SDK[ClaudeSDKClient]
        QUERY[query()]
        MCPS[MCP Servers]
    end

    UC --> CA
    UC --> CL
    CA --> CL
    CA --> TOOLS
    CL --> CORE
    CL --> SDK
    CL --> QUERY
    TOOLS --> MCP
    TOOLS --> MCPS
    STRUCT --> CA
    STREAM --> QUERY
```

## Core Components

### 1. Client Layer

The `Client` and `AsyncClient` classes provide the low-level interface to the Claude Agent SDK.

```mermaid
classDiagram
    class Client {
        +model: str
        +use: Usage
        +result: Message
        +stop_reason: str
        +setting_sources: List[str]
        +__call__(msgs, sp, temp, maxtok, ...) Message
        +structured(msgs, tools, ...) List
        +cost: float
        -_build_options() ClaudeAgentOptions
        -_r(r, prefill) Message
    }

    class AsyncClient {
        +__call__(msgs, ...) Message
        +structured(msgs, ...) List
    }

    AsyncClient --|> Client
```

### 2. Chat Layer

The `Chat` and `AsyncChat` classes maintain conversation history and provide tool support.

```mermaid
classDiagram
    class Chat {
        +h: List[Dict]
        +sp: str
        +tools: List
        +c: Client
        +ns: Dict
        +last: List[Dict]
        +__call__(pr, temp, ...) Message
        +toolloop(pr, max_steps, ...) ToolLoopResult
        +stream(pr, ...) AsyncIterator
        -_build_conversation_prompt() str
        -_build_options() ClaudeAgentOptions
        -_call_with_tools() Message
        -_call_simple() Message
    }

    class AsyncChat {
        +__call__(pr, ...) Message
        +toolloop(pr, ...) ToolLoopResult
    }

    AsyncChat --|> Chat
```

## Message Flow

### Simple Query (No Tools)

```mermaid
sequenceDiagram
    participant User
    participant Chat
    participant Client
    participant SDK as sdk_query()
    participant Claude

    User->>Chat: await chat("Hello")
    Chat->>Chat: _append_pr()
    Chat->>Chat: _build_conversation_prompt()
    Chat->>Chat: _build_options()
    Chat->>Client: _call_simple()
    Client->>SDK: sdk_query(prompt, options)
    SDK->>Claude: API Request
    Claude-->>SDK: Response
    SDK-->>Client: AssistantMessage, ResultMessage
    Client-->>Chat: Message
    Chat->>Chat: Update history
    Chat-->>User: Message
```

### Query with Tools

```mermaid
sequenceDiagram
    participant User
    participant Chat
    participant Client
    participant SDK as ClaudeSDKClient
    participant MCP as MCP Server
    participant Claude

    User->>Chat: await chat("Calculate 2+2")
    Chat->>Chat: _setup_tools()
    Chat->>MCP: create_sdk_mcp_server(tools)
    Chat->>Chat: _build_options(mcp_servers)
    Chat->>SDK: ClaudeSDKClient(options)
    SDK->>Claude: Initial Request
    Claude-->>SDK: ToolUse Request
    SDK->>MCP: Execute Tool
    MCP-->>SDK: Tool Result
    SDK->>Claude: Tool Result
    Claude-->>SDK: Final Response
    SDK-->>Chat: AssistantMessage, ResultMessage
    Chat->>Chat: Update history
    Chat-->>User: Message
```

## Extended Thinking

Extended thinking is enabled via the `MAX_THINKING_TOKENS` environment variable in `ClaudeAgentOptions`:

```mermaid
flowchart LR
    A[maxthinktok=2048] --> B[_build_options]
    B --> C["env={'MAX_THINKING_TOKENS': '2048'}"]
    C --> D[ClaudeAgentOptions]
    D --> E[SDK Request]
    E --> F[ThinkingBlock in Response]
```

## Tool Loop

The `toolloop()` method handles multi-step tool execution:

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> SendPrompt
    SendPrompt --> CheckStopReason
    CheckStopReason --> ToolUse: stop_reason == 'tool_use'
    CheckStopReason --> Done: stop_reason != 'tool_use'
    ToolUse --> ExecuteTool
    ExecuteTool --> SendToolResult
    SendToolResult --> CheckMaxSteps
    CheckMaxSteps --> CheckStopReason: steps < max_steps
    CheckMaxSteps --> AddFinalPrompt: steps == max_steps - 1
    AddFinalPrompt --> SendPrompt
    Done --> ReturnToolLoopResult
    ReturnToolLoopResult --> [*]
```

## Image Processing

Images are processed in `mk_msg()`:

```mermaid
flowchart TB
    A[Input] --> B{Type?}
    B -->|str| C[Text Block]
    B -->|bytes| D[Detect MIME Type]
    D --> E[Base64 Encode]
    E --> F[Image Block]
    B -->|list| G[Process Each Item]
    G --> H[Mixed Content Blocks]
    C --> I[Message Dict]
    F --> I
    H --> I
```

## Text Editor Tool

The text editor provides file manipulation operations:

```mermaid
flowchart LR
    subgraph Commands
        V[view]
        CR[create]
        I[insert]
        SR[str_replace]
        U[undo_edit]
    end

    A[str_replace_editor] --> B{command?}
    B --> V --> R1[Read File/Dir]
    B --> CR --> R2[Write New File]
    B --> I --> R3[Insert at Line]
    B --> SR --> R4[Replace Text]
    B --> U --> R5[Restore Previous]

    R2 --> H[Undo History]
    R3 --> H
    R4 --> H
    H --> U
```

## Key Design Patterns

### 1. Dual Path Architecture

Chat uses different code paths depending on whether tools are needed:

- **With Tools**: Uses `ClaudeSDKClient` with MCP servers
- **Without Tools**: Uses simple `sdk_query()` for efficiency

### 2. SDK Message Extraction

The SDK returns different message types:

- `AssistantMessage`: Contains response content
- `ResultMessage`: Contains usage and cost information

```python
async for msg in sdk_query(...):
    if isinstance(msg, ResultMessage):
        # Extract usage from here
        usage = msg.usage
    elif isinstance(msg, AssistantMessage):
        # Extract content from here
        content = msg.content
```

### 3. Tool Wrapping

Python functions are converted to SDK-compatible tools:

```mermaid
flowchart LR
    A[Python Function] --> B[@tool decorator]
    B --> C[Extract Signature]
    C --> D[Build Schema]
    D --> E[sdk_tool wrapper]
    E --> F[MCP Server]
```

### 4. Mixin Pattern

Structured output support is added via mixins:

```python
class StructuredMixin:
    async def struct(self, msgs, resp_model, ...):
        # Implementation

add_struct_to_chat(Chat)  # Adds .struct() method
```

## File Structure

```
claudette_agent/
├── __init__.py      # Public API exports
├── core.py          # Types, utilities, constants
├── client.py        # Client & AsyncClient
├── chat.py          # Chat & AsyncChat
├── tools.py         # Tool support & MCP
├── structured.py    # Pydantic support
├── streaming.py     # Streaming responses
├── mcp.py           # MCP server config
└── text_editor.py   # Text editor tool
```

## Stateless Mode (`setting_sources`)

By default, claudette-agent runs in stateless mode (`setting_sources=[]`), meaning each query is independent:

```mermaid
flowchart LR
    A[setting_sources] --> B{Value?}
    B -->|"[]"| C[Stateless Mode]
    B -->|"['user', 'project', 'local']"| D[Session Persistence]
    C --> E[No settings loaded<br/>Independent queries<br/>Ideal for API use]
    D --> F[Load ~/.claude/<br/>Load .claude/<br/>Session history]
```

The `setting_sources` parameter is passed through to `ClaudeAgentOptions`:

```python
opts = {
    'system_prompt': sp,
    'setting_sources': self.setting_sources,  # [] for stateless
    # ... other options
}
ClaudeAgentOptions(**opts)
```

## Feature Mapping: claudette → claudette-agent

| claudette Feature | claudette-agent Implementation |
|-------------------|-------------------------------|
| `Client.__call__()` | Uses `sdk_query()` |
| `Chat.__call__()` | Uses `sdk_query()` or `ClaudeSDKClient` |
| `toolloop()` | Returns `ToolLoopResult` with `.value` |
| `maxthinktok` | Via `env={'MAX_THINKING_TOKENS': ...}` |
| `prefill` | Added to prompt as instruction |
| `cache` | Via `cache_control` in messages |
| `struct()` | Via prompt engineering + JSON parsing |
| `stream` | Via SDK async iterator |
| `setting_sources` | Via `ClaudeAgentOptions.setting_sources` |
