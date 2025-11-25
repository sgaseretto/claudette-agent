"""
Client module - Main Client and AsyncClient classes for claudette_agent.
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator, Iterator

from .core import (
    Usage, usage, Message, TextBlock, ToolUseBlock, ThinkingBlock,
    find_block, contents, mk_msg, mk_msgs, mk_toolres, mk_toolres_async,
    get_schema, mk_tool_choice, listify, mk_ns, call_func,
    model_types, pricing, DEFAULT_MODEL
)

try:
    from claude_agent_sdk import (
        query as sdk_query,
        ClaudeSDKClient,
        ClaudeAgentOptions,
        AssistantMessage as SDKAssistantMessage,
        TextBlock as SDKTextBlock,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


def _parse_sdk_message(msg: Any) -> Message:
    """Convert SDK message to our Message format."""
    content_blocks = []
    msg_usage = usage()

    if hasattr(msg, 'content'):
        for block in msg.content:
            if hasattr(block, 'text'):
                content_blocks.append(TextBlock(text=block.text))
            elif hasattr(block, 'type'):
                if block.type == 'text':
                    content_blocks.append(TextBlock(text=getattr(block, 'text', '')))
                elif block.type == 'tool_use':
                    content_blocks.append(ToolUseBlock(
                        id=getattr(block, 'id', ''),
                        name=getattr(block, 'name', ''),
                        input=getattr(block, 'input', {})
                    ))
                elif block.type == 'thinking':
                    content_blocks.append(ThinkingBlock(thinking=getattr(block, 'thinking', '')))

    if hasattr(msg, 'usage'):
        u = msg.usage
        msg_usage = usage(
            inp=getattr(u, 'input_tokens', 0),
            out=getattr(u, 'output_tokens', 0),
            cache_create=getattr(u, 'cache_creation_input_tokens', 0),
            cache_read=getattr(u, 'cache_read_input_tokens', 0)
        )

    return Message(
        id=getattr(msg, 'id', str(uuid.uuid4())),
        role=getattr(msg, 'role', 'assistant'),
        content=content_blocks,
        model=getattr(msg, 'model', ''),
        stop_reason=getattr(msg, 'stop_reason', None),
        stop_sequence=getattr(msg, 'stop_sequence', None),
        usage=msg_usage
    )


def _simple_text_message(text: str) -> Message:
    """Create a simple text message."""
    return Message(
        id=str(uuid.uuid4()),
        role='assistant',
        content=[TextBlock(text=text)],
        usage=usage()
    )


class Client:
    """
    Claude Agent SDK client with Claudette-compatible API.

    This client wraps the claude-agent-sdk to provide the same interface as Claudette.

    Example:
        >>> client = Client('claude-sonnet-4-5-20250929')
        >>> response = await client("What is 2+2?", sp="You are a helpful assistant")
        >>> print(contents(response))
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cli: Any = None,  # For compatibility - not used with agent SDK
        log: bool = False,
        cache: bool = False,
        cwd: str = None,
        allowed_tools: List[str] = None,
        permission_mode: str = "default"
    ):
        """
        Initialize the Client.

        Args:
            model: The model to use (e.g., 'claude-sonnet-4-5-20250929')
            cli: Ignored - for Claudette API compatibility
            log: Whether to log requests/responses
            cache: Whether to use caching
            cwd: Working directory for operations
            allowed_tools: List of allowed tools
            permission_mode: Permission mode for tools
        """
        if not SDK_AVAILABLE:
            raise ImportError(
                "claude-agent-sdk is not installed. "
                "Install it with: pip install claude-agent-sdk"
            )

        self.model = model
        self.use = usage()
        self.log = [] if log else None
        self.cache = cache
        self.cwd = cwd
        self.allowed_tools = allowed_tools
        self.permission_mode = permission_mode
        self.result: Optional[Message] = None
        self.stop_reason: Optional[str] = None
        self.stop_sequence: Optional[str] = None
        self._sdk_tools = []
        self._mcp_servers = {}

    def _r(self, r: Message, prefill: str = '') -> Message:
        """Store the result of the message and accrue total usage."""
        if prefill:
            blk = find_block(r)
            if blk and hasattr(blk, 'text'):
                blk.text = prefill + (blk.text or '')
        self.result = r
        if r.usage:
            self.use = self.use + r.usage
        self.stop_reason = r.stop_reason
        self.stop_sequence = r.stop_sequence
        return r

    def _log_request(self, final: Message, prefill: str, msgs: List, **kwargs) -> Message:
        """Log the request and return the result."""
        self._r(final, prefill)
        if self.log is not None:
            self.log.append({
                "msgs": msgs,
                **kwargs,
                "result": self.result,
                "use": self.use,
                "stop_reason": self.stop_reason,
                "stop_sequence": self.stop_sequence
            })
        return self.result

    def _build_options(
        self,
        sp: str = '',
        tools: Optional[List] = None,
        maxtok: int = 4096,
        **kwargs
    ) -> 'ClaudeAgentOptions':
        """Build ClaudeAgentOptions from parameters."""
        opts = {
            'system_prompt': sp or "You are a helpful assistant.",
            'max_turns': kwargs.get('max_turns', 1),
        }

        if self.cwd:
            opts['cwd'] = self.cwd

        if self.allowed_tools:
            opts['allowed_tools'] = self.allowed_tools

        if self.permission_mode != "default":
            opts['permission_mode'] = self.permission_mode

        if self._mcp_servers:
            opts['mcp_servers'] = self._mcp_servers

        return ClaudeAgentOptions(**opts)

    async def __call__(
        self,
        msgs: Union[str, List],
        sp: str = '',
        temp: float = 0,
        maxtok: int = 4096,
        maxthinktok: int = 0,
        prefill: str = '',
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, bool, Dict]] = None,
        cb: Optional[Callable] = None,
        **kwargs
    ) -> Message:
        """
        Make a call to Claude via the Agent SDK.

        Args:
            msgs: List of messages or a single message string
            sp: System prompt
            temp: Temperature (note: may be limited by SDK)
            maxtok: Maximum tokens
            maxthinktok: Maximum thinking tokens (for extended thinking)
            prefill: Prefill text for Claude's response
            stream: Whether to stream the response
            stop: Stop sequences
            tools: List of tools to make available
            tool_choice: Tool choice configuration
            cb: Callback function for when complete

        Returns:
            Message object with Claude's response
        """
        # Convert single message to list
        if isinstance(msgs, str):
            prompt = msgs
        else:
            # Build prompt from message history
            prompt_parts = []
            for msg in msgs:
                if isinstance(msg, str):
                    prompt_parts.append(msg)
                elif isinstance(msg, dict):
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get('type') == 'text':
                                prompt_parts.append(c.get('text', ''))
                    elif isinstance(content, str):
                        prompt_parts.append(content)
            prompt = "\n\n".join(prompt_parts) if prompt_parts else str(msgs[-1])

        options = self._build_options(sp=sp, tools=tools, maxtok=maxtok, **kwargs)

        collected_text = []
        final_message = None

        try:
            async for msg in sdk_query(prompt=prompt, options=options):
                if hasattr(msg, 'content'):
                    final_message = _parse_sdk_message(msg)
                    for block in msg.content:
                        if hasattr(block, 'text'):
                            collected_text.append(block.text)
        except Exception as e:
            # If SDK call fails, return error message
            final_message = _simple_text_message(f"Error: {str(e)}")

        if final_message is None:
            final_message = _simple_text_message("".join(collected_text) if collected_text else "No response")

        result = self._log_request(final_message, prefill, msgs if isinstance(msgs, list) else [msgs], sp=sp)

        if cb:
            cb(result)

        return result

    def structured(
        self,
        msgs: List,
        tools: Optional[List] = None,
        ns: Optional[Dict[str, Callable]] = None,
        **kwargs
    ) -> List:
        """
        Return the value of all tool calls (generally used for structured outputs).

        Note: This is a sync wrapper around an async operation.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._structured_async(msgs, tools, ns, **kwargs)
        )

    async def _structured_async(
        self,
        msgs: List,
        tools: Optional[List] = None,
        ns: Optional[Dict[str, Callable]] = None,
        **kwargs
    ) -> List:
        """Async implementation of structured."""
        tools = listify(tools)
        res = await self(msgs, tools=tools, tool_choice=tools, **kwargs)

        if ns is None:
            ns = mk_ns(*tools)

        cts = getattr(res, 'content', [])
        results = []

        for block in cts:
            if isinstance(block, ToolUseBlock):
                result = call_func(block.name, block.input, ns=ns)
                results.append(result)

        return results

    @property
    def cost(self) -> float:
        """Calculate the total cost of usage."""
        model_type = model_types.get(self.model, 'sonnet')
        costs = pricing.get(model_type, pricing['sonnet'])
        return self.use.cost(costs)

    def add_mcp_server(self, name: str, server: Any) -> None:
        """Add an MCP server."""
        self._mcp_servers[name] = server

    def _repr_markdown_(self) -> str:
        """Jupyter-friendly representation."""
        if not hasattr(self, 'result') or self.result is None:
            return 'No results yet'

        msg = contents(self.result)
        return f"""{msg}

| Metric | Count | Cost (USD) |
|--------|------:|-----:|
| Input tokens | {self.use.input_tokens:,} | {self.use.input_tokens * 3 / 1e6:.6f} |
| Output tokens | {self.use.output_tokens:,} | {self.use.output_tokens * 15 / 1e6:.6f} |
| Cache tokens | {self.use.cache_creation_input_tokens + self.use.cache_read_input_tokens:,} | {0:.6f} |
| **Total** | **{self.use.total:,}** | **${self.cost:.6f}** |"""


class AsyncClient(Client):
    """
    Async version of the Claude Agent SDK client.

    The base Client is already async-native since the SDK uses async,
    but this class provides explicit async naming for clarity.
    """

    async def __call__(
        self,
        msgs: Union[str, List],
        sp: str = '',
        temp: float = 0,
        maxtok: int = 4096,
        maxthinktok: int = 0,
        prefill: str = '',
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, bool, Dict]] = None,
        cb: Optional[Callable] = None,
        **kwargs
    ) -> Message:
        """Make an async call to Claude."""
        return await super().__call__(
            msgs, sp=sp, temp=temp, maxtok=maxtok, maxthinktok=maxthinktok,
            prefill=prefill, stream=stream, stop=stop, tools=tools,
            tool_choice=tool_choice, cb=cb, **kwargs
        )

    async def structured(
        self,
        msgs: List,
        tools: Optional[List] = None,
        ns: Optional[Dict[str, Callable]] = None,
        **kwargs
    ) -> List:
        """Return the value of all tool calls (async version)."""
        return await self._structured_async(msgs, tools, ns, **kwargs)
