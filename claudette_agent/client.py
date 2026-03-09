"""
Client module - Main Client and AsyncClient classes for claudette_agent.
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator, Iterator, MutableMapping, Literal

from .core import (
    Usage, usage, Message, TextBlock, ToolUseBlock, ThinkingBlock,
    find_block, contents, mk_msg, mk_msgs, mk_toolres, mk_toolres_async,
    get_schema, mk_tool_choice, listify, mk_ns, call_func,
    model_types, pricing, DEFAULT_MODEL,
    _parse_usage, _parse_sdk_message, _simple_text_message,
    AssistantMessage as SDKAssistantMessage,
    ResultMessage as SDKResultMessage,
    StreamEvent,
)

try:
    from claude_agent_sdk import (
        query as sdk_query,
        ClaudeSDKClient,
        ClaudeAgentOptions,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


def _has_option(name: str) -> bool:
    """Check if ClaudeAgentOptions supports a given field."""
    if not SDK_AVAILABLE:
        return False
    import dataclasses
    return name in {f.name for f in dataclasses.fields(ClaudeAgentOptions)}


class Client:
    """
    Claude Agent SDK client with Claudette-compatible API.

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
        permission_mode: str = "default",
        setting_sources: List[str] = None,
        env: MutableMapping[str, str] = None,
        extra_args: Dict[str, Any] = None,
        # New SDK features
        max_turns: int = None,
        max_budget_usd: float = None,
        fallback_model: str = None,
        can_use_tool: Callable = None,
        hooks: Dict = None,
        agents: Dict = None,
        enable_file_checkpointing: bool = False,
        thinking: Any = None,
        effort: Literal["low", "medium", "high", "max"] = None,
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
            setting_sources: List of setting sources to load ('user', 'project', 'local').
                           Default [] = stateless (no settings loaded).
            env: Environment variables to pass to the Claude CLI process.
            extra_args: Additional CLI arguments to pass to ClaudeAgentOptions.
            max_turns: Maximum agentic turns (tool-use round trips)
            max_budget_usd: Maximum budget in USD for the session
            fallback_model: Fallback model if primary fails
            can_use_tool: Custom permission callback for tool use
            hooks: Hook configurations for intercepting events
            agents: Subagent definitions
            enable_file_checkpointing: Enable file change tracking for rewinding
            thinking: Extended thinking config - can be:
                      {"type": "adaptive"}, {"type": "enabled", "budget_tokens": N},
                      {"type": "disabled"}, or None
            effort: Effort level for thinking depth ("low", "medium", "high", "max")
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
        self.setting_sources = setting_sources if setting_sources is not None else []
        self.env = dict(env) if env else {}
        self.extra_args = dict(extra_args) if extra_args else {}
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd
        self.fallback_model = fallback_model
        self.can_use_tool = can_use_tool
        self.hooks = hooks
        self.agents = agents
        self.enable_file_checkpointing = enable_file_checkpointing
        self.thinking = thinking
        self.effort = effort
        self.result: Optional[Message] = None
        self.stop_reason: Optional[str] = None
        self.stop_sequence: Optional[str] = None
        self._sdk_tools = []
        self._mcp_servers = {}

    def _r(self, r: Message, prefill: str = '') -> Message:
        """Store the result of the message and accrue total usage."""
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
        maxthinktok: int = 0,
        stream: bool = False,
        **kwargs
    ) -> 'ClaudeAgentOptions':
        """Build ClaudeAgentOptions for SDK query."""
        # System prompt: support string or dict (preset)
        if isinstance(sp, dict):
            system_prompt = sp
        else:
            system_prompt = sp or "You are a helpful assistant."

        opts = {
            'system_prompt': system_prompt,
            'max_turns': kwargs.get('max_turns') or self.max_turns or 1,
            'setting_sources': self.setting_sources,
        }

        if self.cwd:
            opts['cwd'] = self.cwd

        if self.allowed_tools:
            opts['allowed_tools'] = self.allowed_tools

        if self.permission_mode != "default":
            opts['permission_mode'] = self.permission_mode

        if self._mcp_servers:
            opts['mcp_servers'] = self._mcp_servers

        # Merge environment variables from client instance
        if self.env:
            opts['env'] = opts.get('env', {})
            opts['env'].update(self.env)

        # Extended thinking via native SDK support
        if maxthinktok and maxthinktok > 0:
            if stream:
                raise ValueError(
                    "Streaming is incompatible with extended thinking in the Claude Agent SDK. "
                    "Use stream=False when using maxthinktok, or set maxthinktok=0 for streaming."
                )
            # Use max_thinking_tokens (current SDK) or thinking (future SDK)
            if _has_option('thinking'):
                opts['thinking'] = {"type": "enabled", "budget_tokens": maxthinktok}
            else:
                opts['max_thinking_tokens'] = maxthinktok
        elif self.thinking:
            if _has_option('thinking'):
                opts['thinking'] = self.thinking
            elif isinstance(self.thinking, dict) and self.thinking.get('type') == 'enabled':
                opts['max_thinking_tokens'] = self.thinking.get('budget_tokens', 0)

        # Effort level (if SDK supports it)
        if self.effort and _has_option('effort'):
            opts['effort'] = self.effort

        # Streaming: enable partial messages for char-by-char streaming
        if stream:
            opts['include_partial_messages'] = True

        # New SDK features
        if self.max_budget_usd is not None:
            opts['max_budget_usd'] = self.max_budget_usd

        if self.fallback_model:
            opts['fallback_model'] = self.fallback_model

        if self.can_use_tool:
            opts['can_use_tool'] = self.can_use_tool

        if self.hooks:
            opts['hooks'] = self.hooks

        if self.agents:
            opts['agents'] = self.agents

        if self.enable_file_checkpointing:
            opts['enable_file_checkpointing'] = True

        # Merge extra_args into SDK's extra_args for CLI arguments
        if self.extra_args:
            opts['extra_args'] = opts.get('extra_args', {})
            opts['extra_args'].update(self.extra_args)

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
            sp: System prompt (string or dict for preset, e.g.
                {"type": "preset", "preset": "claude_code", "append": "..."})
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

        # Add prefill instruction to prompt if provided
        if prefill:
            prompt = f"{prompt}\n\n[Start your response with: {prefill}]"

        options = self._build_options(
            sp=sp, tools=tools, maxtok=maxtok, maxthinktok=maxthinktok,
            stream=stream, **kwargs
        )

        # If streaming, return a StreamingResponse
        if stream:
            from .streaming import StreamingResponse
            async_iter = sdk_query(prompt=prompt, options=options)
            return StreamingResponse(
                async_iter=async_iter,
                prefill=prefill,
                callback=cb
            )

        collected_text = []
        final_message = None
        total_usage = usage()
        total_cost_usd = None

        try:
            async for msg in sdk_query(prompt=prompt, options=options):
                if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                    if hasattr(msg, 'usage') and msg.usage:
                        total_usage = _parse_usage(msg.usage)
                    if hasattr(msg, 'total_cost_usd'):
                        total_cost_usd = msg.total_cost_usd
                    continue

                if SDKAssistantMessage is not None and isinstance(msg, SDKAssistantMessage):
                    if hasattr(msg, 'content'):
                        final_message = _parse_sdk_message(msg)
                        for block in msg.content:
                            if hasattr(block, 'text'):
                                collected_text.append(block.text)
                elif hasattr(msg, 'content'):
                    final_message = _parse_sdk_message(msg)
                    for block in msg.content:
                        if hasattr(block, 'text'):
                            collected_text.append(block.text)

        except Exception as e:
            final_message = _simple_text_message(f"Error: {str(e)}")

        if final_message is None:
            final_message = _simple_text_message("".join(collected_text) if collected_text else "No response")

        if total_usage.total > 0:
            final_message.usage = total_usage

        result = self._log_request(final_message, prefill, msgs if isinstance(msgs, list) else [msgs], sp=sp)

        if total_cost_usd is not None:
            self._last_cost_usd = total_cost_usd

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
