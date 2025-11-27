"""
Chat module - Chat and AsyncChat classes with conversation history.

This module provides a Claudette-compatible API using the Claude Agent SDK.
Key differences from Claudette:
- Tools require MCP server registration (handled automatically)
- Streaming returns complete message blocks, not text chunks
- Uses ClaudeSDKClient for tool support, query() for simple prompts
"""
import asyncio
import uuid
import inspect
from typing import Any, Dict, List, Optional, Union, Callable, get_type_hints

from .core import (
    Usage, usage, Message, TextBlock, ToolUseBlock,
    contents, mk_msg, mk_msgs, mk_toolres, mk_toolres_async,
    get_schema, mk_tool_choice, listify, mk_ns, get_costs,
    model_types, pricing, DEFAULT_MODEL
)
from .client import Client, AsyncClient

try:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        query as sdk_query,
        tool as sdk_tool,
        create_sdk_mcp_server,
        AssistantMessage as SDKAssistantMessage,
        ResultMessage as SDKResultMessage,
        TextBlock as SDKTextBlock,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    SDKAssistantMessage = None
    SDKResultMessage = None


def _parse_usage(u):
    """Parse usage from SDK message - handles both dict and object formats."""
    if u is None:
        return usage()

    # SDK returns usage as a dict
    if isinstance(u, dict):
        return usage(
            inp=u.get('input_tokens', 0),
            out=u.get('output_tokens', 0),
            cache_create=u.get('cache_creation_input_tokens', 0),
            cache_read=u.get('cache_read_input_tokens', 0)
        )

    # Fall back to attribute access for compatibility
    return usage(
        inp=getattr(u, 'input_tokens', 0),
        out=getattr(u, 'output_tokens', 0),
        cache_create=getattr(u, 'cache_creation_input_tokens', 0),
        cache_read=getattr(u, 'cache_read_input_tokens', 0)
    )


def nested_idx(lst: List, *indices) -> Any:
    """Get nested index from list."""
    result = lst
    for idx in indices:
        if result is None:
            return None
        try:
            result = result[idx]
        except (IndexError, KeyError, TypeError):
            return None
    return result


def _convert_to_sdk_tool(func: Callable) -> Any:
    """
    Convert a regular Python function to an SDK tool.

    The SDK expects tools in a specific format created by @tool decorator.
    """
    if not SDK_AVAILABLE:
        raise ImportError("claude-agent-sdk is required")

    # Get function metadata
    name = func.__name__
    doc = inspect.getdoc(func) or f"Function {name}"
    description = doc.split("\n")[0]  # First line of docstring

    # Build parameter schema from type hints
    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    sig = inspect.signature(func)

    params = {}
    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue
        param_type = hints.get(param_name, str)
        # Map Python types to simple types for SDK
        if param_type == int:
            params[param_name] = int
        elif param_type == float:
            params[param_name] = float
        elif param_type == bool:
            params[param_name] = bool
        else:
            params[param_name] = str

    # Create the SDK tool wrapper
    @sdk_tool(name, description, params)
    async def sdk_wrapper(args):
        # Call the original function with the args
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            return {
                "content": [{"type": "text", "text": str(result)}]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "is_error": True
            }

    # Store reference to original function
    sdk_wrapper._original_func = func
    sdk_wrapper._tool_name = name

    return sdk_wrapper


class Chat:
    """
    Claude chat client with conversation history.

    Maintains a conversation history and supports tools, system prompts,
    and message streaming.

    IMPORTANT: When using tools, the Chat class uses ClaudeSDKClient which
    requires tools to be packaged as MCP servers. This is handled automatically.

    Example:
        >>> chat = Chat(model='claude-sonnet-4-5-20250929', sp="You are a helpful assistant")
        >>> response = await chat("Hello!")
        >>> print(contents(response))
        >>> response = await chat("What did I just say?")  # Remembers context

    With tools:
        >>> @tool
        ... def add(a: int, b: int) -> int:
        ...     '''Add two numbers'''
        ...     return a + b
        >>> chat = Chat(model='claude-sonnet-4-5-20250929', tools=[add])
        >>> response = await chat("What is 2 + 3?")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        cli: Optional[Client] = None,
        sp: str = '',
        tools: Optional[List] = None,
        temp: float = 0,
        cont_pr: Optional[str] = None,
        cache: bool = False,
        hist: Optional[List] = None,
        ns: Optional[Dict[str, Callable]] = None,
        cwd: str = None,
        allowed_tools: List[str] = None,
        permission_mode: str = "default"
    ):
        """
        Initialize the Chat.

        Args:
            model: Model to use (leave empty if passing `cli`)
            cli: Client to use (leave empty if passing `model`)
            sp: Optional system prompt
            tools: List of tools to make available to Claude
            temp: Temperature
            cont_pr: User prompt to continue an assistant response
            cache: Use Claude cache?
            hist: Initialize history
            ns: Namespace to search for tools
            cwd: Working directory for operations
            allowed_tools: List of allowed SDK tools
            permission_mode: Permission mode for tools
        """
        if not SDK_AVAILABLE:
            raise ImportError(
                "claude-agent-sdk is not installed. "
                "Install it with: pip install claude-agent-sdk"
            )

        assert model or cli, "Must provide either model or cli"
        assert cont_pr != "", "cont_pr may not be an empty string"

        self.c = cli or Client(
            model or DEFAULT_MODEL,
            cache=cache,
            cwd=cwd,
            allowed_tools=allowed_tools,
            permission_mode=permission_mode
        )

        if hist is None:
            hist = []

        self.h = hist  # Conversation history
        self.sp = sp  # System prompt
        self.cont_pr = cont_pr
        self.temp = temp
        self.cache = cache
        self.last: List[Dict] = []  # Last response messages

        # Process tools - convert to SDK format and create MCP server
        self._original_tools = listify(tools) if tools else []
        self._sdk_tools = []
        self._mcp_server = None
        self._allowed_tools = allowed_tools or []

        if self._original_tools:
            self._setup_tools()

        # Create namespace for tool results
        if ns is None:
            ns = {t.__name__: t for t in self._original_tools} if self._original_tools else {}
        self.ns = ns

    def _setup_tools(self):
        """Set up tools as an MCP server for the SDK."""
        # Convert each tool to SDK format
        for func in self._original_tools:
            if callable(func):
                sdk_t = _convert_to_sdk_tool(func)
                self._sdk_tools.append(sdk_t)
                # Add to allowed tools list
                tool_name = f"mcp__tools__{func.__name__}"
                if tool_name not in self._allowed_tools:
                    self._allowed_tools.append(tool_name)

        # Create MCP server with all tools
        if self._sdk_tools:
            self._mcp_server = create_sdk_mcp_server(
                name="tools",
                version="1.0.0",
                tools=self._sdk_tools
            )

    @property
    def tools(self):
        """Get the original tools list."""
        return self._original_tools

    @property
    def use(self) -> Usage:
        """Get usage statistics."""
        return self.c.use

    @property
    def cost(self) -> float:
        """Get total cost."""
        return self.c.cost

    @property
    def model(self) -> str:
        """Get model name."""
        return self.c.model

    def _post_pr(self, pr: Any, prev_role: str) -> None:
        """Post-process prompt and add to history."""
        if pr is None and prev_role == 'assistant':
            if self.cont_pr is None:
                raise ValueError("Prompt must be given after completion, or use `self.cont_pr`.")
            pr = self.cont_pr

        if pr:
            self.h.append(mk_msg(pr, cache=self.cache))

    def _append_pr(self, pr: Any = None) -> None:
        """Append prompt to history, handling role alternation."""
        prev_role = nested_idx(self.h, -1, 'role') if self.h else 'assistant'
        self._post_pr(pr, prev_role)

    def _build_options(self, **kwargs) -> 'ClaudeAgentOptions':
        """Build ClaudeAgentOptions for the SDK call."""
        opts = {
            'system_prompt': self.sp or "You are a helpful assistant.",
        }

        if kwargs.get('max_turns'):
            opts['max_turns'] = kwargs['max_turns']

        if self.c.cwd:
            opts['cwd'] = self.c.cwd

        # Add MCP server if we have tools
        if self._mcp_server:
            opts['mcp_servers'] = {"tools": self._mcp_server}

        # Add allowed tools
        if self._allowed_tools:
            opts['allowed_tools'] = self._allowed_tools

        return ClaudeAgentOptions(**opts)

    def _build_conversation_prompt(self) -> str:
        """Build a conversation prompt from history."""
        parts = []

        for msg in self.h:
            role = msg.get('role', 'user')
            content = msg.get('content', [])

            if isinstance(content, str):
                parts.append(f"{role.capitalize()}: {content}")
            elif isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get('type') == 'text':
                            text_parts.append(c.get('text', ''))
                        elif c.get('type') == 'tool_result':
                            text_parts.append(f"[Tool Result: {c.get('content', '')}]")
                    elif isinstance(c, str):
                        text_parts.append(c)

                if text_parts:
                    parts.append(f"{role.capitalize()}: {' '.join(text_parts)}")

        return "\n\n".join(parts)

    def _parse_sdk_message(self, msg: Any) -> Message:
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

        # Parse usage - SDK returns it as a dict
        if hasattr(msg, 'usage') and msg.usage:
            msg_usage = _parse_usage(msg.usage)

        return Message(
            id=getattr(msg, 'id', str(uuid.uuid4())),
            role=getattr(msg, 'role', 'assistant'),
            content=content_blocks,
            model=getattr(msg, 'model', ''),
            stop_reason=getattr(msg, 'stop_reason', None),
            stop_sequence=getattr(msg, 'stop_sequence', None),
            usage=msg_usage
        )

    async def _call_with_tools(
        self,
        conversation_text: str,
        options: 'ClaudeAgentOptions',
        **kwargs
    ) -> Message:
        """Make a call using ClaudeSDKClient (required for tools)."""
        collected_text = []
        final_message = None
        total_usage = usage()

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(conversation_text)

                async for msg in client.receive_response():
                    # Check for ResultMessage which has usage and total_cost_usd
                    if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                        # Extract usage from ResultMessage (this is where usage actually is!)
                        if hasattr(msg, 'usage') and msg.usage:
                            total_usage = _parse_usage(msg.usage)
                        if hasattr(msg, 'total_cost_usd'):
                            self.c._last_cost_usd = msg.total_cost_usd
                        continue

                    # Process AssistantMessage for content only (usage is on ResultMessage)
                    if SDKAssistantMessage is not None and isinstance(msg, SDKAssistantMessage):
                        if hasattr(msg, 'content'):
                            final_message = self._parse_sdk_message(msg)
                            for block in msg.content:
                                if hasattr(block, 'text'):
                                    collected_text.append(block.text)
                    elif hasattr(msg, 'content'):
                        # Fallback for other message types
                        final_message = self._parse_sdk_message(msg)
                        for block in msg.content:
                            if hasattr(block, 'text'):
                                collected_text.append(block.text)

        except Exception as e:
            final_message = Message(
                id=str(uuid.uuid4()),
                role='assistant',
                content=[TextBlock(text=f"Error: {str(e)}")],
                usage=usage()
            )

        if final_message is None:
            final_message = Message(
                id=str(uuid.uuid4()),
                role='assistant',
                content=[TextBlock(text="".join(collected_text) if collected_text else "No response")],
                usage=usage()
            )

        # Attach total usage to the final message
        if total_usage.total > 0:
            final_message.usage = total_usage

        return final_message

    async def _call_simple(
        self,
        conversation_text: str,
        options: 'ClaudeAgentOptions',
        **kwargs
    ) -> Message:
        """Make a simple call using query() (no tools)."""
        collected_text = []
        final_message = None
        total_usage = usage()

        try:
            async for msg in sdk_query(prompt=conversation_text, options=options):
                # Check for ResultMessage which has usage and total_cost_usd
                if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                    # Extract usage from ResultMessage (this is where usage actually is!)
                    if hasattr(msg, 'usage') and msg.usage:
                        total_usage = _parse_usage(msg.usage)
                    if hasattr(msg, 'total_cost_usd'):
                        self.c._last_cost_usd = msg.total_cost_usd
                    continue

                # Process AssistantMessage for content only (usage is on ResultMessage)
                if SDKAssistantMessage is not None and isinstance(msg, SDKAssistantMessage):
                    if hasattr(msg, 'content'):
                        final_message = self._parse_sdk_message(msg)
                        for block in msg.content:
                            if hasattr(block, 'text'):
                                collected_text.append(block.text)
                elif hasattr(msg, 'content'):
                    # Fallback for other message types
                    final_message = self._parse_sdk_message(msg)
                    for block in msg.content:
                        if hasattr(block, 'text'):
                            collected_text.append(block.text)

        except Exception as e:
            final_message = Message(
                id=str(uuid.uuid4()),
                role='assistant',
                content=[TextBlock(text=f"Error: {str(e)}")],
                usage=usage()
            )

        if final_message is None:
            final_message = Message(
                id=str(uuid.uuid4()),
                role='assistant',
                content=[TextBlock(text="".join(collected_text) if collected_text else "No response")],
                usage=usage()
            )

        # Attach total usage to the final message
        if total_usage.total > 0:
            final_message.usage = total_usage

        return final_message

    async def _call_impl(
        self,
        temp: Optional[float] = None,
        maxtok: int = 4096,
        maxthinktok: int = 0,
        stream: bool = False,
        prefill: str = '',
        tool_choice: Optional[Union[str, bool, Dict]] = None,
        **kw
    ) -> Message:
        """Internal implementation of the call."""
        if temp is None:
            temp = self.temp

        # Build the full conversation context
        conversation_text = self._build_conversation_prompt()

        # Build options
        options = self._build_options(**kw)

        # Use ClaudeSDKClient if we have tools, otherwise use query()
        if self._mcp_server:
            final_message = await self._call_with_tools(conversation_text, options, **kw)
        else:
            final_message = await self._call_simple(conversation_text, options, **kw)

        # Update client state
        self.c._r(final_message, prefill)

        # Create tool results and update history
        self.last = mk_toolres(final_message, ns=self.ns)
        self.h.extend(self.last)

        return final_message

    async def __call__(
        self,
        pr: Any = None,
        temp: Optional[float] = None,
        maxtok: int = 4096,
        maxthinktok: int = 0,
        stream: bool = False,
        prefill: str = '',
        tool_choice: Optional[Union[str, bool, Dict]] = None,
        **kw
    ) -> Message:
        """
        Send a message and get a response.

        Args:
            pr: Prompt / message
            temp: Temperature
            maxtok: Maximum tokens
            maxthinktok: Maximum thinking tokens
            stream: Stream response? (Note: SDK streams messages, not text chunks)
            prefill: Optional prefill to pass to Claude as start of its response
            tool_choice: Optionally force use of some tool

        Returns:
            Message object with Claude's response
        """
        if temp is None:
            temp = self.temp

        # Handle history append
        prev_role = nested_idx(self.h, -1, 'role') if self.h else 'assistant'
        if pr and prev_role == 'user':
            # Already have a user request pending, run it first
            await self._call_impl(temp=temp, maxtok=maxtok, maxthinktok=maxthinktok,
                                  stream=stream, prefill=prefill, tool_choice=tool_choice, **kw)
        self._post_pr(pr, prev_role)

        return await self._call_impl(
            temp=temp,
            maxtok=maxtok,
            maxthinktok=maxthinktok,
            stream=stream,
            prefill=prefill,
            tool_choice=tool_choice,
            **kw
        )

    async def toolloop(
        self,
        pr: Any,
        max_steps: int = 10,
        cont_func: Callable = lambda *args: True,
        final_prompt: str = "You have no more tool uses. Please summarize your findings.",
        **kwargs
    ):
        """
        Add prompt and get response, automatically following up with tool_use messages.

        Note: With the Claude Agent SDK, tool execution is handled automatically
        by ClaudeSDKClient. This method provides compatibility with Claudette's API
        but the SDK manages the tool loop internally.

        Args:
            pr: Prompt to pass to Claude
            max_steps: Maximum number of tool requests to loop through
            cont_func: Function that stops loop if returns False
            final_prompt: Prompt to add if last message is a tool call

        Returns:
            List of responses

        Example:
            >>> results = await chat.toolloop("Research Python async programming")
            >>> for result in results:
            ...     print(contents(result))
        """
        results = []
        init_n = len(self.h)

        # With SDK, the tool loop is handled internally
        # We set max_turns to allow multiple tool calls
        kwargs['max_turns'] = max_steps

        r = await self(pr, **kwargs)
        results.append(r)

        if len(self.last) > 1:
            results.append(self.last[1])

        # The SDK handles additional tool calls internally
        # But we can still check for tool_use stop reason for compatibility
        for i in range(max_steps - 1):
            if self.c.stop_reason != 'tool_use':
                break

            prompt = final_prompt if i == max_steps - 2 else None
            r = await self(prompt, **kwargs)
            results.append(r)

            if len(self.last) > 1:
                results.append(self.last[1])

            if not cont_func(*self.h[-3:]):
                break

        return results

    async def stream(
        self,
        pr: Any,
        temp: Optional[float] = None,
        maxtok: int = 4096,
        **kwargs
    ):
        """
        Get a response from Claude, yielding message blocks as they arrive.

        IMPORTANT: The Claude Agent SDK streams complete message blocks, not
        individual text characters like the Anthropic API. Each yield is a
        complete text block from a message.

        Args:
            pr: Prompt / message
            temp: Temperature
            maxtok: Maximum tokens
            **kwargs: Additional options

        Yields:
            Text content from each message block as it arrives

        Example:
            >>> async for text in chat.stream("Tell me a story"):
            ...     print(text)  # Each 'text' is a complete message block
        """
        if temp is None:
            temp = self.temp

        # Add prompt to history
        self._append_pr(pr)

        # Build the conversation context
        conversation_text = self._build_conversation_prompt()

        # Build options
        options = self._build_options(**kwargs)

        collected_text = []
        total_usage = usage()

        # Use appropriate method based on whether we have tools
        if self._mcp_server:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(conversation_text)
                async for msg in client.receive_response():
                    # Check for ResultMessage which has usage and total_cost_usd
                    if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                        # Extract usage from ResultMessage (this is where usage actually is!)
                        if hasattr(msg, 'usage') and msg.usage:
                            total_usage = _parse_usage(msg.usage)
                        if hasattr(msg, 'total_cost_usd'):
                            self.c._last_cost_usd = msg.total_cost_usd
                        continue

                    # Process AssistantMessage for content only (usage is on ResultMessage)
                    if SDKAssistantMessage is not None and isinstance(msg, SDKAssistantMessage):
                        if hasattr(msg, 'content'):
                            for block in msg.content:
                                if hasattr(block, 'text'):
                                    collected_text.append(block.text)
                                    yield block.text
                    elif hasattr(msg, 'content'):
                        for block in msg.content:
                            if hasattr(block, 'text'):
                                collected_text.append(block.text)
                                yield block.text
        else:
            async for msg in sdk_query(prompt=conversation_text, options=options):
                # Check for ResultMessage which has usage and total_cost_usd
                if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                    # Extract usage from ResultMessage (this is where usage actually is!)
                    if hasattr(msg, 'usage') and msg.usage:
                        total_usage = _parse_usage(msg.usage)
                    if hasattr(msg, 'total_cost_usd'):
                        self.c._last_cost_usd = msg.total_cost_usd
                    continue

                # Process AssistantMessage for content only (usage is on ResultMessage)
                if SDKAssistantMessage is not None and isinstance(msg, SDKAssistantMessage):
                    if hasattr(msg, 'content'):
                        for block in msg.content:
                            if hasattr(block, 'text'):
                                collected_text.append(block.text)
                                yield block.text
                elif hasattr(msg, 'content'):
                    for block in msg.content:
                        if hasattr(block, 'text'):
                            collected_text.append(block.text)
                            yield block.text

        # Update usage on client
        if total_usage.total > 0:
            self.c.use = self.c.use + total_usage

        # Update history with the full response
        full_response = "".join(collected_text)
        self.h.append(mk_msg(full_response, role="assistant"))

    def _repr_markdown_(self) -> str:
        """Jupyter-friendly representation."""
        if not hasattr(self.c, 'result') or self.c.result is None:
            return 'No results yet'

        last_msg = contents(self.c.result)

        def fmt_msg(m):
            t = contents(m) if isinstance(m, Message) else m
            if isinstance(t, dict):
                return t.get('content', str(t))
            return str(t)

        history = '\n\n'.join(
            f"**{m.get('role', 'unknown')}**: {fmt_msg(m)}"
            for m in self.h
        )

        det = self.c._repr_markdown_().split('\n\n')[-1]

        if history:
            history = f"""
<details>
<summary>History</summary>

{history}

</details>
"""

        return f"""{last_msg}
{history}
{det}"""


class AsyncChat(Chat):
    """
    Async version of Chat.

    The base Chat is already async-native, but this provides explicit
    async naming and uses AsyncClient.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        cli: Optional[Client] = None,
        **kwargs
    ):
        """Initialize the AsyncChat."""
        super().__init__(model, cli, **kwargs)
        if not cli:
            self.c = AsyncClient(model or DEFAULT_MODEL, **{
                k: v for k, v in kwargs.items()
                if k in ('cache', 'cwd', 'allowed_tools', 'permission_mode')
            })

    async def _append_pr(self, pr: Any = None) -> None:
        """Append prompt to history (async version)."""
        prev_role = nested_idx(self.h, -1, 'role') if self.h else 'assistant'

        if pr and prev_role == 'user':
            await self._call_impl()

        self._post_pr(pr, prev_role)

    async def __call__(
        self,
        pr: Any = None,
        temp: Optional[float] = None,
        maxtok: int = 4096,
        maxthinktok: int = 0,
        stream: bool = False,
        prefill: str = '',
        tool_choice: Optional[Union[str, bool, Dict]] = None,
        **kw
    ) -> Message:
        """Send a message and get a response (async)."""
        if temp is None:
            temp = self.temp

        await self._append_pr(pr)

        return await self._call_impl(
            temp=temp,
            maxtok=maxtok,
            maxthinktok=maxthinktok,
            stream=stream,
            prefill=prefill,
            tool_choice=tool_choice,
            **kw
        )

    async def toolloop(
        self,
        pr: Any,
        max_steps: int = 10,
        cont_func: Callable = lambda *args: True,
        final_prompt: str = "You have no more tool uses. Please summarize your findings.",
        **kwargs
    ):
        """
        Add prompt and get response, automatically following up with tool_use messages (async).

        Args:
            pr: Prompt to pass to Claude
            max_steps: Maximum number of tool requests to loop through
            cont_func: Function that stops loop if returns False
            final_prompt: Prompt to add if last message is a tool call

        Returns:
            List of response messages
        """
        results = []
        init_n = len(self.h)

        # With SDK, set max_turns for tool handling
        kwargs['max_turns'] = max_steps

        r = await self(pr, **kwargs)
        results.append(r)

        if len(self.last) > 1:
            results.append(self.last[1])

        for i in range(max_steps - 1):
            if self.c.stop_reason != 'tool_use':
                break

            prompt = final_prompt if i == max_steps - 2 else None
            r = await self(prompt, **kwargs)
            results.append(r)

            if len(self.last) > 1:
                results.append(self.last[1])

            if not cont_func(*self.h[-3:]):
                break

        return results
