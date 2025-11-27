"""
Chat module - Chat and AsyncChat classes with conversation history.
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union, Callable

from .core import (
    Usage, usage, Message, TextBlock, ToolUseBlock,
    contents, mk_msg, mk_msgs, mk_toolres, mk_toolres_async,
    get_schema, mk_tool_choice, listify, mk_ns, tool, get_costs,
    model_types, pricing, DEFAULT_MODEL
)
from .client import Client, AsyncClient

try:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        AssistantMessage as SDKAssistantMessage,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


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


class Chat:
    """
    Claude chat client with conversation history.

    Maintains a conversation history and supports tools, system prompts,
    and streaming responses.

    Example:
        >>> chat = Chat('claude-sonnet-4-5-20250929', sp="You are a helpful assistant")
        >>> response = await chat("Hello!")
        >>> print(contents(response))
        >>> response = await chat("What did I just say?")  # Remembers context
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

        if tools:
            tools = [tool(t) for t in listify(tools)]

        if ns is None:
            ns = tools

        self.h = hist  # Conversation history
        self.sp = sp  # System prompt
        self.tools = tools
        self.cont_pr = cont_pr
        self.temp = temp
        self.cache = cache
        self.ns = ns
        self.last: List[Dict] = []  # Last response messages
        self._sdk_client: Optional[ClaudeSDKClient] = None

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

        if pr and prev_role == 'user':
            # Already have a user request pending, run it first
            asyncio.get_event_loop().run_until_complete(self._call_impl())

        self._post_pr(pr, prev_role)

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
        options = ClaudeAgentOptions(
            system_prompt=self.sp or "You are a helpful assistant.",
            max_turns=kw.get('max_turns', 1),
        )

        if self.c.cwd:
            options.cwd = self.c.cwd

        if self.c._mcp_servers:
            options.mcp_servers = self.c._mcp_servers

        # Make the SDK call
        from claude_agent_sdk import query as sdk_query

        collected_text = []
        final_message = None

        try:
            async for msg in sdk_query(prompt=conversation_text, options=options):
                if hasattr(msg, 'content'):
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

        # Update client state
        self.c._r(final_message, prefill)

        # Create tool results and update history
        self.last = mk_toolres(final_message, ns=self.ns)
        self.h.extend(self.last)

        return final_message

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
            stream: Stream response?
            prefill: Optional prefill to pass to Claude as start of its response
            tool_choice: Optionally force use of some tool

        Returns:
            Message object with Claude's response
        """
        if temp is None:
            temp = self.temp

        # Handle history append (sync part)
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

    async def stream(
        self,
        pr: Any,
        temp: Optional[float] = None,
        maxtok: int = 4096,
        **kwargs
    ):
        """
        Stream a response from Claude.

        Args:
            pr: Prompt / message
            temp: Temperature
            maxtok: Maximum tokens
            **kwargs: Additional options

        Yields:
            Text chunks as they arrive

        Example:
            >>> async for chunk in chat.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions

        if temp is None:
            temp = self.temp

        # Add prompt to history
        self._append_pr(pr)

        # Build the conversation context
        conversation_text = self._build_conversation_prompt()

        # Build options
        options = ClaudeAgentOptions(
            system_prompt=self.sp or "You are a helpful assistant.",
            max_turns=kwargs.get('max_turns', 1),
        )

        collected_text = []

        async for msg in sdk_query(prompt=conversation_text, options=options):
            if hasattr(msg, 'content'):
                for block in msg.content:
                    if hasattr(block, 'text'):
                        collected_text.append(block.text)
                        yield block.text

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
