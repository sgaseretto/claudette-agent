"""
Chat module - Chat and AsyncChat classes with conversation history.

This module provides a Claudette-compatible API using the Claude Agent SDK.
Key differences from Claudette:
- Tools require MCP server registration (handled automatically)
- Uses ClaudeSDKClient for tool support, query() for simple prompts
"""
import asyncio
import json
import uuid
import inspect
from dataclasses import replace as dataclass_replace
from typing import Any, Dict, List, Optional, Union, Callable, get_type_hints, MutableMapping, Literal

from .core import (
    Usage, usage, Message, TextBlock, ToolUseBlock,
    contents, mk_msg, mk_msgs, mk_toolres, mk_toolres_async,
    get_schema, mk_tool_choice, listify, mk_ns, get_costs,
    model_types, pricing, DEFAULT_MODEL, ToolLoopResult,
    _parse_usage, _parse_sdk_message, _simple_text_message,
    AssistantMessage as SDKAssistantMessage,
    ResultMessage as SDKResultMessage,
    StreamEvent,
)
from .client import Client, AsyncClient

try:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        query as sdk_query,
        tool as sdk_tool,
        create_sdk_mcp_server,
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


def _convert_to_sdk_tool(func: Callable) -> Any:
    """
    Convert a regular Python function to an SDK tool.

    Also accepts pre-created SdkMcpTool instances (pass-through).
    """
    if not SDK_AVAILABLE:
        raise ImportError("claude-agent-sdk is required")

    # Pass through if already an SDK tool
    if hasattr(func, 'name') and hasattr(func, 'handler'):
        return func

    name = func.__name__
    doc = inspect.getdoc(func) or f"Function {name}"
    description = doc.split("\n")[0]

    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    sig = inspect.signature(func)

    params = {}
    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue
        param_type = hints.get(param_name, str)
        if param_type == int:
            params[param_name] = int
        elif param_type == float:
            params[param_name] = float
        elif param_type == bool:
            params[param_name] = bool
        else:
            params[param_name] = str

    @sdk_tool(name, description, params)
    async def sdk_wrapper(args):
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

    sdk_wrapper._original_func = func
    sdk_wrapper._tool_name = name

    return sdk_wrapper


class Chat:
    """
    Claude chat client with conversation history.

    Maintains a conversation history and supports tools, system prompts,
    and message streaming.

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
        if not SDK_AVAILABLE:
            raise ImportError(
                "claude-agent-sdk is not installed. "
                "Install it with: pip install claude-agent-sdk"
            )

        assert model or cli, "Must provide either model or cli"
        assert cont_pr != "", "cont_pr may not be an empty string"

        if cli is not None:
            self.c = cli
            if setting_sources is not None:
                self.c.setting_sources = setting_sources
            if env is not None:
                self.c.env.update(env)
            if extra_args is not None:
                self.c.extra_args.update(extra_args)
        else:
            self.c = Client(
                model or DEFAULT_MODEL,
                cache=cache,
                cwd=cwd,
                allowed_tools=allowed_tools,
                permission_mode=permission_mode,
                setting_sources=setting_sources if setting_sources is not None else [],
                env=env,
                extra_args=extra_args,
                max_turns=max_turns,
                max_budget_usd=max_budget_usd,
                fallback_model=fallback_model,
                can_use_tool=can_use_tool,
                hooks=hooks,
                agents=agents,
                enable_file_checkpointing=enable_file_checkpointing,
                thinking=thinking,
                effort=effort,
            )

        if hist is None:
            hist = []

        self.h = hist
        self.sp = sp
        self.cont_pr = cont_pr
        self.temp = temp
        self.cache = cache
        self.last: List[Dict] = []

        # Process tools
        self._original_tools = listify(tools) if tools else []
        self._sdk_tools = []
        self._mcp_server = None
        self._allowed_tools = allowed_tools or []

        if self._original_tools:
            self._setup_tools()

        if ns is None:
            ns = {t.__name__: t for t in self._original_tools if callable(t)} if self._original_tools else {}
        self.ns = ns

    def _setup_tools(self):
        """Set up tools as an MCP server for the SDK."""
        for func in self._original_tools:
            if callable(func):
                sdk_t = _convert_to_sdk_tool(func)
                self._sdk_tools.append(sdk_t)
                tool_name = f"mcp__tools__{getattr(func, '__name__', getattr(func, 'name', 'unknown'))}"
                if tool_name not in self._allowed_tools:
                    self._allowed_tools.append(tool_name)

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
        return self.c.use

    @property
    def cost(self) -> float:
        return self.c.cost

    @property
    def model(self) -> str:
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

    def _build_options(self, maxthinktok: int = 0, stream: bool = False, **kwargs) -> 'ClaudeAgentOptions':
        """Build ClaudeAgentOptions for the SDK call."""
        # System prompt: support string or dict (preset)
        if isinstance(self.sp, dict):
            system_prompt = self.sp
        else:
            system_prompt = self.sp or "You are a helpful assistant."

        opts = {
            'system_prompt': system_prompt,
            'setting_sources': self.c.setting_sources,
            'continue_conversation': False,
            'resume': None,
        }

        if kwargs.get('max_turns') or self.c.max_turns:
            opts['max_turns'] = kwargs.get('max_turns') or self.c.max_turns

        if self.c.cwd:
            opts['cwd'] = self.c.cwd

        if self._mcp_server:
            opts['mcp_servers'] = {"tools": self._mcp_server}

        if self._allowed_tools:
            opts['allowed_tools'] = self._allowed_tools

        # Environment variables
        if self.c.env:
            opts['env'] = opts.get('env', {})
            opts['env'].update(self.c.env)

        # Extended thinking via native SDK support
        if maxthinktok and maxthinktok > 0:
            if stream:
                raise ValueError(
                    "Streaming is incompatible with extended thinking in the Claude Agent SDK. "
                    "Use stream=False when using maxthinktok, or set maxthinktok=0 for streaming."
                )
            from .client import _has_option
            if _has_option('thinking'):
                opts['thinking'] = {"type": "enabled", "budget_tokens": maxthinktok}
            else:
                opts['max_thinking_tokens'] = maxthinktok
        elif self.c.thinking:
            from .client import _has_option
            if _has_option('thinking'):
                opts['thinking'] = self.c.thinking
            elif isinstance(self.c.thinking, dict) and self.c.thinking.get('type') == 'enabled':
                opts['max_thinking_tokens'] = self.c.thinking.get('budget_tokens', 0)

        # Effort level (if SDK supports it)
        if self.c.effort:
            from .client import _has_option
            if _has_option('effort'):
                opts['effort'] = self.c.effort

        # Streaming: enable partial messages
        if stream:
            opts['include_partial_messages'] = True

        # New SDK features from client
        if self.c.max_budget_usd is not None:
            opts['max_budget_usd'] = self.c.max_budget_usd
        if self.c.fallback_model:
            opts['fallback_model'] = self.c.fallback_model
        if self.c.can_use_tool:
            opts['can_use_tool'] = self.c.can_use_tool
        if self.c.hooks:
            opts['hooks'] = self.c.hooks
        if self.c.agents:
            opts['agents'] = self.c.agents
        if self.c.enable_file_checkpointing:
            opts['enable_file_checkpointing'] = True

        # Extra args
        if self.c.extra_args:
            opts['extra_args'] = opts.get('extra_args', {})
            opts['extra_args'].update(self.c.extra_args)

        return ClaudeAgentOptions(**opts)

    @staticmethod
    def _msg_has_images(msg: Dict) -> bool:
        """Check if a message dict contains image content blocks."""
        content = msg.get('content', [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'image':
                    return True
        return False

    def _has_images(self) -> bool:
        """Check if any message in conversation history contains images."""
        return any(self._msg_has_images(msg) for msg in self.h)

    async def _client_messages(self, options: 'ClaudeAgentOptions', last_msg: Dict):
        """Async generator that wraps ClaudeSDKClient lifecycle for image messages.

        Uses ClaudeSDKClient (not query()) because it keeps the transport connection
        open, which is needed for large base64 image payloads. query() closes stdin
        immediately after sending, which can cause failures with large payloads.
        """
        async with ClaudeSDKClient(options=options) as client:
            msg_dict = {"type": "user", "message": last_msg}
            await client._transport.write(json.dumps(msg_dict) + "\n")
            async for msg in client.receive_response():
                yield msg

    async def _call_with_images(self, options: 'ClaudeAgentOptions', prefill: str = '') -> Message:
        """Handle SDK calls when conversation contains image content blocks.

        Uses ClaudeSDKClient in streaming mode to pass structured content blocks
        (including images) since the default --print mode only accepts plain text.

        Prior conversation context (text-only messages) is included in the system prompt,
        and the last user message (with images) is sent as a structured message.
        """
        # Separate prior context from the last message
        prior_msgs = self.h[:-1]
        last_msg = self.h[-1]

        # Build prior context as text and prepend to system prompt
        if prior_msgs:
            prior_text = self._build_conversation_prompt_from(prior_msgs)
            current_sp = options.system_prompt or ""
            if isinstance(current_sp, str):
                enhanced_sp = f"{current_sp}\n\n[Previous conversation]\n{prior_text}\n[End of previous conversation]"
            else:
                # For preset system prompts, append context
                append = current_sp.get('append', '')
                enhanced_sp = {**current_sp, 'append': f"{append}\n\n[Previous conversation]\n{prior_text}\n[End of previous conversation]"}
            options = dataclass_replace(options, system_prompt=enhanced_sp)

        if prefill:
            # Add prefill instruction to the text content of the last message
            content = last_msg.get('content', [])
            if isinstance(content, list):
                content = content + [{"type": "text", "text": f"[Start your response with: {prefill}]"}]
                last_msg = {**last_msg, "content": content}

        collected_text = []
        final_message = None
        total_usage = usage()

        try:
            async for msg in self._client_messages(options, last_msg):
                if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                    if hasattr(msg, 'usage') and msg.usage:
                        total_usage = _parse_usage(msg.usage)
                    if hasattr(msg, 'total_cost_usd'):
                        self.c._last_cost_usd = msg.total_cost_usd
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

        return final_message

    def _build_conversation_prompt_from(self, msgs: List[Dict]) -> str:
        """Build a text conversation prompt from a list of message dicts."""
        parts = []
        for msg in msgs:
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
                    if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                        if hasattr(msg, 'usage') and msg.usage:
                            total_usage = _parse_usage(msg.usage)
                        if hasattr(msg, 'total_cost_usd'):
                            self.c._last_cost_usd = msg.total_cost_usd
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
                if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                    if hasattr(msg, 'usage') and msg.usage:
                        total_usage = _parse_usage(msg.usage)
                    if hasattr(msg, 'total_cost_usd'):
                        self.c._last_cost_usd = msg.total_cost_usd
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

        options = self._build_options(maxthinktok=maxthinktok, stream=stream, **kw)

        # Use streaming mode for image content (plain text mode drops image blocks)
        if self._has_images():
            final_message = await self._call_with_images(options, prefill=prefill)
        else:
            conversation_text = self._build_conversation_prompt()

            if prefill:
                conversation_text = f"{conversation_text}\n\n[Start your response with: {prefill}]"

            # Use ClaudeSDKClient if we have tools, otherwise use query()
            if self._mcp_server:
                final_message = await self._call_with_tools(conversation_text, options, **kw)
            else:
                final_message = await self._call_simple(conversation_text, options, **kw)

        self.c._r(final_message, prefill)

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
    ):
        """
        Send a message and get a response.

        Args:
            pr: Prompt / message
            temp: Temperature
            maxtok: Maximum tokens
            maxthinktok: Maximum thinking tokens
            stream: Stream response (yields text chunks via StreamEvent)
            prefill: Optional prefill to pass to Claude as start of its response
            tool_choice: Optionally force use of some tool

        Returns:
            Message object (or StreamingResponse if stream=True)
        """
        if temp is None:
            temp = self.temp

        prev_role = nested_idx(self.h, -1, 'role') if self.h else 'assistant'
        if pr and prev_role == 'user':
            await self._call_impl(temp=temp, maxtok=maxtok, maxthinktok=maxthinktok,
                                  stream=False, prefill=prefill, tool_choice=tool_choice, **kw)
        self._post_pr(pr, prev_role)

        # For streaming, build options and return StreamingResponse
        if stream:
            from .streaming import StreamingResponse
            conversation_text = self._build_conversation_prompt()
            if prefill:
                conversation_text = f"{conversation_text}\n\n[Start your response with: {prefill}]"
            options = self._build_options(maxthinktok=maxthinktok, stream=True, **kw)

            if self._mcp_server:
                # Streaming with tools via ClaudeSDKClient
                async def _stream_with_tools():
                    async with ClaudeSDKClient(options=options) as client:
                        await client.query(conversation_text)
                        async for msg in client.receive_response():
                            yield msg
                async_iter = _stream_with_tools()
            else:
                async_iter = sdk_query(prompt=conversation_text, options=options)

            def _on_stream_done(final_msg):
                """Callback to update history after streaming completes."""
                self.c._r(final_msg, prefill='')
                self.h.append(mk_msg(contents(final_msg), role="assistant"))

            return StreamingResponse(
                async_iter=async_iter,
                prefill='',
                callback=_on_stream_done
            )

        return await self._call_impl(
            temp=temp,
            maxtok=maxtok,
            maxthinktok=maxthinktok,
            stream=False,
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
    ) -> ToolLoopResult:
        """
        Add prompt and get response, automatically following up with tool_use messages.

        Args:
            pr: Prompt to pass to Claude
            max_steps: Maximum number of tool requests to loop through
            cont_func: Function that stops loop if returns False
            final_prompt: Prompt to add if last message is a tool call

        Returns:
            ToolLoopResult with iterable messages and .value for final result
        """
        results = ToolLoopResult([])
        init_n = len(self.h)

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

    async def stream(
        self,
        pr: Any,
        temp: Optional[float] = None,
        maxtok: int = 4096,
        maxthinktok: int = 0,
        **kwargs
    ):
        """
        Get a streaming response from Claude, yielding text chunks as they arrive.

        Uses include_partial_messages=True for character-level streaming via StreamEvent.

        Args:
            pr: Prompt / message
            temp: Temperature
            maxtok: Maximum tokens
            maxthinktok: Maximum thinking tokens

        Yields:
            Text content chunks as they arrive (character-level with StreamEvent)
        """
        if temp is None:
            temp = self.temp

        self._append_pr(pr)

        conversation_text = self._build_conversation_prompt()

        options = self._build_options(maxthinktok=maxthinktok, stream=True, **kwargs)

        collected_text = []
        total_usage = usage()

        if self._mcp_server:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(conversation_text)
                async for msg in client.receive_response():
                    # StreamEvent for char-by-char streaming
                    if StreamEvent is not None and isinstance(msg, StreamEvent):
                        event = msg.event
                        if event.get('type') == 'content_block_delta':
                            delta = event.get('delta', {})
                            if delta.get('type') == 'text_delta':
                                text = delta.get('text', '')
                                if text:
                                    collected_text.append(text)
                                    yield text
                        continue

                    if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                        if hasattr(msg, 'usage') and msg.usage:
                            total_usage = _parse_usage(msg.usage)
                        if hasattr(msg, 'total_cost_usd'):
                            self.c._last_cost_usd = msg.total_cost_usd
                        continue

                    # Fallback: complete AssistantMessage blocks
                    if SDKAssistantMessage is not None and isinstance(msg, SDKAssistantMessage):
                        if hasattr(msg, 'content') and not collected_text:
                            for block in msg.content:
                                if hasattr(block, 'text'):
                                    collected_text.append(block.text)
                                    yield block.text
                    elif hasattr(msg, 'content') and not collected_text:
                        for block in msg.content:
                            if hasattr(block, 'text'):
                                collected_text.append(block.text)
                                yield block.text
        else:
            async for msg in sdk_query(prompt=conversation_text, options=options):
                # StreamEvent for char-by-char streaming
                if StreamEvent is not None and isinstance(msg, StreamEvent):
                    event = msg.event
                    if event.get('type') == 'content_block_delta':
                        delta = event.get('delta', {})
                        if delta.get('type') == 'text_delta':
                            text = delta.get('text', '')
                            if text:
                                collected_text.append(text)
                                yield text
                    continue

                if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                    if hasattr(msg, 'usage') and msg.usage:
                        total_usage = _parse_usage(msg.usage)
                    if hasattr(msg, 'total_cost_usd'):
                        self.c._last_cost_usd = msg.total_cost_usd
                    continue

                # Fallback: complete AssistantMessage blocks
                if SDKAssistantMessage is not None and isinstance(msg, SDKAssistantMessage):
                    if hasattr(msg, 'content') and not collected_text:
                        for block in msg.content:
                            if hasattr(block, 'text'):
                                collected_text.append(block.text)
                                yield block.text
                elif hasattr(msg, 'content') and not collected_text:
                    for block in msg.content:
                        if hasattr(block, 'text'):
                            collected_text.append(block.text)
                            yield block.text

        if total_usage.total > 0:
            self.c.use = self.c.use + total_usage

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
        setting_sources: List[str] = None,
        **kwargs
    ):
        super().__init__(model, cli, setting_sources=setting_sources, **kwargs)
        if not cli:
            # Build kwargs for AsyncClient
            client_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ('cache', 'cwd', 'allowed_tools', 'permission_mode', 'env', 'extra_args',
                         'max_turns', 'max_budget_usd', 'fallback_model', 'can_use_tool',
                         'hooks', 'agents', 'enable_file_checkpointing', 'thinking', 'effort')
            }
            self.c = AsyncClient(
                model or DEFAULT_MODEL,
                setting_sources=setting_sources if setting_sources is not None else [],
                **client_kwargs
            )

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
    ) -> ToolLoopResult:
        """
        Add prompt and get response, automatically following up with tool_use messages (async).

        Returns:
            ToolLoopResult with iterable messages and .value for final result
        """
        results = ToolLoopResult([])
        init_n = len(self.h)

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
