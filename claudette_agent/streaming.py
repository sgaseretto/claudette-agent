"""
Streaming module - Support for streaming responses from Claude.

Uses the SDK's include_partial_messages=True to receive StreamEvent objects
with text_delta for real character-by-character streaming.
"""
import asyncio
from typing import Any, AsyncIterator, Iterator, Callable, Optional, List, Union

from .core import (
    Message, TextBlock, usage, contents, _parse_usage,
    AssistantMessage as SDKAssistantMessage,
    ResultMessage as SDKResultMessage,
    StreamEvent,
)


class StreamingResponse:
    """
    A streaming response that yields text chunks as they arrive.

    When include_partial_messages=True is set in SDK options, the SDK yields
    StreamEvent objects containing raw Claude API events. This class processes
    those events to yield individual text chunks (character-level streaming).

    After iteration completes, access the final message via `.value`.

    Example:
        >>> stream = await client("Tell me a story", stream=True)
        >>> async for chunk in stream:
        ...     print(chunk, end="", flush=True)
        >>> final_msg = stream.value
    """

    def __init__(
        self,
        async_iter: AsyncIterator,
        prefill: str = "",
        callback: Optional[Callable] = None
    ):
        self._async_iter = async_iter
        self._prefill = prefill
        self._callback = callback
        self._collected_text: List[str] = []
        self._result_message = None  # SDK ResultMessage
        self._assistant_message = None  # SDK AssistantMessage
        self.value: Optional[Message] = None

    async def __aiter__(self) -> AsyncIterator[str]:
        """Async iteration over text chunks from StreamEvent deltas."""
        if self._prefill:
            self._collected_text.append(self._prefill)
            yield self._prefill

        async for item in self._async_iter:
            # Process StreamEvent for character-level streaming
            if StreamEvent is not None and isinstance(item, StreamEvent):
                event = item.event
                if event.get('type') == 'content_block_delta':
                    delta = event.get('delta', {})
                    if delta.get('type') == 'text_delta':
                        text = delta.get('text', '')
                        if text:
                            self._collected_text.append(text)
                            yield text

            # Capture final AssistantMessage (complete message with all content)
            elif SDKAssistantMessage is not None and isinstance(item, SDKAssistantMessage):
                self._assistant_message = item
                # If we didn't get StreamEvents (fallback), yield block text
                if not self._collected_text:
                    if hasattr(item, 'content'):
                        for block in item.content:
                            if hasattr(block, 'text'):
                                text = block.text
                                self._collected_text.append(text)
                                yield text

            # Capture ResultMessage for usage info
            elif SDKResultMessage is not None and isinstance(item, SDKResultMessage):
                self._result_message = item

        # Build final message
        self.value = self._build_final_message()

        if self._callback:
            if asyncio.iscoroutinefunction(self._callback):
                await self._callback(self.value)
            else:
                self._callback(self.value)

    def __iter__(self) -> Iterator[str]:
        """Sync iteration (uses event loop)."""
        loop = asyncio.get_event_loop()

        async def collect():
            chunks = []
            async for chunk in self:
                chunks.append(chunk)
            return chunks

        chunks = loop.run_until_complete(collect())
        yield from chunks

    def _build_final_message(self) -> Message:
        """Build the final Message from collected data."""
        from .core import _parse_sdk_message

        msg_usage = usage()

        # Extract usage from ResultMessage
        if self._result_message and hasattr(self._result_message, 'usage') and self._result_message.usage:
            msg_usage = _parse_usage(self._result_message.usage)

        # If we have a full AssistantMessage, parse it properly
        if self._assistant_message:
            result = _parse_sdk_message(self._assistant_message)
            if msg_usage.total > 0:
                result.usage = msg_usage
            return result

        # Fallback: build from collected text
        return Message(
            role='assistant',
            content=[TextBlock(text=''.join(self._collected_text))],
            usage=msg_usage
        )

    def get_final_message(self) -> Message:
        """Get the final accumulated message."""
        if self.value:
            return self.value
        return self._build_final_message()

    @property
    def text(self) -> str:
        """Get all collected text."""
        return ''.join(self._collected_text)

    def __str__(self) -> str:
        return self.text


class StreamingMixin:
    """Mixin class that adds streaming support to Client/Chat classes."""

    async def stream(
        self,
        msgs: Union[str, List],
        sp: str = '',
        temp: float = 0,
        maxtok: int = 4096,
        prefill: str = '',
        **kwargs
    ) -> StreamingResponse:
        """
        Stream a response from Claude.

        Returns:
            StreamingResponse that yields text chunks via async iteration.
            Access .value after iteration for the final Message.
        """
        try:
            from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
        except ImportError:
            raise ImportError("claude-agent-sdk is required for streaming")

        if isinstance(msgs, str):
            prompt = msgs
        else:
            prompt = self._build_prompt_from_msgs(msgs)

        # System prompt: support string or dict
        if isinstance(sp, dict):
            system_prompt = sp
        else:
            system_prompt = sp or getattr(self, 'sp', '') or "You are a helpful assistant."

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            max_turns=kwargs.get('max_turns', 1),
            include_partial_messages=True,
        )

        async_iter = sdk_query(prompt=prompt, options=options)

        return StreamingResponse(
            async_iter=async_iter,
            prefill=prefill,
            callback=kwargs.get('cb')
        )

    def _build_prompt_from_msgs(self, msgs: List) -> str:
        """Build a prompt string from messages."""
        parts = []
        for msg in msgs:
            if isinstance(msg, str):
                parts.append(msg)
            elif isinstance(msg, dict):
                content = msg.get('content', '')
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get('type') == 'text':
                            parts.append(c.get('text', ''))
                elif isinstance(content, str):
                    parts.append(content)
        return "\n\n".join(parts) if parts else ""


async def stream_text(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    **kwargs
) -> AsyncIterator[str]:
    """
    Simple function to stream text from Claude.

    Yields:
        Text chunks as they arrive (character-level when SDK supports it)
    """
    try:
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
    except ImportError:
        raise ImportError("claude-agent-sdk is required for streaming")

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=kwargs.get('max_turns', 1),
        include_partial_messages=True,
    )

    got_stream_events = False
    async for msg in sdk_query(prompt=prompt, options=options):
        if StreamEvent is not None and isinstance(msg, StreamEvent):
            event = msg.event
            if event.get('type') == 'content_block_delta':
                delta = event.get('delta', {})
                if delta.get('type') == 'text_delta':
                    text = delta.get('text', '')
                    if text:
                        got_stream_events = True
                        yield text
        elif SDKAssistantMessage is not None and isinstance(msg, SDKAssistantMessage):
            # Only yield from AssistantMessage if we didn't get StreamEvents (fallback)
            if not got_stream_events and hasattr(msg, 'content'):
                for block in msg.content:
                    if hasattr(block, 'text'):
                        yield block.text


def stream_text_sync(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    **kwargs
) -> Iterator[str]:
    """Synchronous wrapper for streaming text."""
    loop = asyncio.get_event_loop()

    async def collect():
        chunks = []
        async for chunk in stream_text(prompt, system_prompt, **kwargs):
            chunks.append(chunk)
        return chunks

    chunks = loop.run_until_complete(collect())
    yield from chunks


class TextStream:
    """
    A text stream that collects chunks and provides the final message via .value.

    This mimics the claudette text stream behavior.

    Example:
        >>> stream = TextStream(client.stream("Hello"))
        >>> for chunk in stream:
        ...     print(chunk, end="")
        >>> print(f"\\nFinal: {contents(stream.value)}")
    """

    def __init__(self, streaming_response: StreamingResponse):
        self._response = streaming_response
        self.value: Optional[Message] = None

    def __iter__(self) -> Iterator[str]:
        for chunk in self._response:
            yield chunk
        self.value = self._response.get_final_message()

    async def __aiter__(self) -> AsyncIterator[str]:
        async for chunk in self._response:
            yield chunk
        self.value = self._response.get_final_message()
