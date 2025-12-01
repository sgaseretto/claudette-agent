"""
Streaming module - Support for streaming responses from Claude.
"""
import asyncio
from typing import Any, AsyncIterator, Iterator, Callable, Optional, List, Union

from .core import Message, TextBlock, usage, contents


class StreamingResponse:
    """
    A streaming response that yields text chunks as they arrive.

    This class wraps the async iterator from the SDK and provides
    both sync and async interfaces.

    Example:
        >>> async for chunk in client.stream("Tell me a story"):
        ...     print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        async_iter: AsyncIterator,
        prefill: str = "",
        callback: Optional[Callable] = None
    ):
        """
        Initialize the streaming response.

        Args:
            async_iter: The async iterator from the SDK
            prefill: Optional prefill text to prepend
            callback: Optional callback for when streaming completes
        """
        self._async_iter = async_iter
        self._prefill = prefill
        self._callback = callback
        self._collected_text: List[str] = []
        self._final_message: Optional[Message] = None
        self._started = False

    async def __aiter__(self) -> AsyncIterator[str]:
        """Async iteration over text chunks."""
        if self._prefill:
            self._collected_text.append(self._prefill)
            yield self._prefill

        async for item in self._async_iter:
            if hasattr(item, 'content'):
                for block in item.content:
                    if hasattr(block, 'text'):
                        text = block.text
                        self._collected_text.append(text)
                        yield text
            elif isinstance(item, str):
                self._collected_text.append(item)
                yield item

        # Store final message if available
        if hasattr(self._async_iter, 'value'):
            self._final_message = self._async_iter.value

        if self._callback:
            await self._callback(self.get_final_message())

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

    def get_final_message(self) -> Message:
        """Get the final accumulated message."""
        if self._final_message:
            return self._final_message

        return Message(
            role='assistant',
            content=[TextBlock(text=''.join(self._collected_text))],
            usage=usage()
        )

    @property
    def text(self) -> str:
        """Get all collected text."""
        return ''.join(self._collected_text)

    def __str__(self) -> str:
        return self.text


class StreamingMixin:
    """
    Mixin class that adds streaming support to Client/Chat classes.

    This mixin adds the `stream` method for getting streamed responses.
    """

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

        Args:
            msgs: Messages or prompt to send
            sp: System prompt
            temp: Temperature
            maxtok: Maximum tokens
            prefill: Prefill text

        Returns:
            StreamingResponse that can be iterated
        """
        try:
            from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
        except ImportError:
            raise ImportError("claude-agent-sdk is required for streaming")

        # Build prompt
        if isinstance(msgs, str):
            prompt = msgs
        else:
            prompt = self._build_prompt_from_msgs(msgs)

        # Build options
        options = ClaudeAgentOptions(
            system_prompt=sp or getattr(self, 'sp', '') or "You are a helpful assistant.",
            max_turns=kwargs.get('max_turns', 1),
        )

        # Get the async iterator
        async_iter = sdk_query(prompt=prompt, options=options)

        # Wrap in StreamingResponse
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

    Args:
        prompt: The prompt to send
        system_prompt: System prompt
        **kwargs: Additional options

    Yields:
        Text chunks as they arrive
    """
    try:
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
    except ImportError:
        raise ImportError("claude-agent-sdk is required for streaming")

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=kwargs.get('max_turns', 1),
    )

    async for msg in sdk_query(prompt=prompt, options=options):
        if hasattr(msg, 'content'):
            for block in msg.content:
                if hasattr(block, 'text'):
                    yield block.text


def stream_text_sync(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    **kwargs
) -> Iterator[str]:
    """
    Synchronous wrapper for streaming text.

    Args:
        prompt: The prompt to send
        system_prompt: System prompt
        **kwargs: Additional options

    Yields:
        Text chunks
    """
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
    A text stream that collects chunks and provides the final message.

    This mimics the claudette text stream behavior where you can iterate
    over chunks and then access the final message via `.value`.

    Example:
        >>> stream = TextStream(client.stream("Hello"))
        >>> for chunk in stream:
        ...     print(chunk, end="")
        >>> print(f"\\nFinal: {contents(stream.value)}")
    """

    def __init__(self, streaming_response: StreamingResponse):
        """Initialize with a streaming response."""
        self._response = streaming_response
        self.value: Optional[Message] = None

    def __iter__(self) -> Iterator[str]:
        """Iterate and collect the final value."""
        for chunk in self._response:
            yield chunk
        self.value = self._response.get_final_message()

    async def __aiter__(self) -> AsyncIterator[str]:
        """Async iterate and collect the final value."""
        async for chunk in self._response:
            yield chunk
        self.value = self._response.get_final_message()
