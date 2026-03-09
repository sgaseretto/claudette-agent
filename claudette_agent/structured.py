"""
Structured outputs module - Pydantic model support for Claude responses.

Supports two approaches:
1. Native SDK output_format (recommended) - uses ResultMessage.structured_output
2. Tool-call based (legacy claudette compat) - uses tool forcing + schema extraction
"""
import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

try:
    from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
    from claude_agent_sdk import ResultMessage as SDKResultMessage
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    SDKResultMessage = None

from .core import (
    Message, TextBlock, ToolUseBlock, contents, mk_msg, mk_tool_choice,
    find_block, usage
)

T = TypeVar('T', bound='BaseModel')


def _filter_title(obj: Any) -> Any:
    """Recursively remove 'title' keys from a schema dict."""
    if isinstance(obj, dict):
        return {k: _filter_title(v) for k, v in obj.items() if k != 'title'}
    elif isinstance(obj, list):
        return [_filter_title(item) for item in obj]
    else:
        return obj


def claude_schema(cls: Type[T]) -> Dict[str, Any]:
    """
    Create a Claude tool schema from a Pydantic model.

    Args:
        cls: The Pydantic BaseModel class

    Returns:
        Dict containing the tool schema for Claude
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("pydantic is required for structured outputs")

    schema = cls.model_json_schema()
    name = schema.pop('title')

    try:
        description = schema.pop('description')
    except KeyError:
        description = cls.__doc__ or f"Schema for {name}"

    return {
        "name": name,
        "description": description,
        "input_schema": _filter_title(schema)
    }


def _escape_json_newlines(json_string: str) -> str:
    """Escape newlines within JSON string values."""
    def replace_newline(match):
        if match.group(1):
            return match.group(1).replace('\n', '\\n')
        else:
            return match.group(0)

    pattern = r'("(?:[^"\\]|\\.)*")|\n'
    return re.sub(pattern, replace_newline, json_string)


def _mk_struct(inp: Dict, resp_model: Type[T]) -> T:
    """Create a Pydantic model instance from input dict."""
    try:
        return resp_model(**inp)
    except ValidationError:
        return resp_model(**{
            k: json.loads(_escape_json_newlines(v)) if isinstance(v, str) else v
            for k, v in inp.items()
        })


def _extract_tool_input(response: Message) -> Dict:
    """Extract tool input from a response message."""
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            return block.input
        elif hasattr(block, 'type') and block.type == 'tool_use':
            return getattr(block, 'input', {})
    return {}


# Patch Pydantic BaseModel with claude_schema class method
if PYDANTIC_AVAILABLE:
    @classmethod
    def _claude_schema_method(cls):
        return claude_schema(cls)

    BaseModel.claude_schema = _claude_schema_method


class StructuredMixin:
    """
    Mixin class that adds structured output support to Client/Chat classes.

    This mixin adds the `struct` method for getting responses as Pydantic models.
    """

    async def struct(
        self,
        msgs: Union[str, List],
        resp_model: Type[T],
        **kwargs
    ) -> T:
        """
        Parse Claude output into a Pydantic model.

        Args:
            msgs: Messages or prompt to send
            resp_model: The Pydantic BaseModel class to parse into

        Returns:
            Instance of resp_model with parsed data
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic is required for structured outputs")

        # Force tool choice to the model's name
        kwargs["tool_choice"] = mk_tool_choice(resp_model.__name__)
        kwargs["tools"] = [claude_schema(resp_model)]

        # Call the underlying method
        response = await self(msgs, **kwargs)

        # Extract tool input and create model instance
        inp = _extract_tool_input(response)
        return _mk_struct(inp, resp_model)


def add_struct_to_client(client_cls):
    """Add structured output support to a Client class (both native and legacy)."""
    original_init = client_cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

    client_cls.__init__ = new_init

    async def struct(
        self,
        msgs: Union[str, List],
        resp_model: Type[T],
        **kwargs
    ) -> T:
        """Parse Claude output into a Pydantic model (legacy tool-call approach)."""
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic is required for structured outputs")

        kwargs["tool_choice"] = mk_tool_choice(resp_model.__name__)
        kwargs["tools"] = [claude_schema(resp_model)]

        response = await self(msgs, **kwargs)
        inp = _extract_tool_input(response)
        return _mk_struct(inp, resp_model)

    async def struct_native(
        self,
        msgs: Union[str, List],
        resp_model: Type[T],
        **kwargs
    ) -> T:
        """
        Parse Claude output into a Pydantic model using native SDK output_format.

        This uses the SDK's built-in structured output validation via JSON Schema.
        The SDK validates the output and returns it in ResultMessage.structured_output.

        Args:
            msgs: Messages or prompt to send
            resp_model: The Pydantic BaseModel class to parse into

        Returns:
            Instance of resp_model with validated data
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic is required for structured outputs")
        if not SDK_AVAILABLE:
            raise ImportError("claude-agent-sdk is required for native structured outputs")

        # Build prompt
        if isinstance(msgs, str):
            prompt = msgs
        else:
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

        # Build options with output_format
        sp = kwargs.pop('sp', getattr(self, 'sp', '') or "You are a helpful assistant.")
        if isinstance(sp, dict):
            system_prompt = sp
        else:
            system_prompt = sp

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            max_turns=kwargs.pop('max_turns', 3),
            setting_sources=getattr(self, 'setting_sources', []),
            output_format={
                "type": "json_schema",
                "schema": resp_model.model_json_schema()
            }
        )

        if getattr(self, 'cwd', None):
            options.cwd = self.cwd

        structured_output = None

        async for msg in sdk_query(prompt=prompt, options=options):
            if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
                if hasattr(msg, 'structured_output') and msg.structured_output:
                    structured_output = msg.structured_output
                # Check for errors
                if hasattr(msg, 'subtype') and msg.subtype == 'error_max_structured_output_retries':
                    raise ValueError(
                        "Agent could not produce valid structured output after multiple attempts. "
                        "Try simplifying the schema or making fields optional."
                    )

        if structured_output is None:
            raise ValueError("No structured output received from Claude")

        return resp_model.model_validate(structured_output)

    client_cls.struct = struct
    client_cls.struct_native = struct_native
    return client_cls


def add_struct_to_chat(chat_cls):
    """Add structured output support to a Chat class (both native and legacy)."""

    async def struct(
        self,
        pr: Any,
        resp_model: Type[T],
        treat_as_output: bool = True,
        native: bool = False,
        **kwargs
    ) -> T:
        """
        Parse Claude output into a Pydantic model.

        Args:
            pr: Prompt to send (required)
            resp_model: The Pydantic BaseModel class to parse into
            treat_as_output: Whether to treat result as output (adds to history)
            native: If True, use SDK's native output_format instead of prompt engineering

        Returns:
            Instance of resp_model with parsed data
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic is required for structured outputs")

        if not SDK_AVAILABLE:
            raise ImportError("claude-agent-sdk is required for structured outputs")

        if native:
            return await _struct_native_chat(self, pr, resp_model, treat_as_output, **kwargs)
        else:
            return await _struct_prompt_chat(self, pr, resp_model, treat_as_output, **kwargs)

    chat_cls.struct = struct
    return chat_cls


async def _struct_native_chat(chat, pr, resp_model, treat_as_output, **kwargs):
    """Native SDK output_format approach for Chat.struct()."""
    # Append prompt to history
    chat._append_pr(pr)
    conversation_text = chat._build_conversation_prompt()

    # Build options with output_format
    sp = chat.sp or "You are a helpful assistant."
    if isinstance(sp, dict):
        system_prompt = sp
    else:
        system_prompt = sp

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        setting_sources=chat.c.setting_sources,
        output_format={
            "type": "json_schema",
            "schema": resp_model.model_json_schema()
        }
    )

    if kwargs.get('max_turns'):
        options.max_turns = kwargs['max_turns']

    if chat.c.cwd:
        options.cwd = chat.c.cwd

    structured_output = None

    async for msg in sdk_query(prompt=conversation_text, options=options):
        if SDKResultMessage is not None and isinstance(msg, SDKResultMessage):
            if hasattr(msg, 'structured_output') and msg.structured_output:
                structured_output = msg.structured_output
            if hasattr(msg, 'subtype') and msg.subtype == 'error_max_structured_output_retries':
                raise ValueError(
                    "Agent could not produce valid structured output after multiple attempts. "
                    "Try simplifying the schema or making fields optional."
                )

    if structured_output is None:
        raise ValueError("No structured output received from Claude")

    result = resp_model.model_validate(structured_output)

    # Update history
    if treat_as_output:
        chat.h.append(mk_msg(repr(result), "assistant"))
    else:
        chat.h.append(mk_msg(json.dumps(structured_output), "assistant"))

    return result


async def _struct_prompt_chat(chat, pr, resp_model, treat_as_output, **kwargs):
    """Prompt engineering approach for Chat.struct() (legacy)."""
    json_schema = resp_model.model_json_schema()
    schema_str = json.dumps(json_schema, indent=2)

    structured_prompt = f"""{pr}

Please respond with ONLY valid JSON that matches this schema:
{schema_str}

Important: Output ONLY the JSON object, no markdown code blocks, no explanations."""

    chat._append_pr(structured_prompt)
    conversation_text = chat._build_conversation_prompt()

    opts = {
        'system_prompt': chat.sp or "You are a helpful assistant that outputs valid JSON.",
    }

    if kwargs.get('max_turns'):
        opts['max_turns'] = kwargs['max_turns']

    if chat.c.cwd:
        opts['cwd'] = chat.c.cwd

    options = ClaudeAgentOptions(**opts)

    result_data = None
    last_text = None

    async for msg in sdk_query(prompt=conversation_text, options=options):
        if hasattr(msg, 'structured_output') and msg.structured_output:
            result_data = msg.structured_output

        if hasattr(msg, 'content'):
            for block in msg.content:
                if hasattr(block, 'text'):
                    last_text = block.text

    if result_data is None and last_text:
        text = last_text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

        try:
            result_data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from Claude's response: {e}\nResponse: {last_text[:500]}")

    if result_data is None:
        raise ValueError("No structured output received from Claude")

    result = resp_model.model_validate(result_data)

    # Fix history: replace structured prompt with original
    if chat.h:
        chat.h.pop()
    chat.h.append(mk_msg(pr, cache=chat.cache))

    if treat_as_output:
        chat.h.append(mk_msg(repr(result), "assistant"))
    else:
        chat.h.append(mk_msg(json.dumps(result_data), "assistant"))

    return result


def struct_sync(
    client,
    msgs: Union[str, List],
    resp_model: Type[T],
    **kwargs
) -> T:
    """Synchronous wrapper for structured output."""
    return asyncio.get_event_loop().run_until_complete(
        client.struct(msgs, resp_model, **kwargs)
    )
