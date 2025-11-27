"""
Structured outputs module - Pydantic model support for Claude responses.
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
        # Use the class docstring if no description in schema
        description = cls.__doc__ or f"Schema for {name}"

    return {
        "name": name,
        "description": description,
        "input_schema": _filter_title(schema)
    }


def _escape_json_newlines(json_string: str) -> str:
    """Escape newlines within JSON string values."""
    def replace_newline(match):
        if match.group(1):  # Inside a string
            return match.group(1).replace('\n', '\\n')
        else:  # Outside a string
            return match.group(0)

    pattern = r'("(?:[^"\\]|\\.)*")|\n'
    return re.sub(pattern, replace_newline, json_string)


def _mk_struct(inp: Dict, resp_model: Type[T]) -> T:
    """
    Create a Pydantic model instance from input dict.

    Args:
        inp: Input dictionary
        resp_model: The Pydantic model class

    Returns:
        Instance of resp_model
    """
    try:
        return resp_model(**inp)
    except ValidationError:
        # Try parsing string values as JSON
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
    """
    Add structured output support to a Client class.

    This is a decorator that adds the `struct` method.
    """
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
        """Parse Claude output into a Pydantic model."""
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic is required for structured outputs")

        kwargs["tool_choice"] = mk_tool_choice(resp_model.__name__)
        kwargs["tools"] = [claude_schema(resp_model)]

        response = await self(msgs, **kwargs)
        inp = _extract_tool_input(response)
        return _mk_struct(inp, resp_model)

    client_cls.struct = struct
    return client_cls


def add_struct_to_chat(chat_cls):
    """
    Add structured output support to a Chat class.

    This is a decorator that adds the `struct` method.
    """
    async def struct(
        self,
        pr: Any,
        resp_model: Type[T],
        treat_as_output: bool = True,
        **kwargs
    ) -> T:
        """
        Parse Claude output into a Pydantic model.

        Args:
            pr: Prompt to send (required)
            resp_model: The Pydantic BaseModel class to parse into
            treat_as_output: Whether to treat result as output (adds to history)

        Returns:
            Instance of resp_model with parsed data

        Example:
            >>> person = await chat.struct("Extract: John is 25", Person)
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic is required for structured outputs")

        # Append prompt
        self._append_pr(pr)

        # Build tool choice
        kwargs["tool_choice"] = mk_tool_choice(resp_model.__name__)
        kwargs["tools"] = [claude_schema(resp_model)]

        # Make the call
        response = await self._call_impl(**kwargs)

        # Extract and parse result
        inp = _extract_tool_input(response)
        result = _mk_struct(inp, resp_model)

        # Update history
        if treat_as_output:
            msgs = [mk_msg(repr(result), "assistant")]
        else:
            # Include the tool use in history
            tool_block = find_block(response, ToolUseBlock)
            tool_id = getattr(tool_block, 'id', '') if tool_block else ''
            msgs = [
                mk_msg(response.model_dump(), "assistant"),
                mk_msg([{
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": repr(result)
                }], "user")
            ]

        self.h.extend(msgs)
        return result

    chat_cls.struct = struct
    return chat_cls


# Utility function for sync struct calls
def struct_sync(
    client,
    msgs: Union[str, List],
    resp_model: Type[T],
    **kwargs
) -> T:
    """
    Synchronous wrapper for structured output.

    Args:
        client: Client or Chat instance
        msgs: Messages or prompt to send
        resp_model: The Pydantic BaseModel class

    Returns:
        Instance of resp_model
    """
    return asyncio.get_event_loop().run_until_complete(
        client.struct(msgs, resp_model, **kwargs)
    )
