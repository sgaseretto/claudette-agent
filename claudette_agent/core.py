"""
Core module for claudette_agent - Claude Agent SDK wrapper with Claudette-compatible API.
"""
import asyncio
import inspect
from collections import abc
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Union, Callable, AsyncIterator, Iterator,
    get_type_hints, TypeVar
)
from functools import wraps

try:
    from claude_agent_sdk import (
        query as sdk_query,
        ClaudeSDKClient,
        ClaudeAgentOptions,
        tool as sdk_tool,
        create_sdk_mcp_server,
        AssistantMessage,
        UserMessage,
        TextBlock as SDKTextBlock,
        ToolUseBlock as SDKToolUseBlock,
        ToolResultBlock as SDKToolResultBlock,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# Type variable for generic operations
T = TypeVar('T')

# Default models
DEFAULT_MODEL = 'claude-sonnet-4-5-20250929'
OPUS_MODEL = 'claude-opus-4-5-20251101'
HAIKU_MODEL = 'claude-haiku-4-5'

# Model mapping for compatibility
model_types = {
    'claude-opus-4-5-20251101': 'opus',
    'claude-sonnet-4-5-20250929': 'sonnet',
    'claude-haiku-4-5': 'haiku',
    'claude-opus-4-1-20250805': 'opus',
    'claude-sonnet-4-5': 'sonnet',
    'claude-haiku-4-5': 'haiku',
    'claude-opus-4-20250514': 'opus-4',
    'claude-3-opus-20240229': 'opus-3',
    'claude-sonnet-4-20250514': 'sonnet-4',
    'claude-3-7-sonnet-20250219': 'sonnet-3-7',
    'claude-3-5-sonnet-20241022': 'sonnet-3-5',
    'claude-3-haiku-20240307': 'haiku-3',
    'claude-3-5-haiku-20241022': 'haiku-3-5',
}

all_models = list(model_types.keys())
models = all_models[:6]

# Pricing per million tokens (input, output, cache write, cache read)
pricing = {
    'opus': (15, 75, 18.75, 1.5),
    'sonnet': (3, 15, 3.75, 0.3),
    'haiku': (1, 5, 1.25, 0.1),
    'haiku-3': (0.25, 1.25, 0.3, 0.03),
    'haiku-3-5': (1, 3, 1.25, 0.1),
    'opus-3': (15, 75, 18.75, 1.5),
    'opus-4': (15, 75, 18.75, 1.5),
    'sonnet-4': (3, 15, 3.75, 0.3),
    'sonnet-3-5': (3, 15, 3.75, 0.3),
    'sonnet-3-7': (3, 15, 3.75, 0.3),
}

empty = inspect.Parameter.empty


@dataclass
class Usage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        return (self.input_tokens + self.output_tokens +
                self.cache_creation_input_tokens + self.cache_read_input_tokens)

    def __add__(self, other: 'Usage') -> 'Usage':
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    def __repr__(self) -> str:
        return (f'In: {self.input_tokens}; Out: {self.output_tokens}; '
                f'Cache create: {self.cache_creation_input_tokens}; '
                f'Cache read: {self.cache_read_input_tokens}; '
                f'Total: {self.total}')

    def cost(self, costs: tuple) -> float:
        """Calculate cost based on pricing tuple (input, output, cache_write, cache_read)."""
        return sum([
            self.input_tokens * costs[0],
            self.output_tokens * costs[1],
            self.cache_creation_input_tokens * costs[2],
            self.cache_read_input_tokens * costs[3]
        ]) / 1e6


def usage(inp: int = 0, out: int = 0, cache_create: int = 0, cache_read: int = 0) -> Usage:
    """Create a Usage object."""
    return Usage(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read
    )


@dataclass
class TextBlock:
    """Text content block."""
    type: str = "text"
    text: str = ""

    def __repr__(self) -> str:
        return f"TextBlock(text='{self.text[:50]}...')" if len(self.text) > 50 else f"TextBlock(text='{self.text}')"


@dataclass
class ToolUseBlock:
    """Tool use block."""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    """Tool result block."""
    type: str = "tool_result"
    tool_use_id: str = ""
    content: Any = ""


@dataclass
class ThinkingBlock:
    """Thinking/reasoning block."""
    type: str = "thinking"
    thinking: str = ""


@dataclass
class Message:
    """Response message from Claude."""
    id: str = ""
    type: str = "message"
    role: str = "assistant"
    content: List[Any] = field(default_factory=list)
    model: str = ""
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Optional[Usage] = None

    def model_dump(self) -> Dict[str, Any]:
        """Return dict representation."""
        return {
            'id': self.id,
            'type': self.type,
            'role': self.role,
            'content': [c.__dict__ if hasattr(c, '__dict__') else c for c in self.content],
            'model': self.model,
            'stop_reason': self.stop_reason,
            'stop_sequence': self.stop_sequence,
            'usage': self.usage.__dict__ if self.usage else None
        }


def find_block(r: Message, blk_type: type = TextBlock) -> Optional[Any]:
    """Find the first block of type `blk_type` in `r.content`."""
    for block in getattr(r, 'content', []):
        if isinstance(block, blk_type):
            return block
        if isinstance(blk_type, str) and getattr(block, 'type', None) == blk_type:
            return block
    return None


def find_blocks(r: Message, blk_type: type = TextBlock) -> List[Any]:
    """Find all blocks of type `blk_type` in `r.content`."""
    blocks = []
    for block in getattr(r, 'content', []):
        if isinstance(block, blk_type):
            blocks.append(block)
        elif isinstance(blk_type, str) and getattr(block, 'type', None) == blk_type:
            blocks.append(block)
    return blocks


def contents(r: Message, show_thk: bool = True) -> str:
    """Extract text content from a response message."""
    if isinstance(r, str):
        return r

    content_parts = []
    thinking_part = None

    for block in getattr(r, 'content', []):
        if isinstance(block, TextBlock):
            content_parts.append(block.text)
        elif isinstance(block, ThinkingBlock) and show_thk:
            thinking_part = block.thinking
        elif hasattr(block, 'text'):
            content_parts.append(block.text)
        elif hasattr(block, 'thinking') and show_thk:
            thinking_part = block.thinking
        elif isinstance(block, dict):
            if block.get('type') == 'text':
                content_parts.append(block.get('text', ''))
            elif block.get('type') == 'thinking' and show_thk:
                thinking_part = block.get('thinking', '')

    text = ''.join(content_parts)

    if thinking_part:
        return f"{text}\n\n<details>\n<summary>Thinking</summary>\n{thinking_part}\n</details>"

    return text


def mk_msg(content: Union[str, Dict, List, Message], role: str = "user", cache: bool = False) -> Dict[str, Any]:
    """Create a message dict for the API."""
    if isinstance(content, dict):
        return content
    if isinstance(content, Message):
        return content.model_dump()
    if isinstance(content, str):
        msg_content = [{"type": "text", "text": content}]
        if cache:
            msg_content[-1]["cache_control"] = {"type": "ephemeral"}
        return {"role": role, "content": msg_content}
    if isinstance(content, list):
        return {"role": role, "content": content}
    return {"role": role, "content": [{"type": "text", "text": str(content)}]}


def mk_msgs(msgs: List[Any], cache: bool = False, cache_last_ckpt_only: bool = False) -> List[Dict]:
    """Convert list of messages to API format."""
    result = []
    for i, msg in enumerate(msgs):
        if isinstance(msg, dict):
            result.append(msg)
        elif isinstance(msg, str):
            # Alternate between user and assistant
            role = "assistant" if result and result[-1].get("role") == "user" else "user"
            do_cache = cache and (not cache_last_ckpt_only or i == len(msgs) - 1)
            result.append(mk_msg(msg, role=role, cache=do_cache))
        else:
            result.append(mk_msg(msg, cache=cache))
    return result


def _is_builtin(tp: type) -> bool:
    """Returns True for built-in primitive types or containers."""
    return (tp in (str, int, float, bool, complex) or tp is None
            or getattr(tp, '__origin__', None) is not None)


def _convert(val: Dict, tp: type) -> Any:
    """Convert a dictionary argument to the proper type."""
    if val is None or _is_builtin(tp) or not isinstance(val, dict):
        return val
    return tp(**val)


def tool(func: Callable) -> Callable:
    """Decorator that converts dict arguments to proper types based on type hints."""
    if isinstance(func, dict):
        return func  # It's a schema, don't change

    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        new_args = [_convert(arg, hints.get(p, type(arg))) for p, arg in zip(sig.parameters, args)]
        new_kwargs = {k: _convert(v, hints.get(k, type(v))) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapper


def get_schema(func: Callable) -> Dict[str, Any]:
    """Generate a tool schema from a function."""
    if isinstance(func, dict):
        return func

    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    doc = inspect.getdoc(func) or ""

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ('self', 'cls'):
            continue

        param_type = hints.get(name, str)
        json_type = _python_to_json_type(param_type)

        prop = {"type": json_type}

        # Extract parameter description from docstring if available
        if f":param {name}:" in doc:
            desc_start = doc.index(f":param {name}:") + len(f":param {name}:")
            desc_end = doc.find("\n:", desc_start)
            if desc_end == -1:
                desc_end = len(doc)
            prop["description"] = doc[desc_start:desc_end].strip()

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": doc.split("\n")[0] if doc else f"Function {func.__name__}",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


def _python_to_json_type(py_type: type) -> str:
    """Convert Python type to JSON schema type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    origin = getattr(py_type, '__origin__', None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    if origin is Union:
        return "string"  # Simplification

    return type_map.get(py_type, "string")


def mk_tool_choice(choose: Union[str, bool, None]) -> Dict[str, Any]:
    """Create a tool_choice dict."""
    if isinstance(choose, str):
        return {"type": "tool", "name": choose}
    elif choose:
        return {"type": "any"}
    else:
        return {"type": "auto"}


def listify(o: Any) -> List:
    """Convert to list if not already."""
    if o is None:
        return []
    if isinstance(o, (list, tuple)):
        return list(o)
    return [o]


def mk_ns(*tools: Callable) -> Dict[str, Callable]:
    """Create a namespace dict from tools."""
    ns = {}
    for t in tools:
        if callable(t):
            ns[t.__name__] = t
        elif isinstance(t, dict) and 'name' in t:
            ns[t['name']] = t
    return ns


def call_func(name: str, args: Dict, ns: Dict[str, Callable] = None, raise_on_err: bool = True) -> Any:
    """Call a function by name with given arguments."""
    if ns is None:
        ns = {}

    func = ns.get(name)
    if func is None:
        if raise_on_err:
            raise ValueError(f"Function '{name}' not found in namespace")
        return f"Error: Function '{name}' not found"

    try:
        return func(**args)
    except Exception as e:
        if raise_on_err:
            raise
        return f"Error calling {name}: {e}"


async def call_func_async(name: str, args: Dict, ns: Dict[str, Callable] = None, raise_on_err: bool = True) -> Any:
    """Call a function by name with given arguments (async version)."""
    if ns is None:
        ns = {}

    func = ns.get(name)
    if func is None:
        if raise_on_err:
            raise ValueError(f"Function '{name}' not found in namespace")
        return f"Error: Function '{name}' not found"

    try:
        if asyncio.iscoroutinefunction(func):
            return await func(**args)
        return func(**args)
    except Exception as e:
        if raise_on_err:
            raise
        return f"Error calling {name}: {e}"


class ToolResult:
    """Wrapper for tool results with type information."""
    def __init__(self, result_type: str, data: Any):
        self.result_type = result_type
        self.data = data

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return f"ToolResult(result_type='{self.result_type}', data={self.data!r})"


def mk_funcres(fc: ToolUseBlock, ns: Dict[str, Callable]) -> Dict[str, Any]:
    """Given tool use block, get tool result and create a tool_result response."""
    res = call_func(fc.name, fc.input, ns=ns, raise_on_err=False)

    if isinstance(res, ToolResult) and res.result_type == "image/png":
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": res.data}},
            {"type": "text", "text": "Captured screenshot."}
        ]
    else:
        content = str(res.data) if isinstance(res, ToolResult) else str(res)

    return {"type": "tool_result", "tool_use_id": fc.id, "content": content}


def mk_toolres(r: Message, ns: Dict[str, Callable] = None) -> List[Dict]:
    """Create a tool_result message from response."""
    if ns is None:
        ns = {}

    cts = getattr(r, 'content', [])
    res = [mk_msg(r.model_dump(), role='assistant')]

    tcs = [mk_funcres(o, ns) for o in cts if isinstance(o, ToolUseBlock)]
    if tcs:
        res.append(mk_msg(tcs))

    return res


async def mk_funcres_async(fc: ToolUseBlock, ns: Dict[str, Callable]) -> Dict[str, Any]:
    """Given tool use block, get tool result and create a tool_result response (async version)."""
    res = await call_func_async(fc.name, fc.input, ns=ns, raise_on_err=False)
    return {"type": "tool_result", "tool_use_id": fc.id, "content": str(res)}


async def mk_toolres_async(r: Message, ns: Dict[str, Callable] = None) -> List[Dict]:
    """Create a tool_result message from response (async version)."""
    if ns is None:
        ns = {}

    cts = getattr(r, 'content', [])
    res = [mk_msg(r.model_dump(), role='assistant')]

    tcs = [await mk_funcres_async(o, ns) for o in cts if isinstance(o, ToolUseBlock)]
    if tcs:
        res.append(mk_msg(tcs))

    return res


def think_md(txt: str, thk: str) -> str:
    """Format text with thinking in markdown details block."""
    return f"""{txt}

<details>
<summary>Thinking</summary>
{thk}
</details>
"""


def get_costs(c: 'Client') -> tuple:
    """Get detailed cost breakdown."""
    model_type = model_types.get(c.model, 'sonnet')
    costs = pricing.get(model_type, pricing['sonnet'])

    inp_cost = c.use.input_tokens * costs[0] / 1e6
    out_cost = c.use.output_tokens * costs[1] / 1e6

    cache_w = c.use.cache_creation_input_tokens
    cache_r = c.use.cache_read_input_tokens
    cache_cost = (cache_w * costs[2] + cache_r * costs[3]) / 1e6

    return inp_cost, out_cost, cache_cost, cache_w + cache_r, 0  # Last is server_tool_cost
