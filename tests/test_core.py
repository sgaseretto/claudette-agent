"""Tests for claudette_agent core functionality."""
import pytest
from claudette_agent import (
    Usage, Message, TextBlock, ToolUseBlock,
    usage, find_block, find_blocks, contents,
    mk_msg, mk_msgs, mk_tool_choice, get_schema, tool,
    listify, mk_ns
)


class TestUsage:
    """Tests for Usage class."""

    def test_usage_creation(self):
        u = usage(inp=100, out=50, cache_create=10, cache_read=5)
        assert u.input_tokens == 100
        assert u.output_tokens == 50
        assert u.cache_creation_input_tokens == 10
        assert u.cache_read_input_tokens == 5

    def test_usage_total(self):
        u = usage(inp=100, out=50, cache_create=10, cache_read=5)
        assert u.total == 165

    def test_usage_add(self):
        u1 = usage(inp=100, out=50)
        u2 = usage(inp=200, out=100)
        u3 = u1 + u2
        assert u3.input_tokens == 300
        assert u3.output_tokens == 150


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        msg = Message(
            id="test-id",
            role="assistant",
            content=[TextBlock(text="Hello!")]
        )
        assert msg.id == "test-id"
        assert msg.role == "assistant"
        assert len(msg.content) == 1

    def test_message_model_dump(self):
        msg = Message(
            id="test-id",
            role="assistant",
            content=[TextBlock(text="Hello!")]
        )
        dump = msg.model_dump()
        assert dump['id'] == "test-id"
        assert dump['role'] == "assistant"


class TestContents:
    """Tests for contents extraction."""

    def test_contents_simple(self):
        msg = Message(
            content=[TextBlock(text="Hello, world!")]
        )
        assert contents(msg) == "Hello, world!"

    def test_contents_multiple_blocks(self):
        msg = Message(
            content=[
                TextBlock(text="Hello"),
                TextBlock(text=" world!")
            ]
        )
        assert contents(msg) == "Hello world!"

    def test_contents_string(self):
        assert contents("Hello") == "Hello"


class TestFindBlock:
    """Tests for find_block function."""

    def test_find_text_block(self):
        msg = Message(
            content=[TextBlock(text="Hello")]
        )
        block = find_block(msg, TextBlock)
        assert block is not None
        assert block.text == "Hello"

    def test_find_no_block(self):
        msg = Message(content=[])
        block = find_block(msg, TextBlock)
        assert block is None


class TestMkMsg:
    """Tests for mk_msg function."""

    def test_mk_msg_string(self):
        msg = mk_msg("Hello")
        assert msg['role'] == 'user'
        assert msg['content'][0]['type'] == 'text'
        assert msg['content'][0]['text'] == 'Hello'

    def test_mk_msg_role(self):
        msg = mk_msg("Hello", role="assistant")
        assert msg['role'] == 'assistant'

    def test_mk_msg_cache(self):
        msg = mk_msg("Hello", cache=True)
        assert 'cache_control' in msg['content'][0]


class TestMkToolChoice:
    """Tests for mk_tool_choice function."""

    def test_tool_choice_string(self):
        choice = mk_tool_choice("my_tool")
        assert choice['type'] == 'tool'
        assert choice['name'] == 'my_tool'

    def test_tool_choice_true(self):
        choice = mk_tool_choice(True)
        assert choice['type'] == 'any'

    def test_tool_choice_none(self):
        choice = mk_tool_choice(None)
        assert choice['type'] == 'auto'


class TestGetSchema:
    """Tests for get_schema function."""

    def test_get_schema_simple(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        schema = get_schema(add)
        assert schema['name'] == 'add'
        assert 'Add two numbers' in schema['description']
        assert 'a' in schema['input_schema']['properties']
        assert 'b' in schema['input_schema']['properties']

    def test_get_schema_dict_passthrough(self):
        schema = {'name': 'test', 'description': 'Test schema'}
        result = get_schema(schema)
        assert result == schema


class TestTool:
    """Tests for tool decorator."""

    def test_tool_decorator(self):
        @tool
        def my_func(x: int) -> str:
            return str(x)

        assert callable(my_func)
        assert my_func(5) == "5"


class TestListify:
    """Tests for listify function."""

    def test_listify_none(self):
        assert listify(None) == []

    def test_listify_single(self):
        assert listify(1) == [1]

    def test_listify_list(self):
        assert listify([1, 2, 3]) == [1, 2, 3]


class TestMkNs:
    """Tests for mk_ns function."""

    def test_mk_ns(self):
        def func1(): pass
        def func2(): pass

        ns = mk_ns(func1, func2)
        assert 'func1' in ns
        assert 'func2' in ns
        assert ns['func1'] is func1
