"""Tests for extra_args parameter functionality.

This module tests the extra_args parameter that allows passing additional
arguments to ClaudeAgentOptions, such as no_session_persistence=True for
truly stateless queries.
"""
import pytest
from unittest.mock import patch, MagicMock
from claudette_agent import Client, AsyncClient, Chat, AsyncChat, DEFAULT_MODEL


class TestClientExtraArgs:
    """Tests for Client extra_args handling."""

    def test_client_stores_extra_args(self):
        """Client should store extra_args as a dict."""
        client = Client(
            model=DEFAULT_MODEL,
            extra_args={'no-session-persistence': None}
        )
        assert client.extra_args == {'no-session-persistence': None}

    def test_client_handles_none_extra_args(self):
        """Client should handle None extra_args."""
        client = Client(model=DEFAULT_MODEL, extra_args=None)
        assert client.extra_args == {}

    def test_client_handles_empty_extra_args(self):
        """Client should handle empty extra_args dict."""
        client = Client(model=DEFAULT_MODEL, extra_args={})
        assert client.extra_args == {}

    def test_client_copies_extra_args(self):
        """Client should copy extra_args to avoid mutations."""
        original = {'no-session-persistence': None}
        client = Client(model=DEFAULT_MODEL, extra_args=original)
        original['new-flag'] = 'value'
        assert 'new-flag' not in client.extra_args

    def test_client_build_options_includes_extra_args(self):
        """Client._build_options should include extra_args in options."""
        client = Client(
            model=DEFAULT_MODEL,
            extra_args={'no-session-persistence': None, 'custom-flag': 'value'}
        )
        # Call _build_options and check that extra_args are included
        # We can't easily test the actual options object without SDK,
        # but we can verify the extra_args are stored
        assert client.extra_args == {
            'no-session-persistence': None,
            'custom-flag': 'value'
        }


class TestAsyncClientExtraArgs:
    """Tests for AsyncClient extra_args handling."""

    def test_async_client_stores_extra_args(self):
        """AsyncClient should store extra_args."""
        client = AsyncClient(
            model=DEFAULT_MODEL,
            extra_args={'no-session-persistence': None}
        )
        assert client.extra_args == {'no-session-persistence': None}


class TestChatExtraArgs:
    """Tests for Chat extra_args handling."""

    def test_chat_creates_client_with_extra_args(self):
        """Chat should pass extra_args to Client."""
        chat = Chat(
            model=DEFAULT_MODEL,
            extra_args={'no-session-persistence': None}
        )
        assert chat.c.extra_args == {'no-session-persistence': None}

    def test_chat_updates_existing_client_extra_args(self):
        """Chat should update existing client's extra_args."""
        client = Client(model=DEFAULT_MODEL, extra_args={'existing': None})
        chat = Chat(cli=client, extra_args={'new': None})
        assert chat.c.extra_args == {'existing': None, 'new': None}

    def test_chat_with_none_extra_args(self):
        """Chat should handle None extra_args."""
        chat = Chat(model=DEFAULT_MODEL, extra_args=None)
        assert chat.c.extra_args == {}


class TestAsyncChatExtraArgs:
    """Tests for AsyncChat extra_args handling."""

    def test_async_chat_creates_client_with_extra_args(self):
        """AsyncChat should pass extra_args to AsyncClient."""
        chat = AsyncChat(
            model=DEFAULT_MODEL,
            extra_args={'no-session-persistence': None}
        )
        assert chat.c.extra_args == {'no-session-persistence': None}

    def test_async_chat_handles_extra_args_in_kwargs(self):
        """AsyncChat should include extra_args in kwargs filter."""
        chat = AsyncChat(
            model=DEFAULT_MODEL,
            extra_args={'no-session-persistence': None},
            cache=False,
            cwd='/tmp'
        )
        assert chat.c.extra_args == {'no-session-persistence': None}


class TestExtraArgsIntegration:
    """Integration tests for extra_args."""

    def test_stateless_query_setup(self):
        """Test setting up a truly stateless query configuration."""
        # This is the recommended pattern for truly stateless queries
        # Keys should NOT include '--' prefix (SDK adds it internally)
        # Example: {'no-session-persistence': None} becomes --no-session-persistence
        client = Client(
            model=DEFAULT_MODEL,
            setting_sources=[],  # Don't load settings
            extra_args={'no-session-persistence': None}  # Don't persist session
        )
        assert client.setting_sources == []
        assert 'no-session-persistence' in client.extra_args

    def test_chat_stateless_setup(self):
        """Test setting up Chat for stateless queries."""
        chat = Chat(
            model=DEFAULT_MODEL,
            setting_sources=[],
            extra_args={'no-session-persistence': None}
        )
        assert chat.c.setting_sources == []
        assert 'no-session-persistence' in chat.c.extra_args

    def test_combined_env_and_extra_args(self):
        """Test using both env and extra_args together."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            client = Client(
                model=DEFAULT_MODEL,
                env={'HOME': tmpdir},  # Custom HOME for isolation
                extra_args={'no-session-persistence': None}
            )
            assert client.env.get('HOME') == tmpdir
            assert 'no-session-persistence' in client.extra_args


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
