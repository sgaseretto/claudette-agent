#!/usr/bin/env python3
"""
Manual test script for extra_args functionality.

This script tests the extra_args parameter with real SDK calls.
Run with: uv run python test_extra_args_manual.py
"""
import asyncio
from claudette_agent import (
    Client, AsyncClient, Chat, AsyncChat,
    contents, DEFAULT_MODEL
)


async def test_client_with_extra_args():
    """Test Client with extra_args for truly stateless queries."""
    print("=" * 60)
    print("Test 1: Client with extra_args (--no-session-persistence)")
    print("=" * 60)

    # Create a client with extra_args for stateless queries
    # CLI-style args: key is the flag name, value is None for flags without arguments
    client = Client(
        model=DEFAULT_MODEL,
        setting_sources=[],  # Don't load settings
        extra_args={'no-session-persistence': None}  # Don't persist session (CLI flag)
    )

    print(f"  setting_sources: {client.setting_sources}")
    print(f"  extra_args: {client.extra_args}")

    # Make a simple query
    response = await client("Say 'Hello from stateless client!' and nothing else.", sp="Be extremely brief.")
    print(f"  Response: {contents(response)}")
    print(f"  Usage: {client.use}")
    print()


async def test_chat_with_extra_args():
    """Test Chat with extra_args for truly stateless queries."""
    print("=" * 60)
    print("Test 2: Chat with extra_args (--no-session-persistence)")
    print("=" * 60)

    # Create a Chat with extra_args for stateless queries
    chat = Chat(
        model=DEFAULT_MODEL,
        sp="You are a helpful assistant. Be very brief.",
        setting_sources=[],
        extra_args={'no-session-persistence': None}
    )

    print(f"  setting_sources: {chat.c.setting_sources}")
    print(f"  extra_args: {chat.c.extra_args}")

    # Make a query
    response = await chat("What is 2 + 2? Just give the number.")
    print(f"  Response: {contents(response)}")
    print()


async def test_async_chat_with_extra_args():
    """Test AsyncChat with extra_args."""
    print("=" * 60)
    print("Test 3: AsyncChat with extra_args (--no-session-persistence)")
    print("=" * 60)

    # Create an AsyncChat with extra_args
    chat = AsyncChat(
        model=DEFAULT_MODEL,
        sp="Be extremely concise.",
        setting_sources=[],
        extra_args={'no-session-persistence': None}
    )

    print(f"  setting_sources: {chat.c.setting_sources}")
    print(f"  extra_args: {chat.c.extra_args}")

    # Make a query
    response = await chat("Say 'Hello from AsyncChat!'")
    print(f"  Response: {contents(response)}")
    print()


async def test_client_passed_to_chat():
    """Test passing a Client with extra_args to Chat."""
    print("=" * 60)
    print("Test 4: Client with extra_args passed to Chat")
    print("=" * 60)

    # Create a Client with extra_args
    client = Client(
        model=DEFAULT_MODEL,
        setting_sources=[],
        extra_args={'no-session-persistence': None}
    )

    # Pass it to Chat
    chat = Chat(cli=client, sp="Be brief.")

    print(f"  setting_sources: {chat.c.setting_sources}")
    print(f"  extra_args: {chat.c.extra_args}")

    # Make a query
    response = await chat("Say 'Hello from Chat with Client!'")
    print(f"  Response: {contents(response)}")
    print()


async def test_chat_updating_client_extra_args():
    """Test Chat updating existing client's extra_args."""
    print("=" * 60)
    print("Test 5: Chat updating existing Client's extra_args")
    print("=" * 60)

    # Create a Client without extra_args
    client = Client(
        model=DEFAULT_MODEL,
        setting_sources=[],
        extra_args={'existing_flag': True}
    )

    print(f"  Client extra_args before Chat: {client.extra_args}")

    # Pass it to Chat with additional extra_args
    chat = Chat(cli=client, sp="Be brief.", extra_args={'new_flag': True})

    print(f"  Client extra_args after Chat: {chat.c.extra_args}")

    # Make a query
    response = await chat("Say 'Hello!'")
    print(f"  Response: {contents(response)}")
    print()


async def main():
    """Run all tests."""
    print()
    print("Testing extra_args parameter for claudette-agent")
    print("=" * 60)
    print()

    await test_client_with_extra_args()
    await test_chat_with_extra_args()
    await test_async_chat_with_extra_args()
    await test_client_passed_to_chat()
    await test_chat_updating_client_extra_args()

    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
