#!/usr/bin/env python3
"""
Streaming Test for claudette-agent

This script tests streaming functionality in the terminal.
Run with: python streaming_test.py

Note: The Claude Agent SDK streams complete message blocks, not individual
text characters. You'll see text appear as complete blocks rather than
character-by-character streaming.
"""
import asyncio
import sys

from claudette_agent import Chat, contents

# Model to use
MODEL = "claude-sonnet-4-5-20250929"


async def test_basic_streaming():
    """Test basic streaming without tools."""
    print("=" * 60)
    print("Test 1: Basic Streaming")
    print("=" * 60)

    chat = Chat(model=MODEL, sp="You are a storyteller. Keep responses brief.")

    print("\nPrompt: Tell me a very short story about a robot (2-3 sentences)")
    print("\nStreaming response:\n")

    block_count = 0
    async for block in chat.stream("Tell me a very short story about a robot in 2-3 sentences."):
        block_count += 1
        print(f"[Block {block_count}]: {block}")

    print(f"\n\nTotal blocks received: {block_count}")
    print(f"Usage: {chat.use}")
    print(f"Cost: ${chat.cost:.6f}")


async def test_streaming_with_flush():
    """Test streaming with immediate output flushing."""
    print("\n" + "=" * 60)
    print("Test 2: Streaming with Flush (simulated character output)")
    print("=" * 60)

    chat = Chat(model=MODEL, sp="You are helpful. Keep responses brief.")

    print("\nPrompt: What is Python?")
    print("\nStreaming response:\n")

    async for block in chat.stream("What is Python? Answer in 1-2 sentences."):
        # Print each block immediately
        print(block, end="", flush=True)

    print("\n")
    print(f"Usage: {chat.use}")
    print(f"Cost: ${chat.cost:.6f}")


async def test_regular_call():
    """Test regular (non-streaming) call for comparison."""
    print("\n" + "=" * 60)
    print("Test 3: Regular Call (for comparison)")
    print("=" * 60)

    chat = Chat(model=MODEL, sp="You are helpful. Keep responses brief.")

    print("\nPrompt: What is JavaScript?")

    response = await chat("What is JavaScript? Answer in 1-2 sentences.")

    print(f"\nResponse: {contents(response)}")
    print(f"\nUsage: {chat.use}")
    print(f"Cost: ${chat.cost:.6f}")


async def test_multi_turn_streaming():
    """Test streaming across multiple conversation turns."""
    print("\n" + "=" * 60)
    print("Test 4: Multi-turn Streaming")
    print("=" * 60)

    chat = Chat(model=MODEL, sp="You are helpful. Keep responses very brief (1 sentence).")

    prompts = [
        "What is 2 + 2?",
        "Now multiply that by 3.",
        "What's the square root of that?"
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Prompt: {prompt}")
        print("Response: ", end="")

        async for block in chat.stream(prompt):
            print(block, end="", flush=True)

        print()

    print(f"\nFinal Usage: {chat.use}")
    print(f"Total Cost: ${chat.cost:.6f}")


async def test_sdk_raw_streaming():
    """Test raw SDK streaming to understand message structure."""
    print("\n" + "=" * 60)
    print("Test 5: Raw SDK Streaming (debugging)")
    print("=" * 60)

    try:
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions

        # SDK requires ClaudeAgentOptions object
        options = ClaudeAgentOptions(
            system_prompt="You are helpful. Be very brief."
        )

        print("\nPrompt: Say hello in 5 words or less")
        print("\nRaw messages from SDK:\n")

        msg_count = 0
        async for msg in sdk_query(prompt="Say hello in 5 words or less", options=options):
            msg_count += 1
            msg_type = type(msg).__name__

            print(f"[Message {msg_count}] Type: {msg_type}")

            # Print all attributes for debugging
            print(f"  Dir: {[a for a in dir(msg) if not a.startswith('_')]}")

            # Print relevant attributes
            if hasattr(msg, 'content'):
                print(f"  Content: {msg.content}")
            if hasattr(msg, 'usage'):
                print(f"  Usage: {msg.usage}")
                print(f"  Usage type: {type(msg.usage)}")
            if hasattr(msg, 'total_cost_usd'):
                print(f"  Total Cost USD: {msg.total_cost_usd}")
            if hasattr(msg, 'id'):
                print(f"  ID: {msg.id}")
            if hasattr(msg, 'structured_output'):
                print(f"  Structured Output: {msg.structured_output}")

            print()

        print(f"Total messages: {msg_count}")

    except ImportError as e:
        print(f"SDK not available: {e}")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


async def main():
    """Run all streaming tests."""
    print("\nClaudette-Agent Streaming Test")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print("=" * 60)

    try:
        await test_basic_streaming()
        await test_streaming_with_flush()
        await test_regular_call()
        await test_multi_turn_streaming()
        await test_sdk_raw_streaming()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during tests: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
