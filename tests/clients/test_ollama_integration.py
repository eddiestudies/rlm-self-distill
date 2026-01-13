"""Test script for OllamaClient."""

import asyncio

from self_distill import OllamaClient


def test_sync_completion():
    """Test synchronous completion."""
    client = OllamaClient(model_name="llama3.2:3b")

    # List available models
    print("Available models:", client.list_models())

    # Simple completion
    response = client.completion("What is 2 + 2? Answer briefly.")
    print(f"Response: {response}")

    # Check usage
    usage = client.get_usage_summary()
    print(f"Usage: {usage}")


async def test_async_completion():
    """Test asynchronous completion."""
    client = OllamaClient(model_name="llama3.2:3b")

    # Async completion
    response = await client.acompletion(
        "What is the capital of France? Answer briefly."
    )
    print(f"Async Response: {response}")

    # Check usage
    usage = client.get_last_usage()
    print(f"Last call usage: {usage}")


def test_with_messages():
    """Test with message list format."""
    client = OllamaClient(model_name="llama3.2:3b")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Explain Python in one sentence."},
    ]

    response = client.completion(messages)
    print(f"Messages format response: {response}")


if __name__ == "__main__":
    print("=== Testing OllamaClient ===\n")

    print("--- Sync Completion ---")
    test_sync_completion()

    print("\n--- Async Completion ---")
    asyncio.run(test_async_completion())

    print("\n--- Messages Format ---")
    test_with_messages()

    print("\n=== All tests passed ===")
