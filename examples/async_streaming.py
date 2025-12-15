"""Async streaming examples for streaming-tts.

This pattern is ideal for async frameworks like FastAPI, aiohttp,
or any asyncio-based application.

Run with: python examples/async_streaming.py
"""

import asyncio

from streaming_tts import TTSStream, TTSConfig


async def basic_async_iteration():
    """Basic async iterator pattern."""
    stream = TTSStream()
    stream.feed("Hello from async iteration! This is the recommended pattern for async apps.")

    chunk_count = 0
    total_bytes = 0

    async for chunk in stream.stream_async():
        chunk_count += 1
        total_bytes += len(chunk)
        print(f"Async chunk {chunk_count}: {len(chunk)} bytes")

    print(f"\nTotal: {chunk_count} chunks, {total_bytes} bytes")
    stream.shutdown()


async def simulated_fastapi_endpoint():
    """Simulate a FastAPI streaming response.

    In a real FastAPI app, this would be:

    @app.get("/tts")
    async def text_to_speech(text: str):
        stream = TTSStream()
        stream.feed(text)

        async def generate():
            async for chunk in stream.stream_async():
                yield chunk
            stream.shutdown()

        return StreamingResponse(generate(), media_type="audio/pcm")
    """

    class FakeStreamingResponse:
        """Simulates FastAPI's StreamingResponse."""

        def __init__(self):
            self.chunks_sent = 0
            self.bytes_sent = 0

        async def send(self, chunk: bytes):
            self.chunks_sent += 1
            self.bytes_sent += len(chunk)
            # Simulate network latency
            await asyncio.sleep(0.01)
            print(f"  Response chunk #{self.chunks_sent} ({len(chunk)} bytes)")

    response = FakeStreamingResponse()

    stream = TTSStream()
    stream.feed("This simulates a FastAPI streaming endpoint sending audio to a client.")

    async for chunk in stream.stream_async():
        await response.send(chunk)

    print(f"\nResponse complete: {response.chunks_sent} chunks, {response.bytes_sent} bytes")
    stream.shutdown()


async def concurrent_synthesis():
    """Generate multiple audio streams concurrently."""

    async def synthesize(text: str, voice: str) -> tuple[str, int, int]:
        config = TTSConfig(voice=voice)
        stream = TTSStream(config=config)
        stream.feed(text)

        chunks = 0
        total_bytes = 0
        async for chunk in stream.stream_async():
            chunks += 1
            total_bytes += len(chunk)

        stream.shutdown()
        return voice, chunks, total_bytes

    texts_and_voices = [
        ("Hello from voice one.", "af_heart"),
        ("Greetings from voice two.", "am_adam"),
        ("Hi there from voice three.", "bf_alice"),
    ]

    # Run all syntheses concurrently
    tasks = [synthesize(text, voice) for text, voice in texts_and_voices]
    results = await asyncio.gather(*tasks)

    print("Concurrent synthesis results:")
    for voice, chunks, total_bytes in results:
        print(f"  {voice}: {chunks} chunks, {total_bytes} bytes")


async def with_timeout():
    """Demonstrate async iteration with timeout."""
    stream = TTSStream()
    stream.feed("This is a longer text that demonstrates timeout handling during async streaming.")

    try:
        async with asyncio.timeout(2.0):  # 2 second timeout
            chunk_count = 0
            async for chunk in stream.stream_async():
                chunk_count += 1
                print(f"Chunk {chunk_count}: {len(chunk)} bytes")
            print(f"Completed: {chunk_count} chunks")
    except asyncio.TimeoutError:
        print("Timeout reached - stopping synthesis")
        stream.stop()

    stream.shutdown()


async def producer_consumer_pattern():
    """Demonstrate producer-consumer pattern with async queue."""

    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def producer():
        """Produce audio chunks from TTS."""
        stream = TTSStream()
        stream.feed("Producer consumer pattern demonstration with async queues.")

        async for chunk in stream.stream_async():
            await queue.put(chunk)
            print(f"  Producer: queued {len(chunk)} bytes")

        await queue.put(None)  # Signal completion
        stream.shutdown()
        print("  Producer: done")

    async def consumer():
        """Consume audio chunks from queue."""
        chunks_consumed = 0
        bytes_consumed = 0

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            chunks_consumed += 1
            bytes_consumed += len(chunk)
            # Simulate processing
            await asyncio.sleep(0.01)

        print(f"  Consumer: processed {chunks_consumed} chunks, {bytes_consumed} bytes")

    # Run producer and consumer concurrently
    await asyncio.gather(producer(), consumer())


async def sync_iterator_for_comparison():
    """Show sync iterator (useful when you don't need async)."""
    stream = TTSStream()
    stream.feed("This uses the synchronous iterator for comparison.")

    chunk_count = 0
    for chunk in stream.stream():
        chunk_count += 1
        print(f"Sync chunk {chunk_count}: {len(chunk)} bytes")

    print(f"Total: {chunk_count} chunks")
    stream.shutdown()


async def main():
    print("=== Basic Async Iteration ===")
    await basic_async_iteration()

    print("\n=== Simulated FastAPI Endpoint ===")
    await simulated_fastapi_endpoint()

    print("\n=== Concurrent Synthesis ===")
    await concurrent_synthesis()

    print("\n=== With Timeout ===")
    await with_timeout()

    print("\n=== Producer-Consumer Pattern ===")
    await producer_consumer_pattern()

    print("\n=== Sync Iterator (for comparison) ===")
    await sync_iterator_for_comparison()


if __name__ == "__main__":
    asyncio.run(main())
