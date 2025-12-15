"""Callback-based streaming examples for streaming-tts.

This pattern is ideal for WebSocket streaming where you want to
send audio chunks to a client as they're generated.

Run with: python examples/callback_streaming.py
"""

import io
import wave
from pathlib import Path

from streaming_tts import TTSStream, TTSConfig


def simple_callback():
    """Basic callback pattern - process each chunk as it arrives."""
    chunks_received = []

    def on_chunk(chunk: bytes):
        chunks_received.append(chunk)
        print(f"Received chunk: {len(chunk)} bytes")

    stream = TTSStream()
    stream.feed("Hello world! This demonstrates the callback pattern.")
    stream.play(on_chunk=on_chunk, muted=True)

    total_bytes = sum(len(c) for c in chunks_received)
    print(f"\nTotal: {len(chunks_received)} chunks, {total_bytes} bytes")

    stream.shutdown()


def with_lifecycle_callbacks():
    """Use start/stop callbacks for setup and cleanup."""

    def on_start():
        print(">>> Audio generation started")

    def on_chunk(chunk: bytes):
        print(f"  Chunk: {len(chunk)} bytes")

    def on_stop():
        print(">>> Audio generation completed")

    stream = TTSStream()
    stream.feed("This example shows lifecycle callbacks.")
    stream.play(
        on_chunk=on_chunk,
        on_start=on_start,
        on_stop=on_stop,
        muted=True,
    )

    stream.shutdown()


def save_to_wav_file():
    """Collect chunks and save to a WAV file."""
    chunks = []

    def collect_chunk(chunk: bytes):
        chunks.append(chunk)

    config = TTSConfig(voice="af_heart")
    stream = TTSStream(config=config)

    stream.feed("This audio will be saved to a WAV file.")
    stream.play(on_chunk=collect_chunk, muted=True)

    # Combine all chunks
    audio_data = b"".join(chunks)

    # Save as WAV
    output_path = Path("output.wav")
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(config.sample_rate)
        wav_file.writeframes(audio_data)

    print(f"Saved {len(audio_data)} bytes to {output_path}")
    print(f"Duration: {len(audio_data) / (config.sample_rate * 2):.2f} seconds")

    stream.shutdown()


def non_blocking_synthesis():
    """Run synthesis in background thread."""
    import time

    stream = TTSStream()
    chunks = []

    def on_chunk(chunk: bytes):
        chunks.append(chunk)

    def on_stop():
        print(f"Background synthesis complete: {len(chunks)} chunks")

    stream.feed("This runs in the background while the main thread continues.")

    # Start non-blocking synthesis
    thread = stream.play(
        on_chunk=on_chunk,
        on_stop=on_stop,
        muted=True,
        blocking=False,
    )

    # Main thread continues immediately
    print("Main thread: synthesis started in background")
    for i in range(5):
        print(f"Main thread: doing other work... ({i + 1}/5)")
        time.sleep(0.3)

    # Wait for synthesis to complete if needed
    if thread:
        thread.join()
        print("Main thread: synthesis thread joined")

    stream.shutdown()


def simulated_websocket_streaming():
    """Simulate streaming audio over a WebSocket connection."""

    class FakeWebSocket:
        """Simulates a WebSocket for demonstration."""

        def __init__(self):
            self.messages_sent = 0
            self.bytes_sent = 0

        def send_bytes(self, data: bytes):
            self.messages_sent += 1
            self.bytes_sent += len(data)
            # In a real app, this would send to the client
            print(f"  WS: sent message #{self.messages_sent} ({len(data)} bytes)")

    ws = FakeWebSocket()

    stream = TTSStream()
    stream.feed("This demonstrates WebSocket-style streaming. Each chunk is sent as a separate message.")
    stream.play(on_chunk=ws.send_bytes, muted=True)

    print(f"\nWebSocket stats: {ws.messages_sent} messages, {ws.bytes_sent} bytes total")

    stream.shutdown()


def multiple_utterances():
    """Stream multiple utterances sequentially."""
    stream = TTSStream()

    texts = [
        "First message.",
        "Second message with more content.",
        "Third and final message.",
    ]

    for i, text in enumerate(texts):
        print(f"\n--- Utterance {i + 1} ---")
        chunk_count = 0

        def on_chunk(chunk: bytes):
            nonlocal chunk_count
            chunk_count += 1

        stream.feed(text)
        stream.play(on_chunk=on_chunk, muted=True)

        print(f"Generated {chunk_count} chunks")

    stream.shutdown()


if __name__ == "__main__":
    print("=== Simple Callback ===")
    simple_callback()

    print("\n=== Lifecycle Callbacks ===")
    with_lifecycle_callbacks()

    print("\n=== Save to WAV File ===")
    save_to_wav_file()

    print("\n=== Non-Blocking Synthesis ===")
    non_blocking_synthesis()

    print("\n=== Simulated WebSocket Streaming ===")
    simulated_websocket_streaming()

    print("\n=== Multiple Utterances ===")
    multiple_utterances()
