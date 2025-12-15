"""Format conversion examples for streaming-tts.

Demonstrates converting PCM audio to various formats (MP3, Opus, WAV, etc.)
on-the-fly during streaming.

Requires: pip install streaming-tts[formats]

Run with: python examples/format_conversion.py
"""

import asyncio
from pathlib import Path

from streaming_tts import TTSStream, TTSConfig, StreamingAudioWriter, get_content_type


def stream_as_mp3():
    """Stream audio as MP3 format."""
    stream = TTSStream()
    stream.feed("This audio is being encoded as MP3 in real-time.")

    chunks = []
    for chunk in stream.stream(format="mp3"):
        chunks.append(chunk)
        print(f"MP3 chunk: {len(chunk)} bytes")

    # Save to file
    output = Path("output.mp3")
    output.write_bytes(b"".join(chunks))
    print(f"\nSaved {output.stat().st_size} bytes to {output}")

    stream.shutdown()


def stream_as_opus():
    """Stream audio as Opus format (great for WebRTC/low bandwidth)."""
    stream = TTSStream()
    stream.feed("Opus is excellent for voice streaming due to its low latency and small size.")

    chunks = []
    for chunk in stream.stream(format="opus"):
        chunks.append(chunk)
        print(f"Opus chunk: {len(chunk)} bytes")

    # Save to file (Opus uses Ogg container)
    output = Path("output.ogg")
    output.write_bytes(b"".join(chunks))
    print(f"\nSaved {output.stat().st_size} bytes to {output}")

    stream.shutdown()


def stream_as_wav():
    """Stream audio as WAV format."""
    stream = TTSStream()
    stream.feed("WAV format is uncompressed but universally compatible.")

    chunks = []
    for chunk in stream.stream(format="wav"):
        chunks.append(chunk)
        print(f"WAV chunk: {len(chunk)} bytes")

    output = Path("output.wav")
    output.write_bytes(b"".join(chunks))
    print(f"\nSaved {output.stat().st_size} bytes to {output}")

    stream.shutdown()


def stream_as_flac():
    """Stream audio as FLAC format (lossless compression)."""
    stream = TTSStream()
    stream.feed("FLAC provides lossless compression for high quality audio.")

    chunks = []
    for chunk in stream.stream(format="flac"):
        chunks.append(chunk)
        print(f"FLAC chunk: {len(chunk)} bytes")

    output = Path("output.flac")
    output.write_bytes(b"".join(chunks))
    print(f"\nSaved {output.stat().st_size} bytes to {output}")

    stream.shutdown()


def callback_with_format():
    """Use format conversion with callback pattern."""
    chunks = []

    def on_chunk(chunk: bytes):
        chunks.append(chunk)
        print(f"Callback received MP3 chunk: {len(chunk)} bytes")

    stream = TTSStream()
    stream.feed("Format conversion works with callbacks too.")
    stream.play(on_chunk=on_chunk, format="mp3", muted=True)

    print(f"\nTotal: {len(chunks)} chunks, {sum(len(c) for c in chunks)} bytes")
    stream.shutdown()


async def async_stream_with_format():
    """Use format conversion with async streaming."""
    stream = TTSStream()
    stream.feed("Async streaming also supports format conversion.")

    chunks = []
    async for chunk in stream.stream_async(format="opus"):
        chunks.append(chunk)
        print(f"Async Opus chunk: {len(chunk)} bytes")

    print(f"\nTotal: {len(chunks)} chunks, {sum(len(c) for c in chunks)} bytes")
    stream.shutdown()


def compare_format_sizes():
    """Compare file sizes across different formats."""
    text = "This is a test sentence to compare audio format sizes. " * 3

    results = {}

    for fmt in ["pcm", "wav", "mp3", "opus", "flac"]:
        stream = TTSStream()
        stream.feed(text)

        chunks = []
        try:
            for chunk in stream.stream(format=fmt):
                chunks.append(chunk)
            total_size = sum(len(c) for c in chunks)
            results[fmt] = total_size
        except ImportError:
            results[fmt] = "N/A (PyAV not installed)"

        stream.shutdown()

    print("Format size comparison:")
    print("-" * 40)
    for fmt, size in results.items():
        if isinstance(size, int):
            print(f"  {fmt:6}: {size:,} bytes")
        else:
            print(f"  {fmt:6}: {size}")


def low_level_format_writer():
    """Use StreamingAudioWriter directly for custom workflows."""
    from streaming_tts.engine import KokoroTTS

    # Create engine directly
    engine = KokoroTTS(TTSConfig(voice="af_heart"))

    # Create format writer
    writer = StreamingAudioWriter("mp3", sample_rate=24000, bitrate=128000)

    print(f"Content-Type: {get_content_type('mp3')}")

    chunks = []
    for audio_float32 in engine.synthesize("Direct engine access with format conversion."):
        # Writer handles float32 to int16 conversion
        encoded = writer.write_chunk(audio_float32)
        if encoded:
            chunks.append(encoded)
            print(f"Encoded chunk: {len(encoded)} bytes")

    # Finalize to flush encoder buffer
    final = writer.finalize()
    if final:
        chunks.append(final)
        print(f"Final chunk: {len(final)} bytes")

    # Save result
    output = Path("output_direct.mp3")
    output.write_bytes(b"".join(chunks))
    print(f"\nSaved {output.stat().st_size} bytes to {output}")

    engine.shutdown()


def http_content_types():
    """Show MIME content types for HTTP responses."""
    print("Content-Type headers for HTTP responses:")
    print("-" * 40)
    for fmt in ["pcm", "wav", "mp3", "opus", "flac", "aac"]:
        print(f"  {fmt:6}: {get_content_type(fmt)}")


def main():
    print("=== Stream as MP3 ===")
    try:
        stream_as_mp3()
    except ImportError as e:
        print(f"Skipped: {e}")

    print("\n=== Stream as Opus ===")
    try:
        stream_as_opus()
    except ImportError as e:
        print(f"Skipped: {e}")

    print("\n=== Stream as WAV ===")
    try:
        stream_as_wav()
    except ImportError as e:
        print(f"Skipped: {e}")

    print("\n=== Stream as FLAC ===")
    try:
        stream_as_flac()
    except ImportError as e:
        print(f"Skipped: {e}")

    print("\n=== Callback with Format ===")
    try:
        callback_with_format()
    except ImportError as e:
        print(f"Skipped: {e}")

    print("\n=== Async Stream with Format ===")
    try:
        asyncio.run(async_stream_with_format())
    except ImportError as e:
        print(f"Skipped: {e}")

    print("\n=== Compare Format Sizes ===")
    compare_format_sizes()

    print("\n=== HTTP Content Types ===")
    http_content_types()

    print("\n=== Low-Level Format Writer ===")
    try:
        low_level_format_writer()
    except ImportError as e:
        print(f"Skipped: {e}")


if __name__ == "__main__":
    main()
