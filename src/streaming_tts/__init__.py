"""
streaming-tts: A streamlined, Kokoro-based text-to-speech library with streaming support.

Example usage:
    from streaming_tts import TTSStream, TTSConfig

    # Simple playback
    stream = TTSStream()
    stream.feed("Hello world").play()

    # Callback pattern (for WebSocket streaming)
    stream.feed("Hello world").play(on_chunk=ws.send_bytes, muted=True)

    # Async iterator pattern
    async for chunk in stream.stream_async():
        await ws.send_bytes(chunk)

    # Low-latency voice assistant pattern (with LLM streaming)
    from streaming_tts import FragmentConfig

    async def on_first_audio(chunk):
        await ws.send_json({"type": "audio", "start": True})

    async for chunk in tts.stream_text_async(
        llm.stream_response(prompt),
        on_first_chunk=on_first_audio
    ):
        await ws.send_bytes(chunk)

    # Format conversion (requires: pip install streaming-tts[formats])
    from streaming_tts import StreamingAudioWriter
    writer = StreamingAudioWriter("mp3")
    for chunk in engine.synthesize(text):
        encoded = writer.write_chunk(chunk)
        send(encoded)
    send(writer.finalize())
"""

from .config import FragmentConfig, PlaybackConfig, TTSConfig
from .formats import AudioFormat, StreamingAudioWriter, get_content_type
from .stream import TTSStream

__all__ = [
    "TTSConfig",
    "PlaybackConfig",
    "FragmentConfig",
    "TTSStream",
    "StreamingAudioWriter",
    "AudioFormat",
    "get_content_type",
]

__version__ = "0.1.0"
