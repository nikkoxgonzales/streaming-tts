# streaming-tts Examples

This directory contains example scripts demonstrating various features of the streaming-tts library.

## Prerequisites

```bash
# Basic installation
pip install streaming-tts

# For playback examples
pip install streaming-tts[playback]

# For format conversion examples
pip install streaming-tts[formats]

# Or install everything
pip install streaming-tts[all]
```

## Examples

### basic_usage.py
Simple examples to get started:
- Simple playback
- Chained text
- Custom voice and speed
- Context manager usage
- Listing available voices

```bash
python examples/basic_usage.py
```

### callback_streaming.py
Callback-based streaming pattern (ideal for WebSocket):
- Basic callback pattern
- Lifecycle callbacks (on_start, on_stop)
- Saving to WAV file
- Non-blocking synthesis
- Simulated WebSocket streaming

```bash
python examples/callback_streaming.py
```

### async_streaming.py
Async iterator pattern (ideal for FastAPI/aiohttp):
- Basic async iteration
- Simulated FastAPI endpoint
- Concurrent synthesis
- Timeout handling
- Producer-consumer pattern

```bash
python examples/async_streaming.py
```

### format_conversion.py
Converting to various audio formats:
- Stream as MP3, Opus, WAV, FLAC
- Format with callbacks and async
- Comparing format sizes
- Low-level StreamingAudioWriter usage
- HTTP content types

```bash
python examples/format_conversion.py
```

### voice_blending.py
Creating blended voices:
- Equal blend (new syntax): `af_sarah+af_jessica`
- Weighted blend: `af_sarah(0.3)+af_jessica(0.7)`
- Old syntax: `0.3*af_sarah + 0.7*am_adam`
- Multi-voice blends
- Voice subtraction (experimental)

```bash
python examples/voice_blending.py
```

## Output Files

Some examples create output files in the current directory:
- `output.wav` - WAV audio file
- `output.mp3` - MP3 audio file
- `output.ogg` - Opus audio in Ogg container
- `output.flac` - FLAC audio file
- `heart_pure.wav`, `heart_adam_blend.wav` - Voice blend comparisons

## Quick Reference

### Streaming Patterns

```python
# 1. Callback (WebSocket style)
stream.feed("text").play(on_chunk=send_bytes, muted=True)

# 2. Sync iterator
for chunk in stream.stream():
    process(chunk)

# 3. Async iterator
async for chunk in stream.stream_async():
    await send(chunk)
```

### Format Conversion

```python
# With iterator
for chunk in stream.stream(format="mp3"):
    send(chunk)

# With callback
stream.play(on_chunk=callback, format="opus", muted=True)

# With async
async for chunk in stream.stream_async(format="wav"):
    await send(chunk)
```

### Voice Blending

```python
# New syntax (recommended)
config = TTSConfig(voice="af_sarah+af_jessica")           # Equal
config = TTSConfig(voice="af_sarah(0.3)+af_jessica(0.7)") # Weighted

# Old syntax
config = TTSConfig(voice="0.3*af_sarah + 0.7*am_adam")

# Dynamic
stream.set_voice("af_heart+am_adam")
```
