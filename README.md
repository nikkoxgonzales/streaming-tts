# streaming-tts

A streamlined, Kokoro-based text-to-speech library with streaming support.

Extracted and simplified from [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS), focused on the excellent [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS model.

## Features

- **Streaming-first design**: Generate audio chunks as they're synthesized
- **Multiple patterns**: Callback-based, async iterator, or sync iterator
- **Kokoro TTS**: High-quality, lightweight local TTS (82M parameters)
- **Voice blending**: Mix voices with flexible syntax
- **Format conversion**: Output PCM, WAV, MP3, Opus, FLAC, or AAC
- **GPU support**: Auto-detects CUDA, MPS (Apple Silicon), or CPU with memory management
- **Optional playback**: Local audio playback for development/testing
- **Modern Python**: Requires Python 3.12+

## Installation

```bash
pip install streaming-tts
```

For audio playback support:

```bash
pip install streaming-tts[playback]
```

For format conversion (MP3, Opus, etc.):

```bash
pip install streaming-tts[formats]
```

For development:

```bash
pip install streaming-tts[dev]
```

For everything:

```bash
pip install streaming-tts[all]
```

## Quick Start

### Simple Playback

```python
from streaming_tts import TTSStream

stream = TTSStream()
stream.feed("Hello world").play()
```

### Callback Pattern (WebSocket Streaming)

```python
from streaming_tts import TTSStream

stream = TTSStream()

def send_to_client(chunk: bytes):
    websocket.send_bytes(chunk)

stream.feed("Hello world").play(on_chunk=send_to_client, muted=True)
```

### Async Iterator Pattern

```python
from streaming_tts import TTSStream

stream = TTSStream()
stream.feed("Hello world")

async for chunk in stream.stream_async():
    await websocket.send_bytes(chunk)
```

### Sync Iterator Pattern

```python
from streaming_tts import TTSStream

stream = TTSStream()
stream.feed("Hello world")

for chunk in stream.stream():
    process(chunk)
```

## Configuration

### TTS Configuration

```python
from streaming_tts import TTSStream, TTSConfig

config = TTSConfig(
    voice="af_heart",           # Voice name or blend formula
    speed=1.0,                  # Speech speed (1.0 = normal)
    device=None,                # "cuda", "mps", "cpu", or None (auto-detect)
    trim_silence=True,          # Trim leading/trailing silence
    silence_threshold=0.005,
    memory_threshold_gb=2.0,    # GPU memory threshold for auto-clearing
)

stream = TTSStream(config=config)
```

### Available Voices

```python
stream = TTSStream()
voices = stream.get_voices()
print(voices)
# ['af_heart', 'af_alloy', 'am_adam', 'bf_alice', 'jf_alpha', ...]
```

Voice prefixes indicate language:
- `af_`, `am_`: American English
- `bf_`, `bm_`: British English
- `jf_`, `jm_`: Japanese
- `zf_`, `zm_`: Mandarin Chinese
- `ef_`, `em_`: Spanish
- `ff_`: French
- `hf_`, `hm_`: Hindi
- `if_`, `im_`: Italian
- `pf_`, `pm_`: Brazilian Portuguese

### Voice Blending

Mix multiple voices with flexible syntax:

```python
stream = TTSStream()

# New style - equal blend (recommended)
stream.set_voice("af_sarah+af_jessica")

# New style - weighted blend
stream.set_voice("af_sarah(0.3)+af_jessica(0.7)")

# Old style - also supported
stream.set_voice("0.3*af_sarah + 0.7*am_adam")

# Subtraction (experimental)
stream.set_voice("af_sarah-af_jessica")

stream.feed("Blended voice speaking").play()
```

### Playback Configuration

```python
from streaming_tts import TTSStream, PlaybackConfig

playback = PlaybackConfig(
    device_index=None,     # Audio device (None = default)
    frames_per_buffer=512, # Buffer size
    muted=False,           # Skip actual playback
)

stream = TTSStream(playback_config=playback)
```

## API Reference

### TTSStream

Main class for TTS streaming.

```python
class TTSStream:
    def __init__(
        self,
        config: TTSConfig | None = None,
        playback_config: PlaybackConfig | None = None,
    ) -> None: ...

    def feed(self, text: str) -> TTSStream: ...
    def clear(self) -> TTSStream: ...
    def play(
        self,
        *,
        on_chunk: Callable[[bytes], None] | None = None,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        muted: bool | None = None,
        blocking: bool = True,
        format: AudioFormat = "pcm",  # pcm, wav, mp3, opus, flac, aac
    ) -> threading.Thread | None: ...
    def play_async(self, **kwargs) -> threading.Thread: ...
    def stream(self, format: AudioFormat = "pcm") -> Iterator[bytes]: ...
    async def stream_async(self, format: AudioFormat = "pcm") -> AsyncIterator[bytes]: ...
    def stop(self) -> None: ...
    def is_playing(self) -> bool: ...
    def set_voice(self, voice: str) -> TTSStream: ...
    def get_voices(self) -> list[str]: ...
    def shutdown(self) -> None: ...
```

### TTSConfig

```python
@dataclass(frozen=True)
class TTSConfig:
    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000
    channels: int = 1
    device: str | None = None  # "cuda", "mps", "cpu", or None (auto-detect)
    trim_silence: bool = True
    silence_threshold: float = 0.005
    fade_in_ms: int = 10
    fade_out_ms: int = 10
    extra_trim_start_ms: int = 15
    extra_trim_end_ms: int = 15
    memory_threshold_gb: float = 2.0  # GPU memory threshold for auto-clearing
```

### PlaybackConfig

```python
@dataclass(frozen=True)
class PlaybackConfig:
    device_index: int | None = None
    frames_per_buffer: int = 512
    muted: bool = False
```

## Audio Format

Default output format:
- Format: PCM16 (16-bit signed integers)
- Sample rate: 24000 Hz
- Channels: 1 (mono)

### Format Conversion

Convert to other formats on-the-fly (requires `pip install streaming-tts[formats]`):

```python
from streaming_tts import TTSStream

stream = TTSStream()
stream.feed("Hello world")

# Stream as MP3
for chunk in stream.stream(format="mp3"):
    send_to_client(chunk)

# Or with async
async for chunk in stream.stream_async(format="opus"):
    await websocket.send_bytes(chunk)

# Or with callback
stream.feed("More text").play(on_chunk=callback, format="wav", muted=True)
```

Supported formats: `pcm` (default), `wav`, `mp3`, `opus`, `flac`, `aac`

### StreamingAudioWriter (Low-level)

For direct format conversion:

```python
from streaming_tts import StreamingAudioWriter, get_content_type

writer = StreamingAudioWriter("mp3", sample_rate=24000)

for audio_chunk in engine.synthesize(text):
    encoded = writer.write_chunk(audio_chunk)
    if encoded:
        send(encoded)

# Get final bytes (codec flush)
final = writer.finalize()
if final:
    send(final)

# Get MIME type for HTTP headers
content_type = get_content_type("mp3")  # "audio/mpeg"
```

## Context Manager

TTSStream supports context manager protocol for automatic cleanup:

```python
with TTSStream() as stream:
    stream.feed("Hello world")
    for chunk in stream.stream():
        process(chunk)
# Resources automatically released
```

## Development

```bash
# Clone and install
git clone https://github.com/yourusername/streaming-tts
cd streaming-tts
pip install -e ".[dev]"

# Run tests
pytest -v tests/

# Type check
mypy src/

# Lint
ruff check src/ tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) - The underlying TTS model
- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) - Original library this was extracted from
