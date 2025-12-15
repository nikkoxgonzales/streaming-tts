"""Core TTSStream class for text-to-speech streaming."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from .buffers import AsyncChunkBuffer
from .config import FragmentConfig, PlaybackConfig, TTSConfig
from .engine import KokoroTTS
from .formats import AudioFormat, StreamingAudioWriter

if TYPE_CHECKING:
    pass


class TTSStream:
    """
    Main TTS streaming interface.

    Supports multiple streaming patterns:
    1. Callback-based: `stream.play(on_chunk=callback)`
    2. Async iterator: `async for chunk in stream.stream_async()`
    3. Sync iterator: `for chunk in stream.stream()`

    Example (callback pattern - for WebSocket streaming):
        stream = TTSStream()
        stream.feed("Hello world")
        stream.play(on_chunk=ws.send_bytes, muted=True)

    Example (async iterator pattern):
        stream = TTSStream()
        stream.feed("Hello world")
        async for chunk in stream.stream_async():
            await ws.send_bytes(chunk)

    Example (simple playback):
        stream = TTSStream()
        stream.feed("Hello world").play()
    """

    def __init__(
        self,
        config: TTSConfig | None = None,
        playback_config: PlaybackConfig | None = None,
    ) -> None:
        """
        Initialize the TTS stream.

        Args:
            config: TTS configuration (voice, speed, etc.)
            playback_config: Audio playback configuration
        """
        self.config = config or TTSConfig()
        self.playback_config = playback_config or PlaybackConfig()

        self._engine = KokoroTTS(self.config)
        self._text_buffer: list[str] = []
        self._stop_event = threading.Event()
        self._playing = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    def feed(self, text: str) -> TTSStream:
        """
        Add text to synthesize.

        Args:
            text: Text to synthesize

        Returns:
            Self for method chaining
        """
        if text.strip():
            self._text_buffer.append(text)
        return self

    def clear(self) -> TTSStream:
        """
        Clear any buffered text.

        Returns:
            Self for method chaining
        """
        self._text_buffer.clear()
        return self

    @property
    def text(self) -> str:
        """Get accumulated text that was fed."""
        return " ".join(self._text_buffer)

    def play(
        self,
        *,
        on_chunk: Callable[[bytes], None] | None = None,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        muted: bool | None = None,
        blocking: bool = True,
        format: AudioFormat = "pcm",
    ) -> threading.Thread | None:
        """
        Synthesize and optionally play audio.

        Args:
            on_chunk: Called with each audio chunk (in specified format)
            on_start: Called when audio generation starts
            on_stop: Called when audio generation completes
            muted: If True, skip actual playback (just generate chunks).
                   Defaults to playback_config.muted if not specified.
            blocking: If True, block until complete. If False, run in background.
            format: Output format for on_chunk callback.
                    "pcm" = raw PCM16 (default, fastest)
                    "wav", "mp3", "opus", "flac", "aac" require PyAV.
                    Note: Local playback always uses PCM regardless of format.

        Returns:
            Thread if blocking=False, None otherwise
        """
        if not self._text_buffer:
            return None

        if muted is None:
            muted = self.playback_config.muted

        text = self.text
        self._text_buffer.clear()
        self._stop_event.clear()

        def generate() -> None:
            self._playing.set()
            writer: StreamingAudioWriter | None = None
            try:
                if on_start:
                    on_start()

                # Create format writer if not PCM
                if on_chunk and format != "pcm":
                    writer = StreamingAudioWriter(
                        format,
                        sample_rate=self.config.sample_rate,
                        channels=self.config.channels,
                    )

                for chunk in self._engine.synthesize_to_bytes(text):
                    if self._stop_event.is_set():
                        break

                    if on_chunk:
                        if writer:
                            # Convert to requested format
                            encoded = writer.write_chunk(chunk)
                            if encoded:
                                on_chunk(encoded)
                        else:
                            on_chunk(chunk)

                    if not muted:
                        # Local playback always uses raw PCM
                        self._play_chunk(chunk)

                # Finalize format writer
                if writer:
                    final = writer.finalize()
                    if final and on_chunk:
                        on_chunk(final)

            finally:
                self._playing.clear()
                if on_stop:
                    on_stop()

        if blocking:
            generate()
            return None
        else:
            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            return thread

    def play_async(
        self,
        **kwargs: Any,
    ) -> threading.Thread:
        """
        Non-blocking version of play().

        Args:
            **kwargs: Same arguments as play()

        Returns:
            The background thread running the synthesis
        """
        kwargs["blocking"] = False
        thread = self.play(**kwargs)
        if thread is None:
            # No text to synthesize, return a dummy completed thread
            thread = threading.Thread(target=lambda: None)
            thread.start()
        return thread

    def stream(self, format: AudioFormat = "pcm") -> Iterator[bytes]:
        """
        Synchronous generator yielding audio chunks.

        Args:
            format: Output format. "pcm" = raw PCM16 (default, fastest).
                    "wav", "mp3", "opus", "flac", "aac" require PyAV.

        Yields:
            Audio chunks as bytes in the specified format

        Example:
            for chunk in stream.stream():
                process(chunk)

            # Or with format conversion:
            for chunk in stream.stream(format="mp3"):
                send(chunk)
        """
        if not self._text_buffer:
            return

        text = self.text
        self._text_buffer.clear()
        self._stop_event.clear()
        self._playing.set()

        writer: StreamingAudioWriter | None = None
        if format != "pcm":
            writer = StreamingAudioWriter(
                format,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
            )

        try:
            for chunk in self._engine.synthesize_to_bytes(text):
                if self._stop_event.is_set():
                    break
                if writer:
                    encoded = writer.write_chunk(chunk)
                    if encoded:
                        yield encoded
                else:
                    yield chunk

            # Finalize format writer
            if writer:
                final = writer.finalize()
                if final:
                    yield final
        finally:
            self._playing.clear()

    async def stream_async(self, format: AudioFormat = "pcm") -> AsyncIterator[bytes]:
        """
        Async generator yielding audio chunks.

        Runs synthesis in a thread pool and yields chunks via asyncio.Queue.

        Args:
            format: Output format. "pcm" = raw PCM16 (default, fastest).
                    "wav", "mp3", "opus", "flac", "aac" require PyAV.

        Yields:
            Audio chunks as bytes in the specified format

        Example:
            async for chunk in stream.stream_async():
                await ws.send_bytes(chunk)

            # Or with format conversion:
            async for chunk in stream.stream_async(format="opus"):
                await ws.send_bytes(chunk)
        """
        if not self._text_buffer:
            return

        text = self.text
        self._text_buffer.clear()
        self._stop_event.clear()

        buffer = AsyncChunkBuffer(max_size=100)
        loop = asyncio.get_running_loop()

        def generate() -> None:
            self._playing.set()
            writer: StreamingAudioWriter | None = None
            if format != "pcm":
                writer = StreamingAudioWriter(
                    format,
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                )
            try:
                for chunk in self._engine.synthesize_to_bytes(text):
                    if self._stop_event.is_set():
                        break
                    if writer:
                        encoded = writer.write_chunk(chunk)
                        if encoded:
                            loop.call_soon_threadsafe(buffer.put_nowait, encoded)
                    else:
                        # Use call_soon_threadsafe to safely add to async queue
                        loop.call_soon_threadsafe(buffer.put_nowait, chunk)

                # Finalize format writer
                if writer:
                    final = writer.finalize()
                    if final:
                        loop.call_soon_threadsafe(buffer.put_nowait, final)
            finally:
                self._playing.clear()
                loop.call_soon_threadsafe(buffer.mark_done)

        # Run synthesis in thread pool
        future = loop.run_in_executor(self._executor, generate)

        # Yield chunks as they become available
        async for chunk in buffer:
            yield chunk

        # Wait for synthesis to complete
        await asyncio.wrap_future(future)

    async def stream_text_async(
        self,
        text_iterator: AsyncIterator[str],
        *,
        on_first_chunk: Callable[[bytes], Any] | None = None,
        fragment_config: FragmentConfig | None = None,
        format: AudioFormat = "pcm",
    ) -> AsyncIterator[bytes]:
        """
        Stream TTS from an async text iterator with fast sentence fragment support.

        This is optimized for low-latency voice assistants. It:
        1. Accepts streaming text (e.g., from an LLM)
        2. Detects sentence fragments for early TTS start
        3. Calls on_first_chunk BEFORE yielding first audio (for audio_start timing)
        4. Yields audio chunks as they're generated

        Args:
            text_iterator: Async iterator yielding text chunks (e.g., LLM tokens)
            on_first_chunk: Called with first audio chunk before it's yielded.
                           Use this to send "audio starting" signal at the right time.
            fragment_config: Fragment detection settings. Uses defaults if None.
            format: Output format. "pcm" = raw PCM16 (default, fastest).

        Yields:
            Audio chunks as bytes

        Example (WebSocket voice assistant):
            async def on_first_audio(chunk: bytes):
                await ws.send_json({"type": "audio", "start": True})

            async for chunk in tts.stream_text_async(
                llm.stream_response(prompt),
                on_first_chunk=on_first_audio
            ):
                await ws.send_bytes(chunk)

            await ws.send_json({"type": "audio", "end": True})
        """
        config = fragment_config or FragmentConfig()
        buffer = AsyncChunkBuffer(max_size=100)
        loop = asyncio.get_running_loop()

        # State for fragment detection
        pending_text = ""
        word_count = 0
        is_first_fragment = True
        first_chunk_sent = False

        # Track TTS tasks
        tts_tasks: list[asyncio.Future[None]] = []

        def synthesize_fragment(text: str) -> None:
            """Synthesize a text fragment and queue audio chunks."""
            nonlocal first_chunk_sent

            if self._stop_event.is_set() or not text.strip():
                return

            self._playing.set()
            writer: StreamingAudioWriter | None = None
            if format != "pcm":
                writer = StreamingAudioWriter(
                    format,
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                )

            try:
                for chunk in self._engine.synthesize_to_bytes(text):
                    if self._stop_event.is_set():
                        break

                    if writer:
                        encoded = writer.write_chunk(chunk)
                        if encoded:
                            # First chunk callback (thread-safe)
                            if not first_chunk_sent and on_first_chunk:
                                first_chunk_sent = True
                                # Call synchronously in this thread
                                on_first_chunk(encoded)
                            loop.call_soon_threadsafe(buffer.put_nowait, encoded)
                    else:
                        # First chunk callback (thread-safe)
                        if not first_chunk_sent and on_first_chunk:
                            first_chunk_sent = True
                            on_first_chunk(chunk)
                        loop.call_soon_threadsafe(buffer.put_nowait, chunk)

                # Finalize format writer
                if writer:
                    final = writer.finalize()
                    if final:
                        loop.call_soon_threadsafe(buffer.put_nowait, final)
            finally:
                self._playing.clear()

        def find_fragment_boundary(text: str, is_first: bool, word_cnt: int) -> int:
            """Find the index to split text at, or -1 if no split point found."""
            delimiters = config.fragment_delimiters if is_first else config.sentence_delimiters

            for i, char in enumerate(text):
                if char in delimiters:
                    if is_first:
                        # For first fragment, yield early at any delimiter after min length
                        if i + 1 >= config.minimum_fragment_length:
                            return i
                    else:
                        # For subsequent, prefer sentence boundaries
                        if i + 1 >= len(text) or text[i + 1] == ' ':
                            return i

            # Force yield after N words even without delimiter
            if word_cnt >= config.force_fragment_after_words:
                last_space = text.rfind(' ')
                if last_space > config.minimum_fragment_length:
                    return last_space

            return -1

        try:
            # Consume text iterator and queue TTS tasks
            async for text_chunk in text_iterator:
                if self._stop_event.is_set():
                    break

                pending_text += text_chunk
                word_count += text_chunk.count(' ')

                # Check for fragment boundary
                yield_idx = find_fragment_boundary(pending_text, is_first_fragment, word_count)

                if yield_idx >= 0:
                    fragment = pending_text[:yield_idx + 1]
                    pending_text = pending_text[yield_idx + 1:].lstrip()
                    word_count = pending_text.count(' ')

                    if is_first_fragment:
                        is_first_fragment = False

                    # Queue TTS for this fragment
                    task = loop.run_in_executor(self._executor, synthesize_fragment, fragment)
                    tts_tasks.append(task)

            # Process any remaining text
            if pending_text.strip():
                task = loop.run_in_executor(self._executor, synthesize_fragment, pending_text)
                tts_tasks.append(task)

            # Wait for all TTS tasks to complete, yielding audio as it arrives
            async def wait_for_tasks() -> None:
                for task in tts_tasks:
                    await task
                buffer.mark_done()

            wait_task = asyncio.create_task(wait_for_tasks())

            # Yield audio chunks as they become available
            async for chunk in buffer:
                yield chunk

            await wait_task

        except asyncio.CancelledError:
            self._stop_event.set()
            buffer.mark_done()
            raise
        finally:
            self._stop_event.clear()

    def stop(self) -> None:
        """Stop ongoing synthesis immediately."""
        self._stop_event.set()

    def is_playing(self) -> bool:
        """Check if synthesis is in progress."""
        return self._playing.is_set()

    def set_voice(self, voice: str) -> TTSStream:
        """
        Change the voice.

        Args:
            voice: Voice name or blend formula

        Returns:
            Self for method chaining
        """
        self._engine.set_voice(voice)
        return self

    def get_voices(self) -> list[str]:
        """Get list of available voice names."""
        return self._engine.get_voices()

    def shutdown(self) -> None:
        """Release all resources."""
        self.stop()
        self._engine.shutdown()
        self._executor.shutdown(wait=False)

    def _play_chunk(self, chunk: bytes) -> None:
        """
        Play a chunk to audio output.

        Args:
            chunk: PCM16 audio data
        """
        # Lazy import audio player to make it optional
        try:
            from .audio import AudioPlayer

            if not hasattr(self, "_player"):
                self._player = AudioPlayer(self.playback_config, self.config.sample_rate)
                self._player.start()

            self._player.write(chunk)
        except ImportError:
            # PyAudio not installed, skip playback
            pass

    def __enter__(self) -> TTSStream:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit - cleanup resources."""
        self.shutdown()

    def __del__(self) -> None:
        """Destructor - ensure cleanup."""
        with contextlib.suppress(Exception):
            self.shutdown()
