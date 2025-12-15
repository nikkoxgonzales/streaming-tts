"""Tests for TTSStream class."""

import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from streaming_tts.config import PlaybackConfig, TTSConfig
from streaming_tts.stream import TTSStream


@pytest.fixture
def mock_pipeline():
    """Create a mock Kokoro pipeline."""
    with patch("kokoro.KPipeline") as mock_cls:
        mock_result = MagicMock()
        mock_result.audio = torch.randn(24000)
        mock_result.tokens = []

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = iter([mock_result])
        mock_pipeline_instance.load_single_voice = MagicMock(
            return_value=torch.randn(256)
        )
        mock_cls.return_value = mock_pipeline_instance

        yield mock_cls


@pytest.fixture
def stream(mock_pipeline):
    """Create a TTSStream with mocked engine."""
    s = TTSStream()
    yield s
    s.shutdown()


class TestTTSStream:
    """Tests for TTSStream class."""

    def test_init_default(self, mock_pipeline) -> None:
        """Test initialization with defaults."""
        stream = TTSStream()
        assert stream.config.voice == "af_heart"
        assert stream.text == ""
        stream.shutdown()

    def test_init_custom_config(self, mock_pipeline) -> None:
        """Test initialization with custom config."""
        config = TTSConfig(voice="am_adam", speed=1.5)
        stream = TTSStream(config=config)
        assert stream.config.voice == "am_adam"
        stream.shutdown()

    def test_feed(self, stream) -> None:
        """Test feeding text."""
        result = stream.feed("Hello")
        assert result is stream  # Returns self for chaining
        assert stream.text == "Hello"

    def test_feed_chaining(self, stream) -> None:
        """Test method chaining with feed."""
        stream.feed("Hello").feed("world")
        assert stream.text == "Hello world"

    def test_feed_empty_string(self, stream) -> None:
        """Test that empty strings are ignored."""
        stream.feed("Hello").feed("").feed("  ").feed("world")
        assert stream.text == "Hello world"

    def test_clear(self, stream) -> None:
        """Test clearing text buffer."""
        stream.feed("Hello world")
        result = stream.clear()
        assert result is stream
        assert stream.text == ""

    def test_set_voice(self, stream) -> None:
        """Test voice changing."""
        result = stream.set_voice("am_adam")
        assert result is stream
        assert stream._engine._current_voice == "am_adam"

    def test_get_voices(self, stream) -> None:
        """Test getting available voices."""
        voices = stream.get_voices()
        assert len(voices) > 0
        assert "af_heart" in voices


class TestPlayMethod:
    """Tests for play() method."""

    def test_play_blocking(self, stream) -> None:
        """Test blocking play."""
        chunks = []
        stream.feed("Hello world")
        stream.play(on_chunk=chunks.append, muted=True)

        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    def test_play_with_callbacks(self, stream) -> None:
        """Test play with start/stop callbacks."""
        started = threading.Event()
        stopped = threading.Event()

        stream.feed("Hello")
        stream.play(
            on_start=started.set,
            on_stop=stopped.set,
            muted=True,
        )

        assert started.is_set()
        assert stopped.is_set()

    def test_play_no_text(self, stream) -> None:
        """Test play with no text returns None."""
        result = stream.play(muted=True)
        assert result is None

    def test_play_clears_buffer(self, stream) -> None:
        """Test that play clears the text buffer."""
        stream.feed("Hello world")
        stream.play(muted=True)
        assert stream.text == ""

    def test_play_async_returns_thread(self, stream) -> None:
        """Test play_async returns a thread."""
        stream.feed("Hello")
        thread = stream.play_async(muted=True)
        assert isinstance(thread, threading.Thread)
        thread.join(timeout=5)

    def test_play_muted_default(self, mock_pipeline) -> None:
        """Test that muted defaults to playback_config.muted."""
        config = PlaybackConfig(muted=True)
        stream = TTSStream(playback_config=config)
        stream.feed("Hello")

        # Should not raise (no audio playback attempted)
        stream.play()
        stream.shutdown()


class TestStreamIterators:
    """Tests for stream() and stream_async() methods."""

    def test_stream_sync(self, stream) -> None:
        """Test synchronous streaming."""
        stream.feed("Hello world")
        chunks = list(stream.stream())

        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    def test_stream_sync_no_text(self, stream) -> None:
        """Test sync stream with no text."""
        chunks = list(stream.stream())
        assert chunks == []

    def test_stream_clears_buffer(self, stream) -> None:
        """Test that stream clears the text buffer."""
        stream.feed("Hello")
        list(stream.stream())
        assert stream.text == ""

    @pytest.mark.asyncio
    async def test_stream_async(self, stream) -> None:
        """Test async streaming."""
        stream.feed("Hello world")
        chunks = [chunk async for chunk in stream.stream_async()]

        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_async_no_text(self, stream) -> None:
        """Test async stream with no text."""
        chunks = [chunk async for chunk in stream.stream_async()]
        assert chunks == []


class TestStopAndStatus:
    """Tests for stop() and is_playing() methods."""

    def test_is_playing_initially_false(self, stream) -> None:
        """Test is_playing is False initially."""
        assert not stream.is_playing()

    def test_is_playing_during_synthesis(self, stream) -> None:
        """Test is_playing is True during synthesis."""
        playing_during = []

        def check_playing(chunk):
            playing_during.append(stream.is_playing())

        stream.feed("Hello world this is a test")
        stream.play(on_chunk=check_playing, muted=True)

        assert any(playing_during)  # At least one check should be True

    def test_stop(self, stream) -> None:
        """Test stopping synthesis."""
        chunks = []

        def on_chunk(chunk):
            chunks.append(chunk)
            if len(chunks) >= 1:
                stream.stop()

        stream.feed("Hello world this is a very long text to synthesize")
        stream.play(on_chunk=on_chunk, muted=True)

        # Should have stopped after first chunk
        assert len(chunks) <= 2  # May get 1-2 chunks before stop takes effect


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager(self, mock_pipeline) -> None:
        """Test using TTSStream as context manager."""
        with TTSStream() as stream:
            stream.feed("Hello")
            chunks = list(stream.stream())
            assert len(chunks) > 0

    def test_context_manager_cleanup(self, mock_pipeline) -> None:
        """Test that context manager cleans up resources."""
        stream = TTSStream()
        stream.__enter__()
        stream.feed("Hello")
        list(stream.stream())
        stream.__exit__(None, None, None)

        # After exit, executor should be shut down
        assert stream._executor._shutdown


class TestShutdown:
    """Tests for shutdown() method."""

    def test_shutdown(self, stream) -> None:
        """Test shutdown releases resources."""
        stream.feed("Hello")
        list(stream.stream())
        stream.shutdown()

        assert stream._executor._shutdown

    def test_shutdown_stops_playing(self, stream) -> None:
        """Test shutdown stops ongoing playback."""
        stream.feed("Hello world")
        thread = stream.play_async(muted=True)
        stream.shutdown()
        thread.join(timeout=1)

        assert not stream.is_playing()


class TestPlayAsyncNoText:
    """Tests for play_async() with no text."""

    def test_play_async_no_text(self, stream) -> None:
        """Test play_async with no text returns dummy thread."""
        thread = stream.play_async(muted=True)
        assert isinstance(thread, threading.Thread)
        thread.join(timeout=1)
        assert not thread.is_alive()


class TestFormatConversion:
    """Tests for format conversion in play/stream methods."""

    def test_play_with_format(self, stream) -> None:
        """Test play with non-PCM format."""
        chunks = []
        stream.feed("Hello world")

        with patch("streaming_tts.stream.StreamingAudioWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer.write_chunk.return_value = b"encoded_chunk"
            mock_writer.finalize.return_value = b"final_chunk"
            mock_writer_cls.return_value = mock_writer

            stream.play(on_chunk=chunks.append, muted=True, format="mp3")

            mock_writer_cls.assert_called_once()
            assert mock_writer.write_chunk.called
            assert mock_writer.finalize.called
            assert b"encoded_chunk" in chunks
            assert b"final_chunk" in chunks

    def test_play_with_format_empty_encoded(self, stream) -> None:
        """Test play with format when writer returns empty encoded."""
        chunks = []
        stream.feed("Hello world")

        with patch("streaming_tts.stream.StreamingAudioWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer.write_chunk.return_value = None  # No output yet
            mock_writer.finalize.return_value = b"final_chunk"
            mock_writer_cls.return_value = mock_writer

            stream.play(on_chunk=chunks.append, muted=True, format="mp3")

            # Only final chunk should be in chunks
            assert b"final_chunk" in chunks

    def test_stream_sync_with_format(self, stream) -> None:
        """Test sync stream with non-PCM format."""
        stream.feed("Hello world")

        with patch("streaming_tts.stream.StreamingAudioWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer.write_chunk.return_value = b"encoded_chunk"
            mock_writer.finalize.return_value = b"final_chunk"
            mock_writer_cls.return_value = mock_writer

            chunks = list(stream.stream(format="opus"))

            mock_writer_cls.assert_called_once()
            assert b"encoded_chunk" in chunks
            assert b"final_chunk" in chunks

    def test_stream_sync_with_format_empty_chunks(self, stream) -> None:
        """Test sync stream with format when writer returns empty chunks."""
        stream.feed("Hello world")

        with patch("streaming_tts.stream.StreamingAudioWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer.write_chunk.return_value = None
            mock_writer.finalize.return_value = None
            mock_writer_cls.return_value = mock_writer

            chunks = list(stream.stream(format="opus"))

            # No chunks should be yielded when all are None
            assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_async_with_format(self, stream) -> None:
        """Test async stream with non-PCM format."""
        stream.feed("Hello world")

        with patch("streaming_tts.stream.StreamingAudioWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer.write_chunk.return_value = b"encoded_chunk"
            mock_writer.finalize.return_value = b"final_chunk"
            mock_writer_cls.return_value = mock_writer

            chunks = [chunk async for chunk in stream.stream_async(format="wav")]

            mock_writer_cls.assert_called_once()
            assert b"encoded_chunk" in chunks
            assert b"final_chunk" in chunks

    @pytest.mark.asyncio
    async def test_stream_async_with_format_empty_chunks(self, stream) -> None:
        """Test async stream with format when writer returns empty chunks."""
        stream.feed("Hello world")

        with patch("streaming_tts.stream.StreamingAudioWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer.write_chunk.return_value = None
            mock_writer.finalize.return_value = None
            mock_writer_cls.return_value = mock_writer

            chunks = [chunk async for chunk in stream.stream_async(format="wav")]

            # No chunks when all returns are None
            assert chunks == []


class TestStreamTextAsync:
    """Tests for stream_text_async() method."""

    @pytest.mark.asyncio
    async def test_stream_text_async_basic(self, stream) -> None:
        """Test basic stream_text_async functionality."""
        async def text_gen():
            yield "Hello "
            yield "world!"

        chunks = [chunk async for chunk in stream.stream_text_async(text_gen())]

        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_text_async_with_sentence_boundary(self, stream) -> None:
        """Test stream_text_async with sentence boundary detection."""
        async def text_gen():
            yield "Hello world. "
            yield "This is a test."

        chunks = [chunk async for chunk in stream.stream_text_async(text_gen())]

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_text_async_with_on_first_chunk(self, stream) -> None:
        """Test stream_text_async with on_first_chunk callback."""
        first_chunk_received = []

        def on_first(chunk):
            first_chunk_received.append(chunk)

        async def text_gen():
            yield "Hello world!"

        chunks = [chunk async for chunk in stream.stream_text_async(
            text_gen(),
            on_first_chunk=on_first
        )]

        assert len(first_chunk_received) == 1
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_text_async_with_format(self, stream) -> None:
        """Test stream_text_async with non-PCM format."""
        async def text_gen():
            yield "Hello world!"

        with patch("streaming_tts.stream.StreamingAudioWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer.write_chunk.return_value = b"encoded_chunk"
            mock_writer.finalize.return_value = b"final_chunk"
            mock_writer_cls.return_value = mock_writer

            chunks = [chunk async for chunk in stream.stream_text_async(
                text_gen(),
                format="mp3"
            )]

            assert b"encoded_chunk" in chunks

    @pytest.mark.asyncio
    async def test_stream_text_async_with_format_on_first_chunk(self, stream) -> None:
        """Test stream_text_async with format and on_first_chunk callback."""
        first_chunk_received = []

        def on_first(chunk):
            first_chunk_received.append(chunk)

        async def text_gen():
            yield "Hello world!"

        with patch("streaming_tts.stream.StreamingAudioWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer.write_chunk.return_value = b"encoded_chunk"
            mock_writer.finalize.return_value = b"final_chunk"
            mock_writer_cls.return_value = mock_writer

            async for _ in stream.stream_text_async(
                text_gen(),
                on_first_chunk=on_first,
                format="mp3"
            ):
                pass

            assert len(first_chunk_received) == 1
            assert first_chunk_received[0] == b"encoded_chunk"

    @pytest.mark.asyncio
    async def test_stream_text_async_empty_text(self, stream) -> None:
        """Test stream_text_async with empty text."""
        async def text_gen():
            yield "   "  # Only whitespace

        chunks = [chunk async for chunk in stream.stream_text_async(text_gen())]

        # Should handle gracefully
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_stream_text_async_cancelled(self, stream) -> None:
        """Test stream_text_async cancellation."""
        import asyncio

        async def slow_text_gen():
            yield "Hello "
            await asyncio.sleep(10)  # Long delay
            yield "world!"

        async def run_and_cancel():
            gen = stream.stream_text_async(slow_text_gen())
            task = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        await run_and_cancel()

    @pytest.mark.asyncio
    async def test_stream_text_async_force_fragment(self, stream) -> None:
        """Test stream_text_async with force fragment after N words."""
        from streaming_tts.config import FragmentConfig

        config = FragmentConfig(force_fragment_after_words=3)

        async def text_gen():
            # Long text without delimiters to trigger force fragment
            yield "one two three four five six seven eight"

        chunks = [chunk async for chunk in stream.stream_text_async(
            text_gen(),
            fragment_config=config
        )]

        assert len(chunks) > 0


class TestStopDuringStream:
    """Tests for stop event during streaming."""

    def test_stream_sync_stop(self, stream) -> None:
        """Test stopping sync stream."""
        stream.feed("Hello world this is a long text for testing")
        chunks = []

        for i, chunk in enumerate(stream.stream()):
            chunks.append(chunk)
            if i >= 0:
                stream.stop()
                break

        # Should have stopped early
        assert len(chunks) <= 2

    @pytest.mark.asyncio
    async def test_stream_async_stop(self, stream) -> None:
        """Test stopping async stream."""
        stream.feed("Hello world this is a long text for testing")
        chunks = []

        async for chunk in stream.stream_async():
            chunks.append(chunk)
            stream.stop()
            break

        # Should have stopped
        assert len(chunks) >= 1


class TestPlayChunk:
    """Tests for _play_chunk method."""

    def test_play_chunk_creates_player(self, stream) -> None:
        """Test that _play_chunk creates an AudioPlayer."""
        # Remove any existing player
        if hasattr(stream, "_player"):
            del stream._player

        mock_player = MagicMock()
        mock_audio_module = MagicMock()
        mock_audio_module.AudioPlayer.return_value = mock_player

        with patch.dict("sys.modules", {"streaming_tts.audio": mock_audio_module}):
            stream._play_chunk(b"test_audio_data")

            mock_audio_module.AudioPlayer.assert_called_once()
            mock_player.start.assert_called_once()
            mock_player.write.assert_called_once_with(b"test_audio_data")

    def test_play_chunk_reuses_player(self, stream) -> None:
        """Test that _play_chunk reuses existing player."""
        # Remove any existing player first
        if hasattr(stream, "_player"):
            del stream._player

        mock_player = MagicMock()
        mock_audio_module = MagicMock()
        mock_audio_module.AudioPlayer.return_value = mock_player

        with patch.dict("sys.modules", {"streaming_tts.audio": mock_audio_module}):
            # First call creates player
            stream._play_chunk(b"chunk1")
            # Second call should reuse
            stream._play_chunk(b"chunk2")

            # Player should be created once
            assert mock_audio_module.AudioPlayer.call_count == 1
            # But write called twice
            assert mock_player.write.call_count == 2

    def test_play_chunk_import_error(self, stream) -> None:
        """Test _play_chunk handles ImportError gracefully."""
        # Remove any existing player
        if hasattr(stream, "_player"):
            del stream._player

        with patch.dict("sys.modules", {"streaming_tts.audio": None}):
            # Should not raise
            stream._play_chunk(b"test_audio_data")

    def test_play_unmuted(self, stream) -> None:
        """Test play with muted=False calls _play_chunk."""
        with patch.object(stream, "_play_chunk") as mock_play_chunk:
            stream.feed("Hello")
            stream.play(muted=False)

            assert mock_play_chunk.called
