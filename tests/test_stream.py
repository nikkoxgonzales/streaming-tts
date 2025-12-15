"""Tests for TTSStream class."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
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
