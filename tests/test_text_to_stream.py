"""Tests for streaming_tts.text_to_stream module (integration tests)."""

import pytest
import queue
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_engine():
    """Create a mock TTS engine."""
    engine = MagicMock()
    engine.get_stream_info.return_value = (8, 1, 24000)  # paInt16, mono, 24kHz
    engine.queue = queue.Queue()
    engine.timings = queue.Queue()
    engine.can_consume_generators = False
    engine.engine_name = "mock_engine"
    engine.stop_synthesis_event = MagicMock()
    engine.stop_synthesis_event.is_set.return_value = False
    return engine


@pytest.fixture
def mock_stream_player():
    """Mock StreamPlayer class."""
    with patch('streaming_tts.text_to_stream.StreamPlayer') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.muted = False
        yield mock


@pytest.fixture
def mock_dependencies():
    """Mock all heavy dependencies."""
    with patch('streaming_tts.text_to_stream.StreamPlayer') as mock_player, \
         patch('streaming_tts.text_to_stream.s2s') as mock_s2s:

        mock_player_instance = MagicMock()
        mock_player.return_value = mock_player_instance
        mock_player_instance.muted = False

        yield {
            'player': mock_player,
            'player_instance': mock_player_instance,
            's2s': mock_s2s
        }


class TestTextToAudioStreamInit:
    """Tests for TextToAudioStream initialization."""

    def test_init_single_engine(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine)
        assert stream.engine == mock_engine

    def test_init_with_muted(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine, muted=True)
        assert stream.global_muted is True

    def test_init_with_output_device(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine, output_device_index=2)
        assert stream.output_device_index == 2


class TestTextToAudioStreamFeed:
    """Tests for feed functionality."""

    def test_feed_text_returns_self(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine)
        result = stream.feed("Hello world")

        assert result is stream  # For chaining

    def test_feed_iterator(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        def gen():
            yield "Hello "
            yield "world"

        stream = TextToAudioStream(engine=mock_engine)
        result = stream.feed(gen())

        assert result is stream


class TestTextToAudioStreamCallbacks:
    """Tests for callback functionality."""

    def test_callback_on_text_stream_start(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        callback_called = []

        def on_start():
            callback_called.append(True)

        stream = TextToAudioStream(
            engine=mock_engine,
            on_text_stream_start=on_start
        )

        assert stream.on_text_stream_start is not None

    def test_callback_on_text_stream_stop(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        callback_called = []

        def on_stop():
            callback_called.append(True)

        stream = TextToAudioStream(
            engine=mock_engine,
            on_text_stream_stop=on_stop
        )

        assert stream.on_text_stream_stop is not None

    def test_callback_on_audio_stream_start(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(
            engine=mock_engine,
            on_audio_stream_start=lambda: None
        )

        assert stream.on_audio_stream_start is not None

    def test_callback_on_audio_stream_stop(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(
            engine=mock_engine,
            on_audio_stream_stop=lambda: None
        )

        assert stream.on_audio_stream_stop is not None

    def test_callback_on_character(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        chars = []

        stream = TextToAudioStream(
            engine=mock_engine,
            on_character=lambda c: chars.append(c)
        )

        assert stream.on_character is not None


class TestTextToAudioStreamPlayback:
    """Tests for playback functionality."""

    def test_is_playing_initially_false(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine)
        assert stream.is_playing() is False

    def test_stop_method_exists(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine)
        # Should not raise
        stream.stop()

    def test_pause_method_exists(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine)
        # Should not raise
        stream.pause()

    def test_resume_method_exists(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine)
        # Should not raise
        stream.resume()


class TestTextToAudioStreamText:
    """Tests for text accumulation."""

    def test_text_property(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine)
        # text() should return accumulated text
        result = stream.text()
        assert isinstance(result, str)


class TestTextToAudioStreamConfiguration:
    """Tests for configuration options."""

    def test_tokenizer_options(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(
            engine=mock_engine,
            tokenizer="stanza",
            language="en"
        )
        assert stream.tokenizer == "stanza"
        assert stream.language == "en"

    def test_frames_per_buffer(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(
            engine=mock_engine,
            frames_per_buffer=1024
        )
        assert stream.frames_per_buffer == 1024

    def test_playout_chunk_size(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(
            engine=mock_engine,
            playout_chunk_size=512
        )
        assert stream.playout_chunk_size == 512

    def test_log_characters(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(
            engine=mock_engine,
            log_characters=True
        )
        assert stream.log_characters is True


class TestTextToAudioStreamOutput:
    """Tests for output options."""

    def test_output_wavfile_attribute_initialized(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(engine=mock_engine)
        # output_wavfile is initialized to None in __init__
        # and set during play() method calls
        assert stream.output_wavfile is None

    def test_output_device_index(self, mock_engine, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        stream = TextToAudioStream(
            engine=mock_engine,
            output_device_index=1
        )
        assert stream.output_device_index == 1


class TestTextToAudioStreamMultiEngine:
    """Tests for multi-engine support."""

    def test_fallback_engine_list(self, mock_dependencies):
        from streaming_tts.text_to_stream import TextToAudioStream

        engine1 = MagicMock()
        engine1.get_stream_info.return_value = (8, 1, 24000)
        engine1.queue = queue.Queue()
        engine1.timings = queue.Queue()
        engine1.can_consume_generators = False
        engine1.engine_name = "engine1"
        engine1.stop_synthesis_event = MagicMock()

        engine2 = MagicMock()
        engine2.get_stream_info.return_value = (8, 1, 24000)
        engine2.queue = queue.Queue()
        engine2.timings = queue.Queue()
        engine2.can_consume_generators = False
        engine2.engine_name = "engine2"
        engine2.stop_synthesis_event = MagicMock()

        stream = TextToAudioStream(engine=[engine1, engine2])
        # Should use first engine
        assert stream.engine == engine1
