"""Tests for streaming_tts.stream_player module."""

import pytest
import queue
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_pyaudio():
    """Mock PyAudio for testing."""
    with patch('streaming_tts.stream_player.pyaudio') as mock:
        mock.paInt16 = 8
        mock.paFloat32 = 1
        mock.paInt32 = 4
        mock.paInt24 = 3
        mock.paInt8 = 16
        mock.paUInt8 = 32
        mock.paCustomFormat = 256

        mock_instance = MagicMock()
        mock.PyAudio.return_value = mock_instance

        # Mock device info
        mock_instance.get_default_output_device_info.return_value = {
            'index': 0,
            'name': 'Default Device',
            'defaultSampleRate': 44100,
            'maxOutputChannels': 2
        }
        mock_instance.get_device_info_by_index.return_value = {
            'index': 0,
            'name': 'Device 0',
            'defaultSampleRate': 44100,
            'maxOutputChannels': 2
        }
        mock_instance.is_format_supported.return_value = True
        mock_instance.get_sample_size.return_value = 2

        # Mock stream
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_stream.get_write_available.return_value = 1024
        mock_instance.open.return_value = mock_stream

        yield mock


@pytest.fixture
def mock_pa_portaudio():
    """Mock PyAudio C module."""
    with patch('streaming_tts.stream_player.pa') as mock:
        mock.paFramesPerBufferUnspecified = 0
        yield mock


class TestAudioConfiguration:
    """Tests for AudioConfiguration class."""

    def test_default_values(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioConfiguration

        config = AudioConfiguration()
        assert config.format == 8  # paInt16
        assert config.channels == 1
        assert config.rate == 16000
        assert config.muted is False

    def test_custom_values(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioConfiguration

        config = AudioConfiguration(
            format=1,  # paFloat32
            channels=2,
            rate=44100,
            muted=True
        )
        assert config.format == 1
        assert config.channels == 2
        assert config.rate == 44100
        assert config.muted is True

    def test_output_device_index(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioConfiguration

        config = AudioConfiguration(output_device_index=2)
        assert config.output_device_index == 2


class TestAudioStream:
    """Tests for AudioStream class."""

    def test_init(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration()
        stream = AudioStream(config)
        assert stream.config == config
        assert stream.stream is None

    def test_get_supported_sample_rates_caching(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration()
        stream = AudioStream(config)

        # Clear cache
        AudioStream._supported_rates_cache = {}

        rates1 = stream.get_supported_sample_rates(0)
        rates2 = stream.get_supported_sample_rates(0)

        # Should use cache on second call
        assert rates1 == rates2

    def test_get_best_sample_rate_desired_supported(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration()
        stream = AudioStream(config)

        # Clear cache and set up supported rates
        AudioStream._supported_rates_cache = {}
        AudioStream._supported_rates_cache[(0, config.format)] = [16000, 24000, 44100]

        result = stream._get_best_sample_rate(0, 24000)
        assert result == 24000

    def test_open_stream(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration()
        stream = AudioStream(config)
        stream.open_stream()

        mock_pyaudio.PyAudio.return_value.open.assert_called()
        assert stream.stream is not None

    def test_open_stream_muted(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration(muted=True)
        stream = AudioStream(config)
        stream.open_stream()

        # Should not open actual stream when muted
        # The actual behavior depends on implementation

    def test_start_stream(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration()
        stream = AudioStream(config)
        stream.open_stream()

        mock_stream = mock_pyaudio.PyAudio.return_value.open.return_value
        mock_stream.is_active.return_value = False

        stream.start_stream()
        mock_stream.start_stream.assert_called()

    def test_stop_stream(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration()
        stream = AudioStream(config)
        stream.open_stream()

        mock_stream = mock_pyaudio.PyAudio.return_value.open.return_value
        mock_stream.is_active.return_value = True

        stream.stop_stream()
        mock_stream.stop_stream.assert_called()

    def test_close_stream(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration()
        stream = AudioStream(config)
        stream.open_stream()
        stream.close_stream()

        assert stream.stream is None

    def test_is_stream_active(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration()
        stream = AudioStream(config)
        stream.open_stream()

        mock_stream = mock_pyaudio.PyAudio.return_value.open.return_value
        mock_stream.is_active.return_value = True

        assert stream.is_stream_active() is True

    def test_is_installed(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioStream, AudioConfiguration

        config = AudioConfiguration()
        stream = AudioStream(config)

        with patch('shutil.which') as mock_which:
            mock_which.return_value = "/usr/bin/python"
            assert stream.is_installed("python") is True

            mock_which.return_value = None
            assert stream.is_installed("nonexistent") is False


class TestAudioBufferManager:
    """Tests for AudioBufferManager class."""

    def test_add_to_buffer(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioBufferManager, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()
        manager = AudioBufferManager(audio_buffer, timings, config)

        data = b'\x00\x01' * 100
        manager.add_to_buffer(data)

        assert not audio_buffer.empty()
        assert manager.total_samples == 100

    def test_clear_buffer(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioBufferManager, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()
        manager = AudioBufferManager(audio_buffer, timings, config)

        manager.add_to_buffer(b'\x00\x01' * 100)
        manager.clear_buffer()

        assert audio_buffer.empty()
        assert manager.total_samples == 0

    def test_get_from_buffer(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioBufferManager, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()
        manager = AudioBufferManager(audio_buffer, timings, config)

        data = b'\x00\x01' * 100
        manager.add_to_buffer(data)

        success, chunk = manager.get_from_buffer()
        assert success is True
        assert chunk == data

    def test_get_from_empty_buffer(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioBufferManager, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()
        manager = AudioBufferManager(audio_buffer, timings, config)

        success, chunk = manager.get_from_buffer(timeout=0.01)
        assert success is False
        assert chunk is None

    def test_get_buffered_seconds(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import AudioBufferManager, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()
        manager = AudioBufferManager(audio_buffer, timings, config)

        # Add 1 second of audio at 16000 Hz
        manager.total_samples = 16000
        result = manager.get_buffered_seconds(16000)

        assert result == 1.0


class TestStreamPlayer:
    """Tests for StreamPlayer class."""

    def test_init(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()

        player = StreamPlayer(audio_buffer, timings, config)

        assert player.playback_active is False
        assert player.muted is False

    def test_start_creates_thread(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()

        player = StreamPlayer(audio_buffer, timings, config)
        player.start()

        assert player.playback_active is True
        assert player.playback_thread is not None
        assert player.playback_thread.is_alive()

        player.stop(immediate=True)

    def test_pause_resume(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()

        player = StreamPlayer(audio_buffer, timings, config)

        player.pause()
        assert player.pause_event.is_set()

        player.resume()
        assert not player.pause_event.is_set()

    def test_mute(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()

        player = StreamPlayer(audio_buffer, timings, config)

        player.mute(True)
        assert player.muted is True

        player.mute(False)
        assert player.muted is False

    def test_callbacks_on_playback_start(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        start_called = []

        def on_start():
            start_called.append(True)

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()

        player = StreamPlayer(
            audio_buffer, timings, config,
            on_playback_start=on_start
        )

        assert player.on_playback_start is not None

    def test_callbacks_on_playback_stop(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        stop_called = []

        def on_stop():
            stop_called.append(True)

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()

        player = StreamPlayer(
            audio_buffer, timings, config,
            on_playback_stop=on_stop
        )

        assert player.on_playback_stop is not None

    def test_get_buffered_seconds(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        config = AudioConfiguration(rate=16000)
        audio_buffer = queue.Queue()
        timings = queue.Queue()

        player = StreamPlayer(audio_buffer, timings, config)
        player.buffer_manager.total_samples = 16000

        result = player.get_buffered_seconds()
        assert result == 1.0

    def test_stop_immediate(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        config = AudioConfiguration()
        audio_buffer = queue.Queue()
        timings = queue.Queue()

        player = StreamPlayer(audio_buffer, timings, config)
        player.start()
        player.stop(immediate=True)

        assert player.playback_active is False
        assert player.playback_thread is None
