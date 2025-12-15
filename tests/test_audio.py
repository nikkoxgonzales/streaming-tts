"""Tests for audio playback module."""

from unittest.mock import MagicMock, patch

import pytest

from streaming_tts.config import PlaybackConfig


class TestAudioPlayer:
    """Tests for AudioPlayer class."""

    @pytest.fixture
    def mock_pyaudio(self):
        """Mock PyAudio module."""
        with patch.dict("sys.modules", {"pyaudio": MagicMock()}):
            # Reload the audio module to pick up the mock
            import importlib

            import streaming_tts.audio as audio_module

            # Mock the module-level check
            audio_module.PYAUDIO_AVAILABLE = True
            audio_module.pyaudio = MagicMock()

            # Create mock stream
            mock_stream = MagicMock()
            mock_stream.is_active.return_value = True

            # Create mock PyAudio instance
            mock_pa = MagicMock()
            mock_pa.open.return_value = mock_stream
            mock_pa.get_device_count.return_value = 2
            mock_pa.get_device_info_by_index.side_effect = [
                {"name": "Speaker", "maxOutputChannels": 2, "defaultSampleRate": 44100},
                {"name": "Headphones", "maxOutputChannels": 2, "defaultSampleRate": 48000},
            ]

            audio_module.pyaudio.PyAudio.return_value = mock_pa
            audio_module.pyaudio.paInt16 = 8

            yield audio_module, mock_pa, mock_stream

    def test_init(self, mock_pyaudio) -> None:
        """Test AudioPlayer initialization."""
        audio_module, _, _ = mock_pyaudio
        config = PlaybackConfig()

        player = audio_module.AudioPlayer(config, sample_rate=24000)
        assert player._sample_rate == 24000
        assert player._channels == 1
        assert not player._started

    def test_start(self, mock_pyaudio) -> None:
        """Test starting audio stream."""
        audio_module, mock_pa, _ = mock_pyaudio
        config = PlaybackConfig()

        player = audio_module.AudioPlayer(config)
        player.start()

        assert player._started
        mock_pa.open.assert_called_once()

    def test_start_idempotent(self, mock_pyaudio) -> None:
        """Test that start() only opens stream once."""
        audio_module, mock_pa, _ = mock_pyaudio
        config = PlaybackConfig()

        player = audio_module.AudioPlayer(config)
        player.start()
        player.start()

        assert mock_pa.open.call_count == 1

    def test_write(self, mock_pyaudio) -> None:
        """Test writing audio data."""
        audio_module, _, mock_stream = mock_pyaudio
        config = PlaybackConfig()

        player = audio_module.AudioPlayer(config)
        player.start()
        player.write(b"audio data")

        mock_stream.write.assert_called_once_with(b"audio data")

    def test_write_muted(self, mock_pyaudio) -> None:
        """Test that write is skipped when muted."""
        audio_module, _, mock_stream = mock_pyaudio
        config = PlaybackConfig(muted=True)

        player = audio_module.AudioPlayer(config)
        player.start()
        player.write(b"audio data")

        mock_stream.write.assert_not_called()

    def test_write_auto_starts(self, mock_pyaudio) -> None:
        """Test that write auto-starts the stream."""
        audio_module, mock_pa, _ = mock_pyaudio
        config = PlaybackConfig()

        player = audio_module.AudioPlayer(config)
        player.write(b"audio data")

        assert player._started
        mock_pa.open.assert_called_once()

    def test_stop(self, mock_pyaudio) -> None:
        """Test stopping audio stream."""
        audio_module, mock_pa, mock_stream = mock_pyaudio
        config = PlaybackConfig()

        player = audio_module.AudioPlayer(config)
        player.start()
        player.stop()

        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pa.terminate.assert_called_once()
        assert not player._started

    def test_is_playing(self, mock_pyaudio) -> None:
        """Test is_playing check."""
        audio_module, _, mock_stream = mock_pyaudio
        config = PlaybackConfig()

        player = audio_module.AudioPlayer(config)
        assert not player.is_playing()

        player.start()
        assert player.is_playing()

        mock_stream.is_active.return_value = False
        assert not player.is_playing()

    def test_context_manager(self, mock_pyaudio) -> None:
        """Test using AudioPlayer as context manager."""
        audio_module, mock_pa, mock_stream = mock_pyaudio
        config = PlaybackConfig()

        with audio_module.AudioPlayer(config) as player:
            player.write(b"audio data")

        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_get_devices(self, mock_pyaudio) -> None:
        """Test getting available devices."""
        audio_module, _, _ = mock_pyaudio
        config = PlaybackConfig()

        player = audio_module.AudioPlayer(config)
        devices = player.get_devices()

        assert len(devices) == 2
        assert devices[0]["name"] == "Speaker"
        assert devices[1]["name"] == "Headphones"

    def test_custom_device_index(self, mock_pyaudio) -> None:
        """Test using custom device index."""
        audio_module, mock_pa, _ = mock_pyaudio
        config = PlaybackConfig(device_index=1)

        player = audio_module.AudioPlayer(config)
        player.start()

        call_kwargs = mock_pa.open.call_args[1]
        assert call_kwargs["output_device_index"] == 1


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_is_playback_available_with_pyaudio(self) -> None:
        """Test is_playback_available when PyAudio is installed."""
        with patch.dict("sys.modules", {"pyaudio": MagicMock()}):
            import streaming_tts.audio as audio_module

            audio_module.PYAUDIO_AVAILABLE = True
            assert audio_module.is_playback_available()

    def test_is_playback_available_without_pyaudio(self) -> None:
        """Test is_playback_available when PyAudio is not installed."""
        import streaming_tts.audio as audio_module

        original = audio_module.PYAUDIO_AVAILABLE
        audio_module.PYAUDIO_AVAILABLE = False
        try:
            assert not audio_module.is_playback_available()
        finally:
            audio_module.PYAUDIO_AVAILABLE = original

    def test_list_audio_devices_without_pyaudio(self) -> None:
        """Test list_audio_devices when PyAudio is not installed."""
        import streaming_tts.audio as audio_module

        original = audio_module.PYAUDIO_AVAILABLE
        audio_module.PYAUDIO_AVAILABLE = False
        try:
            devices = audio_module.list_audio_devices()
            assert devices == []
        finally:
            audio_module.PYAUDIO_AVAILABLE = original


class TestPlaybackConfig:
    """Tests for PlaybackConfig with audio module."""

    def test_frames_per_buffer(self, ) -> None:
        """Test custom frames_per_buffer."""
        config = PlaybackConfig(frames_per_buffer=1024)
        assert config.frames_per_buffer == 1024

    def test_muted_config(self) -> None:
        """Test muted configuration."""
        config = PlaybackConfig(muted=True)
        assert config.muted is True
