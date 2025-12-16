"""Tests for streaming_tts.base_engine module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from streaming_tts.base_engine import BaseEngine, TimingInfo


class ConcreteEngine(BaseEngine):
    """Concrete implementation of BaseEngine for testing."""

    def __init__(self):
        self._sample_rate = 24000

    def get_stream_info(self):
        return (8, 1, self._sample_rate)  # pyaudio.paInt16 = 8

    def get_voices(self):
        return ["voice1", "voice2"]

    def set_voice(self, voice):
        self.current_voice = voice

    def set_voice_parameters(self, **params):
        self.voice_params = params


class TestTimingInfo:
    """Tests for TimingInfo class."""

    def test_init(self):
        timing = TimingInfo(start_time=0.0, end_time=1.0, word="hello")
        assert timing.start_time == 0.0
        assert timing.end_time == 1.0
        assert timing.word == "hello"

    def test_str_representation(self):
        timing = TimingInfo(start_time=0.5, end_time=1.5, word="world")
        result = str(timing)
        assert "world" in result
        assert "0.5" in result
        assert "1.5" in result
        assert "Word" in result
        assert "Start Time" in result
        assert "End Time" in result


class TestBaseEngine:
    """Tests for BaseEngine abstract class methods."""

    @pytest.fixture
    def engine(self):
        """Create a concrete engine instance for testing."""
        return ConcreteEngine()

    def test_engine_name_default(self, engine):
        assert engine.engine_name == "unknown"

    def test_can_consume_generators_default(self, engine):
        assert engine.can_consume_generators is False

    def test_queue_initialized(self, engine):
        assert engine.queue is not None
        assert engine.queue.empty()

    def test_timings_queue_initialized(self, engine):
        assert engine.timings is not None
        assert engine.timings.empty()

    def test_reset_audio_duration(self, engine):
        engine.audio_duration = 10.5
        engine.reset_audio_duration()
        assert engine.audio_duration == 0

    def test_verify_sample_rate_provided(self, engine):
        result = engine.verify_sample_rate(16000)
        assert result == 16000

    def test_verify_sample_rate_from_stream_info(self, engine):
        result = engine.verify_sample_rate(-1)
        assert result == 24000  # From get_stream_info

    def test_apply_fade_in(self, engine):
        audio = np.ones(24000, dtype=np.float32)
        result = engine.apply_fade_in(audio, fade_duration_ms=15)

        # First samples should be quieter
        assert result[0] < audio[0]
        # Later samples should be near original
        assert np.isclose(result[-1], audio[-1])

    def test_apply_fade_in_preserves_length(self, engine):
        audio = np.ones(24000, dtype=np.float32)
        result = engine.apply_fade_in(audio)
        assert len(result) == len(audio)

    def test_apply_fade_in_short_audio(self, engine):
        # Audio shorter than fade duration
        audio = np.ones(100, dtype=np.float32)
        result = engine.apply_fade_in(audio, fade_duration_ms=100)
        # Should handle gracefully
        assert len(result) == len(audio)

    def test_apply_fade_out(self, engine):
        audio = np.ones(24000, dtype=np.float32)
        result = engine.apply_fade_out(audio, fade_duration_ms=15)

        # Last samples should be quieter
        assert result[-1] < audio[-1]
        # Earlier samples should be near original
        assert np.isclose(result[0], audio[0])

    def test_apply_fade_out_preserves_length(self, engine):
        audio = np.ones(24000, dtype=np.float32)
        result = engine.apply_fade_out(audio)
        assert len(result) == len(audio)

    def test_apply_fade_out_short_audio(self, engine):
        audio = np.ones(100, dtype=np.float32)
        result = engine.apply_fade_out(audio, fade_duration_ms=100)
        assert len(result) == len(audio)

    def test_trim_silence_start(self, engine):
        # Create audio with leading silence
        audio = np.zeros(24000, dtype=np.float32)
        audio[12000:] = 0.5  # Second half has audio

        result = engine.trim_silence_start(audio, silence_threshold=0.01)
        # Should be shorter (leading silence trimmed)
        assert len(result) < len(audio)

    def test_trim_silence_start_no_silence(self, engine):
        # Audio with no leading silence
        audio = np.ones(24000, dtype=np.float32) * 0.5
        result = engine.trim_silence_start(audio, silence_threshold=0.01)
        # Length should be similar (minus extra_ms trimming)
        assert len(result) > 0

    def test_trim_silence_end(self, engine):
        # Create audio with trailing silence
        audio = np.zeros(24000, dtype=np.float32)
        audio[:12000] = 0.5  # First half has audio

        result = engine.trim_silence_end(audio, silence_threshold=0.01)
        # Should be shorter (trailing silence trimmed)
        assert len(result) < len(audio)

    def test_trim_silence_both_ends(self, engine):
        # Audio with silence at both ends
        audio = np.zeros(48000, dtype=np.float32)
        audio[12000:36000] = 0.5  # Middle has audio

        result = engine._trim_silence(audio, silence_threshold=0.01)
        # Should trim both ends
        assert len(result) < len(audio)

    def test_is_installed_with_existing_command(self, engine):
        # 'python' should be installed
        with patch('shutil.which') as mock_which:
            mock_which.return_value = "/usr/bin/python"
            assert engine.is_installed("python") is True

    def test_is_installed_with_missing_command(self, engine):
        with patch('shutil.which') as mock_which:
            mock_which.return_value = None
            assert engine.is_installed("nonexistent_command_xyz") is False

    def test_stop_sets_event(self, engine):
        engine.stop_synthesis_event.clear()
        engine.stop()
        assert engine.stop_synthesis_event.is_set()

    def test_synthesize_clears_stop_event(self, engine):
        engine.stop_synthesis_event.set()
        engine.synthesize("test")
        assert not engine.stop_synthesis_event.is_set()

    def test_get_stream_info(self, engine):
        format_type, channels, sample_rate = engine.get_stream_info()
        assert format_type == 8
        assert channels == 1
        assert sample_rate == 24000

    def test_get_voices(self, engine):
        voices = engine.get_voices()
        assert "voice1" in voices
        assert "voice2" in voices

    def test_set_voice(self, engine):
        engine.set_voice("voice1")
        assert engine.current_voice == "voice1"

    def test_set_voice_parameters(self, engine):
        engine.set_voice_parameters(speed=1.5, pitch=0.8)
        assert engine.voice_params["speed"] == 1.5
        assert engine.voice_params["pitch"] == 0.8

    def test_shutdown_exists(self, engine):
        # Should not raise
        engine.shutdown()


class TestBaseEngineFadeEffects:
    """Additional tests for fade effects."""

    @pytest.fixture
    def engine(self):
        return ConcreteEngine()

    def test_fade_in_linear_ramp(self, engine):
        audio = np.ones(1000, dtype=np.float32)
        result = engine.apply_fade_in(audio, sample_rate=24000, fade_duration_ms=15)

        fade_samples = int(24000 * 15 / 1000)  # 360 samples
        # Check that fade ramps from 0 to 1
        assert result[0] < 0.01  # Near zero at start
        # End of fade should be close to original
        if fade_samples < len(result):
            assert result[fade_samples - 1] > 0.9

    def test_fade_out_linear_ramp(self, engine):
        audio = np.ones(1000, dtype=np.float32)
        result = engine.apply_fade_out(audio, sample_rate=24000, fade_duration_ms=15)

        # Check that fade ramps from 1 to 0
        assert result[-1] < 0.01  # Near zero at end

    def test_fade_does_not_modify_original(self, engine):
        original = np.ones(1000, dtype=np.float32)
        original_copy = original.copy()
        engine.apply_fade_in(original, sample_rate=24000)
        np.testing.assert_array_equal(original, original_copy)

    def test_fade_out_does_not_modify_original(self, engine):
        original = np.ones(1000, dtype=np.float32)
        original_copy = original.copy()
        engine.apply_fade_out(original, sample_rate=24000)
        np.testing.assert_array_equal(original, original_copy)
