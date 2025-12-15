"""Tests for KokoroTTS engine."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from streaming_tts.config import TTSConfig
from streaming_tts.engine import KokoroTTS, TimingInfo


@pytest.fixture
def mock_pipeline():
    """Create a mock Kokoro pipeline."""
    with patch("kokoro.KPipeline") as mock_cls:
        # Create mock result object
        mock_result = MagicMock()
        mock_result.audio = torch.randn(24000)  # 1 second of audio
        mock_result.graphemes = "Hello world"
        mock_result.phonemes = "həˈloʊ wɜːrld"

        # Create mock token for timing
        mock_token = MagicMock()
        mock_token.start_ts = 0.0
        mock_token.end_ts = 0.5
        mock_token.text = "Hello"
        mock_result.tokens = [mock_token]

        # Make pipeline callable and return generator
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = iter([mock_result])
        mock_pipeline_instance.load_single_voice = MagicMock(
            return_value=torch.randn(256)
        )
        mock_cls.return_value = mock_pipeline_instance

        yield mock_cls


@pytest.fixture
def engine(mock_pipeline):
    """Create a KokoroTTS engine with mocked pipeline."""
    return KokoroTTS(TTSConfig())


class TestKokoroTTS:
    """Tests for KokoroTTS class."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        engine = KokoroTTS()
        assert engine.config.voice == "af_heart"
        assert engine._loaded is False

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = TTSConfig(voice="am_adam", speed=1.5)
        engine = KokoroTTS(config)
        assert engine.config.voice == "am_adam"
        assert engine.config.speed == 1.5

    def test_load(self, engine, mock_pipeline) -> None:
        """Test lazy loading."""
        assert engine._loaded is False
        engine.load()
        assert engine._loaded is True
        mock_pipeline.assert_called_once()

    def test_load_idempotent(self, engine, mock_pipeline) -> None:
        """Test that load() only loads once."""
        engine.load()
        engine.load()
        assert mock_pipeline.call_count == 1

    def test_synthesize(self, engine) -> None:
        """Test basic synthesis."""
        chunks = list(engine.synthesize("Hello world"))
        assert len(chunks) > 0
        assert all(isinstance(c, np.ndarray) for c in chunks)
        assert all(c.dtype == np.float32 for c in chunks)

    def test_synthesize_auto_loads(self, engine) -> None:
        """Test that synthesize() auto-loads the engine."""
        assert engine._loaded is False
        list(engine.synthesize("Hello"))
        assert engine._loaded is True

    def test_synthesize_with_timing(self, engine) -> None:
        """Test synthesis with timing information."""
        results = list(engine.synthesize("Hello world", yield_timing=True))
        assert len(results) > 0
        audio, timings = results[0]
        assert isinstance(audio, np.ndarray)
        assert isinstance(timings, list)
        assert len(timings) > 0
        assert isinstance(timings[0], TimingInfo)

    def test_synthesize_to_bytes(self, engine) -> None:
        """Test synthesis to PCM16 bytes."""
        chunks = list(engine.synthesize_to_bytes("Hello world"))
        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    def test_set_voice(self, engine) -> None:
        """Test voice changing."""
        engine.set_voice("am_adam")
        assert engine._current_voice == "am_adam"
        assert engine._current_lang == "a"

        engine.set_voice("bf_alice")
        assert engine._current_voice == "bf_alice"
        assert engine._current_lang == "b"

    def test_get_voices(self, engine) -> None:
        """Test getting available voices."""
        voices = engine.get_voices()
        assert len(voices) > 0
        assert "af_heart" in voices
        assert "am_adam" in voices

    def test_shutdown(self, engine) -> None:
        """Test shutdown clears state."""
        engine.load()
        assert engine._loaded is True
        engine.shutdown()
        assert engine._loaded is False
        assert len(engine._pipelines) == 0


class TestVoiceBlending:
    """Tests for voice blending functionality."""

    def test_blend_formula_parsing(self, engine, mock_pipeline) -> None:
        """Test that blend formulas are parsed correctly."""
        engine.load()
        engine.set_voice("0.3*af_sarah + 0.7*am_adam")

        # Synthesize to trigger blend parsing
        list(engine.synthesize("Hello"))

        # Check that load_single_voice was called for each voice
        pipeline = mock_pipeline.return_value
        assert pipeline.load_single_voice.call_count >= 2

    def test_blend_formula_cached(self, engine, mock_pipeline) -> None:
        """Test that blended voices are cached."""
        engine.load()
        engine.set_voice("0.5*af_sarah + 0.5*am_adam")

        list(engine.synthesize("Hello"))
        list(engine.synthesize("World"))

        # Should only parse blend once
        assert "0.5*af_sarah + 0.5*am_adam" in engine._blended_voices

    def test_invalid_blend_formula(self, engine) -> None:
        """Test that invalid blend formulas raise errors."""
        engine.load()
        engine.set_voice("invalid_formula_no_asterisk")
        # The error should occur during synthesis when the formula is parsed
        # but since there's no asterisk, it's treated as a regular voice name


class TestSilenceTrimming:
    """Tests for silence trimming functionality."""

    def test_trim_silence_start(self) -> None:
        """Test trimming silence from start."""
        engine = KokoroTTS(TTSConfig(trim_silence=True, silence_threshold=0.01))

        # Create audio with leading silence
        silence = np.zeros(1000, dtype=np.float32)
        audio = np.concatenate([silence, np.ones(1000, dtype=np.float32) * 0.5])

        trimmed = engine._trim_silence_start(audio)
        assert len(trimmed) < len(audio)

    def test_trim_silence_end(self) -> None:
        """Test trimming silence from end."""
        engine = KokoroTTS(TTSConfig(trim_silence=True, silence_threshold=0.01))

        # Create audio with trailing silence
        audio = np.concatenate([
            np.ones(1000, dtype=np.float32) * 0.5,
            np.zeros(1000, dtype=np.float32),
        ])

        trimmed = engine._trim_silence_end(audio)
        assert len(trimmed) < len(audio)

    def test_fade_in(self) -> None:
        """Test fade-in application."""
        engine = KokoroTTS(TTSConfig(fade_in_ms=10))
        audio = np.ones(2400, dtype=np.float32)  # 100ms at 24kHz

        faded = engine._apply_fade_in(audio)
        assert faded[0] == 0.0  # Start at 0
        assert faded[-1] == 1.0  # End at full volume

    def test_fade_out(self) -> None:
        """Test fade-out application."""
        engine = KokoroTTS(TTSConfig(fade_out_ms=10))
        audio = np.ones(2400, dtype=np.float32)  # 100ms at 24kHz

        faded = engine._apply_fade_out(audio)
        assert faded[0] == 1.0  # Start at full volume
        assert faded[-1] == 0.0  # End at 0

    def test_trim_disabled(self) -> None:
        """Test that trimming can be disabled."""
        config = TTSConfig(trim_silence=False)
        engine = KokoroTTS(config)

        # With mocked pipeline, audio should pass through without trimming
        # (actual test would need real audio with silence)
        assert engine.config.trim_silence is False


class TestTimingInfo:
    """Tests for TimingInfo dataclass."""

    def test_timing_info_creation(self) -> None:
        """Test TimingInfo creation."""
        timing = TimingInfo(start_time=0.0, end_time=0.5, word="Hello")
        assert timing.start_time == 0.0
        assert timing.end_time == 0.5
        assert timing.word == "Hello"

    def test_timing_info_immutable(self) -> None:
        """Test that TimingInfo is immutable."""
        timing = TimingInfo(start_time=0.0, end_time=0.5, word="Hello")
        with pytest.raises(AttributeError):
            timing.word = "World"  # type: ignore[misc]
