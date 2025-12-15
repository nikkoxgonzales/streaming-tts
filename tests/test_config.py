"""Tests for configuration dataclasses."""

import pytest

from streaming_tts.config import (
    KOKORO_VOICES,
    PlaybackConfig,
    TTSConfig,
    get_lang_code,
)


class TestTTSConfig:
    """Tests for TTSConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TTSConfig()
        assert config.voice == "af_heart"
        assert config.speed == 1.0
        assert config.sample_rate == 24000
        assert config.channels == 1
        assert config.trim_silence is True
        assert config.silence_threshold == 0.005
        assert config.fade_in_ms == 10
        assert config.fade_out_ms == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TTSConfig(
            voice="am_adam",
            speed=1.5,
            trim_silence=False,
        )
        assert config.voice == "am_adam"
        assert config.speed == 1.5
        assert config.trim_silence is False

    def test_immutable(self) -> None:
        """Test that config is immutable (frozen)."""
        config = TTSConfig()
        with pytest.raises(AttributeError):
            config.voice = "am_adam"  # type: ignore[misc]

    def test_invalid_speed(self) -> None:
        """Test that invalid speed raises ValueError."""
        with pytest.raises(ValueError, match="speed must be positive"):
            TTSConfig(speed=0)
        with pytest.raises(ValueError, match="speed must be positive"):
            TTSConfig(speed=-1.0)

    def test_invalid_sample_rate(self) -> None:
        """Test that invalid sample_rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TTSConfig(sample_rate=0)

    def test_invalid_channels(self) -> None:
        """Test that invalid channels raises ValueError."""
        with pytest.raises(ValueError, match="channels must be 1 or 2"):
            TTSConfig(channels=3)

    def test_invalid_silence_threshold(self) -> None:
        """Test that negative silence_threshold raises ValueError."""
        with pytest.raises(ValueError, match="silence_threshold must be non-negative"):
            TTSConfig(silence_threshold=-0.1)


class TestPlaybackConfig:
    """Tests for PlaybackConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default playback configuration values."""
        config = PlaybackConfig()
        assert config.device_index is None
        assert config.frames_per_buffer == 512
        assert config.muted is False

    def test_custom_values(self) -> None:
        """Test custom playback configuration values."""
        config = PlaybackConfig(
            device_index=1,
            frames_per_buffer=1024,
            muted=True,
        )
        assert config.device_index == 1
        assert config.frames_per_buffer == 1024
        assert config.muted is True


class TestGetLangCode:
    """Tests for get_lang_code function."""

    @pytest.mark.parametrize(
        ("voice", "expected"),
        [
            # American English
            ("af_heart", "a"),
            ("am_adam", "a"),
            # British English
            ("bf_alice", "b"),
            ("bm_daniel", "b"),
            # Japanese
            ("jf_alpha", "j"),
            ("jm_kumo", "j"),
            # Mandarin Chinese
            ("zf_xiaobei", "z"),
            ("zm_yunjian", "z"),
            # Spanish
            ("ef_dora", "e"),
            ("em_alex", "e"),
            # French
            ("ff_siwis", "f"),
            # Hindi
            ("hf_alpha", "h"),
            ("hm_omega", "h"),
            # Italian
            ("if_sara", "i"),
            ("im_nicola", "i"),
            # Brazilian Portuguese
            ("pf_dora", "p"),
            ("pm_alex", "p"),
        ],
    )
    def test_single_voice(self, voice: str, expected: str) -> None:
        """Test language detection for single voices."""
        assert get_lang_code(voice) == expected

    def test_blend_formula(self) -> None:
        """Test language detection for blend formulas."""
        assert get_lang_code("0.3*af_sarah + 0.7*am_adam") == "a"
        assert get_lang_code("0.5*bf_alice + 0.5*bm_daniel") == "b"

    def test_unknown_voice(self) -> None:
        """Test fallback for unknown voices."""
        assert get_lang_code("unknown") == "a"
        assert get_lang_code("") == "a"


class TestKokoroVoices:
    """Tests for KOKORO_VOICES constant."""

    def test_voices_not_empty(self) -> None:
        """Test that voices list is not empty."""
        assert len(KOKORO_VOICES) > 0

    def test_af_heart_in_voices(self) -> None:
        """Test that default voice is in the list."""
        assert "af_heart" in KOKORO_VOICES

    def test_all_voices_have_valid_lang_code(self) -> None:
        """Test that all voices have valid language codes."""
        valid_codes = {"a", "b", "j", "z", "e", "f", "h", "i", "p"}
        for voice in KOKORO_VOICES:
            code = get_lang_code(voice)
            assert code in valid_codes, f"Voice {voice} has invalid lang code {code}"
