"""Tests for streaming_tts.kokoro_engine module."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Mock heavy dependencies before importing the module
with patch.dict('sys.modules', {
    'kokoro': MagicMock(),
    'torch': MagicMock(),
}):
    from streaming_tts.kokoro_engine import (
        get_lang_code_from_voice,
        KokoroVoice,
    )


class TestGetLangCodeFromVoice:
    """Tests for get_lang_code_from_voice function."""

    def test_american_english_female(self):
        assert get_lang_code_from_voice("af_heart") == "a"
        assert get_lang_code_from_voice("af_sarah") == "a"

    def test_american_english_male(self):
        assert get_lang_code_from_voice("am_adam") == "a"
        assert get_lang_code_from_voice("am_michael") == "a"

    def test_british_english_female(self):
        assert get_lang_code_from_voice("bf_alice") == "b"
        assert get_lang_code_from_voice("bf_emma") == "b"

    def test_british_english_male(self):
        assert get_lang_code_from_voice("bm_daniel") == "b"
        assert get_lang_code_from_voice("bm_george") == "b"

    def test_japanese_female(self):
        assert get_lang_code_from_voice("jf_alpha") == "j"

    def test_japanese_male(self):
        assert get_lang_code_from_voice("jm_kumo") == "j"

    def test_chinese_female(self):
        assert get_lang_code_from_voice("zf_xiaobei") == "z"

    def test_chinese_male(self):
        assert get_lang_code_from_voice("zm_yunjian") == "z"

    def test_spanish_female(self):
        assert get_lang_code_from_voice("ef_dora") == "e"

    def test_spanish_male(self):
        assert get_lang_code_from_voice("em_alex") == "e"

    def test_french_female(self):
        assert get_lang_code_from_voice("ff_siwis") == "f"

    def test_hindi_female(self):
        assert get_lang_code_from_voice("hf_alpha") == "h"

    def test_hindi_male(self):
        assert get_lang_code_from_voice("hm_omega") == "h"

    def test_italian_female(self):
        assert get_lang_code_from_voice("if_sara") == "i"

    def test_italian_male(self):
        assert get_lang_code_from_voice("im_nicola") == "i"

    def test_portuguese_female(self):
        assert get_lang_code_from_voice("pf_dora") == "p"

    def test_portuguese_male(self):
        assert get_lang_code_from_voice("pm_alex") == "p"

    def test_formula_voice_simple(self):
        result = get_lang_code_from_voice("0.3*af_sarah + 0.7*am_adam")
        assert result == "a"

    def test_formula_voice_british(self):
        result = get_lang_code_from_voice("0.5*bf_alice + 0.5*bm_daniel")
        assert result == "b"

    def test_fallback_unknown_prefix(self):
        result = get_lang_code_from_voice("unknown_voice")
        assert result == "a"  # Falls back to American English

    def test_empty_string(self):
        result = get_lang_code_from_voice("")
        # Empty string returns empty string (no fallback applied for empty input)
        assert result == ""


class TestKokoroVoice:
    """Tests for KokoroVoice class."""

    def test_init_with_name_only(self):
        voice = KokoroVoice(name="af_heart")
        assert voice.name == "af_heart"
        assert voice.language_code == "a"

    def test_init_with_explicit_lang(self):
        voice = KokoroVoice(name="custom_voice", language_code="z")
        assert voice.name == "custom_voice"
        assert voice.language_code == "z"

    def test_language_code_auto_detection(self):
        voice = KokoroVoice(name="bf_alice")
        assert voice.language_code == "b"

    def test_repr(self):
        voice = KokoroVoice(name="af_heart")
        result = repr(voice)
        assert "af_heart" in result
        assert "a" in result


@pytest.fixture
def mock_kokoro_dependencies():
    """Mock all Kokoro engine dependencies."""
    with patch('streaming_tts.kokoro_engine.KPipeline') as mock_pipeline, \
         patch('streaming_tts.kokoro_engine.torch') as mock_torch, \
         patch('streaming_tts.kokoro_engine.pyaudio') as mock_pyaudio:

        mock_pyaudio.paInt16 = 8

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        yield {
            'pipeline': mock_pipeline,
            'pipeline_instance': mock_pipeline_instance,
            'torch': mock_torch,
            'pyaudio': mock_pyaudio,
        }


class TestKokoroEngineInit:
    """Tests for KokoroEngine initialization."""

    def test_init_validates_speed_zero(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        with pytest.raises(ValueError, match="default_speed must be positive"):
            KokoroEngine(default_speed=0)

    def test_init_validates_speed_negative(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        with pytest.raises(ValueError, match="default_speed must be positive"):
            KokoroEngine(default_speed=-1.0)

    def test_init_validates_silence_threshold_high(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        with pytest.raises(ValueError, match="silence_threshold must be between 0 and 1"):
            KokoroEngine(silence_threshold=1.5)

    def test_init_validates_silence_threshold_low(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        with pytest.raises(ValueError, match="silence_threshold must be between 0 and 1"):
            KokoroEngine(silence_threshold=-0.1)

    def test_init_validates_extra_start_ms(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        with pytest.raises(ValueError, match="extra_start_ms must be non-negative"):
            KokoroEngine(extra_start_ms=-5)

    def test_init_validates_extra_end_ms(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        with pytest.raises(ValueError, match="extra_end_ms must be non-negative"):
            KokoroEngine(extra_end_ms=-5)

    def test_init_validates_fade_in_ms(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        with pytest.raises(ValueError, match="fade_in_ms must be non-negative"):
            KokoroEngine(fade_in_ms=-1)

    def test_init_validates_fade_out_ms(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        with pytest.raises(ValueError, match="fade_out_ms must be non-negative"):
            KokoroEngine(fade_out_ms=-1)

    def test_init_default_values(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        assert engine.speed == 1.0
        assert engine.trim_silence is True
        # engine_name is set in __init__ after super().__init__()
        # but BaseEngine.__init__ sets it to "unknown" and then KokoroEngine sets it to "kokoro"
        assert engine.engine_name in ("unknown", "kokoro")

    def test_init_custom_values(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine(
            default_speed=1.5,
            trim_silence=False,
            silence_threshold=0.01
        )
        assert engine.speed == 1.5
        assert engine.trim_silence is False
        assert engine.silence_threshold == 0.01


class TestKokoroEngineVoices:
    """Tests for KokoroEngine voice handling."""

    def test_get_stream_info(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        format_type, channels, sample_rate = engine.get_stream_info()

        assert format_type == 8  # paInt16
        assert channels == 1
        assert sample_rate == 24000

    def test_get_voices(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        voices = engine.get_voices()

        assert len(voices) > 0
        assert all(isinstance(v, KokoroVoice) for v in voices)
        # Check some known voices exist
        voice_names = [v.name for v in voices]
        assert "af_heart" in voice_names
        assert "am_adam" in voice_names
        assert "bf_alice" in voice_names

    def test_set_voice_string(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        engine.set_voice("af_heart")
        assert engine.current_voice == "af_heart"
        assert engine.current_lang == "a"

    def test_set_voice_object(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        voice = KokoroVoice(name="bf_emma", language_code="b")
        engine.set_voice(voice)
        assert engine.current_voice == "bf_emma"

    def test_set_voice_partial_match(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        engine.set_voice("heart")  # Partial match for "af_heart"
        assert "heart" in engine.current_voice.lower()

    def test_set_speed(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        engine.set_speed(1.5)
        assert engine.speed == 1.5

    def test_set_voice_parameters_speed(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        engine.set_voice_parameters(speed=2.0)
        assert engine.speed == 2.0


class TestKokoroEngineSynthesis:
    """Tests for KokoroEngine synthesis."""

    def test_synthesize_calls_pipeline(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        # Setup mock result
        mock_result = MagicMock()
        mock_result.audio.cpu.return_value.numpy.return_value = np.zeros(24000, dtype=np.float32)
        mock_result.graphemes = "Hello"
        mock_result.phonemes = "həˈloʊ"
        mock_result.tokens = []

        mock_kokoro_dependencies['pipeline_instance'].return_value = [mock_result]

        engine = KokoroEngine()
        result = engine.synthesize("Hello")

        assert result is True
        mock_kokoro_dependencies['pipeline_instance'].assert_called()

    def test_synthesize_puts_audio_in_queue(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        # Setup mock result
        mock_result = MagicMock()
        mock_result.audio.cpu.return_value.numpy.return_value = np.random.randn(24000).astype(np.float32)
        mock_result.graphemes = "Hello"
        mock_result.phonemes = "həˈloʊ"
        mock_result.tokens = []

        mock_kokoro_dependencies['pipeline_instance'].return_value = [mock_result]

        engine = KokoroEngine()
        engine.synthesize("Hello")

        assert not engine.queue.empty()

    def test_shutdown(self, mock_kokoro_dependencies):
        from streaming_tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        # Should not raise
        engine.shutdown()
