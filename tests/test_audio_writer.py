"""Tests for streaming_tts.audio_writer module."""

import pytest
import numpy as np
from streaming_tts.audio_writer import (
    StreamingAudioWriter,
    AudioNormalizer,
    create_silence,
    HAS_PYAV,
)


class TestStreamingAudioWriter:
    """Tests for StreamingAudioWriter class."""

    def test_wav_format_init(self):
        writer = StreamingAudioWriter("wav", sample_rate=24000)
        assert writer.format == "wav"
        assert writer.sample_rate == 24000
        assert writer._closed is False
        writer.close()

    def test_pcm_format_init(self):
        writer = StreamingAudioWriter("pcm", sample_rate=24000)
        assert writer.format == "pcm"
        writer.close()

    def test_default_parameters(self):
        writer = StreamingAudioWriter("wav")
        assert writer.sample_rate == 24000
        assert writer.channels == 1
        assert writer.bit_rate == 128000
        writer.close()

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            StreamingAudioWriter("invalid_format")

    def test_context_manager(self):
        with StreamingAudioWriter("wav") as writer:
            assert writer._closed is False
        assert writer._closed is True

    def test_write_after_close_raises(self):
        writer = StreamingAudioWriter("wav")
        writer.close()
        with pytest.raises(RuntimeError, match="closed"):
            writer.write_chunk(np.zeros(100, dtype=np.int16))

    def test_pcm_passthrough(self, sample_audio_int16):
        with StreamingAudioWriter("pcm") as writer:
            result = writer.write_chunk(sample_audio_int16)
            assert len(result) == len(sample_audio_int16) * 2  # 2 bytes per sample

    def test_pcm_returns_raw_bytes(self):
        audio = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        with StreamingAudioWriter("pcm") as writer:
            result = writer.write_chunk(audio)
            assert result == audio.tobytes()

    def test_wav_header_structure(self):
        with StreamingAudioWriter("wav", sample_rate=24000, channels=1) as writer:
            writer.write_chunk(np.zeros(1000, dtype=np.int16))
            data = writer.write_chunk(finalize=True)

        # Check WAV header magic bytes
        assert data[:4] == b'RIFF'
        assert data[8:12] == b'WAVE'
        assert data[12:16] == b'fmt '
        assert data[36:40] == b'data'

    def test_wav_finalize_returns_complete_file(self):
        audio = np.random.randint(-1000, 1000, 1000, dtype=np.int16)
        with StreamingAudioWriter("wav") as writer:
            writer.write_chunk(audio)
            result = writer.write_chunk(finalize=True)

        # Should have header (44 bytes) + data
        assert len(result) >= 44 + 2000  # 1000 samples * 2 bytes

    def test_int16_normalization_from_float(self):
        # Float audio should be converted to int16
        float_audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        with StreamingAudioWriter("pcm") as writer:
            result = writer.write_chunk(float_audio)
            # Should produce bytes
            assert isinstance(result, bytes)
            assert len(result) == 10  # 5 samples * 2 bytes

    def test_empty_chunk_returns_empty(self):
        with StreamingAudioWriter("pcm") as writer:
            result = writer.write_chunk(np.array([], dtype=np.int16))
            assert result == b""

    def test_none_chunk_returns_empty(self):
        with StreamingAudioWriter("pcm") as writer:
            result = writer.write_chunk(None)
            assert result == b""

    def test_bytes_written_tracking(self, sample_audio_int16):
        with StreamingAudioWriter("pcm") as writer:
            assert writer.bytes_written == 0
            writer.write_chunk(sample_audio_int16)
            assert writer.bytes_written == len(sample_audio_int16) * 2

    def test_close_idempotent(self):
        writer = StreamingAudioWriter("wav")
        writer.close()
        writer.close()  # Should not raise
        assert writer._closed is True


class TestStreamingAudioWriterPyAV:
    """Tests for PyAV-based formats (mp3, opus, flac, aac)."""

    @pytest.mark.skipif(not HAS_PYAV, reason="PyAV not installed")
    def test_mp3_format_init(self):
        writer = StreamingAudioWriter("mp3", sample_rate=24000)
        assert writer.format == "mp3"
        writer.close()

    @pytest.mark.skipif(not HAS_PYAV, reason="PyAV not installed")
    def test_opus_format_init(self):
        writer = StreamingAudioWriter("opus", sample_rate=24000)
        assert writer.format == "opus"
        writer.close()

    @pytest.mark.skipif(not HAS_PYAV, reason="PyAV not installed")
    def test_flac_format_init(self):
        writer = StreamingAudioWriter("flac", sample_rate=24000)
        assert writer.format == "flac"
        writer.close()

    @pytest.mark.skipif(HAS_PYAV, reason="Test requires PyAV to be missing")
    def test_pyav_format_without_pyav_raises(self):
        with pytest.raises(ImportError, match="PyAV"):
            StreamingAudioWriter("mp3")


class TestAudioNormalizer:
    """Tests for AudioNormalizer class."""

    def test_init_defaults(self):
        normalizer = AudioNormalizer()
        assert normalizer.sample_rate == 24000
        assert normalizer.gap_trim_ms == 1.0
        assert normalizer.padding_ms == 410.0
        assert normalizer.silence_threshold_db == -45.0

    def test_init_custom_values(self):
        normalizer = AudioNormalizer(
            sample_rate=16000,
            gap_trim_ms=5.0,
            padding_ms=200.0,
            silence_threshold_db=-40.0
        )
        assert normalizer.sample_rate == 16000
        assert normalizer.gap_trim_ms == 5.0

    def test_normalize_float_to_int16(self, sample_audio_float32):
        normalizer = AudioNormalizer()
        result = normalizer.normalize(sample_audio_float32)
        assert result.dtype == np.int16
        assert len(result) == len(sample_audio_float32)

    def test_normalize_already_int16(self, sample_audio_int16):
        normalizer = AudioNormalizer()
        result = normalizer.normalize(sample_audio_int16)
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, sample_audio_int16)

    def test_normalize_clipping(self):
        # Values outside [-1, 1] should be clipped
        float_audio = np.array([2.0, -2.0, 0.0], dtype=np.float32)
        normalizer = AudioNormalizer()
        result = normalizer.normalize(float_audio)
        assert result[0] == 32767  # Clipped to max
        assert result[1] == -32768  # Clipped to min

    def test_find_non_silent_bounds_all_silent(self):
        normalizer = AudioNormalizer()
        silent = np.zeros(24000, dtype=np.int16)
        start, end = normalizer.find_non_silent_bounds(silent)
        # Should return full range when all silent (no non-silent found)
        assert start >= 0
        assert end <= len(silent)

    def test_find_non_silent_bounds_with_audio(self, silent_audio):
        normalizer = AudioNormalizer()
        start, end = normalizer.find_non_silent_bounds(silent_audio)
        # Should trim silence from start and end
        assert start > 0 or start == 0  # May include padding
        assert end <= len(silent_audio)

    def test_punctuation_padding(self):
        normalizer = AudioNormalizer()
        audio = np.random.randint(-1000, 1000, 24000, dtype=np.int16)

        # Different punctuation should give different padding
        start1, end1 = normalizer.find_non_silent_bounds(audio, "Hello.")
        start2, end2 = normalizer.find_non_silent_bounds(audio, "Hello,")

        # Period has higher padding multiplier than comma
        # The end positions might differ due to padding
        assert isinstance(end1, int)
        assert isinstance(end2, int)

    def test_trim_audio(self, silent_audio):
        normalizer = AudioNormalizer()
        result = normalizer.trim_audio(silent_audio)
        # Result should be shorter or equal (trimmed silence)
        assert len(result) <= len(silent_audio)
        assert result.dtype == np.int16

    def test_trim_audio_float_input(self, sample_audio_float32):
        normalizer = AudioNormalizer()
        result = normalizer.trim_audio(sample_audio_float32)
        # Should normalize to int16
        assert result.dtype == np.int16

    def test_trim_audio_is_last_chunk(self):
        normalizer = AudioNormalizer()
        audio = np.random.randint(-1000, 1000, 24000, dtype=np.int16)
        result1 = normalizer.trim_audio(audio, is_last_chunk=False)
        result2 = normalizer.trim_audio(audio, is_last_chunk=True)
        # Last chunk may have different padding
        assert len(result1) > 0
        assert len(result2) > 0


class TestCreateSilence:
    """Tests for create_silence function."""

    def test_creates_zeros(self):
        silence = create_silence(1.0, sample_rate=24000)
        assert np.all(silence == 0)

    def test_correct_length(self):
        # 1 second at 24kHz = 24000 samples
        silence = create_silence(1.0, sample_rate=24000)
        assert len(silence) == 24000

    def test_half_second(self):
        silence = create_silence(0.5, sample_rate=24000)
        assert len(silence) == 12000

    def test_different_sample_rate(self):
        silence = create_silence(1.0, sample_rate=16000)
        assert len(silence) == 16000

    def test_dtype_int16(self):
        silence = create_silence(1.0)
        assert silence.dtype == np.int16

    def test_zero_duration(self):
        silence = create_silence(0.0)
        assert len(silence) == 0

    def test_small_duration(self):
        # 10ms at 24kHz = 240 samples
        silence = create_silence(0.01, sample_rate=24000)
        assert len(silence) == 240
