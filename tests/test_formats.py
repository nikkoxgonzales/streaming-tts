"""Tests for streaming audio format conversion."""

import numpy as np
import pytest

from streaming_tts.formats import (
    FORMAT_CODECS,
    FORMAT_CONTAINERS,
    AudioFormat,
    StreamingAudioWriter,
    get_content_type,
)


class TestGetContentType:
    """Tests for get_content_type function."""

    def test_pcm_content_type(self) -> None:
        """PCM returns audio/pcm."""
        assert get_content_type("pcm") == "audio/pcm"

    def test_wav_content_type(self) -> None:
        """WAV returns audio/wav."""
        assert get_content_type("wav") == "audio/wav"

    def test_mp3_content_type(self) -> None:
        """MP3 returns audio/mpeg."""
        assert get_content_type("mp3") == "audio/mpeg"

    def test_opus_content_type(self) -> None:
        """Opus returns audio/ogg."""
        assert get_content_type("opus") == "audio/ogg"

    def test_flac_content_type(self) -> None:
        """FLAC returns audio/flac."""
        assert get_content_type("flac") == "audio/flac"

    def test_aac_content_type(self) -> None:
        """AAC returns audio/aac."""
        assert get_content_type("aac") == "audio/aac"


class TestFormatMappings:
    """Tests for format codec and container mappings."""

    def test_all_formats_have_codecs(self) -> None:
        """All supported formats have codec mappings."""
        expected_formats = {"wav", "mp3", "opus", "flac", "aac"}
        assert set(FORMAT_CODECS.keys()) == expected_formats

    def test_all_formats_have_containers(self) -> None:
        """All supported formats have container mappings."""
        expected_formats = {"wav", "mp3", "opus", "flac", "aac"}
        assert set(FORMAT_CONTAINERS.keys()) == expected_formats


class TestStreamingAudioWriterPCM:
    """Tests for PCM passthrough mode (no PyAV required)."""

    def test_pcm_passthrough_int16(self) -> None:
        """PCM mode passes through int16 data unchanged."""
        writer = StreamingAudioWriter("pcm")
        audio = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)

        result = writer.write_chunk(audio)

        assert result == audio.tobytes()
        assert writer.format == "pcm"

    def test_pcm_passthrough_float32(self) -> None:
        """PCM mode converts float32 to int16."""
        writer = StreamingAudioWriter("pcm")
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

        result = writer.write_chunk(audio)

        # Verify conversion
        expected = (audio * 32767).astype(np.int16)
        assert result == expected.tobytes()

    def test_pcm_finalize_empty(self) -> None:
        """PCM finalize returns empty bytes."""
        writer = StreamingAudioWriter("pcm")
        writer.write_chunk(np.array([1, 2, 3], dtype=np.int16))

        result = writer.finalize()

        assert result == b""

    def test_pcm_properties(self) -> None:
        """PCM writer has correct properties."""
        writer = StreamingAudioWriter("pcm", sample_rate=48000, channels=2)

        assert writer.format == "pcm"
        assert writer.sample_rate == 48000


class TestStreamingAudioWriterValidation:
    """Tests for writer validation and error handling."""

    def test_unsupported_format_raises(self) -> None:
        """Unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            StreamingAudioWriter("invalid_format")  # type: ignore[arg-type]

    def test_write_after_finalize_raises(self) -> None:
        """Writing after finalize raises RuntimeError."""
        writer = StreamingAudioWriter("pcm")
        writer.finalize()

        with pytest.raises(RuntimeError, match="finalized"):
            writer.write_chunk(np.array([1, 2, 3], dtype=np.int16))

    def test_double_finalize_returns_empty(self) -> None:
        """Double finalize returns empty bytes."""
        writer = StreamingAudioWriter("pcm")
        writer.finalize()

        result = writer.finalize()

        assert result == b""


class TestStreamingAudioWriterContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_enter(self) -> None:
        """Context manager returns self on enter."""
        writer = StreamingAudioWriter("pcm")

        with writer as w:
            assert w is writer

    def test_context_manager_closes(self) -> None:
        """Context manager finalizes on exit."""
        writer = StreamingAudioWriter("pcm")

        with writer:
            writer.write_chunk(np.array([1, 2, 3], dtype=np.int16))

        assert writer._finalized


# Tests that require PyAV - skip if not installed
try:
    import av

    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


@pytest.mark.skipif(not PYAV_AVAILABLE, reason="PyAV not installed")
class TestStreamingAudioWriterWithPyAV:
    """Tests for actual format encoding (requires PyAV)."""

    def test_wav_encoding(self) -> None:
        """WAV encoding produces valid WAV header."""
        writer = StreamingAudioWriter("wav", sample_rate=24000)
        audio = np.random.randint(-32768, 32767, 24000, dtype=np.int16)

        chunk = writer.write_chunk(audio)
        final = writer.finalize()
        result = chunk + final

        # WAV files start with RIFF header
        assert result[:4] == b"RIFF"
        assert b"WAVE" in result[:20]

    def test_mp3_encoding(self) -> None:
        """MP3 encoding produces output."""
        writer = StreamingAudioWriter("mp3", sample_rate=24000, bitrate=128000)
        audio = np.random.randint(-32768, 32767, 24000, dtype=np.int16)

        chunk = writer.write_chunk(audio)
        final = writer.finalize()
        result = chunk + final

        # MP3 should have some output
        assert len(result) > 0

    def test_opus_encoding(self) -> None:
        """Opus encoding produces Ogg container."""
        writer = StreamingAudioWriter("opus", sample_rate=24000)
        audio = np.random.randint(-32768, 32767, 24000, dtype=np.int16)

        chunk = writer.write_chunk(audio)
        final = writer.finalize()
        result = chunk + final

        # Ogg files start with OggS
        assert result[:4] == b"OggS"

    def test_flac_encoding(self) -> None:
        """FLAC encoding produces valid FLAC header."""
        writer = StreamingAudioWriter("flac", sample_rate=24000)
        audio = np.random.randint(-32768, 32767, 24000, dtype=np.int16)

        chunk = writer.write_chunk(audio)
        final = writer.finalize()
        result = chunk + final

        # FLAC files start with fLaC
        assert result[:4] == b"fLaC"

    def test_multiple_chunks(self) -> None:
        """Multiple chunks are encoded correctly."""
        writer = StreamingAudioWriter("wav", sample_rate=24000)

        chunks = []
        for _ in range(5):
            audio = np.random.randint(-32768, 32767, 4800, dtype=np.int16)
            chunks.append(writer.write_chunk(audio))

        final = writer.finalize()
        result = b"".join(chunks) + final

        # Should be valid WAV
        assert result[:4] == b"RIFF"

    def test_float32_input(self) -> None:
        """Float32 input is converted correctly."""
        writer = StreamingAudioWriter("wav", sample_rate=24000)
        audio = np.random.uniform(-1.0, 1.0, 24000).astype(np.float32)

        chunk = writer.write_chunk(audio)
        final = writer.finalize()
        result = chunk + final

        assert result[:4] == b"RIFF"

    def test_stereo_encoding(self) -> None:
        """Stereo encoding works correctly."""
        writer = StreamingAudioWriter("wav", sample_rate=24000, channels=2)
        # Note: For stereo, we'd need interleaved samples
        # This test just verifies it doesn't crash
        audio = np.random.randint(-32768, 32767, 24000, dtype=np.int16)

        chunk = writer.write_chunk(audio)
        final = writer.finalize()

        assert len(chunk + final) > 0


@pytest.mark.skipif(PYAV_AVAILABLE, reason="Only run when PyAV is not installed")
class TestStreamingAudioWriterNoPyAV:
    """Tests for behavior when PyAV is not installed."""

    def test_non_pcm_raises_import_error(self) -> None:
        """Non-PCM format raises ImportError without PyAV."""
        with pytest.raises(ImportError, match="PyAV is required"):
            StreamingAudioWriter("mp3")
