"""Tests for CrossfadingBuffer in streaming_tts.stream_player module."""

import pytest
import numpy as np
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
        yield mock


@pytest.fixture
def mock_pa_portaudio():
    """Mock PyAudio C module."""
    with patch('streaming_tts.stream_player.pa') as mock:
        mock.paFramesPerBufferUnspecified = 0
        yield mock


class TestCrossfadingBuffer:
    """Tests for CrossfadingBuffer class."""

    def test_init_default(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer()
        assert buf.crossfade_ms == 25.0
        assert buf.sample_rate == 24000
        assert buf.channels == 1
        assert buf.sample_width == 2
        assert buf.crossfade_samples == 600  # 25ms at 24kHz
        assert buf.normalize_boundaries is True
        assert buf.target_silence_samples == 120  # 5ms at 24kHz

    def test_init_custom(self, mock_pyaudio, mock_pa_portaudio):
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(
            crossfade_ms=20.0,
            sample_rate=48000,
            channels=2,
            sample_width=4,
        )
        assert buf.crossfade_ms == 20.0
        assert buf.sample_rate == 48000
        assert buf.channels == 2
        assert buf.crossfade_samples == 960  # 20ms at 48kHz

    def test_process_first_chunk(self, mock_pyaudio, mock_pa_portaudio):
        """First chunk should have its tail held back."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(crossfade_ms=10.0, sample_rate=24000)
        # crossfade_samples = 240 at 10ms/24kHz

        # Create a chunk with 1000 samples
        audio = np.ones(1000, dtype=np.int16)
        audio_bytes = audio.tobytes()

        result = buf.process(audio_bytes)

        # First chunk: should output (1000 - 240) = 760 samples
        result_array = np.frombuffer(result, dtype=np.int16)
        assert len(result_array) == 760
        assert buf._pending_tail is not None
        assert len(buf._pending_tail) == 240

    def test_process_second_chunk_crossfade(self, mock_pyaudio, mock_pa_portaudio):
        """Second chunk should crossfade with first chunk's tail."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(crossfade_ms=10.0, sample_rate=24000)
        crossfade_samples = buf.crossfade_samples  # 240

        # First chunk: constant value 1000
        chunk1 = np.full(1000, 1000, dtype=np.int16)
        buf.process(chunk1.tobytes())

        # Second chunk: constant value 2000
        chunk2 = np.full(1000, 2000, dtype=np.int16)
        result = buf.process(chunk2.tobytes())
        result_array = np.frombuffer(result, dtype=np.int16)

        # Result should have:
        # - 240 crossfaded samples (blend from 1000 to 2000)
        # - 520 samples of value 2000 (760 - 240 = 520, since 240 held for next crossfade)
        # Total: 240 + 520 = 760

        # Check crossfade region (should blend from 1000 to 2000)
        crossfade_region = result_array[:crossfade_samples]
        assert crossfade_region[0] < 2000  # Start closer to 1000
        assert crossfade_region[-1] > 1000  # End closer to 2000

    def test_process_short_chunk(self, mock_pyaudio, mock_pa_portaudio):
        """Short chunks (< crossfade_samples) should be passed through."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(crossfade_ms=10.0, sample_rate=24000)

        # Create a chunk shorter than crossfade_samples (240)
        short_audio = np.ones(100, dtype=np.int16)
        result = buf.process(short_audio.tobytes())

        # Short chunk should be output directly
        result_array = np.frombuffer(result, dtype=np.int16)
        assert len(result_array) == 100

    def test_flush(self, mock_pyaudio, mock_pa_portaudio):
        """Flush should return pending tail."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(crossfade_ms=10.0, sample_rate=24000)

        # Process a chunk to create pending tail
        audio = np.ones(1000, dtype=np.int16)
        buf.process(audio.tobytes())

        # Flush should return the tail
        result = buf.flush()
        result_array = np.frombuffer(result, dtype=np.int16)
        assert len(result_array) == 240  # crossfade_samples

        # After flush, pending tail should be None
        assert buf._pending_tail is None

    def test_flush_empty(self, mock_pyaudio, mock_pa_portaudio):
        """Flush on empty buffer should return empty bytes."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer()
        result = buf.flush()
        assert result == b""

    def test_reset(self, mock_pyaudio, mock_pa_portaudio):
        """Reset should clear pending tail."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(crossfade_ms=10.0, sample_rate=24000)

        # Process a chunk to create pending tail
        audio = np.ones(1000, dtype=np.int16)
        buf.process(audio.tobytes())
        assert buf._pending_tail is not None

        # Reset should clear it
        buf.reset()
        assert buf._pending_tail is None

    def test_float32_format(self, mock_pyaudio, mock_pa_portaudio):
        """Test crossfading with float32 audio."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(
            crossfade_ms=10.0,
            sample_rate=24000,
            sample_width=4,  # float32
        )

        # Create float32 audio
        audio = np.ones(1000, dtype=np.float32)
        audio_bytes = audio.tobytes()

        result = buf.process(audio_bytes)
        result_array = np.frombuffer(result, dtype=np.float32)

        # Should have 760 samples (1000 - 240)
        assert len(result_array) == 760

    def test_stereo_audio(self, mock_pyaudio, mock_pa_portaudio):
        """Test crossfading with stereo audio."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(
            crossfade_ms=10.0,
            sample_rate=24000,
            channels=2,
        )
        crossfade_samples = buf.crossfade_samples  # 240 frames

        # Create stereo audio (interleaved L/R)
        # 1000 frames * 2 channels = 2000 samples
        audio = np.ones(2000, dtype=np.int16)
        audio_bytes = audio.tobytes()

        result = buf.process(audio_bytes)
        result_array = np.frombuffer(result, dtype=np.int16)

        # Should output (1000 - 240) frames * 2 channels = 1520 samples
        assert len(result_array) == 1520

    def test_continuous_processing(self, mock_pyaudio, mock_pa_portaudio):
        """Test processing multiple chunks continuously."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(crossfade_ms=10.0, sample_rate=24000)
        crossfade_samples = buf.crossfade_samples  # 240

        total_output = 0

        # Process 5 chunks
        for i in range(5):
            audio = np.full(1000, (i + 1) * 1000, dtype=np.int16)
            result = buf.process(audio.tobytes())
            result_array = np.frombuffer(result, dtype=np.int16)
            total_output += len(result_array)

        # Flush final tail
        result = buf.flush()
        result_array = np.frombuffer(result, dtype=np.int16)
        total_output += len(result_array)

        # Crossfading overlaps chunks, reducing total duration
        # Each boundary loses crossfade_samples (4 boundaries between 5 chunks)
        # Total input: 5 * 1000 = 5000 samples
        # Expected output: 5000 - (4 * crossfade_samples) = 5000 - 960 = 4040
        expected_output = 5000 - (4 * crossfade_samples)
        assert total_output == expected_output

    def test_crossfade_smoothness(self, mock_pyaudio, mock_pa_portaudio):
        """Verify crossfade creates smooth transition."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(crossfade_ms=10.0, sample_rate=24000)
        crossfade_samples = buf.crossfade_samples

        # First chunk: zeros
        chunk1 = np.zeros(1000, dtype=np.int16)
        buf.process(chunk1.tobytes())

        # Second chunk: max values (10000)
        chunk2 = np.full(1000, 10000, dtype=np.int16)
        result = buf.process(chunk2.tobytes())
        result_array = np.frombuffer(result, dtype=np.int16)

        # Check crossfade region increases monotonically
        crossfade_region = result_array[:crossfade_samples]
        for i in range(1, len(crossfade_region)):
            # Allow for slight variations due to integer conversion
            assert crossfade_region[i] >= crossfade_region[i - 1] - 1

    def test_disabled_crossfade_passthrough(self, mock_pyaudio, mock_pa_portaudio):
        """When crossfade is disabled (ms=0), chunks pass through unchanged."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(crossfade_ms=0.0, sample_rate=24000)
        assert buf.crossfade_samples == 0

        audio = np.arange(1000, dtype=np.int16)
        result = buf.process(audio.tobytes())
        result_array = np.frombuffer(result, dtype=np.int16)

        # With 0 crossfade, entire chunk should pass through
        assert len(result_array) == 1000
        np.testing.assert_array_equal(result_array, audio)

    def test_boundary_normalization_trims_silence(self, mock_pyaudio, mock_pa_portaudio):
        """Test that boundary normalization trims excess trailing silence."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(
            crossfade_ms=10.0,
            sample_rate=24000,
            normalize_boundaries=True,
            silence_threshold=500,
            target_silence_ms=5.0,
        )

        # Create audio with 50ms of trailing silence
        audio = np.full(1000, 5000, dtype=np.int16)  # Loud audio
        silence = np.zeros(1200, dtype=np.int16)  # 50ms silence at 24kHz
        combined = np.concatenate([audio, silence])

        result = buf.process(combined.tobytes())
        result_array = np.frombuffer(result, dtype=np.int16)

        # Should be trimmed: original 2200 - crossfade held - excess silence trimmed
        # With normalization, trailing silence should be ~5ms (120 samples)
        assert len(result_array) < 2200 - 240  # Significantly trimmed

    def test_boundary_normalization_disabled(self, mock_pyaudio, mock_pa_portaudio):
        """Test that boundary normalization can be disabled."""
        from streaming_tts.stream_player import CrossfadingBuffer

        buf = CrossfadingBuffer(
            crossfade_ms=10.0,
            sample_rate=24000,
            normalize_boundaries=False,
        )

        # Create audio with trailing silence
        audio = np.full(1000, 5000, dtype=np.int16)
        silence = np.zeros(500, dtype=np.int16)
        combined = np.concatenate([audio, silence])

        result = buf.process(combined.tobytes())
        result_array = np.frombuffer(result, dtype=np.int16)

        # Without normalization, length is 1500 - 240 (crossfade held) = 1260
        assert len(result_array) == 1260


class TestStreamPlayerCrossfade:
    """Integration tests for StreamPlayer crossfading."""

    def test_stream_player_crossfade_enabled(self, mock_pyaudio, mock_pa_portaudio):
        """Verify StreamPlayer initializes crossfade buffer when enabled."""
        import queue
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        mock_pyaudio.PyAudio.return_value.get_sample_size.return_value = 2

        audio_buffer = queue.Queue()
        timings = queue.Queue()
        config = AudioConfiguration(rate=24000)

        player = StreamPlayer(
            audio_buffer,
            timings,
            config,
            enable_crossfade=True,
            crossfade_ms=20.0,
        )

        assert player.enable_crossfade is True
        assert player.crossfade_ms == 20.0

    def test_stream_player_crossfade_disabled(self, mock_pyaudio, mock_pa_portaudio):
        """Verify StreamPlayer skips crossfade when disabled."""
        import queue
        from streaming_tts.stream_player import StreamPlayer, AudioConfiguration

        mock_pyaudio.PyAudio.return_value.get_sample_size.return_value = 2

        audio_buffer = queue.Queue()
        timings = queue.Queue()
        config = AudioConfiguration(rate=24000)

        player = StreamPlayer(
            audio_buffer,
            timings,
            config,
            enable_crossfade=False,
        )

        assert player.enable_crossfade is False
        assert player.crossfade_buffer is None
