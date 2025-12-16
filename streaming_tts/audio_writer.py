"""
Streaming audio writer with multi-format support.
Supports WAV, MP3, Opus, FLAC, AAC, and PCM formats.
"""

import struct
from io import BytesIO
from typing import Optional, Literal

import numpy as np

# Optional PyAV import for advanced formats
try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


AudioFormat = Literal["wav", "mp3", "opus", "flac", "aac", "pcm"]


class StreamingAudioWriter:
    """
    Handles streaming audio format conversions.

    Supports multiple output formats:
    - wav: Standard WAV format (always available)
    - mp3: MP3 format (requires PyAV)
    - opus: Opus format (requires PyAV)
    - flac: FLAC format (requires PyAV)
    - aac: AAC format (requires PyAV)
    - pcm: Raw PCM data (always available)

    Example:
        writer = StreamingAudioWriter("mp3", sample_rate=24000)

        for audio_chunk in audio_chunks:
            data = writer.write_chunk(audio_chunk)
            if data:
                send_to_client(data)

        final_data = writer.write_chunk(finalize=True)
        send_to_client(final_data)
        writer.close()
    """

    SUPPORTED_FORMATS = {"wav", "mp3", "opus", "flac", "aac", "pcm"}
    PYAV_FORMATS = {"mp3", "opus", "flac", "aac"}

    def __init__(
        self,
        format: AudioFormat,
        sample_rate: int = 24000,
        channels: int = 1,
        bit_rate: int = 128000,
    ):
        """
        Initialize the streaming audio writer.

        Args:
            format: Output format (wav, mp3, opus, flac, aac, pcm)
            sample_rate: Audio sample rate in Hz (default: 24000)
            channels: Number of audio channels (default: 1 for mono)
            bit_rate: Bit rate for compressed formats (default: 128000)
        """
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_rate = bit_rate
        self.bytes_written = 0
        self.pts = 0
        self._closed = False

        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.format}. "
                           f"Supported: {self.SUPPORTED_FORMATS}")

        if self.format in self.PYAV_FORMATS and not HAS_PYAV:
            raise ImportError(
                f"PyAV is required for {self.format} format. "
                "Install with: pip install av"
            )

        self._setup_writer()

    def _setup_writer(self):
        """Set up the appropriate writer based on format."""
        if self.format == "wav":
            self._setup_wav_writer()
        elif self.format == "pcm":
            pass  # PCM needs no setup
        elif self.format in self.PYAV_FORMATS:
            self._setup_pyav_writer()

    def _setup_wav_writer(self):
        """Set up WAV format writer."""
        self.output_buffer = BytesIO()
        # Write WAV header placeholder (will update on finalize)
        self._write_wav_header(0)
        self._data_start = self.output_buffer.tell()

    def _write_wav_header(self, data_size: int):
        """Write WAV file header."""
        self.output_buffer.seek(0)

        # RIFF header
        self.output_buffer.write(b'RIFF')
        self.output_buffer.write(struct.pack('<I', 36 + data_size))  # File size - 8
        self.output_buffer.write(b'WAVE')

        # fmt chunk
        self.output_buffer.write(b'fmt ')
        self.output_buffer.write(struct.pack('<I', 16))  # Chunk size
        self.output_buffer.write(struct.pack('<H', 1))   # Audio format (PCM)
        self.output_buffer.write(struct.pack('<H', self.channels))
        self.output_buffer.write(struct.pack('<I', self.sample_rate))
        bytes_per_sample = 2  # 16-bit
        self.output_buffer.write(struct.pack('<I', self.sample_rate * self.channels * bytes_per_sample))
        self.output_buffer.write(struct.pack('<H', self.channels * bytes_per_sample))
        self.output_buffer.write(struct.pack('<H', bytes_per_sample * 8))

        # data chunk header
        self.output_buffer.write(b'data')
        self.output_buffer.write(struct.pack('<I', data_size))

    def _setup_pyav_writer(self):
        """Set up PyAV-based writer for compressed formats."""
        self.output_buffer = BytesIO()

        codec_map = {
            "mp3": "mp3",
            "opus": "libopus",
            "flac": "flac",
            "aac": "aac",
        }

        container_format = "adts" if self.format == "aac" else self.format
        container_options = {}

        # Disable Xing VBR header for MP3 (iOS compatibility)
        if self.format == "mp3":
            container_options = {"write_xing": "0"}

        self.container = av.open(
            self.output_buffer,
            mode="w",
            format=container_format,
            options=container_options
        )

        self.stream = self.container.add_stream(
            codec_map[self.format],
            rate=self.sample_rate,
            layout="mono" if self.channels == 1 else "stereo",
        )

        # Set bit rate for lossy codecs
        if self.format in ["mp3", "aac", "opus"]:
            self.stream.bit_rate = self.bit_rate

    def write_chunk(
        self,
        audio_data: Optional[np.ndarray] = None,
        finalize: bool = False
    ) -> bytes:
        """
        Write a chunk of audio data and return bytes in the target format.

        Args:
            audio_data: Audio samples as numpy array (int16). Can be None if finalizing.
            finalize: If True, finalize the stream and return remaining data.

        Returns:
            Encoded audio bytes (may be empty for some formats until finalized)
        """
        if self._closed:
            raise RuntimeError("Writer is closed")

        if finalize:
            return self._finalize()

        if audio_data is None or len(audio_data) == 0:
            return b""

        # Ensure int16 format
        if audio_data.dtype != np.int16:
            audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)

        self.bytes_written += len(audio_data) * 2  # 2 bytes per sample

        if self.format == "pcm":
            return audio_data.tobytes()
        elif self.format == "wav":
            return self._write_wav_chunk(audio_data)
        else:
            return self._write_pyav_chunk(audio_data)

    def _write_wav_chunk(self, audio_data: np.ndarray) -> bytes:
        """Write chunk in WAV format."""
        # For WAV, we accumulate in buffer and return incrementally
        self.output_buffer.seek(0, 2)  # Seek to end
        self.output_buffer.write(audio_data.tobytes())
        return b""  # WAV data returned on finalize

    def _write_pyav_chunk(self, audio_data: np.ndarray) -> bytes:
        """Write chunk using PyAV encoder."""
        frame = av.AudioFrame.from_ndarray(
            audio_data.reshape(1, -1),
            format="s16",
            layout="mono" if self.channels == 1 else "stereo",
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self.pts
        self.pts += frame.samples

        packets = self.stream.encode(frame)
        for packet in packets:
            self.container.mux(packet)

        # Get buffered data
        data = self.output_buffer.getvalue()
        self.output_buffer.seek(0)
        self.output_buffer.truncate(0)
        return data

    def _finalize(self) -> bytes:
        """Finalize the stream and return remaining data."""
        if self.format == "pcm":
            return b""
        elif self.format == "wav":
            return self._finalize_wav()
        else:
            return self._finalize_pyav()

    def _finalize_wav(self) -> bytes:
        """Finalize WAV format."""
        # Update header with actual data size
        data_size = self.output_buffer.tell() - self._data_start
        self._write_wav_header(data_size)
        self.output_buffer.seek(0, 2)  # Back to end
        return self.output_buffer.getvalue()

    def _finalize_pyav(self) -> bytes:
        """Finalize PyAV-encoded stream."""
        # Flush encoder
        packets = self.stream.encode(None)
        for packet in packets:
            self.container.mux(packet)

        # Get final data before closing
        data = self.output_buffer.getvalue()
        return data

    def close(self):
        """Close the writer and release resources."""
        if self._closed:
            return

        self._closed = True

        if hasattr(self, "container"):
            self.container.close()

        if hasattr(self, "output_buffer"):
            self.output_buffer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class AudioNormalizer:
    """
    Handles audio normalization and gap trimming for smoother playback.

    Features:
    - Normalizes audio to int16 range
    - Trims silence from start/end of chunks
    - Applies punctuation-aware padding for natural rhythm
    """

    # Padding multipliers based on punctuation
    PUNCTUATION_PADDING = {
        '.': 1.0, '!': 0.9, '?': 1.0, ',': 0.8, ';': 0.85, ':': 0.85,
    }

    def __init__(
        self,
        sample_rate: int = 24000,
        gap_trim_ms: float = 1.0,
        padding_ms: float = 410.0,
        silence_threshold_db: float = -45.0,
    ):
        """
        Initialize the audio normalizer.

        Args:
            sample_rate: Audio sample rate in Hz
            gap_trim_ms: Milliseconds to trim from chunk boundaries
            padding_ms: Base padding in milliseconds between chunks
            silence_threshold_db: Threshold for silence detection (dBFS)
        """
        self.sample_rate = sample_rate
        self.gap_trim_ms = gap_trim_ms
        self.padding_ms = padding_ms
        self.silence_threshold_db = silence_threshold_db

        self.samples_to_trim = int(gap_trim_ms * sample_rate / 1000)
        self.samples_to_pad_start = int(50 * sample_rate / 1000)

    def normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio data to int16 range.

        Args:
            audio_data: Input audio (float32 or int16)

        Returns:
            Normalized int16 audio
        """
        if audio_data.dtype == np.int16:
            return audio_data
        return np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)

    def find_non_silent_bounds(
        self,
        audio_data: np.ndarray,
        chunk_text: str = "",
        speed: float = 1.0,
        is_last_chunk: bool = False,
    ) -> tuple[int, int]:
        """
        Find the start and end indices of non-silent audio.

        Args:
            audio_data: Audio samples (int16)
            chunk_text: The text that produced this audio (for punctuation-aware padding)
            speed: Speech speed multiplier
            is_last_chunk: Whether this is the final chunk

        Returns:
            Tuple of (start_index, end_index) for non-silent region
        """
        # Determine padding multiplier from trailing punctuation
        pad_multiplier = 1.0
        if chunk_text:
            last_char = chunk_text.strip()[-1] if chunk_text.strip() else ""
            pad_multiplier = self.PUNCTUATION_PADDING.get(last_char, 1.0)

        # Calculate end padding
        if not is_last_chunk:
            samples_to_pad_end = max(
                int((self.padding_ms * self.sample_rate * pad_multiplier) / 1000)
                - self.samples_to_pad_start,
                0
            )
        else:
            samples_to_pad_end = self.samples_to_pad_start

        # Convert dB threshold to amplitude
        max_amplitude = np.iinfo(np.int16).max
        amplitude_threshold = max_amplitude * (10 ** (self.silence_threshold_db / 20))

        # Find first non-silent sample
        start_idx = 0
        for i in range(len(audio_data)):
            if abs(audio_data[i]) > amplitude_threshold:
                start_idx = i
                break

        # Find last non-silent sample
        end_idx = len(audio_data)
        for i in range(len(audio_data) - 1, -1, -1):
            if abs(audio_data[i]) > amplitude_threshold:
                end_idx = i
                break

        # Apply padding
        start_idx = max(start_idx - self.samples_to_pad_start, 0)
        end_idx = min(
            end_idx + int(samples_to_pad_end / speed),
            len(audio_data)
        )

        return start_idx, end_idx

    def trim_audio(
        self,
        audio_data: np.ndarray,
        chunk_text: str = "",
        speed: float = 1.0,
        is_last_chunk: bool = False,
    ) -> np.ndarray:
        """
        Trim silence and normalize audio chunk.

        Args:
            audio_data: Input audio samples
            chunk_text: Text that produced this audio
            speed: Speech speed multiplier
            is_last_chunk: Whether this is the final chunk

        Returns:
            Trimmed and normalized audio
        """
        audio = self.normalize(audio_data)

        # Trim fixed amount from boundaries
        if len(audio) > (2 * self.samples_to_trim):
            audio = audio[self.samples_to_trim:-self.samples_to_trim]

        # Find and trim to non-silent portion
        start, end = self.find_non_silent_bounds(
            audio, chunk_text, speed, is_last_chunk
        )
        return audio[start:end]


def create_silence(duration_seconds: float, sample_rate: int = 24000) -> np.ndarray:
    """
    Create silent audio data.

    Args:
        duration_seconds: Duration of silence in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Silent audio as int16 numpy array
    """
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.int16)
