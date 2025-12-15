"""Streaming audio format conversion using PyAV."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import av

# Supported output formats
AudioFormat = Literal["pcm", "wav", "mp3", "opus", "flac", "aac"]

# Format to codec mapping
FORMAT_CODECS: dict[str, str] = {
    "wav": "pcm_s16le",
    "mp3": "libmp3lame",
    "opus": "libopus",
    "flac": "flac",
    "aac": "aac",
}

# Format to container mapping
FORMAT_CONTAINERS: dict[str, str] = {
    "wav": "wav",
    "mp3": "mp3",
    "opus": "ogg",  # Opus uses Ogg container
    "flac": "flac",
    "aac": "adts",  # AAC uses ADTS container for streaming
}


class StreamingAudioWriter:
    """
    Converts PCM16 audio to various formats on-the-fly using PyAV.

    Supports streaming encoding for wav, mp3, opus, flac, and aac formats.
    Uses a BytesIO buffer to accumulate encoded output.

    Example:
        writer = StreamingAudioWriter("mp3", sample_rate=24000)
        for audio_chunk in audio_generator:
            encoded = writer.write_chunk(audio_chunk)
            send(encoded)
        final = writer.finalize()
        send(final)
    """

    def __init__(
        self,
        format: AudioFormat,
        sample_rate: int = 24000,
        channels: int = 1,
        bitrate: int = 128000,
    ) -> None:
        """
        Initialize the audio writer.

        Args:
            format: Output format (pcm, wav, mp3, opus, flac, aac)
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 = mono, 2 = stereo)
            bitrate: Bitrate for lossy formats (mp3, opus, aac)

        Raises:
            ImportError: If PyAV is not installed
            ValueError: If format is not supported
        """
        if format == "pcm":
            # PCM passthrough - no encoding needed
            self._format = format
            self._container = None
            self._stream = None
            self._buffer = None
            self._sample_rate = sample_rate
            self._channels = channels
            self._finalized = False
            return

        if format not in FORMAT_CODECS:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported: pcm, {', '.join(FORMAT_CODECS.keys())}"
            )

        try:
            import av
        except ImportError as e:
            raise ImportError(
                "PyAV is required for audio format conversion. "
                "Install with: pip install streaming-tts[formats]"
            ) from e

        self._format = format
        self._sample_rate = sample_rate
        self._channels = channels
        self._bitrate = bitrate

        # Create buffer and container
        self._buffer = io.BytesIO()
        self._container: av.container.OutputContainer = av.open(
            self._buffer,
            mode="w",
            format=FORMAT_CONTAINERS[format],
        )

        # Create audio stream with codec
        codec_name = FORMAT_CODECS[format]
        self._stream: av.audio.stream.AudioStream = self._container.add_stream(
            codec_name,
            rate=sample_rate,
        )
        self._stream.channels = channels
        self._stream.layout = "mono" if channels == 1 else "stereo"

        # Set bitrate for lossy codecs
        if format in ("mp3", "opus", "aac"):
            self._stream.bit_rate = bitrate

        # Track samples for timing
        self._samples_written = 0
        self._finalized = False

    @property
    def format(self) -> AudioFormat:
        """Get the output format."""
        return self._format

    @property
    def sample_rate(self) -> int:
        """Get the sample rate."""
        return self._sample_rate

    def write_chunk(self, audio_int16: np.ndarray) -> bytes:
        """
        Encode an audio chunk and return the encoded bytes.

        Args:
            audio_int16: PCM16 audio samples as numpy array (int16 or float32)

        Returns:
            Encoded audio bytes (may be empty if codec buffers internally)

        Raises:
            RuntimeError: If writer has been finalized
        """
        if self._finalized:
            raise RuntimeError("Cannot write to finalized writer")

        # Handle PCM passthrough
        if self._format == "pcm":
            if audio_int16.dtype == np.float32:
                audio_int16 = (audio_int16 * 32767).astype(np.int16)
            return audio_int16.tobytes()

        import av

        # Convert float32 to int16 if needed
        if audio_int16.dtype == np.float32:
            audio_int16 = (audio_int16 * 32767).astype(np.int16)

        # Ensure int16
        if audio_int16.dtype != np.int16:
            audio_int16 = audio_int16.astype(np.int16)

        # Create audio frame
        frame = av.AudioFrame.from_ndarray(
            audio_int16.reshape(1, -1),  # Shape: (channels, samples)
            format="s16",
            layout="mono" if self._channels == 1 else "stereo",
        )
        frame.sample_rate = self._sample_rate
        frame.pts = self._samples_written

        self._samples_written += len(audio_int16)

        # Encode frame
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

        # Get accumulated bytes
        result = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate()

        return result

    def finalize(self) -> bytes:
        """
        Flush the encoder and return final bytes.

        Must be called after all chunks have been written to get
        any remaining buffered data and proper file termination.

        Returns:
            Final encoded bytes (trailer, buffered data, etc.)
        """
        if self._finalized:
            return b""

        self._finalized = True

        # PCM passthrough - nothing to finalize
        if self._format == "pcm":
            return b""

        # Flush encoder
        for packet in self._stream.encode(None):
            self._container.mux(packet)

        # Close container to write trailer
        self._container.close()

        # Get final bytes
        result = self._buffer.getvalue()
        return result

    def close(self) -> None:
        """Close the writer and release resources."""
        if not self._finalized:
            self.finalize()

    def __enter__(self) -> "StreamingAudioWriter":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()


def get_content_type(format: AudioFormat) -> str:
    """
    Get the MIME content type for an audio format.

    Args:
        format: Audio format

    Returns:
        MIME type string
    """
    content_types: dict[AudioFormat, str] = {
        "pcm": "audio/pcm",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "flac": "audio/flac",
        "aac": "audio/aac",
    }
    return content_types.get(format, "application/octet-stream")
