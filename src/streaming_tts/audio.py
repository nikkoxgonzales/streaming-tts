"""Optional audio playback using PyAudio."""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PlaybackConfig

# Try to import PyAudio - it's optional
try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None  # type: ignore[assignment]


class AudioPlayer:
    """
    Audio player using PyAudio.

    This class provides audio playback functionality for development
    and testing. For production use with WebSocket streaming, you
    typically use muted=True and stream chunks directly.

    Example:
        player = AudioPlayer(config, sample_rate=24000)
        player.start()
        for chunk in audio_chunks:
            player.write(chunk)
        player.stop()
    """

    def __init__(
        self,
        config: PlaybackConfig,
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> None:
        """
        Initialize the audio player.

        Args:
            config: Playback configuration
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels

        Raises:
            ImportError: If PyAudio is not installed
        """
        if not PYAUDIO_AVAILABLE:
            raise ImportError(
                "PyAudio is required for audio playback. "
                "Install with: pip install pyaudio"
            )

        self._config = config
        self._sample_rate = sample_rate
        self._channels = channels
        self._pyaudio: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """Open the audio stream."""
        if self._started:
            return

        with self._lock:
            if self._started:
                return

            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                output=True,
                output_device_index=self._config.device_index,
                frames_per_buffer=self._config.frames_per_buffer,
            )
            self._started = True

    def write(self, chunk: bytes) -> None:
        """
        Write audio data to the stream.

        Args:
            chunk: PCM16 audio data
        """
        if self._config.muted:
            return

        if not self._started:
            self.start()

        with self._lock:
            if self._stream is not None:
                self._stream.write(chunk)

    def stop(self) -> None:
        """Close the audio stream."""
        with self._lock:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None

            if self._pyaudio is not None:
                self._pyaudio.terminate()
                self._pyaudio = None

            self._started = False

    def is_playing(self) -> bool:
        """Check if the audio stream is active."""
        with self._lock:
            return self._stream is not None and self._stream.is_active()

    def get_devices(self) -> list[dict]:
        """
        Get list of available audio output devices.

        Returns:
            List of device info dictionaries
        """
        if not PYAUDIO_AVAILABLE:
            return []

        pa = pyaudio.PyAudio()
        try:
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxOutputChannels", 0) > 0:
                    devices.append({
                        "index": i,
                        "name": info.get("name", "Unknown"),
                        "sample_rate": int(info.get("defaultSampleRate", 0)),
                        "channels": info.get("maxOutputChannels", 0),
                    })
            return devices
        finally:
            pa.terminate()

    def __enter__(self) -> AudioPlayer:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()

    def __del__(self) -> None:
        """Destructor - ensure cleanup."""
        with contextlib.suppress(Exception):
            self.stop()


def is_playback_available() -> bool:
    """Check if audio playback is available."""
    return PYAUDIO_AVAILABLE


def list_audio_devices() -> list[dict]:
    """
    List available audio output devices.

    Returns:
        List of device info dictionaries with keys:
        - index: Device index for PlaybackConfig
        - name: Human-readable device name
        - sample_rate: Default sample rate
        - channels: Number of output channels
    """
    if not PYAUDIO_AVAILABLE:
        return []

    player = AudioPlayer.__new__(AudioPlayer)
    return player.get_devices()
