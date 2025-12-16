"""
Stream Management Module

This module provides classes to handle audio streaming using PyAudio. It covers:
  - Audio configuration (AudioConfiguration)
  - Stream control (AudioStream)
  - Data buffering (AudioBufferManager)
  - Playback with pause, resume, and stop (StreamPlayer)

Key Components:
  1. AudioConfiguration: Sets up audio parameters (format, channels, sample rate, device).
  2. AudioStream: Manages opening, starting, stopping, and closing streams, and adapts to device capabilities.
  3. AudioBufferManager: Buffers audio data in a queue and tracks sample counts.
  4. StreamPlayer: Orchestrates playback, handles events, and supports callbacks.

Designed for flexible, real-time audio playback and streaming, with error handling for unsupported configurations.
"""
from pydub import AudioSegment
try:
    import pyaudio._portaudio as pa
except ImportError:
    print("Could not import the PyAudio C module 'pyaudio._portaudio'.")
    raise
from typing import TYPE_CHECKING, Optional
import numpy as np
import subprocess
import threading
import resampy
import pyaudio
import logging
import shutil
import queue
import time
import io

if TYPE_CHECKING:
    from .diagnostics import PlaybackDiagnostics


class AudioConfiguration:
    """
    Defines the configuration for an audio stream.
    """

    def __init__(
        self,
        format: int = pyaudio.paInt16,
        channels: int = 1,
        rate: int = 16000,
        output_device_index=None,
        muted: bool = False,
        frames_per_buffer: int = pa.paFramesPerBufferUnspecified,
        playout_chunk_size: int = -1,
    ):
        """
        Args:
            format (int): Audio format, typically one of PyAudio's predefined constants, e.g., pyaudio.paInt16 (default).
            channels (int): Number of audio channels, e.g., 1 for mono or 2 for stereo. Defaults to 1 (mono).
            rate (int): Sample rate of the audio stream in Hz. Defaults to 16000.
            output_device_index (int): Index of the audio output device. If None, the default output device is used.
            muted (bool): If True, audio playback is muted. Defaults to False.
            frames_per_buffer (int): Number of frames per buffer for PyAudio. Defaults to pa.paFramesPerBufferUnspecified, letting PyAudio choose.
            playout_chunk_size (int): Size of audio chunks (in bytes) to be played out. Defaults to -1, which determines the chunk size based on frames_per_buffer or a default value.

        """
        self.format = format
        self.channels = channels
        self.rate = rate
        self.output_device_index = output_device_index
        self.muted = muted
        self.frames_per_buffer = frames_per_buffer
        self.playout_chunk_size = playout_chunk_size


class AudioStream:
    """
    Handles audio stream operations
    - opening, starting, stopping, and closing
    """

    # Class-level cache for supported sample rates per device
    _supported_rates_cache: dict = {}

    def __init__(self, config: AudioConfiguration):
        """
        Args:
            config (AudioConfiguration): Object containing audio settings.
        """
        self.config = config
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()
        self.actual_sample_rate = 0
        self.mpv_process = None

    def get_supported_sample_rates(self, device_index):
        """
        Test which standard sample rates are supported by the specified device.
        Results are cached per device to avoid repeated testing (~100ms savings).

        Args:
            device_index (int): The index of the audio device to test

        Returns:
            list: List of supported sample rates
        """
        # Check cache first
        cache_key = (device_index, self.config.format)
        if cache_key in AudioStream._supported_rates_cache:
            return AudioStream._supported_rates_cache[cache_key]

        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        supported_rates = []

        device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
        max_channels = device_info.get('maxOutputChannels')

        # Test each standard sample rate
        for rate in standard_rates:
            try:
                if self.pyaudio_instance.is_format_supported(
                    rate,
                    output_device=device_index,
                    output_channels=max_channels,
                    output_format=self.config.format,
                ):
                    supported_rates.append(rate)
            except (ValueError, OSError):
                continue

        # Cache the result
        AudioStream._supported_rates_cache[cache_key] = supported_rates
        return supported_rates

    def _get_best_sample_rate(self, device_index, desired_rate):
        """
        Determines the best available sample rate for the device.

        Args:
            device_index: Index of the audio device
            desired_rate: Preferred sample rate

        Returns:
            int: Best available sample rate
        """
        try:
            # First determine the actual device index to use
            actual_device_index = (device_index if device_index is not None
                                else self.pyaudio_instance.get_default_output_device_info()['index'])

            # Now use the actual_device_index for getting device info and supported rates
            device_info = self.pyaudio_instance.get_device_info_by_index(actual_device_index)
            supported_rates = self.get_supported_sample_rates(actual_device_index)

            # Check if desired rate is supported
            if desired_rate in supported_rates:
                return desired_rate

            # Find the highest supported rate that's lower than desired_rate
            lower_rates = [r for r in supported_rates if r <= desired_rate]
            if lower_rates:
                return max(lower_rates)

            # If no lower rates, get the lowest higher rate
            higher_rates = [r for r in supported_rates if r > desired_rate]
            if higher_rates:
                return min(higher_rates)

            # If nothing else works, return device's default rate
            return int(device_info.get('defaultSampleRate', 44100))

        except Exception as e:
            logging.warning(f"Error determining sample rate: {e}")
            return 44100  # Safe fallback

    def is_installed(self, lib_name: str) -> bool:
        """
        Check if the given library or software is installed and accessible.

        This method uses shutil.which to determine if the given library or software is
        installed and available in the system's PATH.

        Args:
            lib_name (str): Name of the library or software to check.

        Returns:
            bool: True if the library is installed, otherwise False.
        """
        lib = shutil.which(lib_name)
        if lib is None:
            return False
        return True

    def open_stream(self):
        """Opens an audio stream."""

        # check for mpeg format
        pyChannels = self.config.channels
        desired_rate = self.config.rate
        pyOutput_device_index = self.config.output_device_index

        if self.config.muted:
            logging.debug("Muted mode, no opening stream")

        else:
            if self.config.format == pyaudio.paCustomFormat and pyChannels == -1 and desired_rate == -1:
                logging.debug("Opening mpv stream for mpeg audio chunks, no need to determine sample rate")
                if not self.is_installed("mpv"):
                    message = (
                        "mpv not found, necessary to stream audio. "
                        "On mac you can install it with 'brew install mpv'. "
                        "On linux and windows you can install it from https://mpv.io/"
                    )
                    raise ValueError(message)

                mpv_command = [
                    "mpv",
                    "--no-terminal",
                    "--stream-buffer-size=4096",
                    "--demuxer-max-bytes=4096",
                    "--demuxer-max-back-bytes=4096",
                    "--ad-queue-max-bytes=4096",
                    "--cache=no",
                    "--cache-secs=0",
                    "--",
                    "fd://0"
                ]

                self.mpv_process = subprocess.Popen(
                    mpv_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return

            # Determine the best sample rate
            best_rate = self._get_best_sample_rate(pyOutput_device_index, desired_rate)
            self.actual_sample_rate = best_rate

            if self.config.format == pyaudio.paCustomFormat:
                pyFormat = self.pyaudio_instance.get_format_from_width(2)
                logging.debug(
                    "Opening stream for mpeg audio chunks, "
                    f"pyFormat: {pyFormat}, pyChannels: {pyChannels}, "
                    f"pySampleRate: {best_rate}"
                )
            else:
                pyFormat = self.config.format
                logging.debug(
                    "Opening stream for wave audio chunks, "
                    f"pyFormat: {pyFormat}, pyChannels: {pyChannels}, "
                    f"pySampleRate: {best_rate}"
                )
            try:
                self.stream = self.pyaudio_instance.open(
                    format=pyFormat,
                    channels=pyChannels,
                    rate=best_rate,
                    output_device_index=pyOutput_device_index,
                    frames_per_buffer=self.config.frames_per_buffer,
                    output=True,
                )
            except Exception as e:
                error_msg = (
                    f"Error opening stream with parameters: "
                    f"format={pyFormat}, channels={pyChannels}, rate={best_rate}, "
                    f"output_device_index={pyOutput_device_index}. Error: {e}"
                )
                logging.error(error_msg)

                # Log available audio devices for debugging
                device_count = self.pyaudio_instance.get_device_count()
                logging.info("Available Audio Devices:")
                for i in range(device_count):
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    logging.info(
                        f"  Device {i}: {device_info['name']} "
                        f"(rate={device_info['defaultSampleRate']}Hz, "
                        f"out_ch={device_info['maxOutputChannels']})"
                    )

                raise RuntimeError(error_msg) from e

    def start_stream(self):
        """Starts the audio stream."""
        if self.stream and not self.stream.is_active():
            self.stream.start_stream()

    def stop_stream(self):
        """Stops the audio stream."""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()

    def close_stream(self):
        """Closes the audio stream."""
        if self.stream:
            self.stop_stream()
            self.stream.close()
            self.stream = None
        elif self.mpv_process:
            if self.mpv_process.stdin:
                self.mpv_process.stdin.close()
            self.mpv_process.wait()
            self.mpv_process.terminate()

    def is_stream_active(self) -> bool:
        """
        Checks if the audio stream is active.

        Returns:
            bool: True if the stream is active, False otherwise.
        """
        return self.stream and self.stream.is_active()


class AudioBufferManager:
    """
    Manages an audio buffer, allowing addition and retrieval of audio data.
    """

    # Class constant for format to bytes mapping
    FORMAT_BYTES = {
        pyaudio.paCustomFormat: 4,
        pyaudio.paFloat32: 4,
        pyaudio.paInt32: 4,
        pyaudio.paInt24: 3,
        pyaudio.paInt16: 2,
        pyaudio.paInt8: 1,
        pyaudio.paUInt8: 1,
    }

    def __init__(
            self,
            audio_buffer: queue.Queue,
            timings: queue.Queue,
            config: AudioConfiguration
        ):
        """
        Args:
            audio_buffer (queue.Queue): Queue to be used as the audio buffer.
        """
        self.config = config
        self.audio_buffer = audio_buffer
        self.timings = timings
        self.total_samples = 0

    def add_to_buffer(self, audio_data):
        """
        Adds audio data to the buffer.

        Args:
            audio_data: Audio data to be added.
        """
        self.audio_buffer.put(audio_data)
        self.total_samples += len(audio_data) // 2

    def clear_buffer(self):
        """Clears all audio data from the buffer."""
        while not self.audio_buffer.empty():
            try:
                self.audio_buffer.get_nowait()
            except queue.Empty:
                continue
        while not self.timings.empty():
            try:
                self.timings.get_nowait()
            except queue.Empty:
                continue
        self.total_samples = 0

    def get_from_buffer(self, timeout: float = 0.01):
        """
        Retrieves audio data from the buffer.

        Args:
            timeout (float): Time (in seconds) to wait
              before raising a queue.Empty exception. Default reduced to 10ms for lower latency.

        Returns:
            The audio data chunk or None if the buffer is empty.
        """
        try:
            chunk = self.audio_buffer.get(timeout=timeout)

            # Get format and channels from config
            audio_format = self.config.format
            channels = self.config.channels

            # Log if format is unknown
            if audio_format not in self.FORMAT_BYTES:
                logging.warning(f"Unknown audio format {audio_format} (0x{audio_format:x}), defaulting to 4 bytes")
                bytes_per_sample = 4
            else:
                bytes_per_sample = self.FORMAT_BYTES[audio_format]

            # Calculate bytes per frame
            bytes_per_frame = bytes_per_sample * channels

            # Update total samples counter
            if chunk:
                self.total_samples -= len(chunk) // bytes_per_frame
            return True, chunk
        except queue.Empty:
            return False, None

    def get_buffered_seconds(self, rate: int) -> float:
        """
        Calculates the duration (in seconds) of the buffered audio data.

        Args:
            rate (int): Sample rate of the audio data.

        Returns:
            float: Duration of buffered audio in seconds.
        """
        return self.total_samples / rate


class CrossfadingBuffer:
    """
    Manages audio chunks with crossfading at boundaries.

    When a new chunk arrives, it overlaps with the tail of the previous
    chunk using a configurable crossfade duration. This smooths transitions
    between audio chunks and reduces clicks/pops at boundaries.

    Optionally normalizes silence at chunk boundaries for more consistent
    pauses between sentences.
    """

    def __init__(
        self,
        crossfade_ms: float = 25.0,
        sample_rate: int = 24000,
        channels: int = 1,
        sample_width: int = 2,  # bytes per sample (2 for int16)
        normalize_boundaries: bool = True,
        silence_threshold: int = 500,  # int16 amplitude threshold
        target_silence_ms: float = 5.0,  # target silence at boundaries
    ):
        """
        Args:
            crossfade_ms: Duration of crossfade in milliseconds. Default 25ms.
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels.
            sample_width: Bytes per sample (2 for int16, 4 for float32).
            normalize_boundaries: If True, normalizes silence at chunk
                boundaries for consistent pauses. Default True.
            silence_threshold: Amplitude threshold for silence detection.
            target_silence_ms: Target silence duration at chunk end after
                normalization. Default 5ms.
        """
        self.crossfade_ms = crossfade_ms
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        self.normalize_boundaries = normalize_boundaries
        self.silence_threshold = silence_threshold
        self.target_silence_samples = int(target_silence_ms * sample_rate / 1000)
        self._pending_tail: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def _normalize_chunk_boundaries(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize silence at chunk boundaries for consistent pauses.

        Trims trailing silence and adds a consistent target silence duration.

        Args:
            audio: Audio samples as float32 array.

        Returns:
            Normalized audio array.
        """
        if len(audio) == 0:
            return audio

        # For float32 audio, convert threshold
        threshold = self.silence_threshold
        if self.sample_width == 4:
            threshold = self.silence_threshold / 32768.0

        # Handle multi-channel: work on mono representation for silence detection
        if self.channels > 1 and len(audio.shape) > 1:
            mono = np.mean(np.abs(audio), axis=1)
        else:
            mono = np.abs(audio)

        # Trim trailing silence
        non_silent = np.where(mono > threshold)[0]
        if len(non_silent) > 0:
            last_non_silent = non_silent[-1]
            # Keep a small amount of audio after last non-silent sample
            trim_point = min(last_non_silent + self.target_silence_samples, len(audio))
            audio = audio[:trim_point]

        # Trim leading silence (but keep minimal)
        if len(non_silent) > 0:
            first_non_silent = non_silent[0]
            # Allow up to target_silence_samples of leading silence
            if first_non_silent > self.target_silence_samples:
                audio = audio[first_non_silent - self.target_silence_samples:]

        return audio

    def process(self, chunk_bytes: bytes) -> bytes:
        """
        Process an audio chunk and apply crossfading with the previous chunk.

        The crossfading algorithm:
        1. Normalize boundaries (trim excess silence, add consistent silence)
        2. Extract the first crossfade_samples from the new chunk
        3. If there's a pending tail from the previous chunk:
           - Apply fade-out to pending tail
           - Apply fade-in to new chunk head
           - Sum them together (crossfade)
        4. Store the last crossfade_samples as the new pending tail
        5. Return the processed audio

        Args:
            chunk_bytes: Raw audio data as bytes.

        Returns:
            Processed audio data with crossfading applied.
        """
        with self._lock:
            # Handle zero crossfade (passthrough mode)
            if self.crossfade_samples == 0:
                return chunk_bytes

            # Convert bytes to numpy array
            if self.sample_width == 2:
                audio = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
            else:
                audio = np.frombuffer(chunk_bytes, dtype=np.float32)

            # Handle multi-channel audio
            if self.channels > 1:
                audio = audio.reshape(-1, self.channels)

            # Normalize boundaries before crossfading
            if self.normalize_boundaries:
                audio = self._normalize_chunk_boundaries(audio)

            output_parts = []

            # If we have a pending tail and enough samples in new chunk, crossfade
            if self._pending_tail is not None and len(audio) >= self.crossfade_samples:
                # Create crossfade curves
                fade_out = np.linspace(1.0, 0.0, self.crossfade_samples)
                fade_in = np.linspace(0.0, 1.0, self.crossfade_samples)

                # Reshape for multi-channel
                if self.channels > 1:
                    fade_out = fade_out.reshape(-1, 1)
                    fade_in = fade_in.reshape(-1, 1)

                # Apply crossfade
                tail_faded = self._pending_tail * fade_out
                head_faded = audio[: self.crossfade_samples] * fade_in
                crossfaded = tail_faded + head_faded

                output_parts.append(crossfaded)
                audio = audio[self.crossfade_samples :]

            elif self._pending_tail is not None:
                # Chunk too short for crossfading, just output the tail
                output_parts.append(self._pending_tail)

            # Store new tail if chunk is long enough
            if len(audio) > self.crossfade_samples:
                self._pending_tail = audio[-self.crossfade_samples :].copy()
                audio = audio[: -self.crossfade_samples]
                output_parts.append(audio)
            elif len(audio) > 0:
                # Chunk too short, don't hold a tail
                output_parts.append(audio)
                self._pending_tail = None
            else:
                # No audio left after crossfading
                pass

            # Combine all parts
            if output_parts:
                combined = np.concatenate(output_parts)
            else:
                combined = np.array([], dtype=np.float32)

            # Flatten multi-channel
            if self.channels > 1 and len(combined) > 0:
                combined = combined.flatten()

            # Convert back to bytes
            if self.sample_width == 2:
                return np.clip(combined, -32768, 32767).astype(np.int16).tobytes()
            else:
                return combined.astype(np.float32).tobytes()

    def flush(self) -> bytes:
        """
        Return any pending tail as final output.

        Call this when playback is stopping to ensure the last
        crossfade_samples worth of audio is played.

        Returns:
            Final audio data or empty bytes.
        """
        with self._lock:
            if self._pending_tail is not None:
                tail = self._pending_tail
                self._pending_tail = None

                # Flatten multi-channel
                if self.channels > 1 and len(tail) > 0:
                    tail = tail.flatten()

                if self.sample_width == 2:
                    return tail.astype(np.int16).tobytes()
                else:
                    return tail.astype(np.float32).tobytes()
            return b""

    def reset(self) -> None:
        """Clear the crossfade buffer."""
        with self._lock:
            self._pending_tail = None


class StreamPlayer:
    """
    Manages audio playback operations such as start, stop, pause, and resume.
    """

    def __init__(
        self,
        audio_buffer: queue.Queue,
        timings: queue.Queue,
        config: AudioConfiguration,
        on_playback_start=None,
        on_playback_stop=None,
        on_audio_chunk=None,
        on_word_spoken=None,
        muted=False,
        diagnostics: Optional["PlaybackDiagnostics"] = None,
        enable_crossfade: bool = True,
        crossfade_ms: float = 25.0,
    ):
        """
        Args:
            audio_buffer (queue.Queue): Queue to be used as the audio buffer.
            timings (queue.Queue): Queue for word timing information.
            config (AudioConfiguration): Object containing audio settings.
            on_playback_start (Callable, optional): Callback function to be
              called at the start of playback. Defaults to None.
            on_playback_stop (Callable, optional): Callback function to be
              called at the stop of playback. Defaults to None.
            on_audio_chunk (Callable, optional): Callback function called with
              each audio chunk *after* effects and resampling, just before
              being written to the output device/file. Defaults to None.
            on_word_spoken (Callable, optional): Callback for word timing events.
            muted (bool): Initial muted state.
            diagnostics (PlaybackDiagnostics, optional): Diagnostics collector
              for tracking playback timing. Defaults to None.
            enable_crossfade (bool): If True, enables crossfading between audio
              chunks to smooth transitions. Defaults to True.
            crossfade_ms (float): Duration of crossfade in milliseconds.
              Defaults to 15.0ms.
        """
        self.buffer_manager = AudioBufferManager(audio_buffer, timings, config)
        self.timings = timings
        self.timings_list = []
        self.audio_stream = AudioStream(config)
        self.playback_active = False
        self.immediate_stop = threading.Event()
        self.pause_event = threading.Event()
        self.playback_done = threading.Event()
        self.playback_thread = None
        self.on_playback_start = on_playback_start
        self.on_playback_stop = on_playback_stop
        self.on_audio_chunk = on_audio_chunk
        self.on_word_spoken = on_word_spoken
        self.first_chunk_played = False
        self.muted = muted
        self.seconds_played = 0
        self.diagnostics = diagnostics
        self._current_chunk_id = 0  # Track which chunk is being played
        self._last_buffer_log_time = 0.0  # For periodic buffer logging
        self.enable_crossfade = enable_crossfade
        self.crossfade_ms = crossfade_ms
        self.crossfade_buffer: Optional[CrossfadingBuffer] = None

    def _play_mpeg_chunk(self, chunk):
        """
        Plays a chunk of audio data using mpv.

        Args:
            chunk: Chunk of audio data to be played.
        """
        try:
            # Pause playback if the event is set
            if not self.first_chunk_played and self.on_playback_start:
                self.on_playback_start()
                self.first_chunk_played = True

            if not self.muted:
                if self.audio_stream.mpv_process and self.audio_stream.mpv_process.stdin:
                    self.audio_stream.mpv_process.stdin.write(chunk)
                    self.audio_stream.mpv_process.stdin.flush()

            if self.on_audio_chunk:
                self.on_audio_chunk(chunk)

            import time
            while self.pause_event.is_set():
                time.sleep(0.01)

        except Exception as e:
            print(f"Error sending audio data to mpv: {e}")

    def _play_wav_chunk(self, chunk):
        if self.audio_stream.config.format == pyaudio.paCustomFormat:
            segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
            chunk = segment.raw_data
            sample_width = segment.sample_width
            channels = segment.channels
        else:
            sample_width = self.audio_stream.pyaudio_instance.get_sample_size(self.audio_stream.config.format)
            channels = self.audio_stream.config.channels

        if self.audio_stream.config.rate != self.audio_stream.actual_sample_rate and self.audio_stream.actual_sample_rate > 0:
            if self.audio_stream.config.format == pyaudio.paFloat32:
                audio_data = np.frombuffer(chunk, dtype=np.float32)
                resampled_data = resampy.resample(audio_data, self.audio_stream.config.rate, self.audio_stream.actual_sample_rate)
                chunk = resampled_data.astype(np.float32).tobytes()
            else:
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
                resampled_data = resampy.resample(audio_data, self.audio_stream.config.rate, self.audio_stream.actual_sample_rate)
                chunk = (resampled_data * 32768.0).astype(np.int16).tobytes()

        if self.audio_stream.config.playout_chunk_size > 0:
            sub_chunk_size = self.audio_stream.config.playout_chunk_size
        else:
            if self.audio_stream.config.frames_per_buffer == pa.paFramesPerBufferUnspecified:
                sub_chunk_size = 512
            else:
                sub_chunk_size = self.audio_stream.config.frames_per_buffer * sample_width * channels

        for i in range(0, len(chunk), sub_chunk_size):
            sub_chunk = chunk[i : i + sub_chunk_size]

            if not self.first_chunk_played and self.on_playback_start:
                self.on_playback_start()
                self.first_chunk_played = True

            if not self.muted:
                try:
                    import time

                    # Define the timeout duration in seconds
                    timeout = 0.1

                    # Record the start time
                    start_time = time.time()

                    frames_in_sub_chunk = len(sub_chunk) // (sample_width * channels)

                    # Wait until there's space in the buffer or the timeout is reached
                    while self.audio_stream.stream.get_write_available() < frames_in_sub_chunk:
                        if time.time() - start_time > timeout:
                            print(f"Wait aborted: Timeout of {timeout}s exceeded. "
                                f"Buffer availability: {self.audio_stream.stream.get_write_available()}, "
                                f"Frames in sub-chunk: {frames_in_sub_chunk}")
                            break
                        time.sleep(0.001)  # Small sleep to let the stream process audio

                    self.audio_stream.stream.write(sub_chunk)
                    self.seconds_played += len(sub_chunk) / (self.audio_stream.config.rate * sample_width * channels)
                    while (True):
                        try:
                            timing = self.timings.get_nowait()
                            self.timings_list.append(timing)
                        except queue.Empty:
                            break

                    for timing in self.timings_list:
                        if timing.start_time <= self.seconds_played:
                            if self.on_word_spoken:
                                self.on_word_spoken(timing)
                            self.timings_list.remove(timing)
                            break
                except Exception as e:
                    print(f"streaming-tts error sending audio data: {e}")

            if self.on_audio_chunk:
                self.on_audio_chunk(sub_chunk)

            # Pause playback if the event is set
            while self.pause_event.is_set():
                time.sleep(0.01)

            if self.immediate_stop.is_set():
                break

    def _play_chunk(self, chunk):
        """
        Plays a chunk of audio data.

        Args:
            chunk: Chunk of audio data to be played.
        """
        # --- Handle Raw MPEG Stream (MPV) ---
        is_mpeg_stream = (
            self.audio_stream.config.format == pyaudio.paCustomFormat and
            self.audio_stream.config.channels == -1 and
            self.audio_stream.config.rate == -1
        )

        if is_mpeg_stream:
            self._play_mpeg_chunk(chunk)
            return  # Finished processing MPEG chunk

        # Apply crossfading if enabled
        if self.crossfade_buffer is not None:
            chunk = self.crossfade_buffer.process(chunk)
            if not chunk:  # Chunk absorbed into crossfade buffer
                return

        self._play_wav_chunk(chunk)

    def _process_buffer(self):
        """
        Processes and plays audio data from the buffer
        until it's empty or playback is stopped.
        """
        try:
            while self.playback_active or not self.buffer_manager.audio_buffer.empty():
                success, chunk = self.buffer_manager.get_from_buffer()
                if chunk:
                    # Record playback start for this chunk
                    if self.diagnostics:
                        self.diagnostics.record_playback_start(self._current_chunk_id)

                    self._play_chunk(chunk)

                    # Record playback end and move to next chunk
                    if self.diagnostics:
                        self.diagnostics.record_playback_end(self._current_chunk_id)
                        self._current_chunk_id += 1

                        # Periodic buffer state logging (every 100ms)
                        now = time.perf_counter()
                        if now - self._last_buffer_log_time > 0.1:
                            buffered_secs = self.get_buffered_seconds()
                            sample_count = self.buffer_manager.total_samples
                            self.diagnostics.record_buffer_state(sample_count, buffered_secs)
                            self._last_buffer_log_time = now

                if self.immediate_stop.is_set():
                    logging.info("Immediate stop requested, aborting playback")
                    break

            if self.on_playback_stop:
                self.on_playback_stop()
        finally:
            self.playback_done.set()

    def get_buffered_seconds(self) -> float:
        """
        Calculates the duration (in seconds) of the buffered audio data.

        Returns:
            float: Duration of buffered audio in seconds.
        """
        if self.audio_stream.config.rate > 0:
            return self.buffer_manager.get_buffered_seconds(self.audio_stream.config.rate)
        else: # mpeg
            return self.buffer_manager.get_buffered_seconds(16000)

    def start(self):
        """Starts audio playback."""
        self.first_chunk_played = False
        self.playback_active = True
        self.playback_done.clear()
        self._current_chunk_id = 0  # Reset chunk counter for diagnostics
        self._last_buffer_log_time = 0.0
        if not self.audio_stream.stream:
            self.audio_stream.open_stream()

        # Initialize crossfade buffer if enabled
        if self.enable_crossfade and self.audio_stream.config.rate > 0:
            sample_width = 2  # Default to int16
            if self.audio_stream.config.format == pyaudio.paFloat32:
                sample_width = 4
            self.crossfade_buffer = CrossfadingBuffer(
                crossfade_ms=self.crossfade_ms,
                sample_rate=self.audio_stream.actual_sample_rate or self.audio_stream.config.rate,
                channels=self.audio_stream.config.channels,
                sample_width=sample_width,
            )
        else:
            self.crossfade_buffer = None

        self.audio_stream.start_stream()

        if not self.playback_thread or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._process_buffer)
            self.playback_thread.start()

    def stop(self, immediate: bool = False):
        """
        Stops audio playback.

        Args:
            immediate (bool): If True, stops playback immediately
              without waiting for buffer to empty.
        """
        if not self.playback_thread:
            logging.warning("No playback thread found, cannot stop playback")
            return

        if immediate:
            self.immediate_stop.set()
            # Wait for playback to finish with timeout instead of busy-wait
            if not self.playback_done.wait(timeout=5.0):
                logging.warning("Playback did not stop within timeout")
            self.playback_active = False
        else:
            self.playback_active = False
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=10.0)

        # Flush crossfade buffer before closing stream
        if self.crossfade_buffer is not None and not immediate:
            final_audio = self.crossfade_buffer.flush()
            if final_audio:
                self._play_wav_chunk(final_audio)
            self.crossfade_buffer.reset()

        self.audio_stream.close_stream()
        self.immediate_stop.clear()
        self.buffer_manager.clear_buffer()
        self.playback_thread = None

    def pause(self):
        """Pauses audio playback."""
        self.pause_event.set()

    def resume(self):
        """Resumes paused audio playback."""
        self.pause_event.clear()

    def mute(self, muted: bool = True):
        """Mutes audio playback."""
        self.muted = muted
