"""Shared fixtures for streaming-tts tests."""

import pytest
import numpy as np
import queue


# --- Sample Data Fixtures ---

@pytest.fixture
def sample_text():
    """Standard test text."""
    return "Hello, world! This is a test. How are you?"


@pytest.fixture
def sample_text_with_pauses():
    """Text containing pause tags."""
    return "Hello. [pause:1s] World! [pause:500ms] Goodbye."


@pytest.fixture
def sample_audio_int16():
    """Sample int16 audio data (1 second at 24kHz)."""
    return np.random.randint(-32768, 32767, 24000, dtype=np.int16)


@pytest.fixture
def sample_audio_float32():
    """Sample float32 audio data (-1.0 to 1.0)."""
    return np.random.uniform(-1.0, 1.0, 24000).astype(np.float32)


@pytest.fixture
def silent_audio():
    """Audio with silence at start and end."""
    audio = np.zeros(24000, dtype=np.int16)
    # Add some audio in the middle
    audio[2400:21600] = np.random.randint(-1000, 1000, 19200, dtype=np.int16)
    return audio


@pytest.fixture
def audio_with_silence_edges():
    """Audio with leading and trailing silence."""
    audio = np.zeros(48000, dtype=np.float32)
    # Silence: 0-12000, audio: 12000-36000, silence: 36000-48000
    audio[12000:36000] = np.random.uniform(-0.5, 0.5, 24000).astype(np.float32)
    return audio


# --- Mock Fixtures ---

@pytest.fixture
def mock_queue():
    """Thread-safe queue for testing."""
    return queue.Queue()


# --- PyAV Availability ---

@pytest.fixture
def has_pyav():
    """Check if PyAV is available."""
    import importlib.util
    return importlib.util.find_spec("av") is not None


# --- Pytest Markers ---

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_pyav: marks tests that require PyAV"
    )
