"""Pytest configuration and shared fixtures."""

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def mock_kokoro_pipeline():
    """
    Create a mock Kokoro pipeline for testing without GPU.

    This fixture patches the KPipeline class from kokoro to return
    mock audio data, allowing tests to run in CI without GPU or
    model downloads.
    """
    with patch("kokoro.KPipeline") as mock_cls:
        # Create mock result that mimics Kokoro's output
        mock_result = MagicMock()
        mock_result.audio = torch.randn(24000)  # 1 second at 24kHz
        mock_result.graphemes = "Hello world"
        mock_result.phonemes = "həˈloʊ wɜːrld"

        # Mock tokens for timing info
        mock_token = MagicMock()
        mock_token.start_ts = 0.0
        mock_token.end_ts = 0.5
        mock_token.text = "Hello"
        mock_result.tokens = [mock_token]

        # Create mock pipeline instance
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter([mock_result])
        mock_pipeline.load_single_voice = MagicMock(
            return_value=torch.randn(256)
        )
        mock_cls.return_value = mock_pipeline

        yield mock_cls


@pytest.fixture
def mock_pyaudio():
    """
    Create a mock PyAudio module for testing audio playback.

    This fixture patches PyAudio to avoid requiring audio hardware
    in test environments.
    """
    mock_pa_module = MagicMock()
    mock_pa_module.paInt16 = 8

    mock_stream = MagicMock()
    mock_stream.is_active.return_value = True

    mock_pa_instance = MagicMock()
    mock_pa_instance.open.return_value = mock_stream
    mock_pa_instance.get_device_count.return_value = 1
    mock_pa_instance.get_device_info_by_index.return_value = {
        "name": "Default",
        "maxOutputChannels": 2,
        "defaultSampleRate": 44100,
    }

    mock_pa_module.PyAudio.return_value = mock_pa_instance

    with patch.dict("sys.modules", {"pyaudio": mock_pa_module}):
        yield mock_pa_module, mock_pa_instance, mock_stream
