"""Integration tests for streaming-tts."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from streaming_tts import TTSConfig, TTSStream


@pytest.fixture
def mock_pipeline():
    """Create a mock Kokoro pipeline."""
    with patch("kokoro.KPipeline") as mock_cls:
        # Create multiple chunks to simulate longer synthesis
        def generate_chunks():
            for i in range(3):
                mock_result = MagicMock()
                mock_result.audio = torch.randn(8000)  # ~333ms per chunk
                mock_result.tokens = []
                yield mock_result

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = generate_chunks()
        mock_pipeline_instance.load_single_voice = MagicMock(
            return_value=torch.randn(256)
        )
        mock_cls.return_value = mock_pipeline_instance

        yield mock_cls


class TestEndToEndCallbackPattern:
    """Integration tests for callback-based streaming pattern."""

    def test_basic_callback_flow(self, mock_pipeline) -> None:
        """Test basic callback-based synthesis flow."""
        received_chunks = []
        start_called = threading.Event()
        stop_called = threading.Event()

        with TTSStream() as stream:
            stream.feed("Hello world")
            stream.play(
                on_chunk=received_chunks.append,
                on_start=start_called.set,
                on_stop=stop_called.set,
                muted=True,
            )

        assert start_called.is_set()
        assert stop_called.is_set()
        assert len(received_chunks) > 0
        assert all(isinstance(c, bytes) for c in received_chunks)

    def test_websocket_simulation(self, mock_pipeline) -> None:
        """Simulate WebSocket streaming pattern."""
        sent_messages = []

        class MockWebSocket:
            def send_bytes(self, data: bytes) -> None:
                sent_messages.append(data)

        ws = MockWebSocket()

        with TTSStream() as stream:
            stream.feed("Hello from the TTS server")
            stream.play(on_chunk=ws.send_bytes, muted=True)

        assert len(sent_messages) > 0
        total_bytes = sum(len(m) for m in sent_messages)
        assert total_bytes > 0

    def test_voice_change_mid_session(self, mock_pipeline) -> None:
        """Test changing voice between syntheses."""
        chunks_voice1 = []
        chunks_voice2 = []

        with TTSStream() as stream:
            # First synthesis with default voice
            stream.feed("Hello")
            stream.play(on_chunk=chunks_voice1.append, muted=True)

            # Change voice
            stream.set_voice("am_adam")

            # Reset mock to generate new chunks
            mock_pipeline.return_value.return_value = iter([
                MagicMock(audio=torch.randn(8000), tokens=[])
            ])

            # Second synthesis with new voice
            stream.feed("World")
            stream.play(on_chunk=chunks_voice2.append, muted=True)

        assert len(chunks_voice1) > 0
        assert len(chunks_voice2) > 0


class TestEndToEndAsyncPattern:
    """Integration tests for async iterator pattern."""

    @pytest.mark.asyncio
    async def test_basic_async_flow(self, mock_pipeline) -> None:
        """Test basic async streaming flow."""
        chunks = []

        async with asyncio.timeout(10):
            with TTSStream() as stream:
                stream.feed("Hello world")
                async for chunk in stream.stream_async():
                    chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    @pytest.mark.asyncio
    async def test_async_websocket_simulation(self, mock_pipeline) -> None:
        """Simulate async WebSocket streaming pattern."""
        sent_messages = []

        class MockAsyncWebSocket:
            async def send_bytes(self, data: bytes) -> None:
                await asyncio.sleep(0.001)  # Simulate network delay
                sent_messages.append(data)

        ws = MockAsyncWebSocket()

        async with asyncio.timeout(10):
            with TTSStream() as stream:
                stream.feed("Hello from async TTS")
                async for chunk in stream.stream_async():
                    await ws.send_bytes(chunk)

        assert len(sent_messages) > 0


class TestEndToEndSyncIterator:
    """Integration tests for sync iterator pattern."""

    def test_basic_sync_iterator(self, mock_pipeline) -> None:
        """Test basic sync iterator flow."""
        with TTSStream() as stream:
            stream.feed("Hello world")
            chunks = list(stream.stream())

        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    def test_sync_iterator_processing(self, mock_pipeline) -> None:
        """Test processing chunks from sync iterator."""
        total_samples = 0

        with TTSStream() as stream:
            stream.feed("Hello world")
            for chunk in stream.stream():
                # Each chunk is PCM16 (2 bytes per sample)
                total_samples += len(chunk) // 2

        assert total_samples > 0


class TestStopBehavior:
    """Integration tests for stop functionality."""

    def test_stop_during_callback(self, mock_pipeline) -> None:
        """Test stopping during callback-based synthesis."""
        chunks_received = []
        stop_after = 1

        with TTSStream() as stream:
            def on_chunk(chunk):
                chunks_received.append(chunk)
                if len(chunks_received) >= stop_after:
                    stream.stop()

            stream.feed("This is a long text that should be interrupted")
            stream.play(on_chunk=on_chunk, muted=True)

        # Should have stopped early
        assert len(chunks_received) <= stop_after + 1

    def test_stop_during_async_iteration(self, mock_pipeline) -> None:
        """Test stopping during async iteration."""

        @pytest.mark.asyncio
        async def run_test():
            chunks = []

            async with asyncio.timeout(10):
                with TTSStream() as stream:
                    stream.feed("Long text to be interrupted")
                    async for chunk in stream.stream_async():
                        chunks.append(chunk)
                        if len(chunks) >= 1:
                            stream.stop()
                            break

            return chunks

        # Run the async test
        chunks = asyncio.get_event_loop().run_until_complete(run_test())
        assert len(chunks) >= 1


class TestConcurrentUsage:
    """Integration tests for concurrent usage patterns."""

    def test_background_synthesis(self, mock_pipeline) -> None:
        """Test background synthesis with play_async."""
        with TTSStream() as stream:
            stream.feed("Background synthesis")
            thread = stream.play_async(muted=True)

            # Do other work while synthesis runs
            time.sleep(0.01)

            # Wait for completion
            thread.join(timeout=5)

            assert not thread.is_alive()
            assert not stream.is_playing()

    def test_multiple_sequential_syntheses(self, mock_pipeline) -> None:
        """Test multiple sequential syntheses."""
        all_chunks = []

        with TTSStream() as stream:
            for text in ["First", "Second", "Third"]:
                # Reset mock for each synthesis
                mock_pipeline.return_value.return_value = iter([
                    MagicMock(audio=torch.randn(8000), tokens=[])
                ])

                chunks = []
                stream.feed(text)
                stream.play(on_chunk=chunks.append, muted=True)
                all_chunks.extend(chunks)

        # Should have chunks from all three syntheses
        assert len(all_chunks) >= 3


class TestConfigurationIntegration:
    """Integration tests for configuration options."""

    def test_custom_voice(self, mock_pipeline) -> None:
        """Test synthesis with custom voice."""
        config = TTSConfig(voice="am_adam")

        with TTSStream(config=config) as stream:
            stream.feed("Hello with custom voice")
            chunks = list(stream.stream())

        assert len(chunks) > 0

    def test_speed_configuration(self, mock_pipeline) -> None:
        """Test synthesis with custom speed."""
        config = TTSConfig(speed=1.5)

        with TTSStream(config=config) as stream:
            stream.feed("Fast speech")
            chunks = list(stream.stream())

        assert len(chunks) > 0

    def test_trim_silence_disabled(self, mock_pipeline) -> None:
        """Test synthesis with silence trimming disabled."""
        config = TTSConfig(trim_silence=False)

        with TTSStream(config=config) as stream:
            stream.feed("No trimming")
            chunks = list(stream.stream())

        assert len(chunks) > 0


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_empty_text(self, mock_pipeline) -> None:
        """Test handling of empty text."""
        with TTSStream() as stream:
            # Feed empty text
            stream.feed("")
            stream.feed("   ")

            # Should handle gracefully
            chunks = list(stream.stream())
            assert chunks == []

    def test_shutdown_cleanup(self, mock_pipeline) -> None:
        """Test that shutdown properly cleans up resources."""
        stream = TTSStream()
        stream.feed("Hello")
        list(stream.stream())
        stream.shutdown()

        # Should be safe to call shutdown multiple times
        stream.shutdown()

        # Executor should be shut down
        assert stream._executor._shutdown
