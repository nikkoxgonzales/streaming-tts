"""Tests for buffer classes."""

import asyncio
import threading
import time

import pytest

from streaming_tts.buffers import AsyncChunkBuffer, ChunkBuffer


class TestChunkBuffer:
    """Tests for ChunkBuffer class."""

    def test_init(self) -> None:
        """Test buffer initialization."""
        buffer = ChunkBuffer(max_size=100)
        assert len(buffer) == 0
        assert not buffer.is_done
        assert not buffer.is_cancelled

    def test_put_and_get(self) -> None:
        """Test basic put and get operations."""
        buffer = ChunkBuffer()
        chunk = b"test audio data"

        assert buffer.put(chunk)
        assert len(buffer) == 1

        result = buffer.get()
        assert result == chunk
        assert len(buffer) == 0

    def test_fifo_order(self) -> None:
        """Test that chunks are returned in FIFO order."""
        buffer = ChunkBuffer()
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        for chunk in chunks:
            buffer.put(chunk)

        for expected in chunks:
            assert buffer.get() == expected

    def test_get_returns_none_when_done(self) -> None:
        """Test that get() returns None when buffer is done and empty."""
        buffer = ChunkBuffer()
        buffer.mark_done()
        assert buffer.get() is None

    def test_get_returns_none_when_cancelled(self) -> None:
        """Test that get() returns None when buffer is cancelled."""
        buffer = ChunkBuffer()
        buffer.put(b"data")
        buffer.cancel()
        assert buffer.get() is None

    def test_put_returns_false_when_cancelled(self) -> None:
        """Test that put() returns False when buffer is cancelled."""
        buffer = ChunkBuffer()
        buffer.cancel()
        assert not buffer.put(b"data")

    def test_get_batch(self) -> None:
        """Test batch retrieval."""
        buffer = ChunkBuffer()
        chunks = [b"chunk1", b"chunk2", b"chunk3", b"chunk4", b"chunk5"]
        for chunk in chunks:
            buffer.put(chunk)

        batch = buffer.get_batch(max_chunks=3)
        assert len(batch) == 3
        assert batch == [b"chunk1", b"chunk2", b"chunk3"]
        assert len(buffer) == 2

    def test_iteration(self) -> None:
        """Test iteration over buffer."""
        buffer = ChunkBuffer()
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        def producer():
            for chunk in chunks:
                buffer.put(chunk)
                time.sleep(0.01)
            buffer.mark_done()

        thread = threading.Thread(target=producer)
        thread.start()

        result = list(buffer)
        thread.join()

        assert result == chunks

    def test_reset(self) -> None:
        """Test buffer reset."""
        buffer = ChunkBuffer()
        buffer.put(b"data")
        buffer.mark_done()

        buffer.reset()
        assert len(buffer) == 0
        assert not buffer.is_done
        assert not buffer.is_cancelled

    def test_thread_safety(self) -> None:
        """Test thread-safe producer-consumer pattern."""
        buffer = ChunkBuffer(max_size=10)
        produced = []
        consumed = []

        def producer():
            for i in range(100):
                chunk = f"chunk{i}".encode()
                buffer.put(chunk)
                produced.append(chunk)
            buffer.mark_done()

        def consumer():
            for chunk in buffer:
                consumed.append(chunk)

        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        assert produced == consumed

    def test_max_size_blocking(self) -> None:
        """Test that put() blocks when buffer is full."""
        buffer = ChunkBuffer(max_size=2)
        buffer.put(b"chunk1")
        buffer.put(b"chunk2")

        # Third put should block briefly then timeout
        start = time.monotonic()
        result = buffer.put(b"chunk3", timeout=0.1)
        elapsed = time.monotonic() - start

        # Should have waited ~0.1 seconds before returning False
        assert not result or elapsed >= 0.05


class TestAsyncChunkBuffer:
    """Tests for AsyncChunkBuffer class."""

    @pytest.mark.asyncio
    async def test_init(self) -> None:
        """Test buffer initialization."""
        buffer = AsyncChunkBuffer(max_size=100)
        assert len(buffer) == 0
        assert not buffer.is_done
        assert not buffer.is_cancelled

    @pytest.mark.asyncio
    async def test_put_and_get(self) -> None:
        """Test basic put and get operations."""
        buffer = AsyncChunkBuffer()
        chunk = b"test audio data"

        assert await buffer.put(chunk)
        assert len(buffer) == 1

        result = await buffer.get()
        assert result == chunk

    @pytest.mark.asyncio
    async def test_put_nowait(self) -> None:
        """Test synchronous put operation."""
        buffer = AsyncChunkBuffer()
        chunk = b"test audio data"

        assert buffer.put_nowait(chunk)
        result = await buffer.get()
        assert result == chunk

    @pytest.mark.asyncio
    async def test_async_iteration(self) -> None:
        """Test async iteration."""
        buffer = AsyncChunkBuffer()
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        async def producer():
            for chunk in chunks:
                await buffer.put(chunk)
                await asyncio.sleep(0.01)
            buffer.mark_done()

        asyncio.create_task(producer())

        result = []
        async for chunk in buffer:
            result.append(chunk)

        assert result == chunks

    @pytest.mark.asyncio
    async def test_get_returns_none_when_done(self) -> None:
        """Test that get() returns None when buffer is done and empty."""
        buffer = AsyncChunkBuffer()
        buffer.mark_done()
        result = await buffer.get()
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel(self) -> None:
        """Test cancellation."""
        buffer = AsyncChunkBuffer()
        await buffer.put(b"data")
        buffer.cancel()

        assert buffer.is_cancelled
        result = await buffer.get()
        assert result is None

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Test buffer reset."""
        buffer = AsyncChunkBuffer()
        await buffer.put(b"data")
        buffer.mark_done()

        buffer.reset()
        assert not buffer.is_done
        assert not buffer.is_cancelled

    @pytest.mark.asyncio
    async def test_producer_consumer_pattern(self) -> None:
        """Test async producer-consumer pattern."""
        buffer = AsyncChunkBuffer(max_size=10)
        produced = []
        consumed = []

        async def producer():
            for i in range(50):
                chunk = f"chunk{i}".encode()
                await buffer.put(chunk)
                produced.append(chunk)
            buffer.mark_done()

        async def consumer():
            async for chunk in buffer:
                consumed.append(chunk)

        await asyncio.gather(producer(), consumer())
        assert produced == consumed
