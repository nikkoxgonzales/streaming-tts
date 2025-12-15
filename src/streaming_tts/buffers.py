"""Thread-safe buffer classes for audio streaming."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


class ChunkBuffer:
    """
    Thread-safe buffer for audio chunks.

    This buffer is designed for producer-consumer patterns where
    audio chunks are generated in one thread and consumed in another.

    Example:
        buffer = ChunkBuffer(max_size=100)

        # Producer thread
        buffer.put(chunk)
        buffer.mark_done()

        # Consumer thread
        for chunk in buffer:
            process(chunk)
    """

    def __init__(self, max_size: int = 500) -> None:
        """
        Initialize the chunk buffer.

        Args:
            max_size: Maximum number of chunks to buffer (0 = unlimited)
        """
        self._max_size = max_size
        self._buffer: deque[bytes] = deque(maxlen=max_size if max_size > 0 else None)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._done = threading.Event()
        self._cancelled = threading.Event()

    def put(self, chunk: bytes, timeout: float | None = 5.0) -> bool:
        """
        Add a chunk to the buffer.

        Args:
            chunk: Audio data to add
            timeout: Maximum time to wait if buffer is full (None = block forever)

        Returns:
            True if chunk was added, False if cancelled or timed out
        """
        if self._cancelled.is_set():
            return False

        with self._not_empty:
            # Wait if buffer is full (only if max_size > 0)
            if self._max_size > 0:
                deadline = None
                if timeout is not None:
                    import time

                    deadline = time.monotonic() + timeout

                while len(self._buffer) >= self._max_size:
                    if self._cancelled.is_set():
                        return False

                    remaining = None
                    if deadline is not None:
                        remaining = deadline - __import__("time").monotonic()
                        if remaining <= 0:
                            return False

                    self._not_empty.wait(timeout=remaining or 0.1)

            self._buffer.append(chunk)
            self._not_empty.notify_all()
            return True

    def get(self, timeout: float | None = 0.1) -> bytes | None:
        """
        Get a chunk from the buffer.

        Args:
            timeout: Maximum time to wait for a chunk (None = block forever)

        Returns:
            Audio chunk, or None if buffer is done/cancelled/timed out
        """
        with self._not_empty:
            while len(self._buffer) == 0:
                if self._done.is_set() or self._cancelled.is_set():
                    return None

                if not self._not_empty.wait(timeout=timeout):
                    # Timeout - check if we should continue waiting
                    if self._done.is_set() or self._cancelled.is_set():
                        return None
                    if len(self._buffer) == 0:
                        return None

            chunk = self._buffer.popleft()
            self._not_empty.notify_all()
            return chunk

    def get_batch(self, max_chunks: int = 8) -> list[bytes]:
        """
        Get multiple chunks at once for batch processing.

        Args:
            max_chunks: Maximum number of chunks to retrieve

        Returns:
            List of chunks (may be empty)
        """
        with self._lock:
            chunks: list[bytes] = []
            while len(self._buffer) > 0 and len(chunks) < max_chunks:
                chunks.append(self._buffer.popleft())
            return chunks

    def mark_done(self) -> None:
        """Signal that no more chunks will be added."""
        self._done.set()
        with self._not_empty:
            self._not_empty.notify_all()

    def cancel(self) -> None:
        """Cancel all operations and clear the buffer."""
        self._cancelled.set()
        with self._not_empty:
            self._buffer.clear()
            self._not_empty.notify_all()

    def reset(self) -> None:
        """Reset the buffer for reuse."""
        with self._lock:
            self._buffer.clear()
            self._done.clear()
            self._cancelled.clear()

    @property
    def is_done(self) -> bool:
        """Check if buffer is marked as done."""
        return self._done.is_set()

    @property
    def is_cancelled(self) -> bool:
        """Check if buffer is cancelled."""
        return self._cancelled.is_set()

    def __len__(self) -> int:
        """Return number of chunks in buffer."""
        with self._lock:
            return len(self._buffer)

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over chunks until done or cancelled."""
        while True:
            chunk = self.get()
            if chunk is None:
                break
            yield chunk


class AsyncChunkBuffer:
    """
    Async-compatible buffer for audio chunks.

    This buffer uses asyncio.Queue for async/await patterns.

    Example:
        buffer = AsyncChunkBuffer()

        # Producer (can be sync via put_nowait or async via put)
        await buffer.put(chunk)
        buffer.mark_done()

        # Consumer
        async for chunk in buffer:
            await process(chunk)
    """

    def __init__(self, max_size: int = 500) -> None:
        """
        Initialize the async chunk buffer.

        Args:
            max_size: Maximum number of chunks to buffer (0 = unlimited)
        """
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=max_size if max_size > 0 else 0
        )
        self._done = False
        self._cancelled = False

    async def put(self, chunk: bytes) -> bool:
        """
        Add a chunk to the buffer (async).

        Args:
            chunk: Audio data to add

        Returns:
            True if added, False if cancelled
        """
        if self._cancelled:
            return False
        await self._queue.put(chunk)
        return True

    def put_nowait(self, chunk: bytes) -> bool:
        """
        Add a chunk without waiting (for use from sync code).

        Args:
            chunk: Audio data to add

        Returns:
            True if added, False if cancelled or full
        """
        if self._cancelled:
            return False
        try:
            self._queue.put_nowait(chunk)
            return True
        except asyncio.QueueFull:
            return False

    async def get(self) -> bytes | None:
        """
        Get a chunk from the buffer.

        Returns:
            Audio chunk, or None if done/cancelled
        """
        if self._cancelled:
            return None

        try:
            chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            return chunk
        except TimeoutError:
            if self._done and self._queue.empty():
                return None
            return await self.get() if not self._cancelled else None

    def mark_done(self) -> None:
        """Signal that no more chunks will be added."""
        self._done = True
        # Put sentinel value to unblock waiting consumers
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)

    def cancel(self) -> None:
        """Cancel all operations."""
        self._cancelled = True
        self._done = True
        # Try to unblock consumers
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)

    def reset(self) -> None:
        """Reset the buffer for reuse."""
        self._done = False
        self._cancelled = False
        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @property
    def is_done(self) -> bool:
        """Check if buffer is marked as done."""
        return self._done

    @property
    def is_cancelled(self) -> bool:
        """Check if buffer is cancelled."""
        return self._cancelled

    def __len__(self) -> int:
        """Return approximate number of chunks in buffer."""
        return self._queue.qsize()

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Return async iterator."""
        return self

    async def __anext__(self) -> bytes:
        """Get next chunk or raise StopAsyncIteration."""
        chunk = await self.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk
