"""
Diagnostics module for measuring and analyzing audio playback timing.

Provides tools to track chunk timing through the synthesis-to-playback pipeline
and identify gaps between audio chunks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import threading
import time


@dataclass
class ChunkTiming:
    """Timing information for a single audio chunk through the pipeline."""

    chunk_id: int
    text_preview: str = ""  # First 30 chars of synthesized text
    synthesis_start: float = 0.0  # time.perf_counter() when synthesis began
    synthesis_end: float = 0.0  # time.perf_counter() when synthesis finished
    queue_time: float = 0.0  # time.perf_counter() when chunk was queued
    playback_start: float = 0.0  # time.perf_counter() when playback began
    playback_end: float = 0.0  # time.perf_counter() when playback finished
    audio_duration_ms: float = 0.0  # Duration of the audio chunk in ms
    sample_count: int = 0  # Number of audio samples

    @property
    def synthesis_duration_ms(self) -> float:
        """Time spent synthesizing this chunk."""
        if self.synthesis_end > 0 and self.synthesis_start > 0:
            return (self.synthesis_end - self.synthesis_start) * 1000
        return 0.0

    @property
    def queue_latency_ms(self) -> float:
        """Time from synthesis end to queue."""
        if self.queue_time > 0 and self.synthesis_end > 0:
            return (self.queue_time - self.synthesis_end) * 1000
        return 0.0

    @property
    def playback_latency_ms(self) -> float:
        """Time from queue to playback start."""
        if self.playback_start > 0 and self.queue_time > 0:
            return (self.playback_start - self.queue_time) * 1000
        return 0.0


@dataclass
class GapInfo:
    """Information about a gap between two consecutive audio chunks."""

    prev_chunk_id: int
    curr_chunk_id: int
    gap_ms: float  # Actual gap duration in milliseconds
    expected_gap_ms: float = 0.0  # Expected gap (0 for continuous playback)
    gap_type: str = "normal"  # "synthesis_delay", "buffer_underrun", "normal"

    @property
    def excess_gap_ms(self) -> float:
        """Gap beyond what was expected."""
        return max(0.0, self.gap_ms - self.expected_gap_ms)


@dataclass
class BufferSnapshot:
    """Snapshot of buffer state at a point in time."""

    timestamp: float  # time.perf_counter()
    sample_count: int
    buffered_seconds: float


class PlaybackDiagnostics:
    """
    Thread-safe diagnostics collector for audio playback analysis.

    Tracks timing information for each audio chunk through the pipeline:
    synthesis -> queue -> playback

    Usage:
        diagnostics = PlaybackDiagnostics()

        # During synthesis
        diagnostics.record_synthesis_start(chunk_id, text)
        # ... synthesize ...
        diagnostics.record_synthesis_end(chunk_id, audio_duration_ms)

        # During queue
        diagnostics.record_queue_put(chunk_id)

        # During playback
        diagnostics.record_playback_start(chunk_id)
        # ... play ...
        diagnostics.record_playback_end(chunk_id)

        # Analysis
        diagnostics.print_report()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._chunk_timings: Dict[int, ChunkTiming] = {}
        self._buffer_snapshots: List[BufferSnapshot] = []
        self._lock = threading.Lock()
        self._next_chunk_id = 0
        self._start_time: float = 0.0

    def reset(self) -> None:
        """Clear all recorded diagnostics."""
        with self._lock:
            self._chunk_timings.clear()
            self._buffer_snapshots.clear()
            self._next_chunk_id = 0
            self._start_time = 0.0

    def start_session(self) -> None:
        """Mark the start of a diagnostics session."""
        with self._lock:
            self._start_time = time.perf_counter()

    def get_next_chunk_id(self) -> int:
        """Get the next available chunk ID (thread-safe)."""
        with self._lock:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            return chunk_id

    def record_synthesis_start(self, chunk_id: int, text: str = "") -> None:
        """Record when synthesis begins for a chunk."""
        if not self.enabled:
            return
        with self._lock:
            if chunk_id not in self._chunk_timings:
                self._chunk_timings[chunk_id] = ChunkTiming(chunk_id=chunk_id)
            self._chunk_timings[chunk_id].synthesis_start = time.perf_counter()
            self._chunk_timings[chunk_id].text_preview = text[:30] if text else ""

    def record_synthesis_end(
        self, chunk_id: int, audio_duration_ms: float = 0.0, sample_count: int = 0
    ) -> None:
        """Record when synthesis completes for a chunk."""
        if not self.enabled:
            return
        with self._lock:
            if chunk_id not in self._chunk_timings:
                self._chunk_timings[chunk_id] = ChunkTiming(chunk_id=chunk_id)
            self._chunk_timings[chunk_id].synthesis_end = time.perf_counter()
            self._chunk_timings[chunk_id].audio_duration_ms = audio_duration_ms
            self._chunk_timings[chunk_id].sample_count = sample_count

    def record_queue_put(self, chunk_id: int) -> None:
        """Record when a chunk is added to the playback queue."""
        if not self.enabled:
            return
        with self._lock:
            if chunk_id not in self._chunk_timings:
                self._chunk_timings[chunk_id] = ChunkTiming(chunk_id=chunk_id)
            self._chunk_timings[chunk_id].queue_time = time.perf_counter()

    def record_playback_start(self, chunk_id: int) -> None:
        """Record when playback begins for a chunk."""
        if not self.enabled:
            return
        with self._lock:
            if chunk_id not in self._chunk_timings:
                self._chunk_timings[chunk_id] = ChunkTiming(chunk_id=chunk_id)
            self._chunk_timings[chunk_id].playback_start = time.perf_counter()

    def record_playback_end(self, chunk_id: int) -> None:
        """Record when playback completes for a chunk."""
        if not self.enabled:
            return
        with self._lock:
            if chunk_id not in self._chunk_timings:
                self._chunk_timings[chunk_id] = ChunkTiming(chunk_id=chunk_id)
            self._chunk_timings[chunk_id].playback_end = time.perf_counter()

    def record_buffer_state(self, sample_count: int, buffered_seconds: float) -> None:
        """Record a snapshot of the buffer state."""
        if not self.enabled:
            return
        with self._lock:
            self._buffer_snapshots.append(
                BufferSnapshot(
                    timestamp=time.perf_counter(),
                    sample_count=sample_count,
                    buffered_seconds=buffered_seconds,
                )
            )

    def get_chunk_timings(self) -> List[ChunkTiming]:
        """Get all chunk timings sorted by chunk_id."""
        with self._lock:
            return sorted(self._chunk_timings.values(), key=lambda c: c.chunk_id)

    def get_buffer_snapshots(self) -> List[BufferSnapshot]:
        """Get all buffer snapshots."""
        with self._lock:
            return list(self._buffer_snapshots)

    def analyze_gaps(self) -> List[GapInfo]:
        """
        Analyze gaps between consecutive chunks.

        A gap occurs when there's time between one chunk's playback ending
        and the next chunk's playback starting.

        Returns:
            List of GapInfo objects describing each gap.
        """
        gaps: List[GapInfo] = []
        timings = self.get_chunk_timings()

        for i in range(1, len(timings)):
            prev = timings[i - 1]
            curr = timings[i]

            # Skip if we don't have playback timing
            if prev.playback_end <= 0 or curr.playback_start <= 0:
                continue

            gap_seconds = curr.playback_start - prev.playback_end
            gap_ms = gap_seconds * 1000

            # Classify the gap
            if gap_ms < 0:
                # Overlapping playback (crossfade or buffer)
                gap_type = "overlap"
            elif curr.synthesis_end > prev.playback_end:
                # Synthesis wasn't ready when needed
                gap_type = "synthesis_delay"
            elif gap_ms > 50:  # Threshold for "significant" gap
                gap_type = "buffer_underrun"
            else:
                gap_type = "normal"

            gaps.append(
                GapInfo(
                    prev_chunk_id=prev.chunk_id,
                    curr_chunk_id=curr.chunk_id,
                    gap_ms=gap_ms,
                    gap_type=gap_type,
                )
            )

        return gaps

    def get_summary(self) -> Dict:
        """
        Get a summary of diagnostics data.

        Returns:
            Dict with summary statistics.
        """
        timings = self.get_chunk_timings()
        gaps = self.analyze_gaps()

        if not timings:
            return {
                "chunk_count": 0,
                "total_audio_ms": 0,
                "avg_synthesis_ms": 0,
                "gap_count": 0,
                "avg_gap_ms": 0,
                "max_gap_ms": 0,
                "underrun_count": 0,
                "synthesis_delay_count": 0,
            }

        # Timing stats
        synthesis_times = [t.synthesis_duration_ms for t in timings if t.synthesis_duration_ms > 0]
        total_audio_ms = sum(t.audio_duration_ms for t in timings)

        # Gap stats
        gap_values = [g.gap_ms for g in gaps if g.gap_ms > 0]
        underruns = [g for g in gaps if g.gap_type == "buffer_underrun"]
        synthesis_delays = [g for g in gaps if g.gap_type == "synthesis_delay"]

        return {
            "chunk_count": len(timings),
            "total_audio_ms": total_audio_ms,
            "avg_synthesis_ms": sum(synthesis_times) / len(synthesis_times) if synthesis_times else 0,
            "max_synthesis_ms": max(synthesis_times) if synthesis_times else 0,
            "gap_count": len(gap_values),
            "avg_gap_ms": sum(gap_values) / len(gap_values) if gap_values else 0,
            "max_gap_ms": max(gap_values) if gap_values else 0,
            "min_gap_ms": min(gap_values) if gap_values else 0,
            "underrun_count": len(underruns),
            "synthesis_delay_count": len(synthesis_delays),
        }

    def print_report(self) -> str:
        """
        Generate and print a human-readable diagnostics report.

        Returns:
            The report string.
        """
        summary = self.get_summary()
        gaps = self.analyze_gaps()
        timings = self.get_chunk_timings()

        lines = [
            "=" * 60,
            "PLAYBACK DIAGNOSTICS REPORT",
            "=" * 60,
            "",
            f"Chunks processed: {summary['chunk_count']}",
            f"Total audio: {summary['total_audio_ms']:.1f}ms ({summary['total_audio_ms']/1000:.2f}s)",
            "",
            "SYNTHESIS TIMING:",
            f"  Average: {summary['avg_synthesis_ms']:.1f}ms",
            f"  Maximum: {summary['max_synthesis_ms']:.1f}ms",
            "",
            "GAP ANALYSIS:",
            f"  Total gaps: {summary['gap_count']}",
            f"  Average gap: {summary['avg_gap_ms']:.1f}ms",
            f"  Max gap: {summary['max_gap_ms']:.1f}ms",
            f"  Min gap: {summary['min_gap_ms']:.1f}ms",
            f"  Buffer underruns: {summary['underrun_count']}",
            f"  Synthesis delays: {summary['synthesis_delay_count']}",
        ]

        # Show significant gaps
        significant_gaps = [g for g in gaps if g.gap_ms > 20]
        if significant_gaps:
            lines.append("")
            lines.append("SIGNIFICANT GAPS (>20ms):")
            for gap in significant_gaps[:10]:  # Show top 10
                lines.append(
                    f"  Chunk {gap.prev_chunk_id}->{gap.curr_chunk_id}: "
                    f"{gap.gap_ms:.1f}ms ({gap.gap_type})"
                )

        # Show chunk details
        if timings:
            lines.append("")
            lines.append("CHUNK DETAILS:")
            for t in timings[:20]:  # Show first 20
                lines.append(
                    f"  [{t.chunk_id}] synth:{t.synthesis_duration_ms:.0f}ms "
                    f"audio:{t.audio_duration_ms:.0f}ms "
                    f'"{t.text_preview}..."'
                )

        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)
        print(report)
        return report

    def to_dict(self) -> Dict:
        """Export all diagnostics data as a dictionary."""
        return {
            "summary": self.get_summary(),
            "chunks": [
                {
                    "chunk_id": t.chunk_id,
                    "text_preview": t.text_preview,
                    "synthesis_start": t.synthesis_start,
                    "synthesis_end": t.synthesis_end,
                    "synthesis_duration_ms": t.synthesis_duration_ms,
                    "queue_time": t.queue_time,
                    "queue_latency_ms": t.queue_latency_ms,
                    "playback_start": t.playback_start,
                    "playback_end": t.playback_end,
                    "playback_latency_ms": t.playback_latency_ms,
                    "audio_duration_ms": t.audio_duration_ms,
                    "sample_count": t.sample_count,
                }
                for t in self.get_chunk_timings()
            ],
            "gaps": [
                {
                    "prev_chunk_id": g.prev_chunk_id,
                    "curr_chunk_id": g.curr_chunk_id,
                    "gap_ms": g.gap_ms,
                    "gap_type": g.gap_type,
                }
                for g in self.analyze_gaps()
            ],
            "buffer_snapshots": [
                {
                    "timestamp": s.timestamp,
                    "sample_count": s.sample_count,
                    "buffered_seconds": s.buffered_seconds,
                }
                for s in self.get_buffer_snapshots()
            ],
        }
