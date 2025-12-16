"""Tests for streaming_tts.diagnostics module."""

import pytest
import time
import threading
from streaming_tts.diagnostics import (
    ChunkTiming,
    GapInfo,
    BufferSnapshot,
    PlaybackDiagnostics,
)


class TestChunkTiming:
    """Tests for ChunkTiming dataclass."""

    def test_default_values(self):
        timing = ChunkTiming(chunk_id=0)
        assert timing.chunk_id == 0
        assert timing.text_preview == ""
        assert timing.synthesis_start == 0.0
        assert timing.synthesis_end == 0.0
        assert timing.audio_duration_ms == 0.0

    def test_synthesis_duration_ms(self):
        timing = ChunkTiming(
            chunk_id=0,
            synthesis_start=1.0,
            synthesis_end=1.5,
        )
        assert timing.synthesis_duration_ms == 500.0

    def test_synthesis_duration_ms_no_data(self):
        timing = ChunkTiming(chunk_id=0)
        assert timing.synthesis_duration_ms == 0.0

    def test_queue_latency_ms(self):
        timing = ChunkTiming(
            chunk_id=0,
            synthesis_end=1.0,
            queue_time=1.1,
        )
        assert timing.queue_latency_ms == pytest.approx(100.0, rel=0.01)

    def test_playback_latency_ms(self):
        timing = ChunkTiming(
            chunk_id=0,
            queue_time=1.0,
            playback_start=1.2,
        )
        assert timing.playback_latency_ms == pytest.approx(200.0, rel=0.01)


class TestGapInfo:
    """Tests for GapInfo dataclass."""

    def test_default_values(self):
        gap = GapInfo(prev_chunk_id=0, curr_chunk_id=1, gap_ms=50.0)
        assert gap.prev_chunk_id == 0
        assert gap.curr_chunk_id == 1
        assert gap.gap_ms == 50.0
        assert gap.gap_type == "normal"

    def test_excess_gap_ms(self):
        gap = GapInfo(
            prev_chunk_id=0,
            curr_chunk_id=1,
            gap_ms=100.0,
            expected_gap_ms=20.0,
        )
        assert gap.excess_gap_ms == 80.0

    def test_excess_gap_ms_no_excess(self):
        gap = GapInfo(
            prev_chunk_id=0,
            curr_chunk_id=1,
            gap_ms=10.0,
            expected_gap_ms=20.0,
        )
        assert gap.excess_gap_ms == 0.0


class TestPlaybackDiagnostics:
    """Tests for PlaybackDiagnostics class."""

    def test_init_enabled(self):
        diag = PlaybackDiagnostics(enabled=True)
        assert diag.enabled is True

    def test_init_disabled(self):
        diag = PlaybackDiagnostics(enabled=False)
        assert diag.enabled is False

    def test_get_next_chunk_id_sequential(self):
        diag = PlaybackDiagnostics()
        assert diag.get_next_chunk_id() == 0
        assert diag.get_next_chunk_id() == 1
        assert diag.get_next_chunk_id() == 2

    def test_record_synthesis_start(self):
        diag = PlaybackDiagnostics()
        diag.record_synthesis_start(0, "Hello world")
        timings = diag.get_chunk_timings()
        assert len(timings) == 1
        assert timings[0].chunk_id == 0
        assert timings[0].text_preview == "Hello world"
        assert timings[0].synthesis_start > 0

    def test_record_synthesis_end(self):
        diag = PlaybackDiagnostics()
        diag.record_synthesis_start(0, "Test")
        diag.record_synthesis_end(0, audio_duration_ms=1000.0, sample_count=24000)
        timings = diag.get_chunk_timings()
        assert timings[0].audio_duration_ms == 1000.0
        assert timings[0].sample_count == 24000
        assert timings[0].synthesis_end > timings[0].synthesis_start

    def test_record_queue_put(self):
        diag = PlaybackDiagnostics()
        diag.record_synthesis_start(0, "Test")
        diag.record_synthesis_end(0, audio_duration_ms=500.0)
        diag.record_queue_put(0)
        timings = diag.get_chunk_timings()
        assert timings[0].queue_time > 0

    def test_record_playback_timing(self):
        diag = PlaybackDiagnostics()
        diag.record_synthesis_start(0, "Test")
        diag.record_synthesis_end(0, audio_duration_ms=500.0)
        diag.record_playback_start(0)
        time.sleep(0.01)  # Small delay
        diag.record_playback_end(0)
        timings = diag.get_chunk_timings()
        assert timings[0].playback_start > 0
        assert timings[0].playback_end > timings[0].playback_start

    def test_record_buffer_state(self):
        diag = PlaybackDiagnostics()
        diag.record_buffer_state(24000, 1.0)
        snapshots = diag.get_buffer_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].sample_count == 24000
        assert snapshots[0].buffered_seconds == 1.0

    def test_disabled_no_recording(self):
        diag = PlaybackDiagnostics(enabled=False)
        diag.record_synthesis_start(0, "Test")
        diag.record_buffer_state(24000, 1.0)
        assert len(diag.get_chunk_timings()) == 0
        assert len(diag.get_buffer_snapshots()) == 0

    def test_reset(self):
        diag = PlaybackDiagnostics()
        diag.record_synthesis_start(0, "Test")
        diag.record_buffer_state(24000, 1.0)
        diag.reset()
        assert len(diag.get_chunk_timings()) == 0
        assert len(diag.get_buffer_snapshots()) == 0
        assert diag.get_next_chunk_id() == 0

    def test_analyze_gaps_normal(self):
        diag = PlaybackDiagnostics()

        # Simulate two chunks with a small gap
        diag.record_synthesis_start(0, "First")
        diag.record_synthesis_end(0, audio_duration_ms=500.0)
        diag.record_playback_start(0)
        diag.record_playback_end(0)

        time.sleep(0.01)  # 10ms gap

        diag.record_synthesis_start(1, "Second")
        diag.record_synthesis_end(1, audio_duration_ms=500.0)
        diag.record_playback_start(1)
        diag.record_playback_end(1)

        gaps = diag.analyze_gaps()
        assert len(gaps) == 1
        assert gaps[0].prev_chunk_id == 0
        assert gaps[0].curr_chunk_id == 1
        assert gaps[0].gap_ms > 0

    def test_analyze_gaps_synthesis_delay(self):
        diag = PlaybackDiagnostics()

        # First chunk
        diag.record_synthesis_start(0, "First")
        diag.record_synthesis_end(0, audio_duration_ms=500.0)
        diag.record_playback_start(0)
        diag.record_playback_end(0)

        # Second chunk where synthesis finished after first playback ended
        diag.record_synthesis_start(1, "Second")
        time.sleep(0.02)  # Simulate slow synthesis
        diag.record_synthesis_end(1, audio_duration_ms=500.0)
        diag.record_playback_start(1)
        diag.record_playback_end(1)

        gaps = diag.analyze_gaps()
        assert len(gaps) == 1
        # Gap type should be synthesis_delay
        assert gaps[0].gap_type in ["synthesis_delay", "normal"]

    def test_get_summary_empty(self):
        diag = PlaybackDiagnostics()
        summary = diag.get_summary()
        assert summary["chunk_count"] == 0
        assert summary["total_audio_ms"] == 0
        assert summary["gap_count"] == 0

    def test_get_summary_with_data(self):
        diag = PlaybackDiagnostics()

        # Add two chunks
        diag.record_synthesis_start(0, "First")
        diag.record_synthesis_end(0, audio_duration_ms=500.0)
        diag.record_playback_start(0)
        diag.record_playback_end(0)

        diag.record_synthesis_start(1, "Second")
        diag.record_synthesis_end(1, audio_duration_ms=600.0)
        diag.record_playback_start(1)
        diag.record_playback_end(1)

        summary = diag.get_summary()
        assert summary["chunk_count"] == 2
        assert summary["total_audio_ms"] == 1100.0

    def test_print_report(self):
        diag = PlaybackDiagnostics()
        diag.record_synthesis_start(0, "Test chunk")
        diag.record_synthesis_end(0, audio_duration_ms=500.0)
        diag.record_playback_start(0)
        diag.record_playback_end(0)

        report = diag.print_report()
        assert "PLAYBACK DIAGNOSTICS REPORT" in report
        assert "Chunks processed: 1" in report

    def test_to_dict(self):
        diag = PlaybackDiagnostics()
        diag.record_synthesis_start(0, "Test")
        diag.record_synthesis_end(0, audio_duration_ms=500.0)

        data = diag.to_dict()
        assert "summary" in data
        assert "chunks" in data
        assert "gaps" in data
        assert "buffer_snapshots" in data
        assert len(data["chunks"]) == 1

    def test_thread_safety(self):
        """Test that diagnostics is thread-safe."""
        diag = PlaybackDiagnostics()
        errors = []

        def record_chunks(start_id, count):
            try:
                for i in range(count):
                    chunk_id = start_id + i
                    diag.record_synthesis_start(chunk_id, f"Chunk {chunk_id}")
                    time.sleep(0.001)
                    diag.record_synthesis_end(chunk_id, audio_duration_ms=100.0)
                    diag.record_playback_start(chunk_id)
                    diag.record_playback_end(chunk_id)
            except Exception as e:
                errors.append(e)

        # Launch multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=record_chunks, args=(i * 10, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        timings = diag.get_chunk_timings()
        assert len(timings) == 50  # 5 threads * 10 chunks each

    def test_get_chunk_timings_sorted(self):
        diag = PlaybackDiagnostics()

        # Add chunks out of order
        diag.record_synthesis_start(2, "Third")
        diag.record_synthesis_start(0, "First")
        diag.record_synthesis_start(1, "Second")

        timings = diag.get_chunk_timings()
        assert timings[0].chunk_id == 0
        assert timings[1].chunk_id == 1
        assert timings[2].chunk_id == 2

    def test_start_session(self):
        diag = PlaybackDiagnostics()
        diag.start_session()
        assert diag._start_time > 0
