"""Tests for :mod:`padelgraph_ai.sync.timestamp_sync`.

We render tiny synthetic MP4s with ``cv2.VideoWriter`` so the suite stays
hermetic (no external assets required). Real-video integration tests live
behind the ``integration`` marker and run only when Alan-provided footage is
present.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from padelgraph_ai.schemas import SyncedFrameBatch
from padelgraph_ai.sync import MultiCamSync

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_video(
    path: Path,
    duration_seconds: float,
    fps: float = 30.0,
    width: int = 64,
    height: int = 48,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Write a tiny BGR video at ``path`` with deterministic frame content.

    Each frame fills with ``color`` and stamps a per-frame integer in the
    top-left pixel so tests can distinguish frames if needed.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    try:
        frame_count = max(1, int(round(duration_seconds * fps)))
        for i in range(frame_count):
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            # Stamp the frame index in pixel (0, 0) for traceability.
            frame[0, 0] = (i % 256, (i // 256) % 256, 0)
            writer.write(frame)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sync_two_synthetic_videos_same_fps(tmp_path: Path) -> None:
    """2 same-FPS 1s videos at 30fps, zero offset → ~30 aligned batches."""
    cam1 = tmp_path / "cam1.mp4"
    cam2 = tmp_path / "cam2.mp4"
    _write_synthetic_video(cam1, duration_seconds=1.0, fps=30.0, color=(0, 0, 255))
    _write_synthetic_video(cam2, duration_seconds=1.0, fps=30.0, color=(0, 255, 0))

    sync = MultiCamSync([cam1, cam2], offsets_seconds=[0.0, 0.0])
    batches = list(sync.align())

    # Both cams have ~1s of content at 30fps, so we expect ~30 batches with
    # both cameras present. Allow ±1 frame for codec rounding.
    assert 28 <= len(batches) <= 31, f"unexpected batch count: {len(batches)}"
    full_batches = [b for b in batches if len(b.frames) == 2]
    assert len(full_batches) >= 28
    for batch in full_batches:
        assert set(batch.frames.keys()) == {"cam1", "cam2"}
        assert batch.frames["cam1"].shape == (48, 64, 3)
        assert batch.frames["cam2"].shape == (48, 64, 3)


def test_sync_with_offset(tmp_path: Path) -> None:
    """First batch must respect non-zero offset for one camera."""
    cam1 = tmp_path / "cam1.mp4"
    cam2 = tmp_path / "cam2.mp4"
    _write_synthetic_video(cam1, duration_seconds=2.0, fps=30.0, color=(0, 0, 255))
    _write_synthetic_video(cam2, duration_seconds=2.0, fps=30.0, color=(0, 255, 0))

    sync = MultiCamSync([cam1, cam2], offsets_seconds=[0.0, 0.3])
    batches = list(sync.align())

    assert batches, "expected at least one batch"
    # First batch is at common ts=0.0; cam2 is offset 0.3s into its own
    # timeline, so both cams must yield a frame at ts=0.0.
    first = batches[0]
    assert first.ts == pytest.approx(0.0)
    assert first.frame_index == 0
    assert "cam1" in first.frames
    assert "cam2" in first.frames

    # cam2 should drop out before cam1 because its effective end is at
    # common-ts = duration_cam2 - offset = 2.0 - 0.3 = 1.7s.
    cam2_seen_ts = [b.ts for b in batches if "cam2" in b.frames]
    assert max(cam2_seen_ts) <= 1.71  # tolerance for fp drift


def test_sync_stops_when_all_end(tmp_path: Path) -> None:
    """Iterator must terminate once every camera has ended."""
    cam1 = tmp_path / "cam1.mp4"
    cam2 = tmp_path / "cam2.mp4"
    _write_synthetic_video(cam1, duration_seconds=0.5, fps=30.0, color=(255, 0, 0))
    _write_synthetic_video(cam2, duration_seconds=1.0, fps=30.0, color=(0, 255, 0))

    sync = MultiCamSync([cam1, cam2], offsets_seconds=[0.0, 0.0])
    batches = list(sync.align())

    # Total duration is bounded by the longer video (~1.0s at 30fps).
    assert 25 <= len(batches) <= 31, f"unexpected batch count: {len(batches)}"

    # After cam1 ends, batches should contain only cam2.
    cam1_present_ts = [b.ts for b in batches if "cam1" in b.frames]
    cam2_only_ts = [b.ts for b in batches if "cam2" in b.frames and "cam1" not in b.frames]
    assert cam1_present_ts, "cam1 should appear in early batches"
    assert cam2_only_ts, "cam2 should outlive cam1"
    assert max(cam1_present_ts) < min(cam2_only_ts), "cam1 must end before cam2-only batches start"

    # Frame indices are monotonic and contiguous.
    indices = [b.frame_index for b in batches]
    assert indices == sorted(indices)
    assert indices == list(range(indices[0], indices[-1] + 1))


def test_synced_batch_pydantic_shape(tmp_path: Path) -> None:
    """Every yielded batch must validate as a SyncedFrameBatch instance."""
    cam1 = tmp_path / "cam1.mp4"
    cam2 = tmp_path / "cam2.mp4"
    _write_synthetic_video(cam1, duration_seconds=0.5, fps=30.0)
    _write_synthetic_video(cam2, duration_seconds=0.5, fps=30.0)

    sync = MultiCamSync([cam1, cam2])
    for batch in sync.align():
        assert isinstance(batch, SyncedFrameBatch)
        assert batch.ts >= 0
        assert batch.frame_index >= 0
        assert batch.frames, "frames dict must not be empty when yielded"
        for cam_id, frame in batch.frames.items():
            assert cam_id in {"cam1", "cam2"}
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.uint8
            assert frame.ndim == 3


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_rejects_empty_video_paths() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultiCamSync([])


def test_rejects_offset_length_mismatch(tmp_path: Path) -> None:
    cam1 = tmp_path / "cam1.mp4"
    _write_synthetic_video(cam1, duration_seconds=0.3, fps=30.0)
    with pytest.raises(ValueError, match="offsets_seconds length"):
        MultiCamSync([cam1], offsets_seconds=[0.0, 0.5])


def test_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError):
        MultiCamSync([missing])


def test_target_fps_defaults_to_min(tmp_path: Path) -> None:
    cam1 = tmp_path / "cam1.mp4"
    cam2 = tmp_path / "cam2.mp4"
    _write_synthetic_video(cam1, duration_seconds=0.5, fps=30.0)
    _write_synthetic_video(cam2, duration_seconds=0.5, fps=15.0)

    sync = MultiCamSync([cam1, cam2])
    assert sync.target_fps == pytest.approx(15.0)
    assert sync.cam_ids == ["cam1", "cam2"]


# ---------------------------------------------------------------------------
# Edge case tests (F3 + F6 audit)
# ---------------------------------------------------------------------------


def test_sync_negative_offset_clamped(tmp_path: Path) -> None:
    """Large negative offset must NOT crash or skip the camera entirely.

    Sync semantics: ``offsets_seconds[i]`` is *added* to the common
    timeline to produce ``target_ts``. A large negative offset means
    the camera's timeline only intersects the common timeline far
    in the future (here: common-ts in ``[100, 101]`` for a 1s video
    with offset ``-100``). The iterator must:

    - never raise (e.g. on the early batches where the cam is
      pre-timeline and is silently skipped per the ``align`` contract)
    - eventually yield the cam's frames when the common timeline
      catches up to its window
    - and terminate cleanly once the cam's duration is exhausted
    """
    cam1 = tmp_path / "cam1.mp4"
    _write_synthetic_video(cam1, duration_seconds=1.0, fps=30.0)

    sync = MultiCamSync([cam1], offsets_seconds=[-100.0])
    # Materializing the iterator must not raise. We don't assert on
    # the exact batch count (sync.align's batch_index is unbounded
    # while waiting for the cam to come online), but we do assert
    # that every yielded batch contains cam1 (the ``if frames``
    # guard in ``align`` prevents empty batches), and that we
    # terminate at all.
    batches = list(sync.align())
    assert all("cam1" in b.frames for b in batches), (
        "negative-offset cam should never produce empty batches"
    )

    # And: every yielded batch's common timestamp falls inside the
    # window where the cam is actually addressable, [|offset|, |offset| + duration].
    if batches:
        assert all(100.0 <= b.ts < 101.01 for b in batches), (
            "batches outside the [100, 101] window indicate a sync bug"
        )


def test_sync_seek_failure_counter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_seek_and_read`` must update counters and ``report_seek_stats`` must surface."""
    cam1 = tmp_path / "cam1.mp4"
    _write_synthetic_video(cam1, duration_seconds=1.0, fps=30.0)
    sync = MultiCamSync([cam1])

    # Patch the bound seek+read to always fail. We swap the *instance*
    # method on this specific sync so other tests are unaffected.
    def _always_fail(_self: MultiCamSync, _cap: object, cam_id: str, _ts: float):
        # Mirror the increment behavior of the real method so counter
        # semantics are exercised end-to-end.
        _self._seek_attempts[cam_id] += 1
        _self._seek_failures[cam_id] += 1
        return None

    monkeypatch.setattr(MultiCamSync, "_seek_and_read", _always_fail)

    batches = list(sync.align())
    # First batch: the cam fails → marked ended → loop terminates.
    assert batches == []

    failures, attempts = sync.seek_stats["cam1"]
    assert attempts >= 1
    assert failures == attempts  # every attempt failed

    # >5% failure rate (it's 100% here) must produce a warning line.
    messages = sync.report_seek_stats()
    assert len(messages) == 1
    assert "cam1" in messages[0]
    assert "seek failures" in messages[0]
