"""Smoke tests for ``padelgraph_ai.detection.Detector``.

These tests are intentionally minimal — story-002 ships the wrapper
scaffold and the COCO class filter. Full accuracy gates (≥50% player
recall on Alan-provided clip) belong to Epic 2 per the story spec.

Per D26 (Verificar-antes-de-Afirmar): tests assert what the code
actually does, not what the README promises. When YOLOX is not
installed, the suite documents that gap explicitly via ``pytest.skip``
rather than papering over it.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from padelgraph_ai.detection import Detector
from padelgraph_ai.schemas import Detection, FrameDetections

YOLOX_INSTALLED = importlib.util.find_spec("yolox") is not None


def test_detector_initializes_without_checkpoint() -> None:
    """Stub mode: ``Detector(checkpoint_path=None)`` must construct and
    expose the documented attributes without touching YOLOX."""
    det = Detector(checkpoint_path=None)
    assert det.checkpoint_path is None
    assert det.device in {"mps", "cpu", "cuda"}
    assert 0.0 <= det.confidence_threshold <= 1.0
    assert 0.0 <= det.nms_threshold <= 1.0


def test_infer_synthetic_frame() -> None:
    """A random uint8 480p frame must not crash ``infer`` and must
    return a list (possibly empty — noise has no padel content)."""
    det = Detector(checkpoint_path=None)
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(low=0, high=256, size=(480, 640, 3), dtype=np.uint8)
    result = det.infer(frame)
    assert isinstance(result, list)
    # In stub mode the list is always empty. If YOLOX is wired in later
    # this assertion will still hold for noise input.
    for item in result:
        assert isinstance(item, Detection)


def test_infer_rejects_bad_shape() -> None:
    """Detector validates input shape — a 2D array is not a frame."""
    det = Detector(checkpoint_path=None)
    bad = np.zeros((480, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected"):
        det.infer(bad)


def test_infer_rejects_bad_dtype() -> None:
    """Detector validates input dtype — float frames are not accepted."""
    det = Detector(checkpoint_path=None)
    bad = np.zeros((480, 640, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected uint8"):
        det.infer(bad)


def test_detection_pydantic_shape() -> None:
    """A ``Detection`` instance built from canonical values has the
    documented shape: 4-tuple bbox, confidence in [0,1], class_id in
    the allowed COCO subset."""
    sample = Detection(
        class_id=0,
        class_name="person",
        bbox=(10.0, 20.0, 100.0, 200.0),
        confidence=0.75,
    )
    assert len(sample.bbox) == 4
    assert all(isinstance(v, float) for v in sample.bbox)
    assert 0.0 <= sample.confidence <= 1.0
    assert sample.class_id in {0, 32}
    assert sample.class_name in {"person", "sports ball"}


def test_frame_detections_default_empty() -> None:
    """``FrameDetections`` defaults to an empty detections list — this
    keeps the iterator length equal to the video length even when no
    objects are detected in a frame."""
    fd = FrameDetections(frame_id=0, ts=0.0)
    assert fd.detections == []


@pytest.mark.skipif(
    YOLOX_INSTALLED,
    reason="YOLOX is installed — the stub-mode contract still holds, "
    "but this test specifically pins behavior when YOLOX is absent",
)
def test_stub_mode_when_yolox_missing() -> None:
    """When YOLOX cannot be imported, the detector silently falls back
    to stub mode regardless of whether a checkpoint path was supplied.
    """
    det = Detector(checkpoint_path="/tmp/does-not-exist.pth")
    rng = np.random.default_rng(seed=1)
    frame = rng.integers(low=0, high=256, size=(240, 320, 3), dtype=np.uint8)
    assert det.infer(frame) == []
