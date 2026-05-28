"""YOLOX detector wrapper.

Wraps YOLOX inference behind a Sephirot-friendly ``Detector`` class that
returns Pydantic ``Detection`` / ``FrameDetections`` instances from
``padelgraph_ai.schemas``.

The class supports a **stub mode** (when YOLOX is not installed *and* no
checkpoint is provided) so that downstream story scaffolding and CI can
import the module without paying for the heavy YOLOX dependency. Stub
mode returns empty detection lists; real inference is gated behind a
checkpoint path + the optional ``yolox`` package.

Apache 2.0.
"""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from padelgraph_ai.schemas import Detection, FrameDetections


class YOLOXStubModeWarning(UserWarning):
    """Emitted when ``Detector`` falls back to stub mode despite a checkpoint.

    Stub mode silently returns zero detections; downstream callers that
    were expecting real inference (because they supplied ``--checkpoint``)
    will get an empty pipeline output that *looks* valid. This warning
    surfaces the fallback so it cannot go unnoticed.
    """


# COCO class ids we keep. Anything else is dropped per story-002 scope
# (single-cam padel: person + sports ball only).
_KEEP_CLASSES: dict[int, str] = {
    0: "person",
    32: "sports ball",
}


def _yolox_available() -> bool:
    """Return True iff the optional ``yolox`` package can be imported."""
    return importlib.util.find_spec("yolox") is not None


def _resolve_device(requested: str) -> str:
    """Resolve the ``device`` constructor argument to a concrete torch device string.

    ``auto`` picks MPS on Apple Silicon when available, else falls back to
    CPU. CUDA is intentionally not auto-selected because the MVP target
    hardware is M2 Air — explicit ``device="cuda"`` is still honored if
    the caller forces it.
    """
    if requested != "auto":
        return requested

    try:
        import torch  # local import: torch is a heavy dep
    except ImportError:  # pragma: no cover — torch is a hard dep of the project
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Detector:
    """YOLOX wrapper producing ``Detection`` / ``FrameDetections`` instances.

    Parameters
    ----------
    checkpoint_path:
        Path to a YOLOX ``.pth`` checkpoint. If ``None``, the detector
        runs in **stub mode**: ``infer`` returns an empty list and
        ``infer_video`` yields ``FrameDetections`` with empty detection
        lists. Stub mode keeps the public API usable for downstream
        scaffolding and CI even when YOLOX cannot be installed.
    device:
        ``"auto"`` (default) picks MPS on Apple Silicon, else CPU. Pass
        ``"cuda"`` / ``"cpu"`` / ``"mps"`` explicitly to override.
    confidence_threshold:
        Minimum confidence to keep a detection. Defaults to ``0.25``
        (YOLOX default). Detections below this are dropped before any
        class filtering.
    nms_threshold:
        NMS IoU threshold. Defaults to ``0.45`` (YOLOX default).
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> None:
        self.checkpoint_path: Path | None = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.device: str = _resolve_device(device)
        self.confidence_threshold: float = confidence_threshold
        self.nms_threshold: float = nms_threshold

        # In stub mode we never load a model.
        self._stub_mode: bool = self.checkpoint_path is None or not _yolox_available()
        self._model: object | None = None

        # If a checkpoint was supplied but YOLOX is unavailable, the
        # detector silently degraded to stub mode (returns 0 detections).
        # That silent fallback is exactly the F1 audit finding — surface
        # it as both a Python warning (programmatic consumers) and rely
        # on the CLI layer to emit a click stderr message (human users).
        if self.checkpoint_path is not None and not _yolox_available():
            warnings.warn(
                (
                    f"STUB MODE active: checkpoint {self.checkpoint_path!s} was "
                    "provided but the optional 'yolox' package is not installed. "
                    "Detector.infer() will return 0 detections for every frame. "
                    "Epic 2 will implement the real YOLOX loader; until then, "
                    "install YOLOX or omit --checkpoint to silence this warning."
                ),
                YOLOXStubModeWarning,
                stacklevel=2,
            )

        if not self._stub_mode:
            # Real-mode load is deferred to first inference to keep
            # construction cheap and to surface import errors close to
            # the actual use site. The story marks fine-tuning as Epic 2;
            # for MVP we ship the loader skeleton.
            self._model = None  # populated lazily by _ensure_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stub_mode(self) -> bool:
        """Whether this detector is running in stub mode (returns 0 detections)."""
        return self._stub_mode

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR uint8 frame.

        Returns a list of ``Detection`` instances filtered to the COCO
        classes we care about (person, sports ball). In stub mode this
        returns ``[]`` so callers can still exercise the pipeline.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) BGR frame, got shape {frame.shape!r}")
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame, got dtype {frame.dtype!r}")

        if self._stub_mode:
            return []

        self._ensure_model()
        raw_outputs = self._run_model(frame)
        return self._postprocess(raw_outputs, frame_shape=frame.shape[:2])

    def infer_video(self, video_path: Path | str) -> Iterator[FrameDetections]:
        """Yield per-frame ``FrameDetections`` for the given video file.

        Uses OpenCV for video I/O. In stub mode each frame yields an
        empty detection list — the loop still walks every frame so that
        callers can rely on the iterator length matching the video.
        """
        try:
            import cv2  # local import: opencv is heavy
        except ImportError as exc:  # pragma: no cover — opencv is a hard dep
            raise ImportError(
                "opencv-python is required for infer_video; install with 'uv sync'"
            ) from exc

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                detections = self.infer(frame)
                yield FrameDetections(
                    frame_id=frame_id,
                    ts=frame_id / fps,
                    detections=detections,
                )
                frame_id += 1
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily load the YOLOX model.

        Real loading lives behind the optional ``yolox`` import so the
        module is importable even when YOLOX failed to install (common
        on Python 3.12 + recent setuptools — see ``.decision-log.md``
        for the upstream incompatibility).
        """
        if self._model is not None:
            return
        if self._stub_mode:  # defense in depth — should never trigger
            return

        # NOTE: the concrete YOLOX loader (Exp.get_model + load_state_dict)
        # is intentionally left as a thin placeholder in this MVP story.
        # The story-002 ZAI worker brief explicitly defers fine-tuning to
        # Epic 2; we ship the wrapper skeleton with a clear extension
        # point so the model loader can be wired in once the upstream
        # install path is unblocked.
        raise RuntimeError(
            "YOLOX checkpoint loading is not implemented in this MVP "
            "scaffolding (story-002). Install YOLOX and wire "
            "Detector._ensure_model() before running real inference."
        )

    def _run_model(self, frame: np.ndarray) -> object:  # pragma: no cover — gated
        """Execute the YOLOX forward pass. Extension point for Epic 2."""
        raise RuntimeError(
            "Detector._run_model() not implemented. See Epic 2 for the real YOLOX inference path."
        )

    def _postprocess(
        self,
        raw_outputs: object,
        frame_shape: tuple[int, int],
    ) -> list[Detection]:  # pragma: no cover — gated until real model lands
        """Convert raw YOLOX outputs into Pydantic ``Detection`` instances.

        Filters to ``_KEEP_CLASSES`` (person + sports ball) and clamps
        any out-of-frame bbox coordinates to the frame bounds before
        returning.
        """
        height, width = frame_shape
        detections: list[Detection] = []
        # Expected raw_outputs shape (post-YOLOX postprocess): tensor of
        # rows [x1, y1, x2, y2, obj_conf, class_conf, class_id].
        for row in raw_outputs:  # type: ignore[union-attr]
            x1, y1, x2, y2, obj_conf, class_conf, class_id = (float(v) for v in row)
            confidence = obj_conf * class_conf
            if confidence < self.confidence_threshold:
                continue
            class_id_int = int(class_id)
            if class_id_int not in _KEEP_CLASSES:
                continue
            x = max(0.0, min(x1, width - 1))
            y = max(0.0, min(y1, height - 1))
            w = max(0.0, min(x2 - x1, width - x))
            h = max(0.0, min(y2 - y1, height - y))
            detections.append(
                Detection(
                    class_id=class_id_int,
                    class_name=_KEEP_CLASSES[class_id_int],
                    bbox=(x, y, w, h),
                    confidence=confidence,
                )
            )
        return detections
