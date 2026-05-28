"""End-to-end inference CLI for padelgraph-ai (story-005).

This is the thin Click-based orchestrator that closes Epic 1 (foundation).
It wires the per-story modules — detection, sync, court, fusion — into
one command:

    padelgraph-ai-infer \\
      --video data/inputs/test_cam1.mp4 \\
      --video data/inputs/test_cam2.mp4 \\
      --calib data/inputs/calib_2cam.json \\
      --out data/outputs/run-001.json

The calibration JSON is a list of :class:`padelgraph_ai.schemas.Calibration`
objects (one per camera, in the same order as ``--video``).

Apache 2.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from padelgraph_ai.detection.yolox_runner import Detector
from padelgraph_ai.fusion import Triangulator
from padelgraph_ai.schemas import (
    BallObservation,
    Calibration,
    Detection,
    FrameAnalysis,
    MatchAnalysis,
    MatchMeta,
)
from padelgraph_ai.sync import MultiCamSync

__all__ = ["main"]

# COCO class id used for the padel ball. Matches Detector's _KEEP_CLASSES.
_BALL_CLASS_ID: int = 32
_PERSON_CLASS_ID: int = 0


def _load_calibrations(calib_path: Path) -> list[Calibration]:
    """Parse a calibration JSON file into a list of :class:`Calibration`.

    The file may be either a top-level JSON list of calibration dicts or a
    dict with a ``"cameras"`` key holding that list.
    """
    payload = json.loads(calib_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "cameras" in payload:
        items = payload["cameras"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(
            f"Unsupported calibration JSON shape at {calib_path}: "
            "expected a list or a dict with a 'cameras' key"
        )
    return [Calibration.model_validate(item) for item in items]


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the (cx, cy) center of an (x, y, w, h) bounding box."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _partition_detections(
    detections: list[Detection],
) -> tuple[list[Detection], Detection | None]:
    """Split detections into (player list, single best ball or None).

    The "best" ball is the highest-confidence ``sports ball`` detection.
    Returning a single ball matches MVP scope — multi-ball tracking is
    Epic 2.
    """
    players: list[Detection] = [d for d in detections if d.class_id == _PERSON_CLASS_ID]
    balls: list[Detection] = [d for d in detections if d.class_id == _BALL_CLASS_ID]
    if not balls:
        return players, None
    best_ball = max(balls, key=lambda d: d.confidence)
    return players, best_ball


def _draw_overlay(
    frame: np.ndarray,
    players: list[Detection],
    ball: Detection | None,
    ball_3d: tuple[float, float, float] | None,
) -> np.ndarray:
    """Draw per-frame bbox + ball + 3D label onto ``frame``. Returns a new array."""
    import cv2  # local import — overlay is optional

    annotated = frame.copy()
    for player in players:
        x, y, w, h = (int(round(v)) for v in player.bbox)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if ball is not None:
        cx, cy = (int(round(v)) for v in _bbox_center(ball.bbox))
        cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
        if ball_3d is not None:
            label = f"3D=({ball_3d[0]:.2f}, {ball_3d[1]:.2f}, {ball_3d[2]:.2f})m"
            cv2.putText(
                annotated,
                label,
                (max(0, cx - 80), max(15, cy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                lineType=cv2.LINE_AA,
            )
    return annotated


@click.command()
@click.option(
    "--video",
    "videos",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    required=True,
    help="Input video file. Pass --video twice for the 2-camera MVP setup.",
)
@click.option(
    "--calib",
    "calib_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Per-camera calibration JSON (list of Calibration entries in --video order).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Where to write the MatchAnalysis JSON.",
)
@click.option(
    "--overlay",
    "overlay_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional output mp4 with bbox + ball overlay drawn on cam1.",
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional YOLOX checkpoint. Omit to run the detector in stub mode.",
)
@click.option(
    "--max-frames",
    "max_frames",
    type=click.IntRange(min=1),
    default=None,
    help="Optional cap on the number of synced batches processed (smoke testing).",
)
@click.option(
    "--match-id",
    "match_id",
    type=str,
    default=None,
    help="Optional match identifier embedded in the output JSON meta block.",
)
def main(
    videos: tuple[Path, ...],
    calib_path: Path,
    out_path: Path,
    overlay_path: Path | None,
    checkpoint_path: Path | None,
    max_frames: int | None,
    match_id: str | None,
) -> None:
    """Run the full padelgraph-ai pipeline end-to-end on N synchronized videos."""
    _run(
        videos=list(videos),
        calib_path=calib_path,
        out_path=out_path,
        overlay_path=overlay_path,
        checkpoint_path=checkpoint_path,
        max_frames=max_frames,
        match_id=match_id,
    )


def _run(
    videos: list[Path],
    calib_path: Path,
    out_path: Path,
    overlay_path: Path | None = None,
    checkpoint_path: Path | None = None,
    max_frames: int | None = None,
    match_id: str | None = None,
) -> MatchAnalysis:
    """Pure-function orchestrator. Returns the assembled :class:`MatchAnalysis`."""
    if not videos:
        raise click.UsageError("at least one --video must be provided")

    calibrations = _load_calibrations(calib_path)
    if len(calibrations) != len(videos):
        raise click.UsageError(
            f"calibration count ({len(calibrations)}) must match video count ({len(videos)})"
        )

    sync = MultiCamSync(videos)
    cam_ids = sync.cam_ids
    fps = sync.target_fps

    # Bind calibrations to the canonical cam_ids derived from the sync module
    # (cam1, cam2, ... in --video order). If a calibration carries its own
    # cam_id, we honor it for the meta block but re-key by cam_ids for
    # the triangulator so the keys stay consistent end-to-end.
    calibrations_by_cam_id: dict[str, Calibration] = {
        cam_id: Calibration(
            cam_id=cam_id,
            intrinsics=calib.intrinsics,
            extrinsics=calib.extrinsics,
            distortion_coeffs=calib.distortion_coeffs,
            image_size=calib.image_size,
        )
        for cam_id, calib in zip(cam_ids, calibrations, strict=True)
    }

    detector = Detector(checkpoint_path=checkpoint_path)
    triangulator = Triangulator(calibrations_by_cam_id)

    video_writer = None
    overlay_size: tuple[int, int] | None = None

    frames: list[FrameAnalysis] = []
    try:
        for batch in sync.align():
            if max_frames is not None and len(frames) >= max_frames:
                break

            players_per_cam: dict[str, list[Detection]] = {}
            ball_pixel_per_cam: dict[str, tuple[float, float]] = {}
            best_ball_per_cam: dict[str, Detection] = {}

            for cam_id, frame in batch.frames.items():
                detections = detector.infer(frame)
                players, ball = _partition_detections(detections)
                if players:
                    players_per_cam[cam_id] = players
                if ball is not None:
                    ball_pixel_per_cam[cam_id] = _bbox_center(ball.bbox)
                    best_ball_per_cam[cam_id] = ball

            ball_3d = triangulator.triangulate(ball_pixel_per_cam)
            ball_confidence = (
                min(d.confidence for d in best_ball_per_cam.values()) if best_ball_per_cam else 0.0
            )
            frames.append(
                FrameAnalysis(
                    frame_id=batch.frame_index,
                    ts=batch.ts,
                    players_per_cam=players_per_cam,
                    ball=BallObservation(
                        position_2d_per_cam=ball_pixel_per_cam,
                        position_3d=ball_3d,
                        confidence=ball_confidence,
                    ),
                )
            )

            if overlay_path is not None:
                # Overlay is drawn on cam1's frame (the canonical viewpoint).
                primary_cam = cam_ids[0]
                if primary_cam in batch.frames:
                    overlay_frame = _draw_overlay(
                        batch.frames[primary_cam],
                        players_per_cam.get(primary_cam, []),
                        best_ball_per_cam.get(primary_cam),
                        (ball_3d.x, ball_3d.y, ball_3d.z) if ball_3d else None,
                    )
                    if video_writer is None:
                        import cv2  # local import

                        height, width = overlay_frame.shape[:2]
                        overlay_size = (width, height)
                        overlay_path.parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, overlay_size)
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Could not open VideoWriter for {overlay_path}")
                    video_writer.write(overlay_frame)
    finally:
        if video_writer is not None:
            video_writer.release()

    meta = MatchMeta(
        match_id=match_id or out_path.stem,
        cameras=list(calibrations_by_cam_id.values()),
        fps=fps,
    )
    analysis = MatchAnalysis(meta=meta, frames=frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        analysis.model_dump_json(indent=2),
        encoding="utf-8",
    )
    click.echo(
        f"wrote {len(frames)} frames to {out_path}"
        + (f" (overlay: {overlay_path})" if overlay_path else "")
    )
    return analysis


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    main()
