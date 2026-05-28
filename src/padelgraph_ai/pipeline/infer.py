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
from pydantic import ValidationError

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


class CalibrationParseError(click.UsageError):
    """A user-friendly error raised when a calibration JSON cannot be parsed.

    Wraps Pydantic ``ValidationError`` with a message that names the
    camera (when known) and the specific field at fault — instead of
    dumping the full Pydantic error tree.
    """


def _format_pydantic_error_for_camera(
    error: ValidationError,
    cam_label: str,
) -> str:
    """Render a Pydantic ``ValidationError`` into a single human message.

    Picks the first error in the list, walks its ``loc`` tuple to find
    the field name, and produces ``Invalid calibration for camera 'X':
    <field> <error type>`` — clear enough to fix without reading the
    Pydantic dump.
    """
    errors = error.errors()
    if not errors:
        return f"Invalid calibration for camera {cam_label!r}: {error}"

    first = errors[0]
    loc = first.get("loc", ())
    field_name = ".".join(str(part) for part in loc) if loc else "<unknown field>"
    msg = first.get("msg", "validation failed")
    err_type = first.get("type", "")
    suffix = f" ({err_type})" if err_type else ""
    return f"Invalid calibration for camera {cam_label!r}: field {field_name!r} — {msg}{suffix}"


def _load_calibrations(calib_path: Path) -> list[Calibration]:
    """Parse a calibration JSON file into a list of :class:`Calibration`.

    The file may be either a top-level JSON list of calibration dicts or a
    dict with a ``"cameras"`` key holding that list.

    Raises
    ------
    CalibrationParseError
        Wraps Pydantic ``ValidationError`` with a friendly per-camera
        message naming the camera and the offending field.
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

    calibrations: list[Calibration] = []
    for index, item in enumerate(items):
        # Prefer the explicit cam_id if the JSON already has one;
        # otherwise fall back to the positional camN label so the
        # user can still locate the bad entry.
        cam_label = (
            str(item.get("cam_id")) if isinstance(item, dict) and item.get("cam_id") else None
        ) or f"cam{index + 1}"
        try:
            calibrations.append(Calibration.model_validate(item))
        except ValidationError as exc:
            raise CalibrationParseError(_format_pydantic_error_for_camera(exc, cam_label)) from exc
    return calibrations


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
@click.option(
    "--verbose/--quiet",
    "verbose",
    default=True,
    help=(
        "Emit per-frame diagnostics to stderr (e.g. ball-not-triangulated "
        "events, seek-failure summaries). Default: --verbose. Pass --quiet "
        "to suppress these messages while keeping the final summary."
    ),
)
def main(
    videos: tuple[Path, ...],
    calib_path: Path,
    out_path: Path,
    overlay_path: Path | None,
    checkpoint_path: Path | None,
    max_frames: int | None,
    match_id: str | None,
    verbose: bool,
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
        verbose=verbose,
    )


def _run(
    videos: list[Path],
    calib_path: Path,
    out_path: Path,
    overlay_path: Path | None = None,
    checkpoint_path: Path | None = None,
    max_frames: int | None = None,
    match_id: str | None = None,
    verbose: bool = True,
) -> MatchAnalysis:
    """Pure-function orchestrator. Returns the assembled :class:`MatchAnalysis`.

    Parameters
    ----------
    verbose:
        When ``True`` (default) the orchestrator emits per-frame
        diagnostics to stderr — namely the ball-triangulation-failed
        events (F2 audit) and the post-run seek-failure summary
        (F3 audit). Pass ``False`` to suppress those messages while
        keeping the final "wrote N frames" summary on stdout.
    """
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

    # F1: surface stub-mode fallback to the human CLI user. The Detector
    # constructor already issued a Python warning for programmatic
    # consumers; this stderr line makes it visible to anyone running the
    # CLI without -W default.
    if checkpoint_path is not None and detector.stub_mode:
        click.echo(
            (
                f"WARNING: --checkpoint {checkpoint_path} was provided but YOLOX "
                "is not installed. Detector fell back to STUB MODE — every frame "
                "will produce 0 detections. Epic 2 will implement the real loader; "
                "install YOLOX or omit --checkpoint to silence this warning."
            ),
            err=True,
        )

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
            # F2: surface silent ball-triangulation failures. Triangulator
            # returns None whenever fewer than 2 cameras observed the
            # ball; without this log the JSON quietly carries
            # position_3d: null and the user has no way to debug it.
            if verbose and ball_3d is None:
                observed_cams = sorted(ball_pixel_per_cam.keys())
                click.echo(
                    (
                        f"Frame {batch.frame_index}: ball not triangulated "
                        f"(cams_observed: {observed_cams})"
                    ),
                    err=True,
                )
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

    # F3: surface seek-failure statistics gathered by the sync module.
    # The sync module retries seeks transparently; only warn when a
    # camera lost more than 5% of its batches to seek failures, which
    # indicates a real codec / file integrity problem.
    if verbose:
        for line in sync.report_seek_stats():
            click.echo(line, err=True)

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
