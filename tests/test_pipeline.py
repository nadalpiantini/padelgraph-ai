"""End-to-end smoke test for the :mod:`padelgraph_ai.pipeline.infer` CLI.

We synthesize two tiny throwaway videos with ``cv2.VideoWriter`` plus a
matching calibration JSON, then invoke the Click CLI via
:class:`click.testing.CliRunner` and assert the output JSON validates as
a :class:`MatchAnalysis`. Detector stub mode is fine — this test verifies
the orchestrator wires the modules together without crashing, not that
detection finds anything (real videos are needed for that).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from padelgraph_ai.pipeline.infer import main
from padelgraph_ai.schemas import MatchAnalysis


def _write_synthetic_video(
    path: Path,
    duration_seconds: float = 1.0,
    fps: float = 30.0,
    width: int = 64,
    height: int = 48,
    seed: int = 0,
) -> None:
    """Write a tiny BGR video filled with random-but-deterministic frames."""
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    try:
        frame_count = max(1, int(round(duration_seconds * fps)))
        for _ in range(frame_count):
            frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _calibration_dict(cam_id: str) -> dict:
    """Plausible 2-cam calibration block for the synthetic videos."""
    width = 64
    height = 48
    focal = 60.0
    intrinsics = [
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ]
    # Two distinct world-to-camera transforms — exact numbers don't matter
    # for the smoke test, only that they're independent matrices.
    if cam_id == "cam1":
        extrinsics = [
            [1.0, 0.0, 0.0, -10.0],
            [0.0, 1.0, 0.0, 8.0],
            [0.0, 0.0, 1.0, -4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        extrinsics = [
            [-1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, -18.0],
            [0.0, 0.0, -1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    return {
        "cam_id": cam_id,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "image_size": [width, height],
    }


def test_cli_end_to_end_synthetic(tmp_path: Path) -> None:
    """Two synthetic videos → CLI runs → JSON validates as MatchAnalysis."""
    cam1 = tmp_path / "cam1.mp4"
    cam2 = tmp_path / "cam2.mp4"
    _write_synthetic_video(cam1, seed=1)
    _write_synthetic_video(cam2, seed=2)

    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        json.dumps([_calibration_dict("cam1"), _calibration_dict("cam2")]),
        encoding="utf-8",
    )

    out_path = tmp_path / "out.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--video",
            str(cam1),
            "--video",
            str(cam2),
            "--calib",
            str(calib_path),
            "--out",
            str(out_path),
            "--max-frames",
            "5",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    analysis = MatchAnalysis.model_validate(payload)
    # Stub mode → no detections → no ball 3D positions. That's fine.
    # The contract we verify is shape, length cap, and frame ordering.
    assert len(analysis.frames) == 5
    assert analysis.meta.match_id  # something non-empty derived from --out stem
    assert {c.cam_id for c in analysis.meta.cameras} == {"cam1", "cam2"}
    frame_ids = [f.frame_id for f in analysis.frames]
    assert frame_ids == sorted(frame_ids)


def test_cli_requires_matching_video_calib_count(tmp_path: Path) -> None:
    """Mismatched --video / calibration count must fail fast."""
    cam1 = tmp_path / "cam1.mp4"
    _write_synthetic_video(cam1, seed=1)

    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        json.dumps([_calibration_dict("cam1"), _calibration_dict("cam2")]),
        encoding="utf-8",
    )

    out_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--video",
            str(cam1),
            "--calib",
            str(calib_path),
            "--out",
            str(out_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code != 0


def test_cli_supports_calib_dict_shape(tmp_path: Path) -> None:
    """A calibration JSON with a top-level ``cameras`` key must also parse."""
    cam1 = tmp_path / "cam1.mp4"
    cam2 = tmp_path / "cam2.mp4"
    _write_synthetic_video(cam1, seed=1)
    _write_synthetic_video(cam2, seed=2)

    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        json.dumps({"cameras": [_calibration_dict("cam1"), _calibration_dict("cam2")]}),
        encoding="utf-8",
    )

    out_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--video",
            str(cam1),
            "--video",
            str(cam2),
            "--calib",
            str(calib_path),
            "--out",
            str(out_path),
            "--max-frames",
            "3",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    analysis = MatchAnalysis.model_validate(payload)
    assert len(analysis.frames) == 3


@pytest.mark.parametrize("flag", ["--video", "--calib", "--out"])
def test_cli_rejects_missing_required_flag(tmp_path: Path, flag: str) -> None:
    """Each required flag must be enforced by Click."""
    runner = CliRunner()
    # Invoke with a single irrelevant flag so Click prints usage rather than
    # entering the orchestrator path.
    result = runner.invoke(main, [flag, "/nonexistent/path"])
    assert result.exit_code != 0
