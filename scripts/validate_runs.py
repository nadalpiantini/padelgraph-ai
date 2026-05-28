"""Anomaly detector + markdown report writer for stress test outputs.

Consumes all ``run_*.json`` files in --runs-dir, parses each against
:class:`padelgraph_ai.schemas.MatchAnalysis`, and classifies anomalies:

- ``crash``      — file unreadable or carries an ``_error`` marker
- ``invalid``    — JSON parses but fails Pydantic validation
- ``nan_3d``     — at least one frame has a 3D ball position with NaN/Inf
- ``empty``      — frames list is empty
- ``ok``         — all checks pass

Writes a markdown report with:
- Total runs · success count · failure breakdown
- Up to 5 example anomalies per category (sim_id + reason)
- Exit code 0 if no *critical* anomalies (crash/invalid), 1 otherwise.

Usage:
    uv run python scripts/validate_runs.py --runs-dir runs/ \\
        --report artifacts/stress-test-20260528-120000.md

Apache 2.0.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import click

# Pydantic import is local to keep `--help` fast and avoid hard dep on padelgraph_ai
# when only checking flags. We import inside main().


def _classify_run(run_path: Path) -> tuple[str, str]:
    """Classify a single run file. Returns ``(category, reason)``.

    Categories: 'ok', 'crash', 'invalid', 'nan_3d', 'empty'.
    """
    # Step 1: can we read + parse JSON?
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ("crash", f"unreadable or invalid JSON: {exc}")

    # Step 2: explicit error marker from upstream phase
    if isinstance(payload, dict) and "_error" in payload:
        return ("crash", str(payload["_error"]))

    # Step 3: validate Pydantic shape
    try:
        from padelgraph_ai.schemas import MatchAnalysis

        analysis = MatchAnalysis.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 — we want to catch all validation failures
        return ("invalid", f"Pydantic validation failed: {exc}")

    # Step 4: empty frames?
    if not analysis.frames:
        return ("empty", "frames list is empty")

    # Step 5: NaN/Inf in 3D positions
    for frame in analysis.frames:
        pos = frame.ball.position_3d
        if pos is None:
            continue
        for component in (pos.x, pos.y, pos.z):
            if math.isnan(component) or math.isinf(component):
                return ("nan_3d", f"frame {frame.frame_id} has non-finite 3D component")

    return ("ok", "all checks passed")


def _build_report(
    classified: list[tuple[str, str, str]],
    runs_dir: Path,
) -> str:
    """Build the markdown report body."""
    total = len(classified)
    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sim_id, category, reason in classified:
        by_category[category].append((sim_id, reason))

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    ok_count = len(by_category.get("ok", []))
    crash_count = len(by_category.get("crash", []))
    invalid_count = len(by_category.get("invalid", []))
    nan_count = len(by_category.get("nan_3d", []))
    empty_count = len(by_category.get("empty", []))

    success_pct = (ok_count / total * 100) if total else 0.0

    lines: list[str] = []
    lines.append(f"# padelgraph-ai stress test report — {timestamp}")
    lines.append("")
    lines.append(f"**Runs analyzed:** {total}")
    lines.append(f"**Source:** `{runs_dir}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Count | Percent |")
    lines.append("|----------|-------|---------|")
    lines.append(f"| ok | {ok_count} | {success_pct:.1f}% |")
    lines.append(f"| crash | {crash_count} | {crash_count / total * 100 if total else 0:.1f}% |")
    lines.append(
        f"| invalid | {invalid_count} | {invalid_count / total * 100 if total else 0:.1f}% |"
    )
    lines.append(f"| nan_3d | {nan_count} | {nan_count / total * 100 if total else 0:.1f}% |")
    lines.append(f"| empty | {empty_count} | {empty_count / total * 100 if total else 0:.1f}% |")
    lines.append("")

    critical = crash_count + invalid_count
    if critical == 0:
        lines.append("**Verdict:** PASS — no critical anomalies (crash / invalid).")
    else:
        lines.append(
            f"**Verdict:** FAIL — {critical} critical anomalies (crash / invalid). See examples below."
        )
    lines.append("")

    # Examples per category (max 5 each)
    for category in ("crash", "invalid", "nan_3d", "empty"):
        examples = by_category.get(category, [])
        if not examples:
            continue
        lines.append(f"## {category} examples (showing up to 5 of {len(examples)})")
        lines.append("")
        for sim_id, reason in examples[:5]:
            lines.append(f"- `{sim_id}` — {reason}")
        lines.append("")

    return "\n".join(lines) + "\n"


@click.command()
@click.option(
    "--runs-dir",
    "runs_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing run_*.json files",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output markdown report path",
)
def main(runs_dir: Path, report_path: Path) -> None:
    """Validate stress test runs + write report. Exit 1 if critical anomalies found."""
    run_files = sorted(runs_dir.glob("run_*.json"))
    if not run_files:
        click.echo(f"no run_*.json files found in {runs_dir}", err=True)
        sys.exit(1)

    classified: list[tuple[str, str, str]] = []
    for run_path in run_files:
        sim_id = run_path.stem.removeprefix("run_")
        category, reason = _classify_run(run_path)
        classified.append((sim_id, category, reason))

    report_text = _build_report(classified, runs_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    critical = sum(1 for _, cat, _ in classified if cat in ("crash", "invalid"))
    total = len(classified)
    ok_count = sum(1 for _, cat, _ in classified if cat == "ok")

    click.echo(f"validated {total} runs: {ok_count} ok, {critical} critical")
    click.echo(f"report written to {report_path}")

    if critical > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
