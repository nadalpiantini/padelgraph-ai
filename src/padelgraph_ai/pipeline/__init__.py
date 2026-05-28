"""End-to-end inference pipeline (story-005).

Thin orchestration layer that wires :mod:`padelgraph_ai.detection`,
:mod:`padelgraph_ai.sync`, :mod:`padelgraph_ai.court`, and
:mod:`padelgraph_ai.fusion` into a single command-line entry point.

Apache 2.0.
"""

from padelgraph_ai.pipeline.infer import main

__all__ = ["main"]
