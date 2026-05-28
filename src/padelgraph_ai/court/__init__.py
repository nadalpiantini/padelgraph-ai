"""Court detection + homography 2D for padelgraph-ai.

Public interface:
- `CourtDetector` — interactive corner picker (MVP) + cache load/save
- `Homography` — pixel ↔ court-coordinate (meters) transform

Apache 2.0.
"""

from padelgraph_ai.court.homography import CourtDetector, Homography

__all__ = ["CourtDetector", "Homography"]
