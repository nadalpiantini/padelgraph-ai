"""Multi-camera synchronization (story-003).

Public API:

- :class:`MultiCamSync` — align N video streams onto a common timestamp grid.

Apache 2.0.
"""

from padelgraph_ai.sync.timestamp_sync import MultiCamSync

__all__ = ["MultiCamSync"]
