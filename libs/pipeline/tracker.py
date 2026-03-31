"""Stub tracker – returns no tracks.

Replace this with a real multi-object tracker (ByteTrack, DeepSORT, …).
"""

from libs.pipeline.contracts import Detection, Track, Tracker


class StubTracker(Tracker):
    """No-op tracker that always returns an empty track list.

    Extension point: subclass Tracker and implement ``update`` with a real algorithm.
    """

    def update(self, detections: list[Detection]) -> list[Track]:
        """Return an empty list – real implementation not yet wired."""
        return []
