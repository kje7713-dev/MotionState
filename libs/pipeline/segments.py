"""Stub temporal segmenter – returns no segments.

Replace this with real temporal segmentation logic.
"""

from libs.pipeline.contracts import MotionFeature, Segment, Segmenter


class StubSegmenter(Segmenter):
    """No-op segmenter that always returns an empty list.

    Extension point: subclass Segmenter and implement ``segment``.
    """

    def segment(self, features: list[MotionFeature]) -> list[Segment]:
        """Return an empty list – real implementation not yet wired."""
        return []
