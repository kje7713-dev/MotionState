"""Stub feature deriver – returns no features.

Replace this with real kinematic or positional feature computations.
"""

from libs.pipeline.contracts import FeatureDeriver, MotionFeature, PoseEstimate, Track


class StubFeatureDeriver(FeatureDeriver):
    """No-op feature deriver that always returns an empty list.

    Extension point: subclass FeatureDeriver and implement ``derive``.
    """

    def derive(
        self,
        tracks: list[Track],
        poses: list[PoseEstimate],
    ) -> list[MotionFeature]:
        """Return an empty list – real implementation not yet wired."""
        return []
