"""Stub pose estimator – returns no pose estimates.

Replace this with a real model (MediaPipe, ViTPose, OpenPose, …).
"""

from libs.pipeline.contracts import Frame, PoseEstimate, PoseEstimator, Track


class StubPoseEstimator(PoseEstimator):
    """No-op pose estimator that always returns an empty list.

    Extension point: subclass PoseEstimator and implement ``estimate``.
    """

    def estimate(self, frame: Frame, tracks: list[Track]) -> list[PoseEstimate]:
        """Return an empty list – real implementation not yet wired."""
        return []
