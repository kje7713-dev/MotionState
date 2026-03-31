"""Stub detector – returns no detections.

Replace this with a real model (YOLOv8, RT-DETR, …) once the CV layer is added.
"""

from libs.pipeline.contracts import Detection, Detector, Frame


class StubDetector(Detector):
    """No-op detector that always returns an empty detection list.

    Extension point: subclass Detector and implement ``detect`` with a real model.
    """

    def detect(self, frame: Frame) -> list[Detection]:
        """Return an empty list – real implementation not yet wired."""
        return []
