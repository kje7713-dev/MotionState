"""YOLOv8-based person detector.

This module provides a concrete :class:`Detector` implementation backed by
`ultralytics` (YOLOv8).  The dependency is **optional**: if ``ultralytics``
is not installed the class still exists but raises :class:`ImportError` when
instantiated, allowing the rest of the codebase (and the stub detector) to
work without it.

Usage::

    from libs.pipeline.detector_yolo import YoloDetector
    detector = YoloDetector()           # uses yolov8n.pt by default
    detections = detector.detect(frame)
"""

from __future__ import annotations

import logging

from libs.pipeline.contracts import BoundingBox, Detection, Detector, Frame

logger = logging.getLogger(__name__)

# Person class index in the COCO dataset (used by standard YOLO weights).
_COCO_PERSON_CLASS_ID = 0
_COCO_PERSON_LABEL = "person"


class YoloDetector(Detector):
    """Person detector backed by YOLOv8 (ultralytics).

    Args:
        model_name: The model checkpoint to load (e.g. ``"yolov8n.pt"``).
            Any name accepted by :func:`ultralytics.YOLO` works.
        confidence_threshold: Minimum confidence to retain a detection.

    Raises:
        ImportError: if ``ultralytics`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required to use YoloDetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        self._model = YOLO(model_name)
        self._conf = confidence_threshold
        logger.info(
            "YoloDetector initialised with model=%s conf=%.2f", model_name, confidence_threshold
        )

    def detect(self, frame: Frame) -> list[Detection]:
        """Run YOLOv8 inference and return person detections.

        Args:
            frame: A :class:`~libs.pipeline.contracts.Frame` whose *data*
                field contains raw image bytes (JPEG or PNG).

        Returns:
            A list of :class:`~libs.pipeline.contracts.Detection` objects,
            one per person detected in the frame.  Only detections whose
            class label is ``"person"`` are returned.
        """
        import io

        import numpy as np
        from PIL import Image  # type: ignore[import]

        # Decode raw bytes → PIL Image → numpy array for ultralytics.
        image = Image.open(io.BytesIO(frame.data)).convert("RGB")
        img_array = np.array(image)

        results = self._model.predict(
            img_array,
            conf=self._conf,
            classes=[_COCO_PERSON_CLASS_ID],
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != _COCO_PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = (v.item() for v in box.xyxy[0])
                conf = float(box.conf[0].item())
                detections.append(
                    Detection(
                        frame_index=frame.index,
                        bbox=BoundingBox(
                            x=x1,
                            y=y1,
                            width=x2 - x1,
                            height=y2 - y1,
                            confidence=conf,
                        ),
                        class_id=_COCO_PERSON_CLASS_ID,
                        class_label=_COCO_PERSON_LABEL,
                    )
                )

        return detections
