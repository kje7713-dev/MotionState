"""MediaPipe BlazePose-backed pose estimator.

This module provides a concrete :class:`PoseEstimator` implementation that uses
MediaPipe's Pose solution to extract 2-D body keypoints from cropped person
regions.

Install the required extras before enabling this backend::

    pip install mediapipe pillow numpy
    # or: pip install -e ".[pose]"

Then set ``POSE_BACKEND=mediapipe`` in your environment.
"""

from __future__ import annotations

import io
import logging

from libs.pipeline.contracts import Frame, Keypoint, PoseEstimate, PoseEstimator, Track

logger = logging.getLogger(__name__)

# MediaPipe BlazePose produces 33 named landmarks.
_LANDMARK_NAMES: list[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


class MediaPipePoseEstimator(PoseEstimator):
    """Pose estimator backed by MediaPipe BlazePose.

    For each tracked person visible in a frame this estimator crops the person
    region and runs MediaPipe Pose.  Keypoint coordinates are reported in the
    full-frame pixel space.

    Args:
        min_confidence: Minimum landmark visibility score to include a keypoint
            in the output.  Landmarks below this threshold are omitted.
    """

    def __init__(self, min_confidence: float = 0.3) -> None:
        try:
            import mediapipe as mp
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "MediaPipePoseEstimator requires 'mediapipe', 'pillow', and 'numpy'. "
                "Install them with: pip install mediapipe pillow numpy  "
                "or: pip install -e '.[pose]'"
            ) from exc

        self._mp = mp
        self._np = np
        self._Image = Image
        self._min_confidence = min_confidence
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=min_confidence,
        )

    def estimate(self, frame: Frame, tracks: list[Track]) -> list[PoseEstimate]:
        """Return pose estimates for all tracks visible in *frame*.

        Only tracks whose most-recent detection falls on ``frame.index`` are
        processed.  A pose is emitted for each such track; tracks for which
        MediaPipe finds no landmarks are skipped silently.
        """
        np = self._np
        Image = self._Image

        # Collect (track, detection) pairs active in this frame.
        active: dict[int, tuple[Track, object]] = {}
        for track in tracks:
            for det in track.detections:
                if det.frame_index == frame.index:
                    active[track.track_id] = (track, det)
                    break

        if not active:
            return []

        # Decode frame bytes to a full-resolution RGB numpy array.
        try:
            img = Image.open(io.BytesIO(frame.data)).convert("RGB")
            img_array = np.array(img)
        except Exception:
            logger.warning("Failed to decode frame %d for pose estimation.", frame.index)
            return []

        h_full, w_full = img_array.shape[:2]
        poses: list[PoseEstimate] = []

        for track_id, (_, det) in active.items():
            x1 = max(0, int(det.bbox.x))
            y1 = max(0, int(det.bbox.y))
            x2 = min(w_full, int(det.bbox.x + det.bbox.width))
            y2 = min(h_full, int(det.bbox.y + det.bbox.height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img_array[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]

            try:
                result = self._pose.process(crop)
            except Exception:
                logger.warning(
                    "Pose estimation failed for track %d in frame %d.",
                    track_id,
                    frame.index,
                )
                continue

            if result.pose_landmarks is None:
                continue

            keypoints: list[Keypoint] = []
            for idx, landmark in enumerate(result.pose_landmarks.landmark):
                if landmark.visibility < self._min_confidence:
                    continue
                # Convert normalized crop coordinates back to full-frame pixels.
                kp_x = x1 + landmark.x * crop_w
                kp_y = y1 + landmark.y * crop_h
                keypoints.append(
                    Keypoint(
                        name=_LANDMARK_NAMES[idx],
                        x=float(kp_x),
                        y=float(kp_y),
                        confidence=float(landmark.visibility),
                    )
                )

            poses.append(
                PoseEstimate(
                    frame_index=frame.index,
                    track_id=track_id,
                    keypoints=keypoints,
                )
            )

        return poses
