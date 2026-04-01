"""Concrete feature deriver – computes generic geometric motion features.

Derives scalar features from tracked pose data.  All features are
domain-agnostic: no sport-specific labels, no ontology, no scoring.
"""

from __future__ import annotations

import math

from libs.pipeline.contracts import FeatureDeriver, MotionFeature, PoseEstimate, Track

# Minimum keypoint confidence to count a landmark as "visible".
_VISIBILITY_THRESHOLD = 0.5


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _angle_deg(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    """Return the interior angle at *B* in the A-B-C triplet, in degrees."""
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    mag_ba = math.sqrt(bax**2 + bay**2)
    mag_bc = math.sqrt(bcx**2 + bcy**2)
    if mag_ba == 0.0 or mag_bc == 0.0:
        return 0.0
    cos_a = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_a))


class BasicFeatureDeriver(FeatureDeriver):
    """Derives generic geometric and temporal motion features.

    Per-pose features (one value per :class:`~libs.pipeline.contracts.PoseEstimate`):

    * ``torso_angle`` – lean of the torso relative to vertical (degrees).
    * ``shoulder_width`` – pixel distance between left and right shoulders.
    * ``hip_width`` – pixel distance between left and right hips.
    * ``left_elbow_angle`` – joint angle at the left elbow (degrees).
    * ``right_elbow_angle`` – joint angle at the right elbow (degrees).
    * ``left_knee_angle`` – joint angle at the left knee (degrees).
    * ``right_knee_angle`` – joint angle at the right knee (degrees).
    * ``keypoint_visibility_count`` – number of high-confidence keypoints.
    * ``bbox_area`` – bounding-box area in pixels² (requires matching track
      detection for the same frame).

    Inter-frame feature (one value per consecutive pose pair per track):

    * ``centroid_velocity`` – keypoint-centroid displacement per millisecond
      between consecutive frames of the same track.
    """

    def derive(
        self,
        tracks: list[Track],
        poses: list[PoseEstimate],
    ) -> list[MotionFeature]:
        """Return a list of :class:`~libs.pipeline.contracts.MotionFeature` objects.

        Args:
            tracks: All tracks produced by the tracker (with full detection
                histories).  Used to look up bounding boxes for ``bbox_area``.
            poses: All pose estimates across all frames and tracks.
        """
        if not poses:
            return []

        features: list[MotionFeature] = []

        # Build (track_id, frame_index) -> BoundingBox lookup for bbox_area.
        _bbox_lookup: dict[tuple[int, int], tuple] = {}
        for track in tracks:
            for det in track.detections:
                _bbox_lookup[(track.track_id, det.frame_index)] = det.bbox

        # Group and sort poses per track for inter-frame features.
        poses_by_track: dict[int, list[PoseEstimate]] = {}
        for pose in poses:
            poses_by_track.setdefault(pose.track_id, []).append(pose)
        for track_poses in poses_by_track.values():
            track_poses.sort(key=lambda p: p.frame_index)

        # --- Per-pose features ---
        for pose in poses:
            kp = {k.name: k for k in pose.keypoints}
            ts = pose.timestamp_ms
            tid = pose.track_id

            # torso_angle – lean of torso relative to vertical (image y-axis).
            if all(n in kp for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
                ls, rs = kp["left_shoulder"], kp["right_shoulder"]
                lh, rh = kp["left_hip"], kp["right_hip"]
                smx = (ls.x + rs.x) / 2.0
                smy = (ls.y + rs.y) / 2.0
                hmx = (lh.x + rh.x) / 2.0
                hmy = (lh.y + rh.y) / 2.0
                dx, dy = smx - hmx, smy - hmy
                angle = math.degrees(math.atan2(abs(dx), abs(dy))) if (dx != 0 or dy != 0) else 0.0
                features.append(
                    MotionFeature(
                        track_id=tid, name="torso_angle", start_ms=ts, end_ms=ts, value=angle
                    )
                )

            # shoulder_width
            if "left_shoulder" in kp and "right_shoulder" in kp:
                ls, rs = kp["left_shoulder"], kp["right_shoulder"]
                features.append(
                    MotionFeature(
                        track_id=tid,
                        name="shoulder_width",
                        start_ms=ts,
                        end_ms=ts,
                        value=_distance(ls.x, ls.y, rs.x, rs.y),
                    )
                )

            # hip_width
            if "left_hip" in kp and "right_hip" in kp:
                lh, rh = kp["left_hip"], kp["right_hip"]
                features.append(
                    MotionFeature(
                        track_id=tid,
                        name="hip_width",
                        start_ms=ts,
                        end_ms=ts,
                        value=_distance(lh.x, lh.y, rh.x, rh.y),
                    )
                )

            # left_elbow_angle
            if all(n in kp for n in ("left_shoulder", "left_elbow", "left_wrist")):
                ls, le, lw = kp["left_shoulder"], kp["left_elbow"], kp["left_wrist"]
                features.append(
                    MotionFeature(
                        track_id=tid,
                        name="left_elbow_angle",
                        start_ms=ts,
                        end_ms=ts,
                        value=_angle_deg(ls.x, ls.y, le.x, le.y, lw.x, lw.y),
                    )
                )

            # right_elbow_angle
            if all(n in kp for n in ("right_shoulder", "right_elbow", "right_wrist")):
                rs, re, rw = kp["right_shoulder"], kp["right_elbow"], kp["right_wrist"]
                features.append(
                    MotionFeature(
                        track_id=tid,
                        name="right_elbow_angle",
                        start_ms=ts,
                        end_ms=ts,
                        value=_angle_deg(rs.x, rs.y, re.x, re.y, rw.x, rw.y),
                    )
                )

            # left_knee_angle
            if all(n in kp for n in ("left_hip", "left_knee", "left_ankle")):
                lh, lk, la = kp["left_hip"], kp["left_knee"], kp["left_ankle"]
                features.append(
                    MotionFeature(
                        track_id=tid,
                        name="left_knee_angle",
                        start_ms=ts,
                        end_ms=ts,
                        value=_angle_deg(lh.x, lh.y, lk.x, lk.y, la.x, la.y),
                    )
                )

            # right_knee_angle
            if all(n in kp for n in ("right_hip", "right_knee", "right_ankle")):
                rh, rk, ra = kp["right_hip"], kp["right_knee"], kp["right_ankle"]
                features.append(
                    MotionFeature(
                        track_id=tid,
                        name="right_knee_angle",
                        start_ms=ts,
                        end_ms=ts,
                        value=_angle_deg(rh.x, rh.y, rk.x, rk.y, ra.x, ra.y),
                    )
                )

            # keypoint_visibility_count
            visible = sum(1 for k in pose.keypoints if k.confidence >= _VISIBILITY_THRESHOLD)
            features.append(
                MotionFeature(
                    track_id=tid,
                    name="keypoint_visibility_count",
                    start_ms=ts,
                    end_ms=ts,
                    value=float(visible),
                )
            )

            # bbox_area (from matching track detection for this frame)
            bbox = _bbox_lookup.get((tid, pose.frame_index))
            if bbox is not None:
                features.append(
                    MotionFeature(
                        track_id=tid,
                        name="bbox_area",
                        start_ms=ts,
                        end_ms=ts,
                        value=bbox.width * bbox.height,
                    )
                )

        # --- Inter-frame feature: centroid_velocity ---
        for tid, track_poses in poses_by_track.items():
            for i in range(1, len(track_poses)):
                p0, p1 = track_poses[i - 1], track_poses[i]
                if not p0.keypoints or not p1.keypoints:
                    continue
                cx0 = sum(k.x for k in p0.keypoints) / len(p0.keypoints)
                cy0 = sum(k.y for k in p0.keypoints) / len(p0.keypoints)
                cx1 = sum(k.x for k in p1.keypoints) / len(p1.keypoints)
                cy1 = sum(k.y for k in p1.keypoints) / len(p1.keypoints)
                dist = _distance(cx0, cy0, cx1, cy1)
                dt = p1.timestamp_ms - p0.timestamp_ms
                velocity = dist / dt if dt > 0.0 else 0.0
                features.append(
                    MotionFeature(
                        track_id=tid,
                        name="centroid_velocity",
                        start_ms=p0.timestamp_ms,
                        end_ms=p1.timestamp_ms,
                        value=velocity,
                    )
                )

        return features
