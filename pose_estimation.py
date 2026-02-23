"""Pose estimation module using MediaPipe BlazePose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseResult:
    """Normalized + pixel-space landmark data."""

    landmarks_px: Dict[str, Tuple[int, int]]
    landmarks_norm: Dict[str, Tuple[float, float, float]]
    raw_result: object


class PoseEstimator:
    """Runs MediaPipe pose model and returns body keypoints."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmark_names = {
            "nose": self.mp_pose.PoseLandmark.NOSE,
            "left_shoulder": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            "right_shoulder": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "left_elbow": self.mp_pose.PoseLandmark.LEFT_ELBOW,
            "right_elbow": self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            "left_wrist": self.mp_pose.PoseLandmark.LEFT_WRIST,
            "right_wrist": self.mp_pose.PoseLandmark.RIGHT_WRIST,
            "left_hip": self.mp_pose.PoseLandmark.LEFT_HIP,
            "right_hip": self.mp_pose.PoseLandmark.RIGHT_HIP,
            "left_knee": self.mp_pose.PoseLandmark.LEFT_KNEE,
            "right_knee": self.mp_pose.PoseLandmark.RIGHT_KNEE,
            "left_ankle": self.mp_pose.PoseLandmark.LEFT_ANKLE,
            "right_ankle": self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            "left_foot_index": self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            "right_foot_index": self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
        }

    def detect(self, frame_bgr: np.ndarray) -> PoseResult | None:
        """Run pose detection and return landmarks in normalized + pixel spaces."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        rgb.flags.writeable = True

        if not result.pose_landmarks:
            return None

        h, w, _ = frame_bgr.shape
        landmarks_px: Dict[str, Tuple[int, int]] = {}
        landmarks_norm: Dict[str, Tuple[float, float, float]] = {}

        for name, idx in self.landmark_names.items():
            lm = result.pose_landmarks.landmark[idx.value]
            x_px, y_px = int(lm.x * w), int(lm.y * h)
            landmarks_px[name] = (x_px, y_px)
            landmarks_norm[name] = (lm.x, lm.y, lm.visibility)

        return PoseResult(landmarks_px=landmarks_px, landmarks_norm=landmarks_norm, raw_result=result)

    def draw_skeleton(self, frame_bgr: np.ndarray, pose_result: PoseResult | None) -> np.ndarray:
        """Draw pose skeleton overlay on frame."""
        if pose_result and pose_result.raw_result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame_bgr,
                pose_result.raw_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
        return frame_bgr

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()
