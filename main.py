"""Main runner for AI fitness trainer using webcam + pose estimation."""

from __future__ import annotations

import time

import cv2

from angle_utils import calculate_angle
from camera import Camera
from data_logger import WorkoutLogger
from exercise_detector import ExerciseDetector
from feedback_system import FeedbackSystem
from pose_estimation import PoseEstimator


def midpoint(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[float, float]:
    """Return midpoint between two 2D points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def compute_angles(landmarks: dict) -> dict:
    """Compute key joint angles used by detector and feedback modules."""
    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    left_elbow = landmarks["left_elbow"]
    right_elbow = landmarks["right_elbow"]
    left_wrist = landmarks["left_wrist"]
    right_wrist = landmarks["right_wrist"]
    left_hip = landmarks["left_hip"]
    right_hip = landmarks["right_hip"]
    left_knee = landmarks["left_knee"]
    right_knee = landmarks["right_knee"]
    left_ankle = landmarks["left_ankle"]
    right_ankle = landmarks["right_ankle"]

    # Average bilateral angles for smoother counting.
    elbow_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    elbow_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
    elbow = (elbow_left + elbow_right) / 2.0

    knee_left = calculate_angle(left_hip, left_knee, left_ankle)
    knee_right = calculate_angle(right_hip, right_knee, right_ankle)
    knee = (knee_left + knee_right) / 2.0

    hip_left = calculate_angle(left_shoulder, left_hip, left_knee)
    hip_right = calculate_angle(right_shoulder, right_hip, right_knee)
    hip = (hip_left + hip_right) / 2.0

    shoulder_left = calculate_angle(left_elbow, left_shoulder, left_hip)
    shoulder_right = calculate_angle(right_elbow, right_shoulder, right_hip)
    shoulder = (shoulder_left + shoulder_right) / 2.0

    # Back and body-line approximations for form checks.
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    knee_mid = midpoint(left_knee, right_knee)
    ankle_mid = midpoint(left_ankle, right_ankle)

    back = calculate_angle(shoulder_mid, hip_mid, knee_mid)
    body_line = calculate_angle(shoulder_mid, hip_mid, ankle_mid)

    # Choose front leg (lower y means visually higher in frame).
    front_knee = left_knee if left_knee[1] > right_knee[1] else right_knee
    front_ankle = left_ankle if front_knee == left_knee else right_ankle
    front_hip = left_hip if front_knee == left_knee else right_hip

    front_knee_angle = calculate_angle(front_hip, front_knee, front_ankle)

    return {
        "elbow": elbow,
        "knee": knee,
        "hip": hip,
        "shoulder": shoulder,
        "back": back,
        "body_line": body_line,
        "front_knee": front_knee_angle,
        "phase_down_score": 1 if knee < 120 else 0,
    }


def draw_ui(frame, exercise: str, reps: int, feedback: str, fps: float) -> None:
    """Overlay workout metadata and feedback text."""
    h, w, _ = frame.shape
    cv2.putText(frame, f"Exercise: {exercise}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Reps: {reps}", (w - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 220, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    x = max(10, (w - text_size[0]) // 2)
    y = h - 30
    cv2.putText(frame, feedback, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)


def main() -> None:
    """Initialize modules, run inference loop, and handle keyboard controls."""
    camera = Camera(src=0, width=1280, height=720)
    estimator = PoseEstimator()
    detector = ExerciseDetector()
    feedback_system = FeedbackSystem()
    logger = WorkoutLogger("workout_log.csv")

    try:
        camera.open()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return

    prev_time = time.time()
    last_logged_reps = {"Squat": 0, "Push-up": 0, "Lunge": 0}

    print("Controls: [1] Squat, [2] Push-up, [3] Lunge, [q] Quit")

    while True:
        frame = camera.read()
        if frame is None:
            print("[WARN] Failed to read frame from camera.")
            break

        pose_result = estimator.detect(frame)

        if pose_result:
            landmarks = pose_result.landmarks_px
            angles = compute_angles(landmarks)
            status = detector.update(angles)

            front_toe = landmarks["left_foot_index"] if landmarks["left_knee"][1] > landmarks["right_knee"][1] else landmarks["right_foot_index"]
            front_knee = landmarks["left_knee"] if landmarks["left_knee"][1] > landmarks["right_knee"][1] else landmarks["right_knee"]

            feedback = feedback_system.get_feedback(
                status.exercise,
                angles,
                joints={"front_knee": front_knee, "front_toe": front_toe},
            )

            if status.reps > last_logged_reps[status.exercise]:
                logger.log(status.exercise, status.reps)
                last_logged_reps[status.exercise] = status.reps
        else:
            status = detector.update({})
            feedback = "Pose not detected. Step into frame."

        estimator.draw_skeleton(frame, pose_result)

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        draw_ui(frame, status.exercise, status.reps, feedback, fps)
        cv2.imshow("AI Fitness Trainer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("1"):
            detector.set_exercise("Squat")
        elif key == ord("2"):
            detector.set_exercise("Push-up")
        elif key == ord("3"):
            detector.set_exercise("Lunge")
        elif key == ord("q"):
            break

    camera.release()
    estimator.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
