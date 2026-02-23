"""Real-time posture feedback based on exercise-specific form rules."""

from __future__ import annotations

from typing import Dict


class FeedbackSystem:
    """Generates human-readable coaching hints from current pose angles."""

    def get_feedback(self, exercise: str, angles: Dict[str, float], joints: Dict[str, tuple]) -> str:
        if exercise == "Squat":
            back_angle = angles.get("back", 180.0)
            knee_angle = angles.get("knee", 180.0)

            if back_angle < 150:
                return "Keep your back straight"
            if knee_angle > 90 and angles.get("phase_down_score", 0) > 0:
                return "Go lower"
            return "Great squat form"

        if exercise == "Push-up":
            elbow_angle = angles.get("elbow", 180.0)
            body_angle = angles.get("body_line", 180.0)

            if elbow_angle > 160:
                return "Lower your body"
            if body_angle < 155:
                return "Keep body straight"
            return "Great push-up form"

        if exercise == "Lunge":
            knee = joints.get("front_knee")
            toe = joints.get("front_toe")
            if knee and toe and knee[0] > toe[0]:
                return "Don't push knee forward"
            return "Great lunge form"

        return "Select an exercise"
