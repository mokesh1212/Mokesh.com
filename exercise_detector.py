"""Exercise detection and repetition counting based on joint angle thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExerciseStatus:
    """Structured response for current exercise state."""

    exercise: str
    reps: int
    phase: str


class ExerciseDetector:
    """State-machine detector supporting squats, push-ups, and lunges."""

    def __init__(self) -> None:
        self.rep_counters: Dict[str, int] = {"Squat": 0, "Push-up": 0, "Lunge": 0}
        self.last_phase: Dict[str, str] = {"Squat": "up", "Push-up": "up", "Lunge": "up"}
        self.current_exercise = "Squat"

    def set_exercise(self, exercise_name: str) -> None:
        if exercise_name in self.rep_counters:
            self.current_exercise = exercise_name

    def update(self, angles: Dict[str, float]) -> ExerciseStatus:
        """Update repetition counts using threshold logic and phase transitions."""
        exercise = self.current_exercise
        phase = self.last_phase[exercise]

        if exercise == "Squat":
            knee = angles.get("knee", 180.0)
            if knee < 70:
                phase = "down"
            elif knee > 160 and self.last_phase[exercise] == "down":
                phase = "up"
                self.rep_counters[exercise] += 1

        elif exercise == "Push-up":
            elbow = angles.get("elbow", 180.0)
            if elbow < 70:
                phase = "down"
            elif elbow > 160 and self.last_phase[exercise] == "down":
                phase = "up"
                self.rep_counters[exercise] += 1

        elif exercise == "Lunge":
            front_knee = angles.get("front_knee", 180.0)
            if front_knee < 80:
                phase = "down"
            elif front_knee > 160 and self.last_phase[exercise] == "down":
                phase = "up"
                self.rep_counters[exercise] += 1

        self.last_phase[exercise] = phase
        return ExerciseStatus(exercise=exercise, reps=self.rep_counters[exercise], phase=phase)
