"""CSV logging module for workout summaries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


class WorkoutLogger:
    """Append workout events (exercise, reps, timestamp) to CSV file."""

    def __init__(self, csv_path: str = "workout_log.csv") -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=["Timestamp", "Exercise Name", "Repetitions"])
            df.to_csv(self.csv_path, index=False)

    def log(self, exercise_name: str, reps: int) -> None:
        """Add a single row to the CSV log."""
        row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Exercise Name": exercise_name,
            "Repetitions": reps,
        }
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode="a", header=False, index=False)
