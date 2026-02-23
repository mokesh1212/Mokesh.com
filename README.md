# AI Fitness Trainer (Webcam + Pose Estimation)

This project uses your webcam and MediaPipe BlazePose to detect body landmarks, classify exercises, count repetitions, and provide real-time posture feedback.

## Features

- Live webcam capture and preprocessing.
- Full-body landmark detection with skeleton overlay.
- Joint angle calculations (knee, hip, elbow, shoulder).
- Exercise detection for:
  - Squats
  - Push-ups
  - Lunges
- Repetition counting via angle-based state machine.
- Real-time form feedback messages.
- UI overlay includes:
  - Exercise name
  - Rep counter
  - Feedback message
  - FPS counter
- Workout data logging to CSV (`workout_log.csv`).

## Project Structure

- `main.py` – app entry point and UI loop.
- `camera.py` – webcam access + frame preprocessing.
- `pose_estimation.py` – MediaPipe pose model wrapper.
- `angle_utils.py` – angle calculation helper.
- `exercise_detector.py` – exercise classification + rep counting.
- `feedback_system.py` – posture correction tips.
- `data_logger.py` – CSV logging.
- `requirements.txt` – Python dependencies.

## Installation

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
python main.py
```

### Controls

- `1`: Select **Squat** mode
- `2`: Select **Push-up** mode
- `3`: Select **Lunge** mode
- `q`: Quit

## Notes

- Ensure your full body is visible in the camera for best results.
- Good lighting improves landmark accuracy.
- Logs are appended to `workout_log.csv` whenever a new rep is counted.
