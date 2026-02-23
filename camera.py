"""Camera utilities for webcam capture and frame preprocessing."""

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    """Simple wrapper around cv2.VideoCapture with preprocessing helpers."""

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720) -> None:
        self.src = src
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the camera stream and set expected frame size."""
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Please connect a camera and try again.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> np.ndarray | None:
        """Read and preprocess a frame from the webcam."""
        if self.cap is None:
            raise RuntimeError("Camera is not opened. Call open() first.")

        ok, frame = self.cap.read()
        if not ok:
            return None

        return self.preprocess(frame)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and mirror frame for a selfie-like UX."""
        frame = cv2.resize(frame, (self.width, self.height))
        frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
