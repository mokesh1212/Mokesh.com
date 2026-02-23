"""Math helpers for calculating biomechanical joint angles."""

from __future__ import annotations

from typing import Tuple

import numpy as np


Point = Tuple[float, float]


def calculate_angle(point1: Point, point2: Point, point3: Point) -> float:
    """Calculate the angle (in degrees) formed by point1-point2-point3.

    The middle point (point2) is treated as the vertex.
    """
    a = np.array(point1, dtype=np.float32)
    b = np.array(point2, dtype=np.float32)
    c = np.array(point3, dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return float(angle)
