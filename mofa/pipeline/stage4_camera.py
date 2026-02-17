from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CameraParams:
    scale: float
    rotation: tuple
    translation: tuple


def solve_cameras(view_count: int) -> List[CameraParams]:
    return [CameraParams(scale=1.0, rotation=(0.0, 0.0, 0.0), translation=(0.0, 0.0)) for _ in range(view_count)]
