from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass
class PreprocessedFace:
    image_tensor: np.ndarray
    pose: Tuple[float, float, float]
    bbox: Tuple[int, int, int, int]


def preprocess_images(images: List[Image.Image], size: int = 224) -> List[PreprocessedFace]:
    outputs = []
    for img in images:
        rgb = img.convert("RGB")
        w, h = rgb.size
        box = (0, 0, w, h)
        resized = rgb.resize((size, size))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        outputs.append(PreprocessedFace(image_tensor=arr, pose=(0.0, 0.0, 0.0), bbox=box))
    return outputs
