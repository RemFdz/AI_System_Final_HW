
from typing import Any
import numpy as np


class FaceDetection:
    def __init__(self):
        pass

    @staticmethod
    def get_kps(landmarks: Any) -> np.ndarray:
        landmarks = np.array(landmarks).reshape(-1, 2).astype(np.float32)
        kps = landmarks
        
        return kps