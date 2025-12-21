
from typing import Any
import numpy as np


class FaceDetection:
    def __init__(self):
        pass

    @staticmethod
    def get_kps(landmarks: Any) -> Any:
        landmarks = np.array(landmarks).reshape(-1, 2)
        
        kps = np.array([
            landmarks[0],  # left_eye
            landmarks[1],  # right_eye
            landmarks[2],  # nose
            landmarks[3],  # mouth_left
            landmarks[4],  # mouth_right
        ])

        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        return kps