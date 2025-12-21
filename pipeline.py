from typing import Any, Tuple

import numpy as np
import cv2
from modules.FaceAlignment import FaceAlignment
from modules.FaceDetection import FaceDetection

from triton_service import run_inference, run_inference_retinaface


def open_image(image_bytes: Any) -> Any:
    try:
        return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        raise Exception("Failed to open image: " + str(e))
    
def to_bytes(img: np.ndarray, fmt: str = ".jpg") -> bytes:
    """
    Convert a numpy OpenCV image to bytes in the given format (default JPEG).
    """
    success, encoded = cv2.imencode(fmt, img)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded.tobytes()


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a_norm = np.linalg.norm(vec_a)
    b_norm = np.linalg.norm(vec_b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (a_norm * b_norm))


def get_embeddings(client: Any, image_a: bytes, image_b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call Triton twice to obtain embeddings for two images.

    Extend this by adding detection/alignment/antispoof when those Triton models
    are available in the repository. For now we assume inputs are already aligned.
    """
    emb_a = run_inference(client, image_a)
    emb_b = run_inference(client, image_b)
    return emb_a.squeeze(0), emb_b.squeeze(0)


def get_faces(client: Any, image_a: bytes, image_b: bytes):
    faces_a = run_inference_retinaface(client, image_a)
    faces_b = run_inference_retinaface(client, image_b)

    return faces_a, faces_b


def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    """
    Minimal end-to-end similarity using Triton-managed FR model.

    Students should swap in detection, alignment, and spoofing once those models
    are added to the Triton repository. This keeps all model execution on Triton.
    """
    face_a, face_b = get_faces(client, image_a, image_b)

    decoded_image_a = open_image(image_a)
    decoded_image_b = open_image(image_b) 

    landmarks_a = face_a['landmarks']
    kps_a = FaceDetection.get_kps(landmarks_a)
    aligned_image_a = FaceAlignment.crop_and_align(face_a, kps_a, decoded_image_a)

    landmarks_b = face_b['landmarks']
    kps_b = FaceDetection.get_kps(landmarks_b)
    aligned_image_b = FaceAlignment.crop_and_align(face_b, kps_b, decoded_image_b)
    
    emb_a, emb_b = get_embeddings(client, to_bytes(aligned_image_a), to_bytes(aligned_image_b))
    return _cosine_similarity(emb_a, emb_b)
