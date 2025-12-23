from typing import Any, Tuple

import io
import numpy as np
import cv2
from modules.FaceAlignment import FaceAlignment
from modules.FaceDetection import FaceDetection
from starlette.responses import StreamingResponse

from triton_service import run_inference, run_inference_retinaface

def face_debug_bbox_landmarks_to_png_response(
    image: np.ndarray,
    bbox: Any,
    landmarks: np.ndarray
) -> StreamingResponse:
    debug_img = image.copy()

    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for (x, y) in landmarks.astype(int):
        cv2.circle(debug_img, (x, y), 3, (0, 0, 255), -1)

    success, buffer = cv2.imencode(".png", debug_img)
    if not success:
        raise RuntimeError("Failed to encode PNG")

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png"
    )


def face_to_png_response(aligned_face: np.ndarray) -> StreamingResponse:
    success, buffer = cv2.imencode(".png", aligned_face)
    if not success:
        raise Exception("Failed to encode image as PNG")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


def open_image(image_bytes: Any) -> Any:
    try:
        return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        raise Exception("Failed to open image: " + str(e))
    
def to_bytes(img: np.ndarray, fmt: str = ".jpg") -> bytes:
    success, encoded = cv2.imencode(fmt, img)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded.tobytes()


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    similarity = float(np.dot(vec_a, vec_b))
    
    return similarity


def get_embeddings(client: Any, image_a: bytes, image_b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    emb_a = run_inference(client, image_a)
    emb_b = run_inference(client, image_b)
    return emb_a.squeeze(0), emb_b.squeeze(0)


def get_faces(client: Any, image_a: bytes, image_b: bytes):
    faces_a = run_inference_retinaface(client, image_a)
    faces_b = run_inference_retinaface(client, image_b)

    return faces_a, faces_b


def get_align_face(client: Any, image: bytes):
    face = run_inference_retinaface(client, image)

    landmarks = face['landmarks']
    landmarks = np.asarray(landmarks).reshape(5, 2)
    aligned_image = FaceAlignment.crop_and_align(face, landmarks, face['image'])

    return face_to_png_response(aligned_image)

def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    face_a, face_b = get_faces(client, image_a, image_b)

    landmarks_a = face_a['landmarks']
    landmarks_a = np.asarray(landmarks_a).reshape(5, 2)
    aligned_image_a = FaceAlignment.crop_and_align(face_a, landmarks_a, face_a['image'])

    landmarks_b = face_b['landmarks']
    landmarks_b = np.asarray(landmarks_b).reshape(5, 2)
    aligned_image_b = FaceAlignment.crop_and_align(face_b, landmarks_b, face_b['image'])
    
    emb_a, emb_b = get_embeddings(client, to_bytes(aligned_image_a), to_bytes(aligned_image_b))
    return _cosine_similarity(emb_a, emb_b)
