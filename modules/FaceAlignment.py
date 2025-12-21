from typing import Any
import cv2
import numpy as np

class FaceAlignment:
    def __init__(self):
        pass

    #todo: refator to return cropx1 and cropy1 and use it in crop and align to avoid code duplication
    @staticmethod
    def crop(image, bbox, margin=0.3):
        x1, y1, x2, y2 = bbox.astype(int)
        box_w, box_h = x2 - x1, y2 - y1

        box_w = int(box_w * (1 + margin))
        box_h = int(box_h * (1 + margin))

        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2

        img_h, img_w, _ = image.shape

        new_w = min(img_w, box_w)
        new_h = min(img_h, box_h)

        crop_x1 = max(0, cx - new_w // 2)
        crop_y1 = max(0, cy - new_h // 2)
        crop_x2 = min(img_w, cx + new_w // 2)
        crop_y2 = min(img_h, cy + new_h // 2)

        cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]

        return cropped_img

    @staticmethod
    def crop_and_align(face: dict, kps, image: np.ndarray, margin=0.2) -> np.ndarray:
        x1, y1, x2, y2 = face["bbox"].astype(int)
        box_w, box_h = x2 - x1, y2 - y1

        box_w = int(box_w * (1 + margin))
        box_h = int(box_h * (1 + margin))

        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2

        img_h, img_w, _ = image.shape

        new_w = min(img_w, box_w)
        new_h = min(img_h, box_h)

        crop_x1 = max(0, cx - new_w // 2)
        crop_y1 = max(0, cy - new_h // 2)
        crop_x2 = min(img_w, cx + new_w // 2)
        crop_y2 = min(img_h, cy + new_h // 2)

        cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]

        orig_h, orig_w = cropped_img.shape[:2]

        zoomed_img = cv2.resize(cropped_img, (112, 112))

        kps = kps - np.array([crop_x1, crop_y1], dtype=np.float32)
        scale_x = 112 / orig_w
        scale_y = 112 / orig_h
        kps[:, 0] *= scale_x
        kps[:, 1] *= scale_y

        return FaceAlignment.align_face(zoomed_img, kps)

    @staticmethod
    def align_face(image: np.ndarray, kps: np.ndarray) -> Any:
        src = np.array([
            [38.2946, 51.6963],  # left_eye
            [73.5318, 51.5014],  # right_eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # mouth_left
            [70.7299, 92.2041],  # mouth_right
        ], dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(kps, src, method=cv2.LMEDS)

        aligned = cv2.warpAffine(
            image,
            M,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderValue=(128, 128, 128)
        )

        return aligned