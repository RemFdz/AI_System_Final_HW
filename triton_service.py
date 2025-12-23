import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np


TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002

#embeddings
MODEL_NAME = "fr_model"
MODEL_VERSION = "1"
MODEL_INPUT_NAME = "input"
MODEL_OUTPUT_NAME = "embedding"
MODEL_IMAGE_SIZE = (112, 112)

#detect_faces
RETINAFACE_MODEL_NAME = "face_detector"
RETINAFACE_INPUT_NAME = "input"
RETINAFACE_OUTPUT_BOXES = "bbox"
RETINAFACE_OUTPUT_LANDMARKS = "landmark"
RETINAFACE_IMAGE_SIZE = (640, 640)

def prepare_model_repository(model_repo: Path) -> None:
    """
    Populate the Triton model repository with the FR ONNX model and config.pbtxt.
    """
    model_dir = model_repo / MODEL_NAME / MODEL_VERSION
    model_path = model_dir / "model.onnx"
    config_path = model_dir.parent / "config.pbtxt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing ONNX model at {model_path}. "
            "Run convert_to_onnx.py first or place your exported model there."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    config_text = textwrap.dedent(
        f"""
        name: "{MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 8
        default_model_filename: "model.onnx"
        input [
          {{
            name: "{MODEL_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [3, {MODEL_IMAGE_SIZE[0]}, {MODEL_IMAGE_SIZE[1]}]
          }}
        ]
        output [
          {{
            name: "{MODEL_OUTPUT_NAME}"
            data_type: TYPE_FP32
            dims: [512]
          }}
        ]
        instance_group [
          {{ kind: KIND_CPU }}
        ]
        """
    ).strip() + "\n"

    config_path.write_text(config_text)
    print(f"[triton] Prepared model repository at {model_dir.parent}")


def start_triton_server(model_repo: Path) -> Any:
    """
    Launch Triton Inference Server (CPU) pointing at model_repo and return a handle/process.
    """
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin:
        raise RuntimeError("Could not find `tritonserver` binary in PATH. Is Triton installed?")

    cmd = [
        triton_bin,
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true",
        "--allow-grpc=true",
        "--allow-metrics=true",
        "--log-verbose=1",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"[triton] Starting Triton server with command: {' '.join(cmd)}")
    time.sleep(3)  # Give the server a moment to load the model
    return process


def stop_triton_server(server_handle: Any) -> None:
    """
    Cleanly stop the Triton server started in start_triton_server.
    """
    if server_handle is None:
        return

    server_handle.terminate()
    try:
        server_handle.wait(timeout=10)
        print("[triton] Triton server stopped.")
    except subprocess.TimeoutExpired:
        server_handle.kill()
        print("[triton] Triton server killed after timeout.")


def create_triton_client(url: str) -> Any:
    """
    Initialize a Triton HTTP client for the FR model endpoint.
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("tritonclient[http] is required; install from requirements.txt") from exc

    client = httpclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live.")
    return client


def run_inference(client: Any, image_bytes: bytes) -> Any:
    """
    Preprocess an input image, call Triton, and decode embeddings or scores.
    """
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Pillow, numpy, and tritonclient[http] are required to run inference.") from exc

    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB").resize(MODEL_IMAGE_SIZE)
        np_img = np.asarray(img, dtype=np.float32) / 255.0

    np_img = np.transpose(np_img, (2, 0, 1))  # HWC -> CHW
    batch = np.expand_dims(np_img, axis=0)

    infer_input = httpclient.InferInput(MODEL_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    infer_output = httpclient.InferRequestedOutput(MODEL_OUTPUT_NAME)
    response = client.infer(model_name=MODEL_NAME, inputs=[infer_input], outputs=[infer_output])
    return response.as_numpy(MODEL_OUTPUT_NAME)


def decode_boxes(boxes, priors):
    boxes_decoded = np.zeros_like(boxes)
    boxes_decoded[:,0] = priors[:,0] + boxes[:,0] * 0.1 * priors[:,2]
    boxes_decoded[:,1] = priors[:,1] + boxes[:,1] * 0.1 * priors[:,3]
    boxes_decoded[:,2] = priors[:,2] * np.exp(boxes[:,2] * 0.2)
    boxes_decoded[:,3] = priors[:,3] * np.exp(boxes[:,3] * 0.2)

    x1 = boxes_decoded[:,0] - boxes_decoded[:,2]/2
    y1 = boxes_decoded[:,1] - boxes_decoded[:,3]/2
    x2 = boxes_decoded[:,0] + boxes_decoded[:,2]/2
    y2 = boxes_decoded[:,1] + boxes_decoded[:,3]/2
    return np.stack([x1,y1,x2,y2], axis=-1)

def decode_landmarks(landmarks, priors):
    landmarks_decoded = np.zeros_like(landmarks)
    for i in range(5):
        landmarks_decoded[:,2*i]   = priors[:,0] + landmarks[:,2*i] * 0.1 * priors[:,2]
        landmarks_decoded[:,2*i+1] = priors[:,1] + landmarks[:,2*i+1] * 0.1 * priors[:,3]
    return landmarks_decoded

def generate_priors(image_size=(640, 640), feature_map_sizes=[80, 40, 20], min_sizes=[[16,32],[64,128],[256,512]], steps=[8,16,32]):
    priors = []

    for k, f in enumerate(feature_map_sizes):
        for i in range(f):
            for j in range(f):
                for min_size in min_sizes[k]:
                    cx = (j + 0.5) * steps[k] / image_size[1]
                    cy = (i + 0.5) * steps[k] / image_size[0]
                    s = min_size / image_size[0]
                    priors.append([cx, cy, s, s])
    return np.array(priors, dtype=np.float32)

def nms(boxes, scores, iou_threshold=0.4):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=int)
          
def run_inference_retinaface(client, image_bytes, score_threshold=0.5, iou_threshold=0.4):
    from io import BytesIO
    from PIL import Image
    import numpy as np
    from tritonclient import http as httpclient

    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img_resized = img.resize((640, 640))
        img_np_uint8 = np.asarray(img_resized, dtype=np.uint8)

    #normalize
    np_img = img_np_uint8.astype(np.float32)
    np_img = np_img[:, :, ::-1]
    np_img -= np.array([104.0, 117.0, 123.0])
    np_img = np.transpose(np_img, (2, 0, 1))
    batch = np.expand_dims(np_img, axis=0)

    infer_input = httpclient.InferInput(RETINAFACE_INPUT_NAME, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)
    outputs = [
        httpclient.InferRequestedOutput(RETINAFACE_OUTPUT_BOXES),
        httpclient.InferRequestedOutput(RETINAFACE_OUTPUT_LANDMARKS),
        httpclient.InferRequestedOutput("confidence"),
    ]
    response = client.infer(
        model_name=RETINAFACE_MODEL_NAME,
        inputs=[infer_input],
        outputs=outputs,
    )

    boxes_raw = response.as_numpy(RETINAFACE_OUTPUT_BOXES)
    landmarks_raw = response.as_numpy(RETINAFACE_OUTPUT_LANDMARKS)
    scores_raw = response.as_numpy("confidence")
    
    if boxes_raw.ndim == 3:
        boxes_raw = boxes_raw[0]
    if landmarks_raw.ndim == 3:
        landmarks_raw = landmarks_raw[0]
    if scores_raw.ndim == 3:
        scores_raw = scores_raw[0]
    
    if scores_raw.ndim == 2 and scores_raw.shape[1] == 2:
        scores = scores_raw[:, 1]
    else:
        scores = scores_raw.flatten()
    
    print(f"scores min: {scores.min():.4f}, max: {scores.max():.4f}, mean: {scores.mean():.4f}")
    print(f"Top 10 scores: {np.sort(scores)[-10:]}")

    #priors
    priors = generate_priors(image_size=RETINAFACE_IMAGE_SIZE)
    
    if priors.shape[0] > boxes_raw.shape[0]:
        priors = priors[:boxes_raw.shape[0]]
    elif priors.shape[0] < boxes_raw.shape[0]:
        raise ValueError(f"Number of priors ({priors.shape[0]}) is smaller than model outputs ({boxes_raw.shape[0]}).")

    #decode
    decoded_boxes = decode_boxes(boxes_raw, priors)
    decoded_landmarks = decode_landmarks(landmarks_raw, priors)

    #filter by score
    mask = scores > score_threshold
    print(f"mask sum: {mask.sum()} detections above threshold {score_threshold}")
    
    decoded_boxes = decoded_boxes[mask]
    decoded_landmarks = decoded_landmarks[mask]
    scores = scores[mask]

    if len(scores) == 0:
        print(f"No face detected above threshold {score_threshold}")
        return None

    decoded_boxes = np.atleast_2d(decoded_boxes)
    decoded_landmarks = np.atleast_2d(decoded_landmarks)

    #nms
    keep = nms(decoded_boxes, scores, iou_threshold=iou_threshold)
    decoded_boxes = decoded_boxes[keep]
    decoded_landmarks = decoded_landmarks[keep]
    scores = scores[keep]

    #largest face
    areas = (decoded_boxes[:,2] - decoded_boxes[:,0]) * (decoded_boxes[:,3] - decoded_boxes[:,1])
    idx = np.argmax(areas)

    bbox = decoded_boxes[idx] * 640
    landmarks = decoded_landmarks[idx] * 640

    return {
        "bbox": bbox,
        "landmarks": landmarks,
        "image": img_np_uint8,
        "score": scores[idx],
    }