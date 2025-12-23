# AI SYSTEM DESIGN CAU FINAL HW

## Export models locally
```bash
# Export your arcface to ONNX
python convert_to_onnx.py \
  --weights-path weights/your_fr_model.pth \
  --onnx-path model_repository/fr_model/1/model.onnx
```

## Run with Docker + FastAPI (all-in-one container)
- Build image (uses `model_repository/*`):
  ```bash
  docker build -t fr-triton -f Docker/Dockerfile .
  ```
- Run container (Triton on 8000/8001/8002, FastAPI on 3000):
  ```bash
  docker run --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -p 3000:3000 \
    --name fr_triton \
    fr-triton
  ```
- Open Swagger UI at:
  ```
  http://0.0.0.0:3000
  ```
  (FastAPI inside the container proxies to Triton on localhost:8000.)
