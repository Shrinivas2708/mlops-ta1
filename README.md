# Edge Device Anomaly Detection (Visual Quality Control)

A lightweight Dockerized inspection container that version-controls image datasets with DVC (MinIO/S3 backend), tracks training with MLflow, and serves a CPU-optimized anomaly detection API with FastAPI.

## Problem
- A manufacturing line photographs parts continuously; image data grows daily
- Edge devices have small disks, so full datasets cannot live on them
- Git cannot version large image binaries; teams lose track of which data trained which model
- Inference must run locally on the edge box (low latency, works if network drops)

## Solution architecture

```
                     FACTORY SERVER / CLOUD
  ┌────────────────────────────────────────────────────┐
  │  MinIO (S3)          MLflow Server                 │
  │  bucket: dvcstore    experiments + model registry  │
  └───────▲──────────────────▲─────────────────────────┘
          │ dvc push/pull    │ mlflow.log_*
  ┌───────┴──────────────────┴─────────────────────────┐
  │  TRAINING MACHINE                                  │
  │  git repo (code + tiny .dvc pointer files)         │
  │  dvc.yaml pipeline -> src/train.py -> model.pt     │
  └───────┬────────────────────────────────────────────┘
          │ docker build (model baked in)
  ┌───────▼────────────────────────────────────────────┐
  │  EDGE DEVICE                                       │
  │  Docker container: FastAPI + PyTorch CPU + OpenCV  │
  │  POST /inspect -> PASS / DEFECT in ~3 ms           │
  └────────────────────────────────────────────────────┘
```

## Tech stack and role of each tool
- **DVC**: versions image binaries OUTSIDE git. Git stores only a tiny `.dvc` pointer file (an MD5 hash); the actual images live in MinIO. `git checkout` + `dvc pull` reproduces any historical dataset exactly
- **MinIO**: self-hosted S3-compatible object store, the DVC remote. Swappable for real AWS S3 with one config line
- **MLflow**: logs every training run (hyperparameters, loss curve, anomaly threshold, model file, and the git commit of the data version). Answers "which data and settings produced the deployed model?"
- **PyTorch**: convolutional autoencoder, ~695K parameters, CPU-only build
- **OpenCV**: identical preprocessing in training and serving (grayscale, resize 128x128, Gaussian denoise, CLAHE lighting equalization)
- **FastAPI**: async inference API with `/health`, `/info`, `/inspect`
- **Docker**: multi-stage build; runtime image contains only the serving deps and the trained model, runs as non-root

## How the ML works (unsupervised anomaly detection)
1. The autoencoder is trained ONLY on defect-free parts (no defect labels needed, which matches reality: defects are rare and unpredictable)
2. It learns to compress and reconstruct normal appearance
3. At inference, a defective part reconstructs poorly; the reconstruction error is the anomaly score
4. Scoring is localized: error is averaged over 8x8 patches and the worst patch is taken, so a small scratch cannot hide in the global average
5. Threshold is auto-calibrated at train time as mean + 3 sigma of scores on good images; scores above it return DEFECT
6. Optional heatmap output shows WHERE the defect is (pixel-wise error map)

## Measured results (synthetic demo dataset, CPU)
- Good parts passed: 50/50
- Defects caught: 49/50
- Latency: ~2.6 ms per image
- Model: 695K params, ~2.8 MB file

## Quickstart

```bash
# 1. infra: MinIO + MLflow + edge API
docker compose up -d --build

# 2. demo data (or drop real camera images in data/raw/)
pip install -r requirements.txt
python scripts/generate_sample_data.py --good 200 --defect 30

# 3. version the dataset (binaries -> MinIO, pointer -> git)
git init && git add . && git commit -m "code"
dvc add data/raw
git add data/raw.dvc .gitignore && git commit -m "dataset v1"
dvc push

# 4. train (tracked in MLflow at http://localhost:5000)
export MLFLOW_TRACKING_URI=http://localhost:5000
dvc repro          # runs the train stage from dvc.yaml

# 5. rebuild and test the edge container
docker compose up -d --build edge-api
curl -F "file=@data/raw/defect/defect_0001.png" "http://localhost:8000/inspect?heatmap=true"
```

## Daily workflow when new images arrive
```bash
dvc pull                 # sync current dataset
cp /camera/today/* data/raw/good/
dvc add data/raw         # new dataset version
git commit -am "dataset v2 (added 2026-09-02 batch)"
dvc push                 # binaries to MinIO, edge disk stays clean
dvc repro                # retrain only if data/code changed (DVC caches)
docker build -t edge-inspect:v2 .   # package new model
```

## API
- `GET /health` -> `{"status":"ok","model_loaded":true}`
- `GET /info` -> model size, threshold, input spec
- `POST /inspect` (multipart image) -> `{"verdict":"DEFECT","anomaly_score":0.0091,"threshold":0.0068,"latency_ms":2.6}`
- `POST /inspect?heatmap=true` -> adds base64 JPEG defect heatmap

## Tests
```bash
pytest tests/    # health check, good part passes, defect flagged
```

## Why this design fits edge constraints
- Edge box stores only the container (~300 MB) and the 2.8 MB model, never the dataset
- CPU-only torch build, 2 threads, no GPU or CUDA libraries
- Multi-stage Dockerfile discards build tooling from the runtime image
- Inference works fully offline; DVC/MLflow are only needed at training time
- Non-root container user and healthcheck for safe fleet deployment
