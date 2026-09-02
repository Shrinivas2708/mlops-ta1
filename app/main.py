"""FastAPI inference engine served inside the edge container.

Endpoints:
    GET  /health   - liveness probe for orchestrators
    GET  /info     - model metadata (threshold, params, device)
    POST /inspect  - upload an image, get PASS/DEFECT verdict + score
    POST /inspect?heatmap=true - also returns a base64 JPEG heatmap

The model loads ONCE at startup (not per request) and runs in
torch.inference_mode on CPU threads capped for edge devices.
"""
import base64
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.model import ConvAutoencoder, anomaly_score
from src.preprocess import anomaly_heatmap, preprocess_image

MODEL_PATH = "models/autoencoder.pt"

torch.set_num_threads(2)  # be a good citizen on a shared edge CPU
app = FastAPI(title="Edge Visual Inspection API", version="1.0.0")

model: ConvAutoencoder | None = None
threshold: float = 0.0


@app.on_event("startup")
def load_model():
    global model, threshold
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = ConvAutoencoder()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = ckpt["threshold"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/info")
def info():
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": "ConvAutoencoder",
        "parameters": n_params,
        "anomaly_threshold": threshold,
        "input_size": "128x128 grayscale",
        "device": "cpu",
    }


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), heatmap: bool = False):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Uploaded file is not a decodable image")

    t0 = time.perf_counter()
    x = preprocess_image(img)
    with torch.inference_mode():
        recon = model(x)
    score = float(anomaly_score(x, recon))
    latency_ms = (time.perf_counter() - t0) * 1000

    result = {
        "verdict": "DEFECT" if score > threshold else "PASS",
        "anomaly_score": round(score, 6),
        "threshold": round(threshold, 6),
        "latency_ms": round(latency_ms, 1),
        "filename": file.filename,
    }
    if heatmap:
        hm = anomaly_heatmap(x, recon)
        ok, buf = cv2.imencode(".jpg", hm)
        result["heatmap_jpeg_b64"] = base64.b64encode(buf).decode()
    return result
