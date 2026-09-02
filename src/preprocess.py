"""OpenCV preprocessing pipeline shared by training and inference.

Keeping one shared function guarantees the API preprocesses images
exactly the same way the model was trained (no train/serve skew).
"""
import cv2
import numpy as np
import torch

from .model import IMG_SIZE


def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 image -> normalized 1x1xHxW float tensor.

    Steps:
    1. Grayscale        - defects are texture/shape, color rarely matters
    2. Resize 128x128   - fixed model input, cheap on edge CPU
    3. Gaussian blur    - suppress sensor noise so it is not flagged as defect
    4. CLAHE            - equalize lighting variation across the line
    5. Scale to [0, 1]  - match Sigmoid output range of the autoencoder
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    x = gray.astype(np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # 1x1x128x128


def anomaly_heatmap(original: torch.Tensor, recon: torch.Tensor) -> np.ndarray:
    """Pixel-wise squared error map, scaled to 0-255 for visualization.
    Shows WHERE the defect is, not just that one exists."""
    err = ((recon - original) ** 2).squeeze().numpy()
    err = (err / (err.max() + 1e-8) * 255).astype(np.uint8)
    return cv2.applyColorMap(err, cv2.COLORMAP_JET)
