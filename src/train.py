"""Train the autoencoder on GOOD images only, tracked with MLflow.

Usage:
    python -m src.train --data data/raw/good --epochs 20

MLflow logs: hyperparameters, per-epoch loss, the computed anomaly
threshold, and the final model artifact. Every run is reproducible:
the DVC data version (git rev of the .dvc file) is logged as a tag.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import ConvAutoencoder, reconstruction_error
from .preprocess import preprocess_image


class GoodPartsDataset(Dataset):
    def __init__(self, root: str):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}. Run 'dvc pull' first.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]))
        return preprocess_image(img).squeeze(0)  # 1xHxW


def git_data_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/good")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/autoencoder.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = GoodPartsDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    mlflow.set_experiment("edge-anomaly-detection")
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.set_tag("data_version", git_data_version())
        mlflow.set_tag("n_train_images", len(ds))

        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for batch in dl:
                batch = batch.to(device)
                opt.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * batch.size(0)
            epoch_loss = total / len(ds)
            mlflow.log_metric("train_mse", epoch_loss, step=epoch)
            print(f"epoch {epoch + 1}/{args.epochs}  mse={epoch_loss:.6f}")

        # Threshold = mean + 3*std of reconstruction error on good images.
        # Anything above it at inference time is flagged as a defect.
        model.eval()
        errors = []
        for batch in DataLoader(ds, batch_size=args.batch_size):
            errors.append(reconstruction_error(model, batch.to(device)).cpu())
        errors = torch.cat(errors).numpy()
        threshold = float(errors.mean() + 3 * errors.std())
        mlflow.log_metric("anomaly_threshold", threshold)

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "threshold": threshold},
            args.out,
        )
        mlflow.log_artifact(args.out)
        print(f"saved {args.out}  threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
