import torch
import torch.nn as nn

IMG_SIZE = 128  

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # Encoder: 1x128x128 -> latent vector
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 16x64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32x32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 64x8x8
            nn.ReLU(inplace=True),
        )
        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
        # Decoder: latent -> 1x128x128 reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 16x64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),   # 1x128x128
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.fc_enc(z.flatten(1))
        y = self.fc_dec(z).view(-1, 64, 8, 8)
        return self.decoder(y)


def anomaly_score(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Localized anomaly score.

    Global mean error dilutes a small scratch across 16K pixels, so we
    instead average the error inside 8x8 patches and take the WORST patch.
    A tiny scratch dominates one patch and is caught; global noise is not.
    """
    err = (recon - x) ** 2                       # Bx1xHxW pixel error
    patch = torch.nn.functional.avg_pool2d(err, kernel_size=8, stride=4)
    return patch.amax(dim=(1, 2, 3))             # worst patch per image


def reconstruction_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Per-image anomaly score for a batch (used to calibrate the threshold)."""
    with torch.no_grad():
        recon = model(x)
    return anomaly_score(x, recon)
