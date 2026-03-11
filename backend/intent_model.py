"""
Intent Model – EEGNet-inspired PyTorch architecture.

Pipeline:
  EEG (8, 256) → Temporal Conv → Depthwise Spatial Conv → BiLSTM → FC → embedding (512-d)

A separate linear head maps the embedding to class logits.
"""

import torch
import torch.nn as nn
from typing import Tuple

N_CHANNELS  = 8
N_SAMPLES   = 256
N_CLASSES   = 5
EMBED_DIM   = 512


class EEGNetEncoder(nn.Module):
    """
    Lightweight EEGNet-inspired temporal + spatial convolution encoder
    followed by a Bidirectional LSTM to capture sequential dynamics.
    """

    def __init__(self, n_channels: int = N_CHANNELS, n_samples: int = N_SAMPLES) -> None:
        super().__init__()

        # ── Block 1: Temporal Convolution ────────────────────────────────────
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # ── Block 2: Depthwise Spatial Convolution ────────────────────────────
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.25),
        )

        # ── Block 3: Separable Convolution ─────────────────────────────────
        self.sep_conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(8),
            nn.Dropout(0.25),
        )

        # After pooling: seq_len = n_samples // (4*8) = 8  (for 256 samples)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # BiLSTM output: 256*2 = 512
        self.fc_embed = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, EMBED_DIM),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, channels, samples)
        returns : (batch, EMBED_DIM)
        """
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.sep_conv(x)          # (batch, 64, seq)
        x = x.permute(0, 2, 1)        # (batch, seq, 64)
        out, _ = self.lstm(x)
        x = out[:, -1, :]             # last timestep
        return self.fc_embed(x)


class IntentClassifier(nn.Module):
    """Maps 512-d embeddings produced by EEGNetEncoder to class logits."""

    def __init__(self, n_classes: int = N_CLASSES, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)


class MindLinkModel(nn.Module):
    """Full end-to-end model: EEG → embedding → class logits."""

    def __init__(self, n_channels: int = N_CHANNELS, n_samples: int = N_SAMPLES, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.encoder = EEGNetEncoder(n_channels, n_samples)
        self.classifier = IntentClassifier(n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        logits    = self.classifier(embedding)
        return logits, embedding


# ─── Convenience helpers ──────────────────────────────────────────────────────

def build_model(device: str = "cpu") -> MindLinkModel:
    model = MindLinkModel()
    model.to(device)
    return model


def save_model(model: MindLinkModel, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str, device: str = "cpu") -> MindLinkModel:
    model = MindLinkModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model
