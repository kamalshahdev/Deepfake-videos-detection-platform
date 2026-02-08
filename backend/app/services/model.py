"""Multimodal neural model for deepfake classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        meta_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.video_encoder = ModalityEncoder(video_dim, hidden_dim, dropout)
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dim, dropout)
        self.meta_encoder = ModalityEncoder(meta_dim, hidden_dim, dropout)

        self.gate = nn.Linear(hidden_dim * 3, 3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        video_x: torch.Tensor,
        audio_x: torch.Tensor,
        meta_x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.video_encoder(video_x)
        a = self.audio_encoder(audio_x)
        m = self.meta_encoder(meta_x)

        concat = torch.cat([v, a, m], dim=1)
        gate_logits = self.gate(concat)
        gate_weights = torch.softmax(gate_logits, dim=1)

        masked = gate_weights * mask
        norm = masked.sum(dim=1, keepdim=True) + 1e-6
        masked = masked / norm

        stacked = torch.stack([v, a, m], dim=1)
        fused = (stacked * masked.unsqueeze(-1)).sum(dim=1)

        logits = self.classifier(fused).squeeze(1)
        return logits, masked


@dataclass
class NormalizationStats:
    video_mean: torch.Tensor
    video_std: torch.Tensor
    audio_mean: torch.Tensor
    audio_std: torch.Tensor
    meta_mean: torch.Tensor
    meta_std: torch.Tensor


@dataclass
class LoadedModel:
    model: MultimodalClassifier
    norm: Optional[NormalizationStats]
    source: str


def _to_tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Optional[LoadedModel]:
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    video_dim = int(checkpoint.get("video_dim", 10))
    audio_dim = int(checkpoint.get("audio_dim", 8))
    meta_dim = int(checkpoint.get("meta_dim", 6))
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    dropout = float(checkpoint.get("dropout", 0.2))

    model = MultimodalClassifier(
        video_dim=video_dim,
        audio_dim=audio_dim,
        meta_dim=meta_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    norm_dict = checkpoint.get("normalization")
    norm = None
    if norm_dict:
        norm = NormalizationStats(
            video_mean=_to_tensor(norm_dict["video_mean"]),
            video_std=_to_tensor(norm_dict["video_std"]),
            audio_mean=_to_tensor(norm_dict["audio_mean"]),
            audio_std=_to_tensor(norm_dict["audio_std"]),
            meta_mean=_to_tensor(norm_dict["meta_mean"]),
            meta_std=_to_tensor(norm_dict["meta_std"]),
        )

    return LoadedModel(model=model, norm=norm, source=str(checkpoint_path))
