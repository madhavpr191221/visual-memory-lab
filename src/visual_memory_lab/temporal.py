"""Trainable temporal head for CLIP frame embeddings.

This module is intentionally independent of video decoding. A preprocessing
job can feed it a tensor of sampled frame embeddings, which keeps experiments
reproducible and lets us compare frozen CLIP with conventional fine-tuning.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalWindowEncoder(nn.Module):
    """Pool a sequence of normalized CLIP frame embeddings into one vector."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        output_dim: int | None = None,
        max_frames: int = 32,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or max_frames < 1:
            raise ValueError("dimensions and max_frames must be positive")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        output_dim = output_dim or input_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, frame_embeddings: Tensor, mask: Tensor | None = None) -> Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError("frame_embeddings must have shape [batch, frames, dimension]")
        batch, frames, _ = frame_embeddings.shape
        if frames > self.position.shape[1]:
            raise ValueError("frame count exceeds max_frames")
        hidden = self.input_projection(frame_embeddings) + self.position[:, :frames]
        padding_mask = None if mask is None else ~mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        output = self.norm(self.output_projection(pooled))
        return output / output.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def symmetric_contrastive_loss(
    video_embeddings: Tensor,
    text_embeddings: Tensor,
    *,
    temperature: float = 0.07,
) -> Tensor:
    """CLIP-style loss for aligned video-window and text embeddings."""
    if video_embeddings.shape != text_embeddings.shape:
        raise ValueError("video and text embeddings must have the same shape")
    if video_embeddings.ndim != 2 or video_embeddings.shape[0] < 2:
        raise ValueError("embeddings must have shape [batch, dimension] with batch >= 2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    video = video_embeddings / video_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    text = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    logits = video @ text.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (nn.functional.cross_entropy(logits, targets) + nn.functional.cross_entropy(logits.T, targets)) / 2
