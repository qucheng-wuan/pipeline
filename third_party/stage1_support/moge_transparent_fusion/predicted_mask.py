"""Calibration and gating primitives for deployable TOM-Net masks."""

from __future__ import annotations

import torch
from torch import Tensor


def foreground_logit(mask_logits: Tensor) -> Tensor:
    """Return the binary foreground log-odds from TOM-Net's two logits."""

    if mask_logits.ndim != 4 or mask_logits.shape[1] != 2:
        raise ValueError("TOM-Net mask logits must be [B,2,H,W]")
    return mask_logits[:, 1:2].float() - mask_logits[:, 0:1].float()


def calibrated_probability(mask_logits: Tensor, temperature: float) -> Tensor:
    if not 0.05 <= float(temperature) <= 20.0:
        raise ValueError("temperature must lie in [0.05,20]")
    return torch.sigmoid(foreground_logit(mask_logits) / float(temperature))


def probability_to_gate(probability: Tensor, threshold: float) -> Tensor:
    """Continuous gate with an exact background dead-zone below ``threshold``."""

    if not 0.0 <= float(threshold) < 1.0:
        raise ValueError("threshold must lie in [0,1)")
    probability = probability.float().clamp(0.0, 1.0)
    return ((probability - float(threshold)) / (1.0 - float(threshold))).clamp(0.0, 1.0)
