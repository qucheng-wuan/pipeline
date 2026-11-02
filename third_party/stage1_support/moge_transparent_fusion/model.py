"""Parallel O1/O2/O3/O4 transparent-region guidance adapter candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


VARIANTS = ("o1", "o2", "o3", "o4")


def _odd_kernel(radius: int) -> int:
    if radius < 0:
        raise ValueError("morphology radius must be non-negative")
    return radius * 2 + 1


def _erode(mask: Tensor, radius: int) -> Tensor:
    return 1.0 - F.max_pool2d(1.0 - mask, _odd_kernel(radius), stride=1, padding=radius)


def _dilate(mask: Tensor, radius: int) -> Tensor:
    return F.max_pool2d(mask, _odd_kernel(radius), stride=1, padding=radius)


def build_region_masks(
    mask: Tensor,
    content: Tensor,
    *,
    edge_radius: int,
    ring_inner_radius: int,
    ring_outer_radius: int,
    soft: bool = False,
) -> dict[str, Tensor]:
    """Build ROI/inside/edge/ring/global weights at image resolution.

    Morphology intentionally happens before patch pooling so thin objects do not
    disappear merely because the DINO grid is coarse.
    """

    if mask.ndim != 4 or mask.shape[1] != 1 or content.shape != mask.shape:
        raise ValueError("mask and content must both be [B,1,H,W]")
    region = mask.clamp(0.0, 1.0) * content if soft else (mask >= 0.5).to(mask.dtype) * content
    eroded = _erode(region, edge_radius) * region * content
    near = _dilate(region, edge_radius) * content
    edge = (near - eroded).clamp_(0.0, 1.0) * content
    inner = _dilate(region, ring_inner_radius)
    outer = _dilate(region, ring_outer_radius)
    ring = (outer - inner).clamp_(0.0, 1.0) * (1.0 - region) * content
    return {
        "roi": region,
        "inside": eroded,
        "edge": edge,
        "ring": ring,
        "global": content,
    }


def _pool_weights(weights: Tensor, size: tuple[int, int]) -> Tensor:
    return F.adaptive_avg_pool2d(weights, size)


def _masked_pool(features: Tensor, weights: Tensor, fallback: Tensor) -> Tensor:
    """Pool [B,L,C,H,W] with [B,R,H,W], returning [B,L,R,C]."""

    if features.ndim != 5 or weights.ndim != 4 or fallback.ndim != 4:
        raise ValueError("invalid feature or region-weight rank")
    denominator = weights.sum(dim=(-2, -1), keepdim=True)
    fallback_denominator = fallback.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    safe_weights = torch.where(
        denominator > 1e-6,
        weights,
        fallback.expand(-1, weights.shape[1], -1, -1),
    )
    safe_denominator = torch.where(
        denominator > 1e-6,
        denominator,
        fallback_denominator.expand(-1, weights.shape[1], -1, -1),
    )
    numerator = torch.einsum("blrhw,blchw->blrc", safe_weights[:, None], features)
    return numerator / safe_denominator[:, None, :, 0, 0, None]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.net(value))


@dataclass(frozen=True)
class AdapterSettings:
    hidden_channels: int = 128
    dense_channels_per_layer: int = 32
    token_channels_per_layer: int = 32
    max_log_delta: float = 0.35
    pixel_edge_radius: int = 5
    pixel_ring_inner_radius: int = 7
    pixel_ring_outer_radius: int = 23
    soft_region_masks: bool = False


class TransparentGuidanceAdapter(nn.Module):
    """A frozen-backbone residual head; DINO is guidance, never a depth head."""

    feature_layers = 4
    feature_width = 1280
    feature_hw = (24, 32)

    def __init__(self, variant: str, settings: AdapterSettings = AdapterSettings()) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.settings = settings
        hidden = settings.hidden_channels
        self.base_stem = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
        )

        if variant in {"o3", "o4"}:
            self.dense_projectors = nn.ModuleList(
                nn.Conv2d(self.feature_width, settings.dense_channels_per_layer, 1)
                for _ in range(self.feature_layers)
            )
            dense_width = self.feature_layers * settings.dense_channels_per_layer
            self.dense_fuse = nn.Sequential(
                nn.Conv2d(hidden + dense_width, hidden, 1),
                nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
                nn.GELU(),
            )

        if variant in {"o2", "o4"}:
            self.token_projectors = nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(self.feature_width),
                    nn.Linear(self.feature_width, settings.token_channels_per_layer),
                    nn.GELU(),
                )
                for _ in range(self.feature_layers)
            )
            region_count = 1 if variant == "o2" else 5
            token_width = self.feature_layers * region_count * settings.token_channels_per_layer
            self.film = nn.Sequential(
                nn.Linear(token_width, hidden * 2),
                nn.Tanh(),
            )
            nn.init.zeros_(self.film[-2].weight)
            nn.init.zeros_(self.film[-2].bias)

        self.body = nn.Sequential(*(ResidualBlock(hidden) for _ in range(4)))
        self.readout = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _validate_features(self, features: Tensor | None, batch: int) -> Tensor:
        if features is None:
            raise ValueError(f"{self.variant} requires DINOv3 features")
        expected = (batch, self.feature_layers, self.feature_width, *self.feature_hw)
        if tuple(features.shape) != expected:
            raise ValueError(f"DINOv3 feature shape {tuple(features.shape)} != {expected}")
        return features

    def _project_tokens(self, descriptors: Tensor) -> Tensor:
        # descriptors: [B,L,R,C]
        projected = [
            projector(descriptors[:, layer])
            for layer, projector in enumerate(self.token_projectors)
        ]
        return torch.cat([item.flatten(1) for item in projected], dim=1)

    def forward(
        self,
        base_depth: Tensor,
        transparent_mask: Tensor,
        known_mask: Tensor,
        features: Tensor | None = None,
        gate_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if base_depth.ndim != 4 or base_depth.shape[1] != 1:
            raise ValueError("base_depth must be [B,1,H,W]")
        if transparent_mask.shape != base_depth.shape or known_mask.shape != base_depth.shape:
            raise ValueError("depth and masks must share [B,1,H,W]")
        if gate_mask is not None and gate_mask.shape != base_depth.shape:
            raise ValueError("gate_mask and depth must share [B,1,H,W]")
        batch = base_depth.shape[0]
        mask = transparent_mask.float().clamp(0.0, 1.0)
        content = known_mask.float().clamp(0.0, 1.0)
        regions = build_region_masks(
            mask,
            content,
            edge_radius=self.settings.pixel_edge_radius,
            ring_inner_radius=self.settings.pixel_ring_inner_radius,
            ring_outer_radius=self.settings.pixel_ring_outer_radius,
            soft=self.settings.soft_region_masks,
        )
        patch_regions = {
            name: _pool_weights(value, self.feature_hw)
            for name, value in regions.items()
        }

        log_base = base_depth.float().clamp_min(1e-6).log()
        mean = (log_base * content).sum(dim=(-2, -1), keepdim=True) / content.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1.0)
        normalized_log_base = log_base - mean
        base_patch = F.interpolate(
            normalized_log_base, self.feature_hw, mode="bilinear", align_corners=False,
            antialias=True,
        )
        base_inputs = torch.cat(
            (
                base_patch,
                patch_regions["roi"],
                patch_regions["edge"],
                patch_regions["ring"],
            ),
            dim=1,
        )
        hidden = self.base_stem(base_inputs)

        dino: Tensor | None = None
        if self.variant != "o1":
            dino = self._validate_features(features, batch).float()

        if self.variant in {"o3", "o4"}:
            dense_roi = patch_regions["roi"][:, None]
            projected_dense = [
                projector(dino[:, layer] * dense_roi[:, 0])
                for layer, projector in enumerate(self.dense_projectors)
            ]
            hidden = self.dense_fuse(torch.cat((hidden, *projected_dense), dim=1))

        if self.variant == "o2":
            descriptors = _masked_pool(
                dino,
                patch_regions["roi"],
                patch_regions["global"],
            )
            gamma, beta = self.film(self._project_tokens(descriptors)).chunk(2, dim=1)
            hidden = hidden * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        elif self.variant == "o4":
            named = ("inside", "edge", "ring", "global")
            weights = torch.cat([patch_regions[name] for name in named], dim=1)
            descriptors = _masked_pool(dino, weights, patch_regions["global"])
            relation = descriptors[:, :, 0:1] - descriptors[:, :, 2:3]
            descriptors = torch.cat((descriptors, relation), dim=2)
            gamma, beta = self.film(self._project_tokens(descriptors)).chunk(2, dim=1)
            hidden = hidden * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        raw_patch = self.readout(self.body(hidden))
        raw = F.interpolate(
            raw_patch, base_depth.shape[-2:], mode="bilinear", align_corners=False,
        )
        bounded = self.settings.max_log_delta * torch.tanh(raw)
        gate = mask if gate_mask is None else gate_mask.float().clamp(0.0, 1.0)
        delta = bounded * gate * content
        prediction = base_depth.float() * torch.exp(delta)
        return {
            "depth": prediction,
            "delta_log": delta,
            "raw_delta": raw,
            "bounded_delta": bounded,
            "patch_roi": patch_regions["roi"],
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

