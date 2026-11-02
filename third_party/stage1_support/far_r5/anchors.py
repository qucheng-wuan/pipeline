"""The verified metric DINO+DPT anchor from the original FAR project.

The frozen backbone is shared with the existing R5 RGB cache.  This module
reproduces the original reassembly, decoder and metric tail, including parameter
names, so no large backbone needs to be loaded to decode its cached features.
FAR's relation experiments did not supply this checkpoint: it is the stronger
RFTrans/ClearGrasp jointly supervised DINO+DPT baseline at update 2800.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_FAR_CHECKPOINT = Path(
    "/ssd/polyu/TransDepth/runs/far_minimal_20260920/"
    "baseline_joint_rftrans_cleargrasp_seed17_u3000_20260923/checkpoints/best.pt"
)
FAR_CHECKPOINT_SHA256 = "221daf822f081685277fc6b9b5c03db88b2e33e89a10b2e3cca72b53244e6d3f"
FAR_CONFIG_SHA256 = "ecb913e26d54b1a8ff6ecb7471e55a8722660abbfbc08be802c94dc5c272e13d"
DINO_CHECKPOINT_SHA256 = "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"
DINO_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
FEATURE_BLOCKS = (7, 15, 23, 31)
FEATURE_SHAPE = (4, 1280, 24, 32)
CANVAS_HW = (384, 512)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dino_cache_contract(contract: Mapping[str, Any]) -> None:
    """Reject caches with another backbone, normalization, tap order or canvas."""
    expected = {
        "canvas_hw": list(CANVAS_HW),
        "dinov3_checkpoint_sha256": DINO_CHECKPOINT_SHA256,
        "dinov3_repo_commit": DINO_COMMIT,
        "feature_blocks_0based": list(FEATURE_BLOCKS),
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"incompatible FAR/R5 DINO cache contract: {key}")
    preprocessing = contract.get("preprocessing", {})
    if (
        preprocessing.get("rgb")
        != "ImageNet-normalized aspect-preserving bilinear-antialiased letterbox"
        or preprocessing.get("dinov3_shape") != list(FEATURE_SHAPE)
    ):
        raise ValueError("incompatible FAR/R5 DINO preprocessing or feature shape")


def _conv_gn_act(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, channels),
        nn.GELU(approximate="none"),
    )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(_conv_gn_act(channels), _conv_gn_act(channels))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.body(inputs)


class _ReassemblePyramid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(1280, 128, 1) for _ in range(4)])
        self.scale1 = nn.Conv2d(128, 128, 3, padding=1)
        self.scale2 = nn.Conv2d(128, 128, 3, padding=1)
        self.scale4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)

    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, ...]:
        projected = [
            layer(item) for layer, item in zip(self.projections, features, strict=True)
        ]
        p1 = self.scale1(F.interpolate(
            projected[0], size=(96, 128), mode="bilinear", align_corners=False
        ))
        p2 = self.scale2(F.interpolate(
            projected[1], size=(48, 64), mode="bilinear", align_corners=False
        ))
        return p1, p2, projected[2], self.scale4(projected[3])


class _DPTFuse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coarse = _ResidualBlock()
        self.lateral = nn.ModuleList([_ResidualBlock() for _ in range(3)])
        self.fusion = nn.ModuleList([_ResidualBlock() for _ in range(3)])

    def forward(self, pyramid: Sequence[Tensor]) -> Tensor:
        output = self.coarse(pyramid[3])
        for index in (2, 1, 0):
            lateral = self.lateral[index](pyramid[index])
            coarse = F.interpolate(
                output, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            output = self.fusion[index](lateral + coarse)
        return output


class _DepthTail(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(128, 32, 3, padding=1)
        self.activation = nn.GELU(approximate="none")
        self.conv2 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        output = F.interpolate(
            inputs, size=(192, 256), mode="bilinear", align_corners=False
        )
        output = self.conv2(self.activation(self.conv1(output)))
        output = F.interpolate(
            output, size=CANVAS_HW, mode="bilinear", align_corners=False
        )
        return F.softplus(output.float()) + 1e-6


class FrozenFARAnchor(nn.Module):
    """Frozen metric optical-Z decoder, consuming RGB-only DINO feature maps."""

    def __init__(self) -> None:
        super().__init__()
        self.reassembly = _ReassemblePyramid()
        self.decoder = _DPTFuse()
        self.tail = _DepthTail()
        self.provenance: dict[str, Any] = {}
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FrozenFARAnchor:
        super().train(False)
        return self

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path = DEFAULT_FAR_CHECKPOINT,
        *,
        expected_sha256: str = FAR_CHECKPOINT_SHA256,
        device: str | torch.device = "cpu",
    ) -> FrozenFARAnchor:
        path = Path(path).expanduser().resolve(strict=True)
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError("FAR checkpoint SHA256 mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("schema_version") != "td_checkpoint_v1"
            or checkpoint.get("experiment_kind") != "t0"
            or checkpoint.get("backbone_sha256") != DINO_CHECKPOINT_SHA256
            or checkpoint.get("config_sha256") != FAR_CONFIG_SHA256
            or checkpoint.get("next_update") != 2800
            or checkpoint.get("model_spec") != {
                "selected_block": None, "selected_head": None,
                "beta": None, "rank": 4, "eta": 1.0,
            }
        ):
            raise ValueError("FAR checkpoint architecture or training identity mismatch")
        state = checkpoint.get("trainable_state")
        if not isinstance(state, dict):
            raise ValueError("FAR checkpoint lacks trainable_state")
        model = cls()
        # All 60 tensors must match; never silently drop a branch or missing layer.
        model.load_state_dict(state, strict=True)
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("FAR checkpoint contains non-finite weights")
        model.provenance = {
            "checkpoint": str(path),
            "checkpoint_sha256": actual_sha256,
            "backbone_sha256": DINO_CHECKPOINT_SHA256,
            "config_sha256": FAR_CONFIG_SHA256,
            "checkpoint_update": 2800,
            "architecture": "frozen_dinov3_h16plus_dpt_joint_metric_baseline",
            "historical_far_relation_enabled": False,
            "depth_coordinate": "optical_z",
            "depth_unit": "m",
            "feature_blocks_0based": list(FEATURE_BLOCKS),
            "cache_feature_dtype": "float16",
            "decoder_parameters": sum(item.numel() for item in model.parameters()),
            "cache_precision_note": (
                "R5 DINO features use FP16 batch-one extraction; historical FAR "
                "used BF16. Re-evaluate this decoder on the shared R5 cache."
            ),
        }
        return model.to(device).requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, features: Tensor | Sequence[Tensor]) -> Tensor:
        if isinstance(features, Tensor):
            if features.ndim != 5 or tuple(features.shape[1:]) != FEATURE_SHAPE:
                raise ValueError("FAR stacked features must be [B,4,1280,24,32]")
            maps = tuple(features[:, index] for index in range(4))
        else:
            maps = tuple(features)
        if len(maps) != 4 or any(
            not isinstance(item, Tensor)
            or item.ndim != 4
            or tuple(item.shape[1:]) != FEATURE_SHAPE[1:]
            for item in maps
        ):
            raise ValueError("FAR requires four [B,1280,24,32] feature maps")
        if any(item.shape[0] != maps[0].shape[0] for item in maps):
            raise ValueError("FAR feature maps have different batch sizes")
        weight = self.reassembly.projections[0].weight
        if any(item.device != weight.device for item in maps):
            raise ValueError("FAR decoder and feature maps must be on the same device")
        maps = tuple(item.to(dtype=weight.dtype) for item in maps)
        depth = self.tail(self.decoder(self.reassembly(maps)))
        if not bool(torch.isfinite(depth).all()) or not bool((depth > 0).all()):
            raise FloatingPointError("FAR anchor depth must be finite and positive")
        return depth


def load_far_anchor(
    checkpoint: str | Path = DEFAULT_FAR_CHECKPOINT,
    *,
    device: str | torch.device = "cpu",
    expected_sha256: str = FAR_CHECKPOINT_SHA256,
) -> FrozenFARAnchor:
    return FrozenFARAnchor.from_checkpoint(
        checkpoint, device=device, expected_sha256=expected_sha256
    )


def predict_far_from_features(model: FrozenFARAnchor, features: Tensor) -> Tensor:
    """Return [B,1,384,512] meters using only the frozen RGB feature tensor."""
    return model(features)
