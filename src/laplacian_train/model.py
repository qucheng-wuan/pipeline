"""Trainable full-image Laplacian fusion with exact global crop centering.

The frozen baseline is the previously selected six-level, strength-0.5
adapter. Each Laplacian band is first lifted through the original sequence of
bilinear expansions to native image coordinates. A learned low-resolution
spatial gate modulates each lifted band. Thus this is spatial modulation of
redundant Laplacian bands, not an orthogonal wavelet or spectral partition.

Training crops sample the SAME full-image gate field used at inference. The
global mean correction is a linear functional of the gate grid, so its exact
coefficients can be cached from full-image bands. Crops never subtract their
own mean and never construct a new crop-local pyramid.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from laplacian_fusion.pyramid import fuse_depth, laplacian_pyramid


def _float(value: Tensor) -> Tensor:
    return value if value.dtype == torch.float64 else value.float()


def _map(value: Tensor, name: str, channels: int | None = None) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4 or min(value.shape) < 1:
        raise ValueError(f"{name} must be a nonempty [B,C,H,W] tensor")
    if channels is not None and value.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def lift_bands(log_difference: Tensor, levels: int = 6) -> Tensor:
    """Return [B,levels,H,W] bands lifted with sequential dyadic expansion.

    A direct resize from a coarse band to native resolution is a different
    operator. The sequential expansions here use exactly the old pyramid's
    shapes and align_corners=False convention, including odd dimensions.
    """
    _map(log_difference, "log_difference", 1)
    with torch.autocast(device_type=log_difference.device.type, enabled=False):
        bands, _ = laplacian_pyramid(_float(log_difference), levels)
        lifted = []
        for index, band in enumerate(bands):
            value = band
            for parent in range(index - 1, -1, -1):
                value = F.interpolate(value, bands[parent].shape[-2:],
                                      mode="bilinear", align_corners=False)
            lifted.append(value)
        return torch.cat(lifted, dim=1)


def fixed_depth_and_bands(v2_depth: Tensor, moge_depth: Tensor,
                          levels: int = 6) -> tuple[Tensor, Tensor]:
    """Build the immutable old baseline and full native modulation bands.

    The baseline is evaluated by the original function, rather than summing
    individually lifted bands in a different floating-point addition order.
    Consequently a zero learned gate delta preserves its bits exactly.
    """
    fixed, _ = fuse_depth(v2_depth, moge_depth, levels=levels, strength=.5)
    with torch.autocast(device_type=v2_depth.device.type, enabled=False):
        dtype = torch.float64 if torch.float64 in (v2_depth.dtype, moge_depth.dtype) else torch.float32
        difference = moge_depth.to(dtype).log() - v2_depth.to(dtype).log()
        return fixed, lift_bands(difference, levels)


def mean_coefficients(bands: Tensor,
                      gate_hw: tuple[int, int] = (24, 32)) -> Tensor:
    """Return I* (bands / HW), the adjoint of full-image gate interpolation.

    For every gate grid g, (coeff*g).sum() equals mean(bands*I(g)).
    This helper intentionally computes the actual interpolation adjoint,
    rather than approximating it with area downsampling. It works inside
    no_grad/inference_mode preprocessing and returns a detached tensor.
    """
    _map(bands, "bands")
    if len(gate_hw) != 2 or any(isinstance(x, bool) or int(x) != x or x < 1 for x in gate_hw):
        raise ValueError("gate_hw must contain two positive integers")
    # enable_grad alone does not override an enclosing inference_mode scope.
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type=bands.device.type, enabled=False):
        source = _float(bands.detach())
        if torch.is_inference(source):
            source = source.clone()
        grid = source.new_zeros((*source.shape[:2], *gate_hw), requires_grad=True)
        full = F.interpolate(grid, source.shape[-2:], mode="bilinear", align_corners=False)
        functional = (full * source).mean(dim=(-2, -1)).sum()
        return torch.autograd.grad(functional, grid, create_graph=False)[0].detach()


def build_gate_context(rgb: Tensor, probability: Tensor, teacher_normal: Tensor,
                       teacher_confidence: Tensor, v2_depth: Tensor,
                       moge_depth: Tensor, bands: Tensor,
                       gate_hw: tuple[int, int] = (24, 32)) -> Tensor:
    """Build RGB-only context: 3+1+3+1+1+1+levels channels.

    Log-depth channels are centered over the complete image. All channels
    are area pooled onto the gate grid; normal vectors are normalized before
    pooling. Their pooled magnitude records local orientation agreement.
    No target, segmentation label, camera matrix, or fitted GT scale is used.
    """
    _map(v2_depth, "v2_depth", 1)
    expected = (v2_depth.shape[0], *v2_depth.shape[-2:])
    rgb = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()
    for name, value, channels in (("rgb", rgb, 3), ("probability", probability, 1),
            ("teacher_normal", teacher_normal, 3), ("teacher_confidence", teacher_confidence, 1),
            ("moge_depth", moge_depth, 1), ("bands", bands, None)):
        _map(value, name, channels)
        if (value.shape[0], *value.shape[-2:]) != expected or value.device != v2_depth.device:
            raise ValueError(f"{name} does not share full-image geometry and device")
    with torch.autocast(device_type=v2_depth.device.type, enabled=False):
        v2, moge = v2_depth.float(), moge_depth.float()
        if not bool(((v2 > 0) & torch.isfinite(v2) & (moge > 0) & torch.isfinite(moge)).all()):
            raise ValueError("depth inputs must be finite and positive")
        v2_log, moge_log = v2.log(), moge.log()
        channels = (rgb, probability.float().clamp(0, 1),
                    F.normalize(teacher_normal.float(), dim=1, eps=1e-6),
                    teacher_confidence.float().clamp(0, 1),
                    v2_log - v2_log.mean((-2, -1), keepdim=True),
                    moge_log - moge_log.mean((-2, -1), keepdim=True),
                    bands.float().abs())
        # Pool separately to avoid allocating a 16-channel native tensor.
        return torch.cat([F.adaptive_avg_pool2d(value, gate_hw) for value in channels], dim=1)


@dataclass(frozen=True)
class ModelConfig:
    levels: int = 6
    hidden_channels: int = 32
    gate_height: int = 24
    gate_width: int = 32

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class LearnableLaplacianFusion(nn.Module):
    """Small trainable V2 fusion layer, initialized to the fixed 0.5 adapter."""

    def __init__(self, config: ModelConfig | Mapping | None = None):
        super().__init__()
        self.settings = (ModelConfig() if config is None else
                         ModelConfig(**dict(config)) if isinstance(config, Mapping) else config)
        if not isinstance(self.settings, ModelConfig):
            raise TypeError("config must be ModelConfig or a mapping")
        cfg = self.settings
        self.spatial = nn.Sequential(nn.Conv2d(10 + cfg.levels, cfg.hidden_channels, 3, padding=1),
                                     nn.SiLU(),
                                     nn.Conv2d(cfg.hidden_channels, cfg.hidden_channels, 3, padding=1),
                                     nn.SiLU(), nn.Conv2d(cfg.hidden_channels, cfg.levels, 1))
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)
        self.global_logits = nn.Parameter(torch.zeros(cfg.levels))

    def config(self) -> dict:
        return asdict(self.settings)

    def forward(self, context: Tensor) -> dict[str, Tensor]:
        _map(context, "context", 10 + self.settings.levels)
        if context.shape[-2:] != (self.settings.gate_height, self.settings.gate_width):
            raise ValueError("context does not match the configured gate grid")
        raw = self.spatial(context).float() + self.global_logits.float()[None, :, None, None]
        delta = .5 * raw.tanh()
        return {"gate_delta": delta, "gate_alpha": .5 + delta,
                "global_alpha": .5 + .5 * self.global_logits.float().tanh(),
                "gate_regularization": delta.square().mean()}


def sample_gate_patch(gate_delta: Tensor, native_hw: Tensor, crop_top_left: Tensor,
                      output_hw: tuple[int, int]) -> Tensor:
    """Sample the native-resolution interpolation at patch pixel centers."""
    _map(gate_delta, "gate_delta")
    b = gate_delta.shape[0]
    native_hw = torch.as_tensor(native_hw, device=gate_delta.device).reshape(b, 2)
    crop_top_left = torch.as_tensor(crop_top_left, device=gate_delta.device).reshape(b, 2)
    if bool((native_hw < 1).any()):
        raise ValueError("native_hw must be positive")
    h, w = output_hw
    if h < 1 or w < 1:
        raise ValueError("output dimensions must be positive")
    dtype = _float(gate_delta).dtype
    nh = native_hw[:, 0].to(dtype).view(b, 1, 1)
    nw = native_hw[:, 1].to(dtype).view(b, 1, 1)
    top = crop_top_left[:, 0].to(dtype).view(b, 1, 1)
    left = crop_top_left[:, 1].to(dtype).view(b, 1, 1)
    yy = torch.arange(h, device=gate_delta.device, dtype=dtype).view(1, h, 1)
    xx = torch.arange(w, device=gate_delta.device, dtype=dtype).view(1, 1, w)
    gy = ((yy + top + .5) / nh * 2 - 1).expand(b, h, w)
    gx = ((xx + left + .5) / nw * 2 - 1).expand(b, h, w)
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(_float(gate_delta), grid, mode="bilinear",
                         padding_mode="border", align_corners=False)


def _check_fusion_inputs(inputs: Mapping[str, Tensor], gate_delta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    fixed, bands, coeff = inputs["fixed_depth"], inputs["bands"], inputs["mean_coeff"]
    _map(fixed, "fixed_depth", 1)
    _map(bands, "bands", gate_delta.shape[1])
    _map(coeff, "mean_coeff", gate_delta.shape[1])
    if (bands.shape[0] != fixed.shape[0] or bands.shape[-2:] != fixed.shape[-2:]
            or coeff.shape != gate_delta.shape or fixed.shape[0] != gate_delta.shape[0]):
        raise ValueError("fusion inputs have incompatible batch, band, image, or grid shapes")
    if any(value.device != gate_delta.device for value in (fixed, bands, coeff)):
        raise ValueError("fusion inputs and gates must share a device")
    return _float(fixed), _float(bands), _float(coeff)


def _apply(fixed: Tensor, bands: Tensor, coeff: Tensor, gate_delta: Tensor,
           gate_native: Tensor) -> dict[str, Tensor]:
    global_mean = (coeff * gate_delta).sum(dim=(1, 2, 3))[:, None, None, None]
    correction = (bands * gate_native).sum(dim=1, keepdim=True) - global_mean
    return {"depth": fixed * correction.exp(), "log_correction": correction,
            "global_log_mean": global_mean, "gate_delta": gate_delta}


def apply_gates_patch(inputs: Mapping[str, Tensor], gate_delta: Tensor) -> dict[str, Tensor]:
    """Apply a crop with full-image bands, grid coordinates, and mean coefficients."""
    with torch.autocast(device_type=gate_delta.device.type, enabled=False):
        delta = _float(gate_delta)
        fixed, bands, coeff = _check_fusion_inputs(inputs, delta)
        native = sample_gate_patch(delta, inputs["native_hw"], inputs["crop_top_left"],
                                   tuple(fixed.shape[-2:]))
        return _apply(fixed, bands, coeff, delta, native)


def apply_gates_full(inputs: Mapping[str, Tensor], gate_delta: Tensor) -> dict[str, Tensor]:
    """Full-image inference with the same gate interpolation and global mean."""
    with torch.autocast(device_type=gate_delta.device.type, enabled=False):
        delta = _float(gate_delta)
        if "mean_coeff" not in inputs:
            inputs = {**inputs, "mean_coeff": mean_coefficients(inputs["bands"], tuple(delta.shape[-2:]))}
        fixed, bands, coeff = _check_fusion_inputs(inputs, delta)
        native = F.interpolate(delta, fixed.shape[-2:], mode="bilinear", align_corners=False)
        return _apply(fixed, bands, coeff, delta, native)


__all__ = ["ModelConfig", "LearnableLaplacianFusion", "lift_bands",
           "fixed_depth_and_bands", "mean_coefficients", "build_gate_context",
           "sample_gate_patch", "apply_gates_patch", "apply_gates_full"]
