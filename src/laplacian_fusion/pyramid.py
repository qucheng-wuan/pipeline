"""Inject RGB-derived MoGe shape bands into a V2 metric-depth prediction.

For ``x = log(moge) - log(v2)``, each pyramid band is
``L_k = x_k - up(down(x_k))``. Downsampling uses the separable binomial
``[1, 4, 6, 4, 1] / 16`` filter with replicate boundaries and samples ``::2``;
upsampling uses bilinear interpolation to the exact preceding shape. Keeping
the shapes makes telescoping reconstruction valid for odd image dimensions.

This is a redundant, nonorthogonal pyramid, not a strict spectral partition.
An optional smoothed probability gate also mixes spatial frequencies. Removing
the final correction's per-image mean preserves the global geometric mean of
depth (up to floating-point roundoff); it does not guarantee a lower MAE,
unchanged background, or preservation of every low-frequency component.
No labels, camera calibration, or GT scale fitting enter this module.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _check_map(value: Tensor, name: str, *, depth: bool = False) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be a [B,C,H,W] tensor")
    if any(size < 1 for size in value.shape):
        raise ValueError(f"{name} must have nonempty dimensions")
    if depth and value.shape[1] != 1:
        raise ValueError(f"{name} must have one depth channel")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    if depth and not bool((value > 0).all()):
        raise ValueError(f"{name} must be strictly positive")


def _working_float(value: Tensor) -> Tensor:
    return value if value.dtype == torch.float64 else value.float()


def _gaussian5(value: Tensor) -> Tensor:
    channels = value.shape[1]
    kernel = value.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0)) / 16.0
    horizontal = kernel.view(1, 1, 1, 5).expand(channels, 1, 1, 5)
    vertical = kernel.view(1, 1, 5, 1).expand(channels, 1, 5, 1)
    value = F.conv2d(F.pad(value, (2, 2, 0, 0), mode="replicate"),
                     horizontal, groups=channels)
    return F.conv2d(F.pad(value, (0, 0, 2, 2), mode="replicate"),
                    vertical, groups=channels)


def gaussian5(value: Tensor) -> Tensor:
    """Separable Gaussian approximation, preserving image size and constants."""
    _check_map(value, "value")
    with torch.autocast(device_type=value.device.type, enabled=False):
        return _gaussian5(_working_float(value))


def _expand(value: Tensor, shape: Sequence[int]) -> Tensor:
    return F.interpolate(value, size=tuple(shape), mode="bilinear", align_corners=False)


def laplacian_pyramid(value: Tensor, levels: int = 3) -> tuple[list[Tensor], Tensor]:
    """Return fine-to-coarse residual bands and the final Gaussian lowpass.

    Each residual keeps its original spatial dimensions. Repeated levels on a
    1x1 lowpass are permitted and contribute exactly zero additional detail.
    """
    _check_map(value, "value")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")
    with torch.autocast(device_type=value.device.type, enabled=False):
        current = _working_float(value)
        bands: list[Tensor] = []
        for _ in range(levels):
            low = _gaussian5(current)[..., ::2, ::2]
            bands.append(current - _expand(low, current.shape[-2:]))
            current = low
        return bands, current


def reconstruct_pyramid(bands: Sequence[Tensor], coarse: Tensor) -> Tensor:
    """Reconstruct in reverse order, using each band's exact spatial shape."""
    _check_map(coarse, "coarse")
    with torch.autocast(device_type=coarse.device.type, enabled=False):
        current = _working_float(coarse)
        for band in reversed(bands):
            _check_map(band, "band")
            if band.shape[:2] != current.shape[:2] or band.device != current.device:
                raise ValueError("bands and coarse must share batch, channels, and device")
            expected = ((band.shape[-2] + 1) // 2, (band.shape[-1] + 1) // 2)
            if current.shape[-2:] != expected:
                raise ValueError("band sizes do not form a dyadic pyramid")
            current = _expand(current, band.shape[-2:]) + _working_float(band)
        return current


def fuse_depth(
    v2_depth: Tensor,
    moge_depth: Tensor,
    levels: int = 3,
    strength: float = 1.0,
    probability: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(fused_depth, zero_mean_log_correction)`` for ``[B,1,H,W]`` maps.

    The cumulative high-frequency correction comes from reconstructing the
    log-depth-difference pyramid with its coarsest lowpass set to zero. A finite
    scalar ``strength`` controls interpolation or extrapolation; it is not
    clipped. A probability map, when supplied, is clamped to [0,1] and smoothed
    before gating. The correction is then centered over each complete image.

    ``strength == 0`` returns the original V2 tensor itself, with an all-zero
    correction. Other calls compute in at least FP32 (FP64 inputs retain FP64).
    MoGe's constant depth scale cancels; scaling V2 scales the fused result by
    the same factor, up to numerical roundoff. Inputs must cover full images if
    the stated global mean preservation is desired; independent tiles would
    instead preserve their own means and can introduce seams.
    """
    _check_map(v2_depth, "v2_depth", depth=True)
    _check_map(moge_depth, "moge_depth", depth=True)
    if moge_depth.shape != v2_depth.shape or moge_depth.device != v2_depth.device:
        raise ValueError("depth inputs must have the same shape and device")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")
    if isinstance(strength, bool) or not math.isfinite(float(strength)):
        raise ValueError("strength must be a finite scalar")
    if probability is not None:
        _check_map(probability, "probability")
        if probability.shape != v2_depth.shape or probability.device != v2_depth.device:
            raise ValueError("probability must share depth shape and device")
    if float(strength) == 0.0:
        return v2_depth, torch.zeros_like(v2_depth)

    with torch.autocast(device_type=v2_depth.device.type, enabled=False):
        dtype = torch.float64 if torch.float64 in (v2_depth.dtype, moge_depth.dtype) else torch.float32
        v2, moge = v2_depth.to(dtype), moge_depth.to(dtype)
        difference = moge.log() - v2.log()
        bands, coarse = laplacian_pyramid(difference, levels)
        correction = reconstruct_pyramid(bands, torch.zeros_like(coarse))
        if probability is not None:
            correction = correction * _gaussian5(probability.to(dtype).clamp(0.0, 1.0))
        correction = correction * float(strength)
        correction = correction - correction.mean(dim=(-2, -1), keepdim=True)
        return v2 * correction.exp(), correction


__all__ = ["gaussian5", "laplacian_pyramid", "reconstruct_pyramid", "fuse_depth"]
