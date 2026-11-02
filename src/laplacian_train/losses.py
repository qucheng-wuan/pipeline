"""Independent-GT normal pyramid supervision for trainable depth fusion.

This is an adaptation of the frequency-supervision idea, not a reproduction of
TransNormal. The input directions are normalized once. Gaussian lowpasses and
Laplacian residuals remain vectors: normalizing a lowpass would make the
decomposition nonlinear and discard its dispersion information.

All masks, edge weights and denominators depend only on target data. Every
5x5 reduction and bilinear expansion requires all of its GT contributors.
Image padding is not valid GT. Invalid predicted normals become zero and incur
error on the unchanged support. The caller must retain its angular objective.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate(prediction: Tensor, target: Tensor, support: Tensor,
              focus: Tensor | None) -> None:
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("prediction must be floating [B,3,H,W] normals")
    if target.shape != prediction.shape or min(prediction.shape) < 1:
        raise ValueError("target and prediction must have the same nonempty shape")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("normal tensors must be floating point")
    if support.shape != prediction[:, :1].shape:
        raise ValueError("valid_support must be [B,1,H,W]")
    for value in (target, support, focus):
        if value is not None and value.device != prediction.device:
            raise ValueError("all inputs must share a device")
    if focus is not None:
        if focus.shape != support.shape:
            raise ValueError("focus_mask must match valid_support")
        if not bool(torch.isfinite(focus).all()) or bool((focus < 0).any()):
            raise ValueError("focus_mask must be finite and nonnegative")


def _unit(value: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    finite = torch.isfinite(value).all(1, keepdim=True)
    safe = torch.where(finite, value, torch.zeros_like(value))
    length = torch.linalg.vector_norm(safe, dim=1, keepdim=True)
    usable = finite & torch.isfinite(length) & (length > eps)
    safe = torch.where(usable, safe, torch.zeros_like(safe))
    length = torch.where(usable, length, torch.ones_like(length))
    return safe / length.clamp_min(eps), usable


def _blur(value: Tensor) -> Tensor:
    channels = value.shape[1]
    kernel = value.new_tensor((1, 4, 6, 4, 1)) / 16
    horizontal = kernel.view(1, 1, 1, 5).expand(channels, 1, 1, 5)
    vertical = kernel.view(1, 1, 5, 1).expand(channels, 1, 5, 1)
    value = F.conv2d(F.pad(value, (2, 2, 0, 0), mode="replicate"),
                     horizontal, groups=channels)
    return F.conv2d(F.pad(value, (0, 0, 2, 2), mode="replicate"),
                    vertical, groups=channels)


def _reduce_support(valid: Tensor) -> Tensor:
    # Explicit invalid padding avoids treating synthetic border samples as GT.
    invalid = F.pad((~valid).float(), (2, 2, 2, 2), value=1)
    return (F.max_pool2d(invalid, 5, stride=1) == 0)[..., ::2, ::2]


def _expand(value: Tensor, shape: Sequence[int]) -> Tensor:
    return F.interpolate(value, size=tuple(shape), mode="bilinear", align_corners=False)


def _expand_support(valid: Tensor, shape: Sequence[int]) -> Tensor:
    # Bilinear weights are nonnegative: interpolated invalid mass is zero iff
    # every source with nonzero interpolation weight is valid (also odd sizes).
    return _expand((~valid).float(), shape) == 0


def _image_mean(value: Tensor, weights: Tensor) -> Tensor:
    denom = weights.flatten(1).sum(1)
    numer = (value * weights).flatten(1).sum(1)
    present = (denom > 0).to(value.dtype)
    return ((numer / denom.clamp_min(1e-12)) * present).sum() / present.sum().clamp_min(1)


def _robust_l1(error: Tensor, epsilon: float) -> Tensor:
    # Subtract epsilon so identical fields have exactly zero loss.
    return (torch.sqrt(error.square() + epsilon * epsilon) - epsilon).mean(1, keepdim=True)


def normal_edge_weights(target: Tensor, valid_support: Tensor, *,
                        floor: float = 1.0, gain: float = 1.0,
                        eps: float = 1e-8) -> Tensor:
    """Detached GT edges with a positive floor; invalid boundaries add no edge.

    Both endpoints receive each valid pair's direction change. A flat field is
    exactly ``floor``. Per-image maximum normalization bounds weights by
    ``floor + gain`` without allowing texture or predicted errors to set them.
    """
    if not math.isfinite(floor) or floor <= 0 or not math.isfinite(gain) or gain < 0:
        raise ValueError("edge floor must be positive and gain nonnegative")
    _validate(target, target, valid_support, None)
    truth, usable = _unit(target.detach().float(), eps)
    valid = valid_support.detach().bool() & usable
    dx = torch.linalg.vector_norm(truth[..., 1:] - truth[..., :-1], dim=1, keepdim=True)
    dy = torch.linalg.vector_norm(truth[..., 1:, :] - truth[..., :-1, :], dim=1, keepdim=True)
    dx = torch.where(valid[..., 1:] & valid[..., :-1], dx, torch.zeros_like(dx))
    dy = torch.where(valid[..., 1:, :] & valid[..., :-1, :], dy, torch.zeros_like(dy))
    edge = (F.pad(dx, (0, 1, 0, 0)) + F.pad(dx, (1, 0, 0, 0))
            + F.pad(dy, (0, 0, 0, 1)) + F.pad(dy, (0, 0, 1, 0))) / 4
    scale = edge.flatten(1).amax(1).view(-1, 1, 1, 1)
    edge = torch.where(scale > eps, edge / scale.clamp_min(eps), torch.zeros_like(edge))
    return (floor + gain * edge.clamp(0, 1)).detach()


def normal_pyramid_loss(
    prediction_normal: Tensor,
    target_normal: Tensor,
    valid_support: Tensor,
    focus_mask: Tensor | None = None,
    *,
    core_mask: Tensor | None = None,
    levels: int = 3,
    level_weights: Sequence[float] | None = None,
    coarse_weight: float = 0.25,
    edge_floor: float = 1.0,
    edge_gain: float = 1.0,
    charbonnier_eps: float = 1e-3,
    normal_eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Return differentiable loss plus diagnostics on fixed GT-only support.

    Detail is the level-weighted mean of image-macro vector Charbonnier losses;
    weights default to 2**(-level). The edge weight multiplies the numerator,
    while the denominator is the GT support/focus mass, so it really increases
    emphasis near edges. Coarse is a vector loss without edge weighting.
    Empty support contributes connected zero, never NaN. ``focus_mask`` is a
    nonnegative weighting map, not a filter-validity mask. ``core_mask`` selects
    loss centers while retaining surrounding halo GT in filter support. Core
    centers follow the same ``::2`` lattice at every scale.
    """
    _validate(prediction_normal, target_normal, valid_support, focus_mask)
    if core_mask is not None and (core_mask.shape != valid_support.shape
                                  or core_mask.device != prediction_normal.device):
        raise ValueError("core_mask must share valid_support shape and device")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")
    weights = tuple(2.0 ** -i for i in range(levels)) if level_weights is None else tuple(level_weights)
    if len(weights) != levels or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("level_weights must match levels and have positive finite total")
    for name, value in (("charbonnier_eps", charbonnier_eps), ("normal_eps", normal_eps)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(coarse_weight) or coarse_weight < 0:
        raise ValueError("coarse_weight must be finite and nonnegative")
    with torch.autocast(device_type=prediction_normal.device.type, enabled=False):
        pred, _ = _unit(prediction_normal.float(), normal_eps)
        truth, target_usable = _unit(target_normal.detach().float(), normal_eps)
        valid = valid_support.detach().bool() & target_usable
        pred = torch.where(valid, pred, torch.zeros_like(pred))
        truth = torch.where(valid, truth, torch.zeros_like(truth))
        focus = (torch.ones_like(valid, dtype=pred.dtype) if focus_mask is None
                 else focus_mask.detach().float())
        core = torch.ones_like(valid) if core_mask is None else core_mask.detach().bool()
        edge = normal_edge_weights(truth, valid, floor=edge_floor, gain=edge_gain, eps=normal_eps)
        result: dict[str, Tensor] = {"normal_pixels": valid.sum(),
                                    "edge_weight_mean": _image_mean(edge, valid.float())}
        detail = pred.sum() * 0
        for index, weight in enumerate(weights):
            low_pred, low_truth = _blur(pred)[..., ::2, ::2], _blur(truth)[..., ::2, ::2]
            low_valid = _reduce_support(valid)
            band_valid = valid & _expand_support(low_valid, pred.shape[-2:])
            difference = ((pred - _expand(low_pred, pred.shape[-2:]))
                          - (truth - _expand(low_truth, truth.shape[-2:])))
            selected = band_valid & core
            roi = selected.float() * focus
            term = _image_mean(_robust_l1(difference, charbonnier_eps) * edge, roi)
            detail = detail + float(weight) * term
            result[f"detail_l{index}"] = term
            result[f"pixels_l{index}"] = selected.sum()
            result[f"weight_mass_l{index}"] = roi.sum()
            pred, truth, valid = low_pred, low_truth, low_valid
            focus, edge = _blur(focus)[..., ::2, ::2], _blur(edge)[..., ::2, ::2]
            core = core[..., ::2, ::2]
        detail = detail / sum(weights)
        coarse_roi = (valid & core).float() * focus
        coarse = _image_mean(_robust_l1(pred - truth, charbonnier_eps), coarse_roi)
        result.update(loss=detail + coarse_weight * coarse, detail=detail, coarse=coarse,
                      coarse_pixels=(valid & core).sum(), coarse_weight_mass=coarse_roi.sum())
        return result


def independent_normal_geometry(
    depth: Tensor,
    batch: Mapping[str, Tensor],
    *,
    normal_log_jump: float = 0.05,
    normal_reference_neighbor_angle_deg: float = 30.0,
) -> dict[str, Tensor]:
    """Prepare differentiable prediction normals and independent-only GT support.

    Uses V2's camera/stencil convention. Images without independent assets are
    excluded, rather than silently substituted with depth-derived labels. The
    native V2 package must be importable (same requirement as the train model).
    """
    from native_fusion.geometry import depth_to_normals, surface_normal_reference

    if depth.ndim != 4 or depth.shape[1] != 1 or depth.shape != batch["target"].shape:
        raise ValueError("depth and target must share [B,1,H,W]")
    target = batch["target"].detach().float()
    valid = (batch["valid"].detach().bool() & batch.get("content", torch.ones_like(target)).bool()
             & torch.isfinite(target) & (target > 0))
    def flag(name: str) -> Tensor:
        value = torch.as_tensor(batch.get(name, False), device=depth.device, dtype=torch.bool).flatten()
        if value.numel() not in (1, depth.shape[0]):
            raise ValueError(f"{name} must be scalar or one flag per image")
        return value.expand(depth.shape[0]).view(-1, 1, 1, 1)
    available, camera = flag("normal_reference_available"), flag("normal_camera_valid")
    geometry_valid = valid & available & camera
    truth = torch.where(valid, target, torch.ones_like(target))
    labels = batch.get("instance_id", batch["transparent"].long())
    reference = surface_normal_reference(
        truth, batch["intrinsics"], geometry_valid,
        reference_normal=batch.get("target_normal"), reference_normal_valid=batch.get("normal_valid"),
        reference_available=available.flatten(), max_log_jump=normal_log_jump,
        max_normal_neighbor_angle_deg=normal_reference_neighbor_angle_deg,
        region_labels=labels, boundary_mask=batch.get("boundary_mask"))
    predicted, _ = depth_to_normals(
        depth.float(), batch["intrinsics"], geometry_valid, reference_depth=truth,
        max_log_jump=normal_log_jump, region_labels=labels, boundary_mask=batch.get("boundary_mask"))
    return {"predicted_normal": predicted, "target_normal": reference.normal.detach(),
            "valid_support": reference.valid & available,
            "independent_images": (available & camera).sum()}


def depth_normal_pyramid_loss(
    depth: Tensor,
    batch: Mapping[str, Tensor],
    *,
    core_mask: Tensor | None = None,
    foreground_balance: float = 0.5,
    normal_log_jump: float = 0.05,
    normal_reference_neighbor_angle_deg: float = 30.0,
    **pyramid_kwargs,
) -> dict[str, Tensor]:
    """Convenience depth-to-normal pyramid wrapper; returns scalar diagnostics.

    For reuse with angular supervision, call ``independent_normal_geometry``
    and ``normal_pyramid_loss`` directly instead of recomputing the geometry.
    Core is loss-only: GT and geometry in the halo remain available.
    """
    if not math.isfinite(foreground_balance) or foreground_balance < 0:
        raise ValueError("foreground_balance must be finite and nonnegative")
    geometry = independent_normal_geometry(
        depth, batch, normal_log_jump=normal_log_jump,
        normal_reference_neighbor_angle_deg=normal_reference_neighbor_angle_deg)
    focus = 1.0 + foreground_balance * batch["transparent"].detach().float()
    result = normal_pyramid_loss(
        geometry["predicted_normal"], geometry["target_normal"], geometry["valid_support"],
        focus_mask=focus, core_mask=core_mask, **pyramid_kwargs)
    result["independent_images"] = geometry["independent_images"]
    return result


__all__ = ["normal_edge_weights", "normal_pyramid_loss", "depth_normal_pyramid_loss",
           "independent_normal_geometry"]
