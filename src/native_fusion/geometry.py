"""Perspective geometry and shared, label-defined depth/surface evaluation.

Contract: optical-Z metres, OpenCV camera (+X right,+Y down,+Z forward),
boundary-coordinate pixel K, sample centres (u+.5,v+.5). Visible normals face
camera: dy cross dx gives (0,0,-1) on a fronto-parallel plane. Authenticated
ClearGrasp cameraNormals are independent renderer-surface references after
the documented [R,-G,-B] conversion. Where unavailable, released rectified Z
and authenticated pinhole K provide explicitly labelled induced normals.
The published ClearGrasp rectification is retained unchanged. Masks/occlusion support come from GT,
never from prediction-dependent jumps (which would permit metric gaming).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor

COORDINATE_CONTRACT = "opencv_xyz_optical_z_boundary_pixel_K_centers_plus_half_front_facing_normals"


def _depth(value: Tensor, name: str = "depth") -> None:
    if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[1] != 1 or not value.is_floating_point():
        raise ValueError(f"{name} must be floating [B,1,H,W]")


def _mask(value: Tensor, reference: Tensor, name: str) -> Tensor:
    if value.shape != reference.shape:
        raise ValueError(f"{name} must match [B,1,H,W]")
    return value.bool()


def canonical_intrinsics(intrinsics: Tensor, *, integer_pixel_centers: bool = False) -> Tensor:
    """Validate K; explicitly convert integer-centre calibrated cx/cy by +.5."""
    if intrinsics.ndim not in (2, 3) or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must be [3,3] or [B,3,3]")
    dtype = torch.float64 if intrinsics.dtype == torch.float64 else torch.float32
    value = intrinsics.to(dtype=dtype).clone()
    if not bool(torch.isfinite(value).all()) or bool((value[..., (0, 1), (0, 1)] <= 0).any()):
        raise ValueError("intrinsics require finite positive focal lengths")
    expected = value.new_tensor((0, 0, 1)).expand_as(value[..., 2, :])
    if not torch.allclose(value[..., 2, :], expected, atol=1e-7, rtol=0):
        raise ValueError("intrinsics bottom row must be [0,0,1]")
    if bool((torch.linalg.det(value).abs() < 1e-12).any()):
        raise ValueError("intrinsics must be invertible")
    if integer_pixel_centers:
        value[..., 0, 2] += .5
        value[..., 1, 2] += .5
    return value


def crop_resize_intrinsics(
    intrinsics: Tensor, crop_top_left: tuple[float, float] | Tensor,
    crop_hw: tuple[int, int], output_hw: tuple[int, int] | None = None,
) -> Tensor:
    """K' = resize @ crop @ K; crop_top_left is (top,left) in source pixels.

    Boundary coordinates make align_corners=False resize exactly diag(sx,sy,1),
    with no extra half-pixel shift. Padding before the crop can be represented
    by negative crop origins. Padding on the right/bottom leaves K unchanged.
    """
    value = canonical_intrinsics(intrinsics)
    output_hw = crop_hw if output_hw is None else output_hw
    if min(*crop_hw, *output_hw) <= 0:
        raise ValueError("crop and output sizes must be positive")
    origin = torch.as_tensor(crop_top_left, dtype=value.dtype, device=value.device)
    if origin.shape not in ((2,), (*value.shape[:-2], 2)) or not bool(torch.isfinite(origin).all()):
        raise ValueError("crop origin must be (top,left) or one origin per K")
    sy, sx = output_hw[0] / crop_hw[0], output_hw[1] / crop_hw[1]
    transform = torch.eye(3, dtype=value.dtype, device=value.device).expand_as(value).clone()
    transform[..., 0, 0], transform[..., 1, 1] = sx, sy
    transform[..., 0, 2], transform[..., 1, 2] = -origin[..., 1] * sx, -origin[..., 0] * sy
    return transform @ value


def camera_rays(intrinsics: Tensor, hw: tuple[int, int], *, batch_size: int | None = None) -> Tensor:
    value = canonical_intrinsics(intrinsics)
    if value.ndim == 2:
        value = value[None]
    if batch_size is not None:
        if value.shape[0] not in (1, batch_size):
            raise ValueError("intrinsics batch size differs from depth")
        value = value.expand(batch_size, -1, -1)
    h, w = hw
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=value.dtype, device=value.device) + .5,
        torch.arange(w, dtype=value.dtype, device=value.device) + .5, indexing="ij",
    )
    pixel = torch.stack((xx, yy, torch.ones_like(xx)))[None]
    rays = torch.linalg.inv(value) @ pixel.flatten(2)
    rays = rays.reshape(value.shape[0], 3, h, w)
    return rays / rays[:, 2:3]


def _five(value: Tensor) -> tuple[Tensor, ...]:
    return (value[..., 1:-1, 1:-1], value[..., 1:-1, :-2], value[..., 1:-1, 2:],
            value[..., :-2, 1:-1], value[..., 2:, 1:-1])


def surface_support(
    reference_depth: Tensor, valid: Tensor | None = None, *, max_log_jump: float | None = .05,
    region_labels: Tensor | None = None, boundary_mask: Tensor | None = None,
) -> Tensor:
    """Five-point interior stencil fixed by label/reference geometry."""
    _depth(reference_depth, "reference_depth")
    if min(reference_depth.shape[-2:]) < 3:
        raise ValueError("surface reconstruction needs at least 3x3 pixels")
    support = torch.isfinite(reference_depth) & (reference_depth > 0)
    if valid is not None:
        support = support & _mask(valid, reference_depth, "valid")
    pieces = _five(support)
    inside = pieces[0].clone()
    for other in pieces[1:]:
        inside = inside & other
    if max_log_jump is not None:
        if not math.isfinite(max_log_jump) or max_log_jump < 0:
            raise ValueError("max_log_jump must be finite nonnegative or None")
        safe = torch.where(support, reference_depth, torch.ones_like(reference_depth)).log()
        center, *neighbors = _five(safe)
        for neighbor in neighbors:
            inside = inside & ((center - neighbor).abs() <= max_log_jump)
    if region_labels is not None:
        if region_labels.shape != reference_depth.shape:
            raise ValueError("region_labels must match depth")
        center, *neighbors = _five(region_labels)
        for neighbor in neighbors:
            inside = inside & (center == neighbor)
    if boundary_mask is not None:
        boundary = _mask(boundary_mask, reference_depth, "boundary_mask")
        for piece in _five(boundary):
            inside = inside & ~piece
    return F.pad(inside, (1, 1, 1, 1), value=False)


def depth_to_normals(
    depth: Tensor, intrinsics: Tensor, valid: Tensor | None = None, *,
    reference_depth: Tensor | None = None, max_log_jump: float | None = .05,
    region_labels: Tensor | None = None, boundary_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Differentiable perspective normals and label-defined stencil validity.

    Pass GT as reference_depth for both predicted and GT normals in losses and
    metrics. Invalid/degenerate predicted normals become zero vectors, but do
    NOT remove a reference-valid pixel from support. Evaluators count these as
    180 degrees; training applies a direction penalty on the same fixed mask.
    Normalization of both tangents before cross product avoids a depth-scale
    dependent epsilon on a squared-metre cross product.
    """
    _depth(depth)
    reference = depth.detach() if reference_depth is None else reference_depth.detach()
    if reference.shape != depth.shape:
        raise ValueError("reference depth must match prediction")
    support = surface_support(reference, valid, max_log_jump=max_log_jump,
                              region_labels=region_labels, boundary_mask=boundary_mask)
    dtype = torch.float64 if depth.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=depth.device.type, enabled=False):
        work = depth.to(dtype=dtype)
        rays = camera_rays(intrinsics.to(device=depth.device, dtype=dtype), depth.shape[-2:], batch_size=depth.shape[0])
        finite = torch.isfinite(work) & (work > 0)
        safe = torch.where(finite, work, torch.ones_like(work))
        points = rays * safe
        dx = points[..., 1:-1, 2:] - points[..., 1:-1, :-2]
        dy = points[..., 2:, 1:-1] - points[..., :-2, 1:-1]
        dx = F.normalize(dx, dim=1, eps=1e-12)
        dy = F.normalize(dy, dim=1, eps=1e-12)
        cross = torch.linalg.cross(dy, dx, dim=1)
        norm = cross.norm(dim=1, keepdim=True)
        normal = cross / norm.clamp_min(1e-12)
        numerical = torch.isfinite(normal).all(1, keepdim=True) & (norm > 1e-10)
        for sample in _five(finite):
            numerical = numerical & sample
        normal = torch.where(numerical, normal, torch.zeros_like(normal))
        normal = F.pad(normal, (1, 1, 1, 1))
        normal = torch.where(support, normal, torch.zeros_like(normal))
    return normal, support


def normal_angles(prediction: Tensor, target: Tensor) -> Tensor:
    """Directed degrees; invalid prediction is 180, never removed from support."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("normal arrays must share [B,3,H,W]")
    finite = torch.isfinite(prediction).all(1, keepdim=True)
    pred = torch.where(finite, prediction, torch.zeros_like(prediction))
    good = finite & (pred.norm(dim=1, keepdim=True) > 1e-8)
    pred = F.normalize(pred, dim=1, eps=1e-12)
    truth = F.normalize(target, dim=1, eps=1e-12)
    sine = torch.linalg.cross(pred, truth, dim=1).norm(dim=1, keepdim=True)
    cosine = (pred * truth).sum(1, keepdim=True).clamp(-1, 1)
    degrees = torch.atan2(sine, cosine) * (180 / math.pi)
    return torch.where(good, degrees, torch.full_like(degrees, 180))


def cleargrasp_normals_to_opencv(raw: Tensor) -> tuple[Tensor, Tensor]:
    """Official cameraNormals RGB XYZ (+right,+up,+viewer) -> OpenCV.

    This conversion applies only to authenticated ClearGrasp cameraNormals
    assets, not arbitrary renderer/world normals or TransNormal output.
    The published [-1,-1,-1] invalid sentinel is rejected explicitly.
    """
    if raw.ndim not in (3, 4) or raw.shape[-3] != 3 or not raw.is_floating_point():
        raise ValueError('cameraNormals must be floating [3,H,W] or [B,3,H,W]')
    dimension = raw.ndim - 3
    finite = torch.isfinite(raw).all(dimension, keepdim=True)
    safe = torch.where(finite, raw, torch.zeros_like(raw))
    length = safe.norm(dim=dimension, keepdim=True)
    valid = finite & (length >= .5) & (length <= 1.5) & ~(safe == -1).all(dimension, keepdim=True)
    signs = raw.new_tensor([1, -1, -1]).reshape(*([1] * dimension), 3, 1, 1)
    normal = safe * signs / length.clamp_min(1e-8)
    return torch.where(valid, normal, torch.zeros_like(normal)), valid


def normal_stencil_support(normal: Tensor, valid: Tensor, max_degrees: float = 30) -> Tensor:
    """Require five valid directions on the same smooth reference surface."""
    if normal.ndim != 4 or normal.shape[1] != 3 or valid.shape != normal[:, :1].shape:
        raise ValueError('normal/valid must share [B,3|1,H,W]')
    if not 0 < max_degrees < 180:
        raise ValueError('normal stencil angle must lie in (0,180)')
    center, *neighbors = _five(normal)
    support, *neighbor_valid = _five(valid.bool())
    support = support.clone()
    cosine = math.cos(math.radians(max_degrees))
    for other, usable in zip(neighbors, neighbor_valid):
        support = support & usable & ((center * other).sum(1, keepdim=True) >= cosine)
    return F.pad(support, (1, 1, 1, 1), value=False)


@dataclass
class NormalReference:
    normal: Tensor
    valid: Tensor
    independent: Tensor
    induced_normal: Tensor
    induced_valid: Tensor


def surface_normal_reference(
    target: Tensor, intrinsics: Tensor, valid: Tensor, *,
    reference_normal: Tensor | None = None, reference_normal_valid: Tensor | None = None,
    reference_available: bool | Tensor | None = None, max_log_jump: float = .05,
    max_normal_neighbor_angle_deg: float = 30, region_labels: Tensor | None = None,
    boundary_mask: Tensor | None = None,
) -> NormalReference:
    """Independent normal GT where available; induced normals elsewhere.

    Selection is per IMAGE. Invalid pixels of an available independent asset
    stay invalid, never silently switch to depth-derived supervision. All
    support uses target geometry and labels only. Pass valid AND camera-valid
    here, since both reference kinds supervise a depth-to-normal operation.
    Independent directions are already in OpenCV coordinates; do not flip.
    """
    induced, induced_valid = depth_to_normals(target, intrinsics, valid,
        reference_depth=target, max_log_jump=max_log_jump, region_labels=region_labels,
        boundary_mask=boundary_mask)
    induced_valid = induced_valid & (induced.norm(dim=1, keepdim=True) > 1e-8)
    available = torch.as_tensor(reference_normal is not None if reference_available is None else reference_available,
                                device=target.device, dtype=torch.bool).flatten()
    if available.numel() not in (1, target.shape[0]):
        raise ValueError('reference_available must be scalar or one flag per image')
    available = available.expand(target.shape[0])[:, None, None, None]
    if reference_normal is None:
        if bool(available.any()) or reference_normal_valid is not None:
            raise ValueError('independent reference availability/validity requires normal tensors')
        return NormalReference(induced, induced_valid, available, induced, induced_valid)
    if reference_normal.shape != induced.shape or reference_normal_valid is None:
        raise ValueError('independent reference needs [B,3,H,W] normals and [B,1,H,W] validity')
    raw = reference_normal.detach().to(device=target.device, dtype=induced.dtype)
    finite = torch.isfinite(raw).all(1, keepdim=True)
    safe = torch.where(finite, raw, torch.zeros_like(raw))
    length = safe.norm(dim=1, keepdim=True)
    usable = _mask(reference_normal_valid, target, 'reference_normal_valid') & finite & (length >= .5) & (length <= 1.5)
    independent = safe / length.clamp_min(1e-8)
    # Do not require induced-normal orientation agreement: its quantization and
    # rectification errors are precisely why independent surface GT is used.
    support = surface_support(target, valid, max_log_jump=max_log_jump,
                              region_labels=region_labels, boundary_mask=boundary_mask)
    support = support & normal_stencil_support(independent, usable, max_normal_neighbor_angle_deg)
    selected = torch.where(available, independent, induced)
    selected_valid = torch.where(available, support, induced_valid)
    return NormalReference(selected, selected_valid, available, induced, induced_valid)


def depth_edges(depth: Tensor, valid: Tensor, threshold: float = .05) -> Tensor:
    """Mark BOTH endpoints of a 4-connected log-Z jump on GT-valid support."""
    _depth(depth)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("edge threshold must be finite positive log-depth jump")
    valid = _mask(valid, depth, "valid")
    safe = torch.where(torch.isfinite(depth) & (depth > 0), depth, torch.ones_like(depth)).log()
    edge = torch.zeros_like(valid)
    dx = (safe[..., 1:] - safe[..., :-1]).abs() > threshold
    dx = dx & valid[..., 1:] & valid[..., :-1]
    dy = (safe[..., 1:, :] - safe[..., :-1, :]).abs() > threshold
    dy = dy & valid[..., 1:, :] & valid[..., :-1, :]
    edge[..., 1:] |= dx
    edge[..., :-1] |= dx
    edge[..., 1:, :] |= dy
    edge[..., :-1, :] |= dy
    return edge


@torch.no_grad()
def evaluate_depth(
    prediction: Tensor, target: Tensor, valid: Tensor, transparent: Tensor,
    intrinsics: Tensor | None = None, *, max_log_jump: float = .05,
    edge_threshold: float = .05, edge_tolerance_px: int = 2,
    normal_camera_valid: bool | Tensor = True,
    reference_normal: Tensor | None = None, reference_normal_valid: Tensor | None = None,
    reference_available: bool | Tensor | None = None,
    max_normal_neighbor_angle_deg: float = 30,
) -> dict[str, dict[str, Any]]:
    """Shared per-image metrics for baselines and final models.

    Call with B=1, then macro-average metrics for image-macro reporting. Each
    median is the exact median of that image's supported normal angles; a mean
    of these medians is NOT a pooled-pixel median. _sums supports pooled means,
    RMS, threshold accuracy and edge counts. K=None omits unavailable normals.
    normal_* uses independent camera normals where explicitly available and
    otherwise released-Z induced normals, with source counts/kind included.
    depth_induced_normal_* always reports the latter as a separate diagnostic;
    it must not be called independent surface ground truth.
    Edge tolerance uses a square Chebyshev neighbourhood in native pixels;
    missing predictions with GT edges get F1=0, empty GT+prediction gives None.
    """
    _depth(prediction, "prediction")
    _depth(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")
    valid = _mask(valid, target, "valid") & torch.isfinite(target) & (target > 0)
    transparent = _mask(transparent, target, "transparent")
    if bool((valid & (~torch.isfinite(prediction) | (prediction <= 0))).any()):
        raise ValueError("invalid predictions cannot shrink GT evaluation support")
    if not isinstance(edge_tolerance_px, int) or edge_tolerance_px < 0:
        raise ValueError("edge tolerance must be a nonnegative integer")
    pred = torch.where(valid, prediction, torch.ones_like(prediction)).double()
    truth = torch.where(valid, target, torch.ones_like(target)).double()
    error = (pred - truth).abs()
    ratio = torch.maximum(pred / truth, truth / pred)
    gt_edge = depth_edges(truth, valid, edge_threshold)
    pred_edge = depth_edges(pred, valid, edge_threshold)
    kernel = edge_tolerance_px * 2 + 1
    gt_dilated = F.max_pool2d(gt_edge.float(), kernel, 1, edge_tolerance_px).bool()
    pred_dilated = F.max_pool2d(pred_edge.float(), kernel, 1, edge_tolerance_px).bool()
    angles = normal_valid = induced_angles = induced_valid = None
    independent = torch.zeros(target.shape[0], 1, 1, 1, device=target.device, dtype=torch.bool)
    camera_valid = torch.as_tensor(normal_camera_valid, device=target.device, dtype=torch.bool).flatten()
    if camera_valid.numel() not in (1, target.shape[0]):
        raise ValueError("normal_camera_valid must be scalar or one flag per image")
    camera_valid = camera_valid.expand(target.shape[0])[:, None, None, None]
    geometry_valid = valid & camera_valid
    if intrinsics is not None and bool(camera_valid.any()):
        reference = surface_normal_reference(truth, intrinsics, geometry_valid,
            reference_normal=reference_normal, reference_normal_valid=reference_normal_valid,
            reference_available=reference_available, max_log_jump=max_log_jump,
            max_normal_neighbor_angle_deg=max_normal_neighbor_angle_deg, region_labels=transparent)
        normal_valid, independent = reference.valid, reference.independent
        pred_normal, _ = depth_to_normals(pred, intrinsics, geometry_valid,
            reference_depth=truth, max_log_jump=max_log_jump, region_labels=transparent)
        angles = normal_angles(pred_normal, reference.normal)
        induced_valid = reference.induced_valid
        induced_angles = normal_angles(pred_normal, reference.induced_normal)
    result = {}
    for region, selection in (("all", valid), ("background", valid & ~transparent), ("transparent", valid & transparent)):
        count = int(selection.sum())
        sums = {"pixels": count, "absolute_m": float(error[selection].sum()),
                "squared_m": float(error[selection].square().sum()),
                "relative": float((error / truth)[selection].sum())}
        metrics = {"pixels": count, "mae_mm": sums["absolute_m"] / count * 1000 if count else None,
                   "rmse_mm": math.sqrt(sums["squared_m"] / count) * 1000 if count else None,
                   "abs_rel": sums["relative"] / count if count else None}
        for threshold in (1.05, 1.10, 1.25):
            key = f"delta_{threshold:.2f}"
            sums[key] = int(((ratio < threshold) & selection).sum())
            metrics[key] = sums[key] / count if count else None
        support = selection & normal_valid if normal_valid is not None else torch.zeros_like(selection)
        for prefix, values, usable in (
            ('normal', angles, support),
            ('depth_induced_normal', induced_angles, selection & induced_valid if induced_valid is not None else torch.zeros_like(selection)),
        ):
            n = int(usable.sum())
            sums[f'{prefix}_pixels'] = metrics[f'{prefix}_pixels'] = n
            sums[f'{prefix}_valid_images'] = metrics[f'{prefix}_valid_images'] = int(usable.flatten(1).any(1).sum())
            sums[f'{prefix}_angle_sum_deg'] = float(values[usable].sum()) if n else 0.0
            metrics[f'{prefix}_mean_deg'] = sums[f'{prefix}_angle_sum_deg'] / n if n else None
            metrics[f'{prefix}_median_deg'] = float(torch.quantile(values[usable], .5)) if n else None
            for threshold in (11.25, 22.5, 30.0):
                key = f'{prefix}_{threshold:g}'
                sums[key] = int((values[usable] < threshold).sum()) if n else 0
                metrics[key] = sums[key] / n if n else None
        for source, flag in (('independent', independent), ('induced', ~independent)):
            source_support = support & flag
            sums[f'normal_{source}_pixels'] = metrics[f'normal_{source}_pixels'] = int(source_support.sum())
            sums[f'normal_{source}_images'] = metrics[f'normal_{source}_images'] = int(source_support.flatten(1).any(1).sum())
        sources = [source for source in ('independent', 'induced') if sums[f'normal_{source}_pixels']]
        metrics['normal_reference_kind'] = ('mixed' if len(sources) > 1 else
            ('independent_camera_normal' if sources == ['independent'] else
             'released_z_induced' if sources else 'none'))
        pe, ge = pred_edge & selection, gt_edge & selection
        npred, ngt = int(pe.sum()), int(ge.sum())
        mp, mg = int((pe & gt_dilated).sum()), int((ge & pred_dilated).sum())
        precision = mp / npred if npred else (0.0 if ngt else None)
        recall = mg / ngt if ngt else (0.0 if npred else None)
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and precision + recall > 0 else (0.0 if npred or ngt else None)
        metrics.update(edge_precision=precision, edge_recall=recall, edge_f1=f1)
        sums.update(pred_edges=npred, gt_edges=ngt, matched_pred_edges=mp, matched_gt_edges=mg)
        metrics["_sums"] = sums
        result[region] = metrics
    return result
