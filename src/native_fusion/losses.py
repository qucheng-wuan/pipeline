"""Final native-depth metric and explicit perspective surface supervision.

GT is confined to this training objective. Authenticated independent camera
normal assets are the primary reference. Released-Z induced normals are used
only for images without independent references, and never claimed to be an
independent measurement of renderer surface orientation.
Teacher trust is a loss-only mask; no GT-conditioned teacher filtering belongs
in model.forward or deployment. Frozen teacher confidence is a heuristic, so
its directions are also checked against the selected reference normals.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from .geometry import depth_to_normals, surface_normal_reference, normal_stencil_support


@dataclass(frozen=True)
class LossConfig:
    metric: float = 1.0
    regional: float = .25
    relative: float = .10
    background_no_regress: float = .10
    background_margin_log: float = .005
    context_metric: float = .50
    normal_gt: float = .20
    normal_teacher: float = .05
    normal_detail: float = .025
    depth_gradient: float = .10
    depth_edge: float = .05
    correction_regularization: float = .001
    normal_log_jump: float = .05
    normal_reference_neighbor_angle_deg: float = 30.0
    teacher_min_confidence: float = .20
    teacher_max_gt_angle_deg: float = 45.0
    teacher_max_neighbor_angle_deg: float = 30.0
    teacher_transparent_only: bool = True
    normal_foreground_balance: float = .50
    depth_edge_threshold: float = .05

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool):
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite nonnegative")
        if not 0 <= self.teacher_min_confidence <= 1:
            raise ValueError("teacher confidence threshold must lie in [0,1]")
        if not 0 < self.teacher_max_gt_angle_deg < 90 or not 0 < self.teacher_max_neighbor_angle_deg < 180:
            raise ValueError("teacher angle limits require meaningful directed angles")
        if not 0 < self.normal_reference_neighbor_angle_deg < 180:
            raise ValueError('normal reference stencil angle must lie in (0,180)')


def _config(value: LossConfig | Mapping | None) -> LossConfig:
    if value is None:
        return LossConfig()
    if isinstance(value, LossConfig):
        return value
    return LossConfig(**dict(value))


def masked_image_mean(value: Tensor, mask: Tensor, weight: Tensor | None = None) -> Tensor:
    """Equal image weight, then normalized confidence weight within each image."""
    selected = mask.bool()
    safe = torch.where(selected, value, torch.zeros_like(value))
    weights = selected.to(value.dtype)
    if weight is not None:
        weights = weights * torch.where(selected, weight.to(value.dtype), torch.zeros_like(weights))
    denom = weights.flatten(1).sum(1)
    numerator = (safe * weights).flatten(1).sum(1)
    means = numerator / denom.clamp_min(1e-12)
    present = (denom > 0).to(value.dtype)
    return (means * present).sum() / present.sum().clamp_min(1)


def _metric_terms(depth: Tensor, target: Tensor, valid: Tensor, transparent: Tensor, cfg: LossConfig):
    valid = valid.bool() & torch.isfinite(target) & (target > 0)
    if bool((valid & (~torch.isfinite(depth) | (depth <= 0))).any()):
        raise FloatingPointError("final depth is not finite-positive on supervision support")
    pred = torch.where(valid, depth.float(), torch.ones_like(depth.float()))
    truth = torch.where(valid, target.float(), torch.ones_like(target.float()))
    error = pred.log() - truth.log()
    huber = F.huber_loss(error, torch.zeros_like(error), delta=.1, reduction="none")
    relative = (pred - truth).abs() / truth
    fg, bg = valid & transparent.bool(), valid & ~transparent.bool()
    metric = masked_image_mean(huber, valid) + cfg.relative * masked_image_mean(relative, valid)
    region = masked_image_mean(huber, fg) + masked_image_mean(huber, bg)
    return metric, region, error, valid, fg, bg, truth


def objective(output: Mapping[str, Tensor], batch: Mapping[str, Tensor], config: LossConfig | Mapping | None = None):
    cfg = _config(config)
    depth, target = output['depth'].float(), batch['target'].float()
    if depth.shape != target.shape or depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("final depth and target must share [B,1,H,W]")
    content = batch.get('content', torch.ones_like(batch['valid'])).bool()
    metric, region, error, valid, fg, bg, truth = _metric_terms(
        depth, target, batch['valid'].bool() & content, batch['transparent'], cfg)
    zero = depth.sum() * 0
    background = zero
    if 'far_depth' in batch:
        far = batch['far_depth'].float()
        if bool((valid & (~torch.isfinite(far) | (far <= 0))).any()):
            raise ValueError("metric anchor is invalid on supervised pixels")
        far_log = torch.where(valid, far, torch.ones_like(far)).log()
        anchor_error = (far_log - truth.log()).abs()
        background = masked_image_mean(F.relu(error.abs() - anchor_error - cfg.background_margin_log), bg)

    # All explicit surface terms supervise FINAL depth, after metric and detail
    # corrections. GT determines the normal stencil; predicted jumps do not.
    camera_valid = torch.as_tensor(batch.get('normal_camera_valid', True), device=depth.device, dtype=torch.bool).flatten()
    if camera_valid.numel() not in (1, depth.shape[0]):
        raise ValueError('normal_camera_valid must be scalar or one flag per image')
    camera_valid = camera_valid.expand(depth.shape[0])[:, None, None, None]
    geometry_valid = valid & camera_valid
    normal_labels = batch.get('instance_id', batch['transparent'].long())
    reference = surface_normal_reference(truth, batch['intrinsics'], geometry_valid,
        reference_normal=batch.get('target_normal'), reference_normal_valid=batch.get('normal_valid'),
        reference_available=batch.get('normal_reference_available'),
        max_log_jump=cfg.normal_log_jump, max_normal_neighbor_angle_deg=cfg.normal_reference_neighbor_angle_deg, region_labels=normal_labels,
        boundary_mask=batch.get('boundary_mask'))
    normal_support, gt_normal = reference.valid, reference.normal.detach()
    predicted_normal, _ = depth_to_normals(depth, batch['intrinsics'], geometry_valid,
        reference_depth=truth, max_log_jump=cfg.normal_log_jump, region_labels=normal_labels,
        boundary_mask=batch.get('boundary_mask'))
    normal_cosine = (predicted_normal * gt_normal).sum(1, keepdim=True).clamp(-1, 1)
    angular_gt = 1 - normal_cosine
    normal_gt = masked_image_mean(angular_gt, normal_support)
    normal_gt += cfg.normal_foreground_balance * masked_image_mean(angular_gt, normal_support & fg)

    teacher_raw = batch['teacher_normal'].detach().float()
    confidence = batch['teacher_confidence'].detach().float()
    if teacher_raw.shape != predicted_normal.shape or confidence.shape != depth.shape:
        raise ValueError("teacher normals/confidence must share native patch geometry")
    teacher_finite = torch.isfinite(teacher_raw).all(1, keepdim=True)
    teacher = torch.where(teacher_finite, teacher_raw, torch.zeros_like(teacher_raw))
    teacher_norm = teacher.norm(dim=1, keepdim=True)
    teacher = F.normalize(teacher, dim=1, eps=1e-8)
    confidence = torch.where(torch.isfinite(confidence), confidence.clamp(0, 1), torch.zeros_like(confidence))
    trusted = teacher_finite & (teacher_norm > 1e-6) & (confidence >= cfg.teacher_min_confidence)
    trusted = normal_stencil_support(teacher, trusted, cfg.teacher_max_neighbor_angle_deg)
    trusted = trusted & normal_support
    trusted = trusted & ((teacher * gt_normal).sum(1, keepdim=True) >= math.cos(math.radians(cfg.teacher_max_gt_angle_deg)))
    if cfg.teacher_transparent_only:
        trusted = trusted & fg
    angular_teacher = 1 - (predicted_normal * teacher).sum(1, keepdim=True).clamp(-1, 1)
    normal_teacher = masked_image_mean(angular_teacher, trusted, confidence)

    # Match normal-field variation on valid same-surface stencils; this is a
    # normal-gradient detail loss, not a mislabeled wavelet or 3D tangent loss.
    normal_detail, gradient, edge = zero.clone(), zero.clone(), zero.clone()
    detail_terms, gradient_terms, edge_terms = 0, 0, 0
    for stride in (1, 2, 4):
        for axis in (-1, -2):
            if stride >= depth.shape[axis]:
                continue
            if axis == -1:
                a, b = (..., slice(stride, None)), (..., slice(None, -stride))
            else:
                a, b = (..., slice(stride, None), slice(None)), (..., slice(None, -stride), slice(None))
            pair = valid[a] & valid[b]
            difference = (error[a] - error[b]) / stride
            gradient += masked_image_mean(difference.abs(), pair)
            gradient_terms += 1
            truth_jump = truth[a].log() - truth[b].log()
            jump = pair & (truth_jump.abs() > cfg.depth_edge_threshold)
            edge += masked_image_mean(F.smooth_l1_loss(error[a] - error[b], torch.zeros_like(difference), beta=.05, reduction='none'), jump)
            edge_terms += 1
            if stride <= 2:
                normal_pair = normal_support[a] & normal_support[b] & (normal_labels[a] == normal_labels[b])
                delta = ((predicted_normal[a] - predicted_normal[b]) - (gt_normal[a] - gt_normal[b])).abs().mean(1, keepdim=True) / stride
                normal_detail += masked_image_mean(delta, normal_pair)
                detail_terms += 1
    normal_detail /= max(detail_terms, 1)
    gradient /= max(gradient_terms, 1)
    edge /= max(edge_terms, 1)

    context = zero
    if 'context_depth' in output:
        needed = ('context_target', 'context_valid', 'context_transparent')
        if any(key not in batch for key in needed):
            raise ValueError("context_depth requires independent full-image context labels")
        context_valid = batch['context_valid'].bool()
        if 'context_content' in batch:
            context_valid = context_valid & batch['context_content'].bool()
        cm, cr, *_ = _metric_terms(output['context_depth'].float(), batch['context_target'], context_valid, batch['context_transparent'], cfg)
        context = cm + cfg.regional * cr

    regularization = zero
    for key in ('low_correction', 'region_correction', 'detail_correction'):
        if key in output:
            value = output[key].float()
            regularization = regularization + (masked_image_mean(value.square(), content) if value.shape == depth.shape else value.square().mean())
    total = (cfg.metric * metric + cfg.regional * region + cfg.background_no_regress * background
             + cfg.context_metric * context + cfg.normal_gt * normal_gt + cfg.normal_teacher * normal_teacher
             + cfg.normal_detail * normal_detail + cfg.depth_gradient * gradient + cfg.depth_edge * edge
             + cfg.correction_regularization * regularization)
    terms = {'loss': total, 'metric': metric, 'regional': region, 'background_no_regress': background,
             'context_metric': context, 'normal_gt': normal_gt, 'normal_teacher': normal_teacher,
             'normal_detail': normal_detail, 'depth_gradient': gradient, 'depth_edge': edge,
             'regularization': regularization, 'normal_camera_images': camera_valid.sum(),
             'normal_pixels': normal_support.sum(), 'normal_independent_pixels': (normal_support & reference.independent).sum(),
             'normal_induced_pixels': (normal_support & ~reference.independent).sum(), 'trusted_teacher_pixels': trusted.sum()}
    return total, {key: value.detach() for key, value in terms.items()}
