"""Spatial TransNormal-guided residual refinement with sealed O2/O4 anchors.

R4 decoded every output point independently.  This module instead keeps the
best existing deployable adapter (O4) as an exact frozen anchor and refines its
*raw* residual with a spatial feature pyramid.  A frozen TransNormal prediction
is an explicit required input to the refiner; it is not exposed as an output
head and the public deployment pipeline can therefore remain RGB-only depth.

The correction readout is zero initialized.  Consequently a newly constructed
R5 model follows the identical arithmetic as O4 and returns an exactly equal O4
depth tensor.  O2 is also loaded and frozen: its disagreement with O4 is given
to the spatial refiner rather than discarding the older learned guidance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .io_utils import sha256_file
from .model import (
    AdapterSettings,
    TransparentGuidanceAdapter,
    build_region_masks,
    trainable_parameter_count,
)

ANCHOR_CHECKPOINT_SCHEMA = "moge_transparent_predicted_adapter_checkpoint_v1"
ANCHOR_VARIANTS = {"o2_tom": "o2", "o4_tom": "o4"}
CONDITION_CHANNELS = 13


@dataclass(frozen=True)
class GiftNormalR5Settings:
    """Capacity of the trainable spatial refinement path."""

    dino_channels_per_layer: int = 16
    spatial_channels: int = 32
    context_dilations: tuple[int, ...] = (1, 2, 4, 8)
    decoder_scales: tuple[int, ...] = (8, 4, 2, 1)
    teacher_min_norm: float = 1e-6

    def __post_init__(self) -> None:
        if self.dino_channels_per_layer <= 0 or self.spatial_channels <= 0:
            raise ValueError("R5 channel counts must be positive")
        if not self.context_dilations or any(
            not isinstance(value, int) or value <= 0
            for value in self.context_dilations
        ):
            raise ValueError("context_dilations must contain positive integers")
        if not self.decoder_scales or any(
            not isinstance(value, int) or value <= 0 for value in self.decoder_scales
        ):
            raise ValueError("decoder_scales must contain positive integers")
        if self.decoder_scales[-1] != 1:
            raise ValueError("the last R5 decoder scale must be full resolution (1)")
        if any(
            first <= second
            for first, second in zip(
                self.decoder_scales, self.decoder_scales[1:], strict=False
            )
        ):
            raise ValueError("decoder_scales must be strictly decreasing")
        if (
            not math.isfinite(float(self.teacher_min_norm))
            or self.teacher_min_norm <= 0
        ):
            raise ValueError("teacher_min_norm must be finite and positive")


@dataclass(frozen=True)
class FrozenAnchor:
    """Loaded anchor module plus immutable checkpoint provenance."""

    model: TransparentGuidanceAdapter
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint: Mapping[str, Any]


def _checkpoint_file(path: str | Path, label: str) -> Path:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{label} checkpoint is not a file: {source}")
    return source


def load_frozen_guidance_anchor(
    checkpoint_path: str | Path,
    *,
    expected_variant: str,
    expected_sha256: str | None = None,
) -> FrozenAnchor:
    """Strictly load and freeze one sealed O2/O4 adapter checkpoint."""

    if expected_variant not in ANCHOR_VARIANTS:
        raise ValueError(f"unsupported R5 anchor variant: {expected_variant}")
    source = _checkpoint_file(checkpoint_path, expected_variant)
    digest = sha256_file(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"{expected_variant} checkpoint SHA256 mismatch: {digest} != {expected_sha256}"
        )
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{expected_variant} checkpoint is not a mapping")
    expected_adapter = ANCHOR_VARIANTS[expected_variant]
    identity = {
        "schema_version": ANCHOR_CHECKPOINT_SCHEMA,
        "variant": expected_variant,
        "adapter_variant": expected_adapter,
    }
    mismatch = [
        key for key, expected in identity.items() if value.get(key) != expected
    ]
    if mismatch:
        raise ValueError(
            f"{expected_variant} checkpoint identity mismatch: {mismatch}"
        )
    raw_settings = value.get("settings")
    state_dict = value.get("state_dict")
    if not isinstance(raw_settings, Mapping) or not isinstance(state_dict, Mapping):
        raise TypeError(f"{expected_variant} checkpoint lacks settings/state_dict")
    try:
        settings = AdapterSettings(**dict(raw_settings))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{expected_variant} checkpoint has invalid adapter settings"
        ) from error
    model = TransparentGuidanceAdapter(expected_adapter, settings)
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"{expected_variant} checkpoint state_dict does not match its adapter"
        ) from error
    if value.get("parameter_count") != trainable_parameter_count(model):
        raise ValueError(f"{expected_variant} checkpoint parameter count mismatch")
    nonfinite = [
        name for name, tensor in model.state_dict().items() if not torch.isfinite(tensor).all()
    ]
    if nonfinite:
        raise ValueError(
            f"{expected_variant} checkpoint contains non-finite tensors: {nonfinite[:4]}"
        )
    model.requires_grad_(False).eval()
    return FrozenAnchor(
        model=model,
        checkpoint_path=source,
        checkpoint_sha256=digest,
        checkpoint=value,
    )


def _groups(channels: int) -> int:
    return max(1, math.gcd(channels, 8))


class _ContextResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        update = self.conv1(F.gelu(self.norm1(value), approximate="none"))
        update = self.conv2(F.gelu(self.norm2(update), approximate="none"))
        return value + update


class _PyramidFusionBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(approximate="none"),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(approximate="none"),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


def _teacher_edge(normal: Tensor, confidence: Tensor) -> Tensor:
    """Return a finite confidence-masked full-image normal edge map."""

    horizontal_valid = (confidence[..., 1:] > 0) & (confidence[..., :-1] > 0)
    vertical_valid = (confidence[..., 1:, :] > 0) & (confidence[..., :-1, :] > 0)
    dx = torch.linalg.vector_norm(normal[..., 1:] - normal[..., :-1], dim=1, keepdim=True)
    dy = torch.linalg.vector_norm(
        normal[..., 1:, :] - normal[..., :-1, :], dim=1, keepdim=True
    )
    dx = torch.where(horizontal_valid, dx, torch.zeros_like(dx))
    dy = torch.where(vertical_valid, dy, torch.zeros_like(dy))
    edge = 0.5 * (
        F.pad(dx, (0, 1, 0, 0), value=0.0)
        + F.pad(dy, (0, 0, 0, 1), value=0.0)
    )
    return edge.clamp(0.0, 1.0) * confidence


def _refine_anchor_raw(anchor_raw: Tensor, refinement: Tensor) -> Tensor:
    """Add a residual without narrowing the frozen anchor's arithmetic.

    CUDA autocast may keep the O4 interpolation output in FP32 while producing
    the R5 convolutional readout in BF16.  Casting the anchor to the readout
    dtype would then change O4 even when ``refinement`` is exactly zero.  The
    frozen anchor defines the deployment arithmetic, so cast only the trainable
    residual; ``Tensor.to`` remains differentiable for nonzero refinements.
    """

    if anchor_raw.shape != refinement.shape:
        raise ValueError("anchor raw delta and refinement must have the same shape")
    if anchor_raw.device != refinement.device:
        raise ValueError("anchor raw delta and refinement must share one device")
    if not anchor_raw.is_floating_point() or not refinement.is_floating_point():
        raise TypeError("anchor raw delta and refinement must be floating point")
    return anchor_raw.detach() + refinement.to(dtype=anchor_raw.dtype)


class GiftNormalR5SpatialAdapter(nn.Module):
    """O4-anchored, O2-aware spatial depth refinement guided by TransNormal.

    The module consumes cached/online frozen-backbone tensors.  ``teacher_normal``
    must already use the project's OpenCV convention (a visible front-facing
    plane is ``(0, 0, -1)``).  It is normalized again here so quantized caches
    cannot change the scale of a condition channel.
    """

    feature_layers = 4
    feature_width = 1280
    feature_hw = (24, 32)
    outline_classes = 3

    def __init__(
        self,
        o2_checkpoint: str | Path,
        o4_checkpoint: str | Path,
        settings: GiftNormalR5Settings | None = None,
        *,
        o2_checkpoint_sha256: str | None = None,
        o4_checkpoint_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if settings is None:
            settings = GiftNormalR5Settings()
        self.settings = settings
        o2 = load_frozen_guidance_anchor(
            o2_checkpoint,
            expected_variant="o2_tom",
            expected_sha256=o2_checkpoint_sha256,
        )
        o4 = load_frozen_guidance_anchor(
            o4_checkpoint,
            expected_variant="o4_tom",
            expected_sha256=o4_checkpoint_sha256,
        )
        if o2.model.settings.max_log_delta != o4.model.settings.max_log_delta:
            raise ValueError("O2 and O4 anchors use different max_log_delta values")
        for key in (
            "pixel_edge_radius",
            "pixel_ring_inner_radius",
            "pixel_ring_outer_radius",
            "soft_region_masks",
        ):
            if getattr(o2.model.settings, key) != getattr(o4.model.settings, key):
                raise ValueError(f"O2 and O4 region setting differs: {key}")

        self.o2_anchor = o2.model
        self.o4_anchor = o4.model
        self.o2_checkpoint_path = str(o2.checkpoint_path)
        self.o4_checkpoint_path = str(o4.checkpoint_path)
        self.o2_checkpoint_sha256 = o2.checkpoint_sha256
        self.o4_checkpoint_sha256 = o4.checkpoint_sha256
        self.max_log_delta = float(o4.model.settings.max_log_delta)

        per_layer = settings.dino_channels_per_layer
        spatial = settings.spatial_channels
        self.feature_projectors = nn.ModuleList(
            nn.Conv2d(self.feature_width, per_layer, 1)
            for _ in range(self.feature_layers)
        )
        projected_channels = self.feature_layers * per_layer
        self.dino_fuse = nn.Sequential(
            nn.Conv2d(projected_channels, spatial, 1),
            nn.GroupNorm(_groups(spatial), spatial),
            nn.GELU(approximate="none"),
        )
        self.context = nn.Sequential(
            *(
                _ContextResidualBlock(spatial, dilation)
                for dilation in settings.context_dilations
            )
        )
        # Re-inject all four projected DINO layers at every spatial scale.  This
        # is an actual spatial FPN, rather than a point-wise sampled descriptor.
        self.lateral_projectors = nn.ModuleList(
            nn.Conv2d(projected_channels, spatial, 1)
            for _ in settings.decoder_scales
        )
        decoder_input = spatial * 2 + CONDITION_CHANNELS
        self.decoder = nn.ModuleList(
            _PyramidFusionBlock(decoder_input, spatial)
            for _ in settings.decoder_scales
        )
        self.outline_head = nn.Sequential(
            nn.Conv2d(spatial, spatial, 3, padding=1),
            nn.GELU(approximate="none"),
            nn.Conv2d(spatial, self.outline_classes, 1),
        )
        # The soft predicted boundary is visible to the residual readout.  The
        # final layer is exactly zero so arbitrary teacher/DINO inputs cannot
        # perturb O4 before training.
        self.refine_readout = nn.Sequential(
            nn.Conv2d(spatial + 1, spatial, 3, padding=1),
            nn.GELU(approximate="none"),
            nn.Conv2d(spatial, 1, 1),
        )
        nn.init.zeros_(self.refine_readout[-1].weight)
        nn.init.zeros_(self.refine_readout[-1].bias)

    def train(self, mode: bool = True) -> GiftNormalR5SpatialAdapter:
        super().train(mode)
        # ``Module.train`` recurses into children; restore the immutable anchors.
        self.o2_anchor.eval()
        self.o4_anchor.eval()
        return self

    def _validate_inputs(
        self,
        base_depth: Tensor,
        transparent_mask: Tensor,
        known_mask: Tensor,
        features: Tensor,
        teacher_normal: Tensor,
        teacher_confidence: Tensor,
        gate_mask: Tensor | None,
    ) -> None:
        if not isinstance(teacher_normal, Tensor) or not isinstance(
            teacher_confidence, Tensor
        ):
            raise TypeError("R5 requires tensor teacher_normal and teacher_confidence")
        if base_depth.ndim != 4 or base_depth.shape[1] != 1:
            raise ValueError("base_depth must be [B,1,H,W]")
        if not base_depth.is_floating_point():
            raise TypeError("base_depth must be floating point")
        batch, _, height, width = base_depth.shape
        scalar_shape = (batch, 1, height, width)
        if tuple(transparent_mask.shape) != scalar_shape or tuple(known_mask.shape) != scalar_shape:
            raise ValueError("transparent_mask and known_mask must match base depth")
        if gate_mask is not None and tuple(gate_mask.shape) != scalar_shape:
            raise ValueError("gate_mask must match base depth")
        expected_features = (
            batch,
            self.feature_layers,
            self.feature_width,
            *self.feature_hw,
        )
        if tuple(features.shape) != expected_features:
            raise ValueError(
                f"DINOv3 feature shape {tuple(features.shape)} != {expected_features}"
            )
        if tuple(teacher_normal.shape) != (batch, 3, height, width):
            raise ValueError("teacher_normal must be full-image [B,3,H,W]")
        if tuple(teacher_confidence.shape) != scalar_shape:
            raise ValueError("teacher_confidence must be full-image [B,1,H,W]")
        values = [
            transparent_mask,
            known_mask,
            features,
            teacher_normal,
            teacher_confidence,
        ]
        if gate_mask is not None:
            values.append(gate_mask)
        if any(value.device != base_depth.device for value in values):
            raise ValueError("all R5 inputs must share one device")
        named = {
            "base_depth": base_depth,
            "transparent_mask": transparent_mask,
            "known_mask": known_mask,
            "DINOv3 features": features,
            "teacher_normal": teacher_normal,
            "teacher_confidence": teacher_confidence,
        }
        if gate_mask is not None:
            named["gate_mask"] = gate_mask
        nonfinite = [name for name, value in named.items() if not torch.isfinite(value).all()]
        if nonfinite:
            raise ValueError(f"R5 inputs contain non-finite values: {nonfinite}")
        content = known_mask.detach() > 0
        if bool((base_depth < 0).any()) or bool(((base_depth <= 0) & content).any()):
            raise ValueError(
                "base_depth must be positive on known pixels and nonnegative elsewhere"
            )
        confidence = teacher_confidence.detach()
        if bool((confidence < 0).any()) or bool((confidence > 1).any()):
            raise ValueError("teacher_confidence must lie in [0,1]")
        teacher_norm = torch.linalg.vector_norm(
            teacher_normal.detach().float(), dim=1, keepdim=True
        )
        invalid_supported = (confidence > 0) & (
            teacher_norm <= float(self.settings.teacher_min_norm)
        )
        if bool(invalid_supported.any()):
            raise ValueError(
                "positive-confidence teacher pixels must contain a nonzero normal"
            )

    def _normalized_teacher(
        self, teacher_normal: Tensor, teacher_confidence: Tensor
    ) -> Tensor:
        value = teacher_normal.detach().float()
        norm = torch.linalg.vector_norm(value, dim=1, keepdim=True)
        normalized = value / norm.clamp_min(float(self.settings.teacher_min_norm))
        fallback = torch.zeros_like(normalized)
        fallback[:, 2:3] = -1.0
        return torch.where(norm > self.settings.teacher_min_norm, normalized, fallback)

    @staticmethod
    def _normalized_log_depth(base_depth: Tensor, content: Tensor) -> Tensor:
        log_depth = base_depth.float().clamp_min(1e-6).log()
        denominator = content.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        mean = (log_depth * content).sum(dim=(-2, -1), keepdim=True) / denominator
        return log_depth - mean

    @staticmethod
    def _target_sizes(
        height: int, width: int, scales: tuple[int, ...]
    ) -> tuple[tuple[int, int], ...]:
        return tuple(
            (max(1, math.ceil(height / scale)), max(1, math.ceil(width / scale)))
            for scale in scales
        )

    def forward(
        self,
        base_depth: Tensor,
        transparent_mask: Tensor,
        known_mask: Tensor,
        features: Tensor,
        teacher_normal: Tensor,
        teacher_confidence: Tensor,
        *,
        gate_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self._validate_inputs(
            base_depth,
            transparent_mask,
            known_mask,
            features,
            teacher_normal,
            teacher_confidence,
            gate_mask,
        )
        probability = transparent_mask.float().clamp(0.0, 1.0)
        content = known_mask.float().clamp(0.0, 1.0)
        gate = probability if gate_mask is None else gate_mask.float().clamp(0.0, 1.0)

        # Frozen anchors deliberately receive the same exact inputs and gate as
        # R5.  no_grad plus requires_grad=False makes their immutability explicit.
        with torch.no_grad():
            o2 = self.o2_anchor(
                base_depth,
                transparent_mask,
                known_mask,
                features,
                gate_mask=gate_mask,
            )
            o4 = self.o4_anchor(
                base_depth,
                transparent_mask,
                known_mask,
                features,
                gate_mask=gate_mask,
            )

        normal = self._normalized_teacher(teacher_normal, teacher_confidence)
        confidence = teacher_confidence.detach().float()
        normal_edge = _teacher_edge(normal, confidence)
        region_settings = self.o4_anchor.settings
        regions = build_region_masks(
            probability,
            content,
            edge_radius=region_settings.pixel_edge_radius,
            ring_inner_radius=region_settings.pixel_ring_inner_radius,
            ring_outer_radius=region_settings.pixel_ring_outer_radius,
            soft=region_settings.soft_region_masks,
        )
        o2_bounded = o2["bounded_delta"].detach().float()
        o4_bounded = o4["bounded_delta"].detach().float()
        conditions = torch.cat(
            (
                self._normalized_log_depth(base_depth, content),
                probability,
                regions["edge"],
                regions["ring"],
                normal,
                confidence,
                normal_edge,
                o2_bounded,
                o4_bounded,
                o2_bounded - o4_bounded,
                content,
            ),
            dim=1,
        )
        if conditions.shape[1] != CONDITION_CHANNELS:
            raise RuntimeError("internal R5 condition-channel contract changed")

        frozen_features = features.detach()
        projection_dtype = self.feature_projectors[0].weight.dtype
        projected = torch.cat(
            tuple(
                projector(frozen_features[:, layer].to(dtype=projection_dtype))
                for layer, projector in enumerate(self.feature_projectors)
            ),
            dim=1,
        )
        state = self.context(self.dino_fuse(projected))
        height, width = base_depth.shape[-2:]
        target_sizes = self._target_sizes(
            height, width, self.settings.decoder_scales
        )
        for size, lateral, decoder in zip(
            target_sizes, self.lateral_projectors, self.decoder, strict=True
        ):
            state = F.interpolate(
                state, size=size, mode="bilinear", align_corners=False
            )
            dino_lateral = F.interpolate(
                lateral(projected), size=size, mode="bilinear", align_corners=False
            )
            resized_conditions = F.interpolate(
                conditions, size=size, mode="bilinear", align_corners=False
            )
            state = decoder(
                torch.cat((state, dino_lateral, resized_conditions), dim=1)
            )

        outline_logits = self.outline_head(state)
        boundary_probability = outline_logits.softmax(dim=1)[:, 1:].sum(
            dim=1, keepdim=True
        )
        refine_raw = self.refine_readout(
            torch.cat((state, boundary_probability), dim=1)
        )
        if not bool(torch.isfinite(refine_raw).all()):
            raise FloatingPointError("R5 spatial refiner produced non-finite values")

        # Refine O4 in raw space and repeat O4's exact tanh/gate arithmetic.
        # At construction refine_raw is exact zero, so every returned depth bit
        # is equal to the frozen O4 result for the same input.
        raw = _refine_anchor_raw(o4["raw_delta"], refine_raw)
        bounded = self.max_log_delta * torch.tanh(raw)
        # Keep gate/content in FP32 exactly as ``TransparentGuidanceAdapter``
        # does.  Casting them to an autocast dtype would preserve FP32 tests but
        # silently break the promised BF16 bit identity with O4.
        delta = bounded * gate * content
        prediction = base_depth.float() * torch.exp(delta)
        valid_prediction = (content <= 0) | (prediction > 0)
        if not bool(torch.isfinite(prediction).all()) or not bool(valid_prediction.all()):
            raise FloatingPointError("R5 predicted depth is not finite-positive")
        return {
            "depth": prediction,
            "delta_log": delta,
            "raw_delta": raw,
            "bounded_delta": bounded,
            "refine_raw": refine_raw,
            "outline_logits": outline_logits,
            "teacher_normal": normal,
            "teacher_confidence": confidence,
            "teacher_edge": normal_edge,
            "o2_depth": o2["depth"].detach(),
            "o4_depth": o4["depth"].detach(),
            "o2_delta_log": o2["delta_log"].detach(),
            "o4_delta_log": o4["delta_log"].detach(),
            "anchor_disagreement": (o2_bounded - o4_bounded).abs(),
            "patch_roi": o4["patch_roi"].detach(),
        }


def r5_trainable_parameter_count(model: nn.Module) -> int:
    """Count only R5 parameters; frozen O2/O4 anchors are excluded naturally."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# Keep the versioned adapter naming parallel to ``GiftNormalR4Adapter`` while
# retaining the more descriptive class name in diagnostics and checkpoints.
GiftNormalR5Adapter = GiftNormalR5SpatialAdapter
