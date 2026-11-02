"""Strict depth-only component facade for GIFT/TransNormal R5.

The train-time R5 model deliberately exposes diagnostics (teacher normals,
anchor predictions and outline logits).  None of those are part of the public
inference product.  This module authenticates the complete R5 checkpoint and
its frozen O2/O4 anchors, binds it to one official TransNormal fingerprint and
returns only corrected metric optical-Z depth.

This facade consumes authenticated frozen-component tensors.  The actual
public RGB-only orchestrator is :mod:`gift_normal_r5_infer`, which runs MoGe,
TOM-Net, project DINOv3, and official TransNormal internally before invoking
this module.

Native inference does not resize corrected depth.  It unletterboxes the
bounded log-residual and TOM gate independently, then applies the residual to
the frozen native MoGe depth.  An explicit ``where`` makes a zero TOM gate an
exact identity, including after interpolation.
"""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .gift_normal_r5_model import (
    GiftNormalR5Settings,
    GiftNormalR5SpatialAdapter,
    r5_trainable_parameter_count,
)
from .io_utils import read_json, sha256_file, target_geometry

CHECKPOINT_SCHEMA = "moge_transparent_gift_normal_r5_checkpoint_v1"
PLAN_SCHEMA = "moge_transparent_transnormal_teacher_r5_plan_v1"
INFERENCE_CONTRACT = (
    "RGB-only corrected metric optical-z depth; official TransNormal teacher "
    "internal; normal output not exposed"
)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in string.hexdigits for character in value
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA256")
    return value.lower()


def _settings_from_plan(plan: Mapping[str, Any]) -> GiftNormalR5Settings:
    value = plan.get("model")
    if not isinstance(value, Mapping):
        raise TypeError("R5 plan lacks model settings")
    try:
        return GiftNormalR5Settings(
            dino_channels_per_layer=int(value["dino_channels_per_layer"]),
            spatial_channels=int(value["spatial_channels"]),
            context_dilations=tuple(int(item) for item in value["context_dilations"]),
            decoder_scales=tuple(int(item) for item in value["decoder_scales"]),
            teacher_min_norm=float(
                value.get(
                    "teacher_min_norm", GiftNormalR5Settings().teacher_min_norm
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("R5 plan contains invalid model settings") from error


def _normalized_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("R5 checkpoint settings must be a mapping")
    result = dict(value)
    for key in ("context_dilations", "decoder_scales"):
        if key in result:
            try:
                result[key] = tuple(int(item) for item in result[key])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid R5 checkpoint setting: {key}") from error
    return result


def _anchor_contract(
    plan: Mapping[str, Any], label: str
) -> tuple[Path, str]:
    assets = plan.get("assets")
    if not isinstance(assets, Mapping):
        raise TypeError("R5 plan lacks assets")
    value = assets.get(f"{label}_checkpoint")
    if not isinstance(value, Mapping):
        raise TypeError(f"R5 plan lacks {label.upper()} checkpoint contract")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"R5 plan has invalid {label.upper()} checkpoint path")
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label.upper()} checkpoint is not a file: {path}")
    expected = _sha256(value.get("sha256"), f"plan {label.upper()} checkpoint")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label.upper()} checkpoint SHA256 mismatch: {actual} != {expected}"
        )
    return path, actual


def _validate_plan(plan_path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(plan_path).expanduser().resolve(strict=True)
    plan = read_json(source)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"unsupported R5 plan schema: {plan.get('schema_version')}")
    inference = plan.get("inference_contract")
    if not isinstance(inference, Mapping):
        raise TypeError("R5 plan lacks inference contract")
    required = {
        "external_output": "corrected metric optical-z depth only",
        "normal_output_exposed": False,
        "teacher_is_required_at_training_and_inference": True,
    }
    changed = [key for key, expected in required.items() if inference.get(key) != expected]
    if changed:
        raise ValueError(f"R5 plan inference contract mismatch: {changed}")
    return source, plan


def _anchor_state(model: GiftNormalR5SpatialAdapter, name: str) -> dict[str, Tensor]:
    anchor = getattr(model, name)
    return {key: tensor.detach().cpu().clone() for key, tensor in anchor.state_dict().items()}


def _verify_anchor_unchanged(
    model: GiftNormalR5SpatialAdapter,
    name: str,
    expected: Mapping[str, Tensor],
) -> None:
    anchor = getattr(model, name)
    actual = anchor.state_dict()
    if actual.keys() != expected.keys() or any(
        not torch.equal(actual[key].detach().cpu(), expected[key]) for key in expected
    ):
        raise ValueError(f"R5 checkpoint attempts to replace frozen {name[:2].upper()} weights")
    if any(parameter.requires_grad for parameter in anchor.parameters()):
        raise ValueError(f"R5 {name[:2].upper()} anchor is not frozen")


def _fingerprint_batch(
    value: str | Sequence[str], expected: str, batch_size: int
) -> None:
    if isinstance(value, str):
        supplied = [_sha256(value, "runtime teacher fingerprint")] * batch_size
    else:
        supplied = [_sha256(item, "runtime teacher fingerprint") for item in value]
        if len(supplied) != batch_size:
            raise ValueError("one teacher fingerprint is required for every batch item")
    if any(item != expected for item in supplied):
        raise ValueError("official TransNormal teacher fingerprint mismatch")


def _padding_batch(
    padding_tblr: Tensor | Sequence[int] | Sequence[Sequence[int]],
    batch_size: int,
) -> list[tuple[int, int, int, int]]:
    if isinstance(padding_tblr, Tensor):
        value = padding_tblr.detach().cpu()
        if value.shape == (4,):
            value = value[None].expand(batch_size, 4)
        if value.shape != (batch_size, 4):
            raise ValueError("padding_tblr must be [4] or [B,4]")
        if value.is_floating_point() and not torch.equal(value, value.round()):
            raise ValueError("padding_tblr must contain integers")
        rows = value.tolist()
    else:
        raw = list(padding_tblr)
        if len(raw) == 4 and all(not isinstance(item, Sequence) for item in raw):
            rows = [raw] * batch_size
        else:
            rows = [list(item) for item in raw]  # type: ignore[arg-type]
            if len(rows) != batch_size:
                raise ValueError("padding_tblr must contain one row per batch item")
    result: list[tuple[int, int, int, int]] = []
    for row in rows:
        if len(row) != 4 or any(isinstance(item, bool) for item in row):
            raise ValueError("every padding row must be (top,bottom,left,right)")
        converted = tuple(int(item) for item in row)
        if any(float(item) != converted[index] for index, item in enumerate(row)):
            raise ValueError("padding_tblr must contain integers")
        if min(converted) < 0:
            raise ValueError("letterbox padding cannot be negative")
        result.append(converted)
    return result


def unletterbox_r5_delta(
    bounded_delta_canvas: Tensor,
    gate_canvas: Tensor,
    padding_tblr: Tensor | Sequence[int] | Sequence[Sequence[int]],
    native_hw: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Restore ungated log-residual and gate to one native image geometry."""

    if (
        bounded_delta_canvas.ndim != 4
        or bounded_delta_canvas.shape[1] != 1
        or gate_canvas.shape != bounded_delta_canvas.shape
    ):
        raise ValueError("canvas delta and gate must share [B,1,H,W]")
    height, width = bounded_delta_canvas.shape[-2:]
    native_height, native_width = (int(native_hw[0]), int(native_hw[1]))
    if native_height <= 0 or native_width <= 0:
        raise ValueError("native_hw must contain positive dimensions")
    paddings = _padding_batch(padding_tblr, bounded_delta_canvas.shape[0])
    native_delta: list[Tensor] = []
    native_gate: list[Tensor] = []
    expected_padding = target_geometry(
        (native_height, native_width), (height, width)
    )[1]
    for index, padding in enumerate(paddings):
        if padding != expected_padding:
            raise ValueError(
                f"padding/native geometry mismatch: {padding} != {expected_padding}"
            )
        top, bottom, left, right = padding
        end_y = height - bottom if bottom else height
        end_x = width - right if right else width
        if top >= end_y or left >= end_x:
            raise ValueError("padding removes the complete canvas")
        delta_crop = bounded_delta_canvas[index : index + 1, :, top:end_y, left:end_x]
        gate_crop = gate_canvas[index : index + 1, :, top:end_y, left:end_x]
        native_delta.append(
            F.interpolate(
                delta_crop.float(),
                size=(native_height, native_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        )
        native_gate.append(
            F.interpolate(
                gate_crop.float(),
                size=(native_height, native_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).clamp(0.0, 1.0)
        )
    return torch.cat(native_delta, dim=0), torch.cat(native_gate, dim=0)


class GiftNormalR5DepthExport(nn.Module):
    """Depth-only facade around the authenticated full R5 model."""

    public_outputs = ("corrected_depth",)

    def __init__(
        self,
        model: GiftNormalR5SpatialAdapter,
        *,
        teacher_fingerprint: str,
        checkpoint_sha256: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.teacher_fingerprint = _sha256(
            teacher_fingerprint, "checkpoint teacher fingerprint"
        )
        self.checkpoint_sha256 = _sha256(
            checkpoint_sha256, "R5 checkpoint"
        )

    @staticmethod
    def _validate_probability(value: Tensor, base_depth: Tensor, label: str) -> None:
        if value.shape != base_depth.shape:
            raise ValueError(f"{label} must match frozen MoGe depth [B,1,H,W]")
        if not value.is_floating_point():
            raise TypeError(f"{label} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} contains non-finite values")
        if bool(((value < 0) | (value > 1)).any()):
            raise ValueError(f"{label} must lie in [0,1]")

    def forward(
        self,
        frozen_moge_depth: Tensor,
        tom_probability: Tensor,
        tom_gate: Tensor,
        dino_features: Tensor,
        teacher_normal: Tensor,
        teacher_confidence: Tensor,
        *,
        teacher_fingerprint: str | Sequence[str],
        native_moge_depth: Tensor | None = None,
        padding_tblr: Tensor | Sequence[int] | Sequence[Sequence[int]] | None = None,
    ) -> Tensor:
        """Return corrected canvas or native depth and no auxiliary tensors."""

        if frozen_moge_depth.ndim != 4 or frozen_moge_depth.shape[1] != 1:
            raise ValueError("frozen_moge_depth must be [B,1,H,W]")
        if not frozen_moge_depth.is_floating_point():
            raise TypeError("frozen_moge_depth must be floating point")
        if not bool(torch.isfinite(frozen_moge_depth).all()):
            raise ValueError("frozen_moge_depth contains non-finite values")
        if bool((frozen_moge_depth < 0).any()):
            raise ValueError("frozen_moge_depth must be nonnegative")
        self._validate_probability(tom_probability, frozen_moge_depth, "tom_probability")
        self._validate_probability(tom_gate, frozen_moge_depth, "tom_gate")
        self._validate_probability(
            teacher_confidence, frozen_moge_depth, "teacher_confidence"
        )
        _fingerprint_batch(
            teacher_fingerprint,
            self.teacher_fingerprint,
            frozen_moge_depth.shape[0],
        )
        known_mask = (frozen_moge_depth > 0).to(dtype=frozen_moge_depth.dtype)
        with torch.inference_mode():
            output = self.model(
                frozen_moge_depth,
                tom_probability,
                known_mask,
                dino_features,
                teacher_normal,
                teacher_confidence,
                gate_mask=tom_gate,
            )
        if not isinstance(output, Mapping) or {"depth", "bounded_delta"} - set(output):
            raise RuntimeError("R5 model did not return its required depth residual")

        if native_moge_depth is None:
            if padding_tblr is not None:
                raise ValueError("padding_tblr is only valid with native_moge_depth")
            active = (tom_gate > 0) & (known_mask > 0)
            corrected = torch.where(active, output["depth"].float(), frozen_moge_depth.float())
        else:
            if padding_tblr is None:
                raise ValueError("native inference requires padding_tblr")
            if (
                native_moge_depth.ndim != 4
                or native_moge_depth.shape[:2] != frozen_moge_depth.shape[:2]
            ):
                raise ValueError("native_moge_depth must be [B,1,H_native,W_native]")
            if not native_moge_depth.is_floating_point():
                raise TypeError("native_moge_depth must be floating point")
            if not bool(torch.isfinite(native_moge_depth).all()) or bool(
                (native_moge_depth < 0).any()
            ):
                raise ValueError("native_moge_depth must be finite and nonnegative")
            native_hw = tuple(int(value) for value in native_moge_depth.shape[-2:])
            bounded_native, gate_native = unletterbox_r5_delta(
                output["bounded_delta"], tom_gate, padding_tblr, native_hw
            )
            native_base = native_moge_depth.float()
            active = (gate_native > 0) & (native_base > 0)
            candidate = native_base * torch.exp(bounded_native * gate_native)
            corrected = torch.where(active, candidate, native_base)
        if not bool(torch.isfinite(corrected).all()) or bool(
            ((corrected <= 0) & (corrected != 0)).any()
        ):
            raise FloatingPointError("R5 exported corrected depth is invalid")
        return corrected


def load_r5_depth_export(
    checkpoint_path: str | Path,
    *,
    plan_path: str | Path,
    expected_teacher_fingerprint: str,
    device: str | torch.device = "cpu",
) -> tuple[GiftNormalR5DepthExport, dict[str, Any]]:
    """Authenticate a complete R5 checkpoint and construct its depth facade."""

    locked_plan, plan = _validate_plan(plan_path)
    teacher_fingerprint = _sha256(
        expected_teacher_fingerprint, "expected teacher fingerprint"
    )
    source = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"R5 checkpoint is not a file: {source}")
    checkpoint_sha256 = sha256_file(source)
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError("R5 checkpoint must be a mapping")
    checkpoint = dict(value)
    identity = {
        "schema_version": CHECKPOINT_SCHEMA,
        "plan": str(locked_plan),
        "plan_sha256": sha256_file(locked_plan),
        "inference_contract": INFERENCE_CONTRACT,
        "normal_head": False,
        "teacher_fingerprint": teacher_fingerprint,
    }
    mismatch = [key for key, expected in identity.items() if checkpoint.get(key) != expected]
    if mismatch:
        raise ValueError(f"R5 checkpoint provenance mismatch: {mismatch}")

    o2_path, o2_sha256 = _anchor_contract(plan, "o2")
    o4_path, o4_sha256 = _anchor_contract(plan, "o4")
    anchor_identity = {
        "o2_checkpoint_sha256": o2_sha256,
        "o4_checkpoint_sha256": o4_sha256,
    }
    changed = [
        key for key, expected in anchor_identity.items() if checkpoint.get(key) != expected
    ]
    if changed:
        raise ValueError(f"R5 checkpoint anchor provenance mismatch: {changed}")

    settings = _settings_from_plan(plan)
    if _normalized_settings(checkpoint.get("settings")) != asdict(settings):
        raise ValueError("R5 checkpoint settings differ from the locked plan")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("R5 checkpoint state_dict must be a nonempty mapping")
    if any(not isinstance(key, str) or not isinstance(tensor, Tensor) for key, tensor in state_dict.items()):
        raise ValueError("R5 state_dict must map string names to tensors")
    nonfinite = [
        key for key, tensor in state_dict.items() if not bool(torch.isfinite(tensor).all())
    ]
    if nonfinite:
        raise ValueError(f"R5 checkpoint contains non-finite state: {nonfinite[:4]}")

    model = GiftNormalR5SpatialAdapter(
        o2_path,
        o4_path,
        settings,
        o2_checkpoint_sha256=o2_sha256,
        o4_checkpoint_sha256=o4_sha256,
    )
    frozen_o2 = _anchor_state(model, "o2_anchor")
    frozen_o4 = _anchor_state(model, "o4_anchor")
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError("R5 checkpoint state_dict does not match the locked model") from error
    _verify_anchor_unchanged(model, "o2_anchor", frozen_o2)
    _verify_anchor_unchanged(model, "o4_anchor", frozen_o4)
    parameter_count = checkpoint.get("parameter_count")
    if parameter_count is not None and (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count != r5_trainable_parameter_count(model)
    ):
        raise ValueError("R5 checkpoint parameter_count mismatch")
    model.to(torch.device(device)).eval()
    if model.o2_anchor.training or model.o4_anchor.training:
        raise ValueError("R5 anchors did not remain in evaluation mode")
    model.requires_grad_(False)
    export = GiftNormalR5DepthExport(
        model,
        teacher_fingerprint=teacher_fingerprint,
        checkpoint_sha256=checkpoint_sha256,
    ).eval()
    return export, checkpoint


# Keep the conventional loader name available to evaluators.
load_r5_checkpoint = load_r5_depth_export
