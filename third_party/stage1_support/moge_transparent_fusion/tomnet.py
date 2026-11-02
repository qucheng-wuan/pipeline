"""Faithful PyTorch port of the official TOM-Net CoarseNet.

The port follows ``models/CoarseNet.lua`` from TOM-Net commit
``f4c60113653ea6f73e0a20a944691888bfb3065e``.  It intentionally keeps the
four coarse-to-fine outputs and all three task branches (flow, two-class
object mask, and attenuation/rho).  The public input contract is RGB in
``[0, 1]`` with spatial dimensions divisible by 64.

Torch7 used one-based class labels: background=1 and object=2.  Consequently,
the transparent-object probability in PyTorch is **channel index 1**.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


OFFICIAL_TOMNET_COMMIT = "f4c60113653ea6f73e0a20a944691888bfb3065e"
OFFICIAL_COARSENET_LUA_SHA256 = (
    "cbd3e5330d38b7e138760575106d21a254f5e4971b5be2c8d5b10debbdd3f53c"
)
OFFICIAL_COARSENET_T7_SHA256 = (
    "258d91c9ea379607f68d7a9015c85216ee486c972fe0e053113178ae6cdb369b"
)
CONVERTED_CHECKPOINT_FORMAT = "tomnet-coarsenet-pytorch-v1"

TORCH_BACKGROUND_CLASS = 1
TORCH_OBJECT_CLASS = 2
BACKGROUND_MASK_CHANNEL = 0
OBJECT_MASK_CHANNEL = 1
TORCH7_FEEDBACK_SOFTMAX_MODE = "CUDNN_SOFTMAX_MODE_INSTANCE"


@dataclass(frozen=True)
class TomNetScaleOutput:
    """One TOM-Net output scale; ``scale`` is 4, 3, 2, or 1.

    ``mask_probability`` is the useful per-pixel, two-class probability exposed
    by this port.  It is deliberately *not* the normalization used internally
    by the original Torch7 cross-scale feedback.  That serialized
    ``cudnn.SoftMax`` uses ``CUDNN_SOFTMAX_MODE_INSTANCE`` and therefore
    normalizes all ``C*H*W`` mask logits together for each sample.
    """

    scale: int
    flow: Tensor
    mask_logits: Tensor
    mask_probability: Tensor
    attenuation: Tensor

    @property
    def object_probability(self) -> Tensor:
        """Transparent-object probability (Torch class 2 / channel 1)."""

        return self.mask_probability[:, OBJECT_MASK_CHANNEL : OBJECT_MASK_CHANNEL + 1]


@dataclass(frozen=True)
class TomNetCoarseOutput:
    """Four outputs in official coarse-to-fine order: scales 4, 3, 2, 1."""

    scales: tuple[TomNetScaleOutput, ...]

    @property
    def final(self) -> TomNetScaleOutput:
        return self.scales[-1]

    @property
    def flow(self) -> Tensor:
        return self.final.flow

    @property
    def mask_logits(self) -> Tensor:
        return self.final.mask_logits

    @property
    def mask_probability(self) -> Tensor:
        return self.final.mask_probability

    @property
    def object_probability(self) -> Tensor:
        """Final transparent-object probability as ``[B,1,H,W]``."""

        return self.final.object_probability

    @property
    def attenuation(self) -> Tensor:
        return self.final.attenuation


class _ConvBnRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(value)))


class _DecoderBlock(_ConvBnRelu):
    def forward(self, value: Tensor) -> Tensor:
        value = super().forward(value)
        return F.interpolate(value, scale_factor=2.0, mode="nearest")


class _OutputHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.flow = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.mask_logits = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.attenuation = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        features: Tensor,
        *,
        scale: int,
        flow_reference_width: float,
    ) -> TomNetScaleOutput:
        flow_ratio = flow_reference_width / float(2 ** (scale - 1))
        flow = torch.tanh(self.flow(features)) * flow_ratio
        mask_logits = self.mask_logits(features)
        mask_probability = torch.softmax(mask_logits, dim=1)
        attenuation = self.attenuation(features)
        return TomNetScaleOutput(
            scale=scale,
            flow=flow,
            mask_logits=mask_logits,
            mask_probability=mask_probability,
            attenuation=attenuation,
        )


class TomNetCoarseNet(nn.Module):
    """Official four-scale, three-branch TOM-Net CoarseNet.

    The original model was trained at width 448 and serialized fixed flow
    multipliers ``56, 112, 224, 448``.  ``flow_reference_width`` therefore
    defaults to 448 even when convolutional inference uses another valid image
    size.  This has no effect on mask logits: normalized flow fed between
    scales cancels the multiplier exactly.
    """

    input_channels = 3
    number_of_scales = 4
    output_scales = (4, 3, 2, 1)
    object_mask_channel = OBJECT_MASK_CHANNEL

    _ENCODER_SPECS = (
        (3, 16, 1),
        (16, 16, 1),
        (16, 16, 2),
        (16, 16, 1),
        (16, 32, 2),
        (32, 32, 1),
        (32, 64, 2),
        (64, 64, 1),
        (64, 128, 2),
        (128, 128, 1),
        (128, 256, 2),
        (256, 256, 1),
        (256, 256, 2),
        (256, 256, 1),
    )

    def __init__(
        self,
        *,
        flow_reference_width: float = 448.0,
        validate_input_range: bool = True,
    ) -> None:
        super().__init__()
        if flow_reference_width <= 0:
            raise ValueError("flow_reference_width must be positive")
        self.flow_reference_width = float(flow_reference_width)
        self.validate_input_range = bool(validate_input_range)

        self.encoder = nn.ModuleList(
            _ConvBnRelu(in_channels, out_channels, stride)
            for in_channels, out_channels, stride in self._ENCODER_SPECS
        )
        self.decoders = nn.ModuleDict(
            {
                "stage6": nn.ModuleList(_DecoderBlock(256, 256, 1) for _ in range(3)),
                "stage5": nn.ModuleList(_DecoderBlock(1024, 128, 1) for _ in range(3)),
                "stage4": nn.ModuleList(_DecoderBlock(512, 64, 1) for _ in range(3)),
                "stage3": nn.ModuleList(_DecoderBlock(256, 32, 1) for _ in range(3)),
                "stage2": nn.ModuleList(_DecoderBlock(133, 16, 1) for _ in range(3)),
                "stage1": nn.ModuleList(_DecoderBlock(69, 16, 1) for _ in range(3)),
            }
        )
        self.heads = nn.ModuleDict(
            {
                "scale4": _OutputHead(256),
                "scale3": _OutputHead(133),
                "scale2": _OutputHead(69),
                "scale1": _OutputHead(69),
            }
        )
        self.provenance: dict[str, Any] | None = None

    def mask_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate only the four two-class mask-head parameters."""

        for scale in self.output_scales:
            yield from self.heads[f"scale{scale}"].mask_logits.parameters()

    def mask_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for scale in self.output_scales:
            prefix = f"heads.scale{scale}.mask_logits"
            for name, parameter in self.heads[f"scale{scale}"].mask_logits.named_parameters():
                yield f"{prefix}.{name}", parameter

    @staticmethod
    def _join(values: Sequence[Tensor]) -> Tensor:
        return torch.cat(tuple(values), dim=1)

    @staticmethod
    def _decode(inputs: Sequence[Tensor], blocks: nn.ModuleList) -> list[Tensor]:
        joined = TomNetCoarseNet._join(inputs)
        return [block(joined) for block in blocks]

    def _head(self, inputs: Sequence[Tensor], scale: int) -> TomNetScaleOutput:
        return self.heads[f"scale{scale}"](
            self._join(inputs),
            scale=scale,
            flow_reference_width=self.flow_reference_width,
        )

    def _feedback(self, output: TomNetScaleOutput) -> Tensor:
        # Exact Torch7 normalizeOutput.  The checkpoint's cudnn.SoftMax stores
        # CUDNN_SOFTMAX_MODE_INSTANCE, whose reduction domain is the complete
        # C*H*W instance (not the two channels independently at every pixel).
        # Confusing this with a channel softmax makes scale-2 drift and drives
        # the final mask logit difference to roughly +214 on the official bull
        # example, yielding an invalid all-foreground prediction.
        flow_ratio = self.flow_reference_width / float(2 ** (output.scale - 1))
        feedback_mask = torch.softmax(output.mask_logits.flatten(start_dim=1), dim=1)
        feedback_mask = feedback_mask.view_as(output.mask_logits)
        normalized = torch.cat(
            (output.flow / flow_ratio, feedback_mask, output.attenuation),
            dim=1,
        )
        return F.interpolate(normalized, scale_factor=2.0, mode="nearest")

    def _validate_input(self, rgb: Tensor) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != self.input_channels:
            raise ValueError("TOM-Net input must have shape [B,3,H,W]")
        if not torch.is_floating_point(rgb):
            raise TypeError("TOM-Net RGB input must be floating point in [0,1]")
        height, width = rgb.shape[-2:]
        if height <= 0 or width <= 0 or height % 64 != 0 or width % 64 != 0:
            raise ValueError("TOM-Net H and W must be positive multiples of 64")
        if self.validate_input_range:
            detached = rgb.detach()
            if not bool(torch.isfinite(detached).all()):
                raise ValueError("TOM-Net RGB input contains non-finite values")
            minimum = float(detached.amin())
            maximum = float(detached.amax())
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError(
                    f"TOM-Net expects RGB in [0,1], observed [{minimum}, {maximum}]"
                )

    def forward(self, rgb: Tensor) -> TomNetCoarseOutput:
        self._validate_input(rgb)

        # Encoder skip tensors correspond to conv0 ... conv6 in CoarseNet.lua.
        encoder_features: list[Tensor] = []
        value = rgb
        for block_index, block in enumerate(self.encoder):
            value = block(value)
            if block_index % 2 == 1:
                encoder_features.append(value)
        conv0, conv1, conv2, conv3, conv4, conv5, conv6 = encoder_features

        # The three independent decoder paths are flow, mask, and rho.  They
        # receive the same concatenated multi-path tensor but never share weights.
        deconv6 = [block(conv6) for block in self.decoders["stage6"]]
        deconv5 = self._decode((*deconv6, conv5), self.decoders["stage5"])
        deconv4 = self._decode((*deconv5, conv4), self.decoders["stage4"])
        scale4_inputs = (*deconv4, conv3)
        scale4 = self._head(scale4_inputs, 4)

        # In the original nngraph, scale-4 feedback is appended after stage-3
        # blocks are constructed.  Thus it enters the scale-3 head and stage-2,
        # not the stage-3 convolution itself.
        deconv3 = self._decode(scale4_inputs, self.decoders["stage3"])
        scale3_inputs = (*deconv3, conv2, self._feedback(scale4))
        scale3 = self._head(scale3_inputs, 3)

        deconv2 = self._decode(scale3_inputs, self.decoders["stage2"])
        scale2_inputs = (*deconv2, conv1, self._feedback(scale3))
        scale2 = self._head(scale2_inputs, 2)

        deconv1 = self._decode(scale2_inputs, self.decoders["stage1"])
        scale1_inputs = (*deconv1, conv0, self._feedback(scale2))
        scale1 = self._head(scale1_inputs, 1)

        return TomNetCoarseOutput(scales=(scale4, scale3, scale2, scale1))


@dataclass(frozen=True)
class _LayerBinding:
    source_path: str
    source: Any
    target_path: str
    target: nn.Module
    kind: str


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _typename(value: Any) -> str:
    name = getattr(value, "_typename", type(value).__name__)
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def _field(value: Any, name: str) -> Any:
    obj = getattr(value, "_obj", None)
    if not isinstance(obj, Mapping):
        raise ValueError(f"{_typename(value)} has no serialized field mapping")
    if name in obj:
        return obj[name]
    encoded = name.encode("utf-8")
    if encoded in obj:
        return obj[encoded]
    raise ValueError(f"{_typename(value)} is missing required field {name!r}")


def _modules(value: Any, path: str) -> list[Any]:
    modules = _field(value, "modules")
    if not isinstance(modules, list):
        raise ValueError(f"{path}.modules must be a Torch7 list")
    return modules


def _require_type(value: Any, allowed: set[str], path: str) -> None:
    actual = _typename(value)
    if actual not in allowed:
        raise ValueError(f"{path}: expected {sorted(allowed)}, found {actual}")


_CONV_TYPES = {"nn.SpatialConvolution", "cudnn.SpatialConvolution"}
_BN_TYPES = {"nn.SpatialBatchNormalization", "cudnn.SpatialBatchNormalization"}
_RELU_TYPES = {"nn.ReLU", "cudnn.ReLU"}
_TANH_TYPES = {"nn.Tanh", "cudnn.Tanh"}
_SOFTMAX_TYPES = {"nn.SoftMax", "cudnn.SoftMax"}


def _binding_for_block(
    source_block: Any,
    target_block: _ConvBnRelu,
    *,
    source_path: str,
    target_path: str,
    decoder: bool,
    bottom: bool = False,
) -> list[_LayerBinding]:
    children = _modules(source_block, source_path)
    expected_length = 4 if decoder and bottom else 5 if decoder else 3
    if len(children) != expected_length:
        raise ValueError(
            f"{source_path}: expected {expected_length} children, found {len(children)}"
        )
    offset = 0
    if decoder and not bottom:
        _require_type(children[0], {"nn.JoinTable"}, f"{source_path}.modules[0]")
        offset = 1
    _require_type(children[offset], _CONV_TYPES, f"{source_path}.modules[{offset}]")
    _require_type(children[offset + 1], _BN_TYPES, f"{source_path}.modules[{offset + 1}]")
    _require_type(children[offset + 2], _RELU_TYPES, f"{source_path}.modules[{offset + 2}]")
    if decoder:
        _require_type(
            children[offset + 3],
            {"nn.SpatialUpSamplingNearest"},
            f"{source_path}.modules[{offset + 3}]",
        )
        scale_factor = float(_field(children[offset + 3], "scale_factor"))
        if scale_factor != 2.0:
            raise ValueError(f"{source_path}: expected nearest upsample factor 2")
    return [
        _LayerBinding(
            source_path=f"{source_path}.modules[{offset}]",
            source=children[offset],
            target_path=f"{target_path}.conv",
            target=target_block.conv,
            kind="conv",
        ),
        _LayerBinding(
            source_path=f"{source_path}.modules[{offset + 1}]",
            source=children[offset + 1],
            target_path=f"{target_path}.bn",
            target=target_block.bn,
            kind="batch_norm",
        ),
    ]


def _binding_for_head(
    source_head: Any,
    target_head: _OutputHead,
    *,
    source_path: str,
    target_path: str,
    scale: int,
) -> list[_LayerBinding]:
    children = _modules(source_head, source_path)
    if len(children) != 2:
        raise ValueError(f"{source_path}: expected JoinTable + ConcatTable")
    _require_type(children[0], {"nn.JoinTable"}, f"{source_path}.modules[0]")
    _require_type(children[1], {"nn.ConcatTable"}, f"{source_path}.modules[1]")
    branches = _modules(children[1], f"{source_path}.modules[1]")
    if len(branches) != 3:
        raise ValueError(f"{source_path}: expected flow/mask/rho branches")

    flow_modules = _modules(branches[0], f"{source_path}.flow")
    rho_modules = _modules(branches[2], f"{source_path}.rho")
    if len(flow_modules) != 3 or len(rho_modules) != 1:
        raise ValueError(f"{source_path}: malformed flow or rho branch")
    _require_type(flow_modules[0], _CONV_TYPES, f"{source_path}.flow.conv")
    _require_type(flow_modules[1], _TANH_TYPES, f"{source_path}.flow.tanh")
    _require_type(flow_modules[2], {"nn.MulConstant"}, f"{source_path}.flow.scale")
    expected_ratio = 448.0 / float(2 ** (scale - 1))
    actual_ratio = float(_field(flow_modules[2], "constant_scalar"))
    if abs(actual_ratio - expected_ratio) > 1e-12:
        raise ValueError(
            f"{source_path}: flow multiplier {actual_ratio} != {expected_ratio}"
        )
    _require_type(branches[1], _CONV_TYPES, f"{source_path}.mask")
    _require_type(rho_modules[0], _CONV_TYPES, f"{source_path}.rho.conv")
    return [
        _LayerBinding(
            source_path=f"{source_path}.flow.conv",
            source=flow_modules[0],
            target_path=f"{target_path}.flow",
            target=target_head.flow,
            kind="conv",
        ),
        _LayerBinding(
            source_path=f"{source_path}.mask",
            source=branches[1],
            target_path=f"{target_path}.mask_logits",
            target=target_head.mask_logits,
            kind="conv",
        ),
        _LayerBinding(
            source_path=f"{source_path}.rho.conv",
            source=rho_modules[0],
            target_path=f"{target_path}.attenuation",
            target=target_head.attenuation,
            kind="conv",
        ),
    ]


def _validate_normalizer(source: Any, *, source_path: str, scale: int) -> None:
    children = _modules(source, source_path)
    if len(children) != 3:
        raise ValueError(f"{source_path}: expected ParallelTable, JoinTable, Upsample")
    _require_type(children[0], {"nn.ParallelTable"}, f"{source_path}.modules[0]")
    _require_type(children[1], {"nn.JoinTable"}, f"{source_path}.modules[1]")
    _require_type(
        children[2],
        {"nn.SpatialUpSamplingNearest"},
        f"{source_path}.modules[2]",
    )
    branches = _modules(children[0], f"{source_path}.modules[0]")
    if len(branches) != 3:
        raise ValueError(f"{source_path}: expected normalized flow/mask/rho branches")
    flow_modules = _modules(branches[0], f"{source_path}.flow")
    if len(flow_modules) != 1:
        raise ValueError(f"{source_path}: malformed normalized-flow branch")
    _require_type(flow_modules[0], {"nn.MulConstant"}, f"{source_path}.flow.scale")
    _require_type(branches[1], _SOFTMAX_TYPES, f"{source_path}.mask.softmax")
    if _typename(branches[1]) == "cudnn.SoftMax":
        mode = _field(branches[1], "mode")
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        if mode != TORCH7_FEEDBACK_SOFTMAX_MODE:
            raise ValueError(
                f"{source_path}: feedback softmax mode {mode!r} != "
                f"{TORCH7_FEEDBACK_SOFTMAX_MODE}"
            )
    _require_type(branches[2], {"nn.Identity"}, f"{source_path}.rho.identity")
    expected = float(2 ** (scale - 1)) / 448.0
    actual = float(_field(flow_modules[0], "constant_scalar"))
    if abs(actual - expected) > 1e-12:
        raise ValueError(f"{source_path}: flow normalizer {actual} != {expected}")


def _torch7_bindings(source_model: Any, target: TomNetCoarseNet) -> list[_LayerBinding]:
    _require_type(source_model, {"nn.gModule"}, "model")
    top = _modules(source_model, "model")
    if len(top) != 40:
        raise ValueError(f"official CoarseNet must expose 40 top modules, found {len(top)}")
    _require_type(top[0], {"nn.Identity"}, "model.modules[0]")
    for index, module in enumerate(top[1:], start=1):
        _require_type(module, {"nn.Sequential"}, f"model.modules[{index}]")

    bindings: list[_LayerBinding] = []
    for index, target_block in enumerate(target.encoder):
        bindings.extend(
            _binding_for_block(
                top[index + 1],
                target_block,
                source_path=f"model.modules[{index + 1}]",
                target_path=f"encoder.{index}",
                decoder=False,
            )
        )

    stage_layout = (
        ("stage6", 15, True),
        ("stage5", 18, False),
        ("stage4", 21, False),
        ("stage3", 25, False),
        ("stage2", 30, False),
        ("stage1", 35, False),
    )
    for stage, first_top_index, bottom in stage_layout:
        for branch, target_block in enumerate(target.decoders[stage]):
            top_index = first_top_index + branch
            bindings.extend(
                _binding_for_block(
                    top[top_index],
                    target_block,
                    source_path=f"model.modules[{top_index}]",
                    target_path=f"decoders.{stage}.{branch}",
                    decoder=True,
                    bottom=bottom,
                )
            )

    head_layout = ((4, 24), (3, 29), (2, 34), (1, 39))
    for scale, top_index in head_layout:
        bindings.extend(
            _binding_for_head(
                top[top_index],
                target.heads[f"scale{scale}"],
                source_path=f"model.modules[{top_index}]",
                target_path=f"heads.scale{scale}",
                scale=scale,
            )
        )
    for scale, top_index in ((4, 28), (3, 33), (2, 38)):
        _validate_normalizer(
            top[top_index], source_path=f"model.modules[{top_index}]", scale=scale
        )
    return bindings


def _all_source_parameter_modules(source_model: Any) -> list[Any]:
    found: list[Any] = []
    seen: set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if _typename(value) in _CONV_TYPES | _BN_TYPES:
            found.append(value)
        obj = getattr(value, "_obj", None)
        if not isinstance(obj, Mapping):
            return
        children = obj.get(b"modules", obj.get("modules"))
        if isinstance(children, list):
            for child in children:
                visit(child)

    visit(source_model)
    return found


def _tensor_from_source(source: Any, field: str, expected: Tensor, path: str) -> Tensor:
    value = _field(source, field)
    try:
        tensor = torch.as_tensor(value).clone().contiguous()
    except Exception as error:  # pragma: no cover - defensive error enrichment
        raise ValueError(f"{path}.{field}: cannot convert Torch7 tensor") from error
    if tuple(tensor.shape) != tuple(expected.shape):
        raise ValueError(
            f"{path}.{field}: source shape {tuple(tensor.shape)} != "
            f"target shape {tuple(expected.shape)}"
        )
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{path}.{field}: expected a floating-point tensor")
    return tensor.to(dtype=expected.dtype)


def _copy_torch7_parameters(
    source_model: Any,
    target: TomNetCoarseNet,
) -> dict[str, Any]:
    bindings = _torch7_bindings(source_model, target)
    source_modules = _all_source_parameter_modules(source_model)
    mapped_source_ids = [id(binding.source) for binding in bindings]
    if len(bindings) != 76 or len(source_modules) != 76:
        raise ValueError(
            f"expected 76 parameterized layers, got bindings={len(bindings)} "
            f"source={len(source_modules)}"
        )
    if len(set(mapped_source_ids)) != len(mapped_source_ids):
        raise ValueError("a Torch7 parameterized layer was mapped more than once")
    if set(mapped_source_ids) != {id(module) for module in source_modules}:
        raise ValueError("Torch7 parameterized-layer mapping is not exhaustive")

    mapping_records: list[dict[str, Any]] = []
    covered_state_keys: set[str] = set()
    generated_state_keys: list[str] = []
    with torch.no_grad():
        for binding in bindings:
            source, target_module = binding.source, binding.target
            if binding.kind == "conv":
                if not isinstance(target_module, nn.Conv2d):
                    raise TypeError(f"{binding.target_path} is not Conv2d")
                expected_geometry = {
                    "nInputPlane": target_module.in_channels,
                    "nOutputPlane": target_module.out_channels,
                    "kH": target_module.kernel_size[0],
                    "kW": target_module.kernel_size[1],
                    "dH": target_module.stride[0],
                    "dW": target_module.stride[1],
                    "padH": target_module.padding[0],
                    "padW": target_module.padding[1],
                }
                for field, expected in expected_geometry.items():
                    actual = int(_field(source, field))
                    if actual != expected:
                        raise ValueError(
                            f"{binding.source_path}.{field}: {actual} != {expected}"
                        )
                fields = ("weight", "bias")
            elif binding.kind == "batch_norm":
                if not isinstance(target_module, nn.BatchNorm2d):
                    raise TypeError(f"{binding.target_path} is not BatchNorm2d")
                source_eps = float(_field(source, "eps"))
                source_momentum = float(_field(source, "momentum"))
                if abs(source_eps - target_module.eps) > 1e-12:
                    raise ValueError(f"{binding.source_path}.eps does not match PyTorch")
                if abs(source_momentum - target_module.momentum) > 1e-12:
                    raise ValueError(
                        f"{binding.source_path}.momentum does not match PyTorch"
                    )
                fields = ("weight", "bias", "running_mean", "running_var")
            else:  # pragma: no cover - all bindings are constructed internally
                raise AssertionError(f"unknown layer binding kind {binding.kind}")

            for field in fields:
                target_tensor = getattr(target_module, field)
                source_tensor = _tensor_from_source(
                    source, field, target_tensor, binding.source_path
                )
                target_tensor.copy_(source_tensor)
                target_key = f"{binding.target_path}.{field}"
                if target_key in covered_state_keys:
                    raise ValueError(f"target tensor mapped twice: {target_key}")
                covered_state_keys.add(target_key)
                mapping_records.append(
                    {
                        "source": f"{binding.source_path}.{field}",
                        "target": target_key,
                        "shape": list(target_tensor.shape),
                    }
                )

            if binding.kind == "batch_norm":
                counter_key = f"{binding.target_path}.num_batches_tracked"
                target_module.num_batches_tracked.zero_()
                covered_state_keys.add(counter_key)
                generated_state_keys.append(counter_key)

    target_state_keys = set(target.state_dict().keys())
    if covered_state_keys != target_state_keys:
        missing = sorted(target_state_keys - covered_state_keys)
        unexpected = sorted(covered_state_keys - target_state_keys)
        raise ValueError(
            f"target state mapping incomplete; missing={missing}, unexpected={unexpected}"
        )
    parameter_keys = set(dict(target.named_parameters()).keys())
    source_backed_keys = {record["target"] for record in mapping_records}
    if not parameter_keys.issubset(source_backed_keys):
        raise ValueError(
            f"target parameters lack Torch7 sources: {sorted(parameter_keys-source_backed_keys)}"
        )

    convolution_count = sum(binding.kind == "conv" for binding in bindings)
    batch_norm_count = sum(binding.kind == "batch_norm" for binding in bindings)
    return {
        "source_layer_count": len(source_modules),
        "source_convolution_count": convolution_count,
        "source_batch_norm_count": batch_norm_count,
        "source_tensor_count": len(mapping_records),
        "target_parameter_key_count": len(parameter_keys),
        "target_state_key_count": len(target_state_keys),
        "generated_state_keys": generated_state_keys,
        "all_target_state_keys_covered": True,
        "tensor_mapping": mapping_records,
    }


def _load_torch7_object(path: Path) -> Any:
    try:
        import torchfile  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "loading a .t7 checkpoint requires the optional 'torchfile' package"
        ) from error
    # torchfile 0.1.0 still references Python 2's xrange when the explicit
    # 64-bit-long reader is selected.  Torch7 Linux checkpoints use 64-bit longs.
    if not hasattr(torchfile, "xrange"):
        torchfile.xrange = range
    checkpoint = torchfile.load(str(path), force_8bytes_long=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Torch7 checkpoint root must be a table")
    source_model = checkpoint.get(b"model", checkpoint.get("model"))
    if source_model is None:
        raise ValueError("Torch7 checkpoint does not contain a 'model' field")
    return source_model


def import_torch7_coarsenet(
    torch7_path: str | Path,
    *,
    strict_official_sha: bool = True,
    source_lua_path: str | Path | None = None,
) -> tuple[TomNetCoarseNet, dict[str, Any]]:
    """Convert an official Torch7 CoarseNet into a CPU PyTorch model.

    Every convolution and batch-normalization tensor is checked for type,
    geometry, shape, uniqueness, and exhaustive source/target coverage.
    """

    path = Path(torch7_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    model_sha = _file_sha256(path)
    if strict_official_sha and model_sha != OFFICIAL_COARSENET_T7_SHA256:
        raise ValueError(
            f"unexpected CoarseNet.t7 SHA-256: {model_sha}; "
            f"expected {OFFICIAL_COARSENET_T7_SHA256}"
        )

    lua_sha: str | None = None
    lua_path_text: str | None = None
    if source_lua_path is not None:
        lua_path = Path(source_lua_path).expanduser().resolve()
        if not lua_path.is_file():
            raise FileNotFoundError(lua_path)
        lua_sha = _file_sha256(lua_path)
        lua_path_text = str(lua_path)
        if lua_sha != OFFICIAL_COARSENET_LUA_SHA256:
            raise ValueError(
                f"unexpected CoarseNet.lua SHA-256: {lua_sha}; "
                f"expected {OFFICIAL_COARSENET_LUA_SHA256}"
            )

    source_model = _load_torch7_object(path)
    model = TomNetCoarseNet()
    coverage = _copy_torch7_parameters(source_model, model)
    provenance: dict[str, Any] = {
        "format": CONVERTED_CHECKPOINT_FORMAT,
        "source_torch7_path": str(path),
        "source_torch7_sha256": model_sha,
        "source_repository_commit": OFFICIAL_TOMNET_COMMIT,
        "source_coarsenet_lua_path": lua_path_text,
        "source_coarsenet_lua_sha256": lua_sha or OFFICIAL_COARSENET_LUA_SHA256,
        "source_coarsenet_lua_sha256_verified": lua_sha is not None,
        "architecture": {
            "input": "RGB float [0,1]",
            "input_channels": 3,
            "batch_norm": True,
            "number_of_scales": 4,
            "output_scales_coarse_to_fine": [4, 3, 2, 1],
            "branches": ["flow", "mask_logits", "attenuation"],
            "flow_reference_width": 448.0,
            "feedback_mask_softmax_mode": TORCH7_FEEDBACK_SOFTMAX_MODE,
            "feedback_mask_softmax_domain": "flattened C*H*W independently per sample",
            "exported_mask_probability_domain": "two channels independently per pixel",
            "torch_object_class": TORCH_OBJECT_CLASS,
            "pytorch_object_mask_channel": OBJECT_MASK_CHANNEL,
        },
        "coverage": coverage,
    }
    model.provenance = provenance
    return model, provenance


def convert_torch7_coarsenet(
    torch7_path: str | Path,
    output_path: str | Path,
    *,
    strict_official_sha: bool = True,
    source_lua_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert ``CoarseNet.t7`` to a self-describing PyTorch checkpoint."""

    model, provenance = import_torch7_coarsenet(
        torch7_path,
        strict_official_sha=strict_official_sha,
        source_lua_path=source_lua_path,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CONVERTED_CHECKPOINT_FORMAT,
        "architecture": {
            "flow_reference_width": model.flow_reference_width,
            "validate_input_range": model.validate_input_range,
        },
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "provenance": provenance,
    }
    torch.save(payload, output)
    return provenance


def load_tomnet_coarsenet(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    *,
    strict_official_sha: bool = True,
    source_lua_path: str | Path | None = None,
) -> TomNetCoarseNet:
    """Load either the official ``.t7`` or a converted ``.pt`` checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    if path.suffix.lower() == ".t7":
        model, _ = import_torch7_coarsenet(
            path,
            strict_official_sha=strict_official_sha,
            source_lua_path=source_lua_path,
        )
        return model.to(map_location)

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch <2.0 does not expose weights_only.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("converted TOM-Net checkpoint must be a mapping")
    if payload.get("format") != CONVERTED_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported TOM-Net checkpoint format: {payload.get('format')!r}"
        )
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("converted checkpoint lacks architecture metadata")
    model = TomNetCoarseNet(
        flow_reference_width=float(architecture.get("flow_reference_width", 448.0)),
        validate_input_range=bool(architecture.get("validate_input_range", True)),
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("converted checkpoint lacks a state_dict")
    model.load_state_dict(state_dict, strict=True)
    provenance = payload.get("provenance")
    model.provenance = dict(provenance) if isinstance(provenance, Mapping) else None
    return model.to(map_location)


__all__ = [
    "BACKGROUND_MASK_CHANNEL",
    "CONVERTED_CHECKPOINT_FORMAT",
    "OBJECT_MASK_CHANNEL",
    "OFFICIAL_COARSENET_LUA_SHA256",
    "OFFICIAL_COARSENET_T7_SHA256",
    "OFFICIAL_TOMNET_COMMIT",
    "TORCH_BACKGROUND_CLASS",
    "TORCH_OBJECT_CLASS",
    "TORCH7_FEEDBACK_SOFTMAX_MODE",
    "TomNetCoarseNet",
    "TomNetCoarseOutput",
    "TomNetScaleOutput",
    "convert_torch7_coarsenet",
    "import_torch7_coarsenet",
    "load_tomnet_coarsenet",
]
