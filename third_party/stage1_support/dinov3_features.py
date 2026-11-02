"""Strict DINOv3 H+/16 loading and four-layer dense features.

The training implementation uses the pinned Meta DINOv3 source layout because the
selected-head FAR adapter is already verified against its packed QKV attention.  The
approved user checkpoint is the official Hugging Face conversion, however.  For a
``.safetensors`` checkpoint we therefore reverse Hugging Face's published conversion
map in memory and require every checkpoint tensor to be consumed.  No converted
checkpoint is written to disk.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
HF_H16PLUS_TENSOR_COUNT = 615


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hf_h16plus_config(weight: Path) -> None:
    config_path = weight.parent / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Hugging Face DINOv3 config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "model_type": "dinov3_vit",
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "num_register_tokens": 4,
        "patch_size": 16,
        "use_gated_mlp": True,
        "hidden_act": "silu",
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "proj_bias": True,
        "rope_theta": 100.0,
        "pos_embed_rescale": 2.0,
        "pos_embed_shift": None,
        "pos_embed_jitter": None,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Hugging Face DINOv3 identity mismatch: {mismatches}")


def _load_hf_safetensors_into_meta_layout(model: nn.Module, weight: Path) -> None:
    """Reverse the official Meta-to-HF key conversion with exhaustive checks."""
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - exercised by deployment environment
        raise RuntimeError("loading the approved HF checkpoint requires safetensors") from error

    _validate_hf_h16plus_config(weight)
    hf_state = load_file(str(weight), device="cpu")
    if len(hf_state) != HF_H16PLUS_TENSOR_COUNT:
        raise RuntimeError(
            f"unexpected Hugging Face tensor count: {len(hf_state)} "
            f"!= {HF_H16PLUS_TENSOR_COUNT}"
        )
    target = model.state_dict()
    used_hf: set[str] = set()
    replaced_meta: set[str] = set()

    def assign(hf_key: str, meta_key: str, *, squeeze_mask: bool = False) -> None:
        if hf_key not in hf_state or meta_key not in target:
            raise RuntimeError(f"missing conversion key: {hf_key} -> {meta_key}")
        value = hf_state[hf_key].squeeze(1) if squeeze_mask else hf_state[hf_key]
        if tuple(value.shape) != tuple(target[meta_key].shape):
            raise RuntimeError(
                f"shape mismatch for {hf_key} -> {meta_key}: "
                f"{tuple(value.shape)} != {tuple(target[meta_key].shape)}"
            )
        target[meta_key] = value
        used_hf.add(hf_key)
        replaced_meta.add(meta_key)

    assign("embeddings.cls_token", "cls_token")
    assign("embeddings.mask_token", "mask_token", squeeze_mask=True)
    assign("embeddings.register_tokens", "storage_tokens")
    assign("embeddings.patch_embeddings.weight", "patch_embed.proj.weight")
    assign("embeddings.patch_embeddings.bias", "patch_embed.proj.bias")
    assign("norm.weight", "norm.weight")
    assign("norm.bias", "norm.bias")

    for index in range(32):
        hf_prefix = f"layer.{index}."
        meta_prefix = f"blocks.{index}."
        for suffix in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            assign(hf_prefix + suffix, meta_prefix + suffix)

        q_weight_key = hf_prefix + "attention.q_proj.weight"
        k_weight_key = hf_prefix + "attention.k_proj.weight"
        v_weight_key = hf_prefix + "attention.v_proj.weight"
        packed_weight_key = meta_prefix + "attn.qkv.weight"
        packed_weight = torch.cat(
            (hf_state[q_weight_key], hf_state[k_weight_key], hf_state[v_weight_key]), dim=0
        )
        if tuple(packed_weight.shape) != tuple(target[packed_weight_key].shape):
            raise RuntimeError(f"packed QKV weight shape mismatch at block {index}")
        target[packed_weight_key] = packed_weight
        used_hf.update((q_weight_key, k_weight_key, v_weight_key))
        replaced_meta.add(packed_weight_key)

        q_bias_key = hf_prefix + "attention.q_proj.bias"
        v_bias_key = hf_prefix + "attention.v_proj.bias"
        forbidden_k_bias = hf_prefix + "attention.k_proj.bias"
        if forbidden_k_bias in hf_state:
            raise RuntimeError("the locked H+/16 checkpoint must not contain K bias")
        packed_bias_key = meta_prefix + "attn.qkv.bias"
        q_bias = hf_state[q_bias_key]
        packed_bias = torch.cat((q_bias, torch.zeros_like(q_bias), hf_state[v_bias_key]), dim=0)
        if tuple(packed_bias.shape) != tuple(target[packed_bias_key].shape):
            raise RuntimeError(f"packed QKV bias shape mismatch at block {index}")
        target[packed_bias_key] = packed_bias
        used_hf.update((q_bias_key, v_bias_key))
        replaced_meta.add(packed_bias_key)

        assign(hf_prefix + "attention.o_proj.weight", meta_prefix + "attn.proj.weight")
        assign(hf_prefix + "attention.o_proj.bias", meta_prefix + "attn.proj.bias")
        assign(hf_prefix + "layer_scale1.lambda1", meta_prefix + "ls1.gamma")
        assign(hf_prefix + "layer_scale2.lambda1", meta_prefix + "ls2.gamma")
        for hf_name, meta_name in (
            ("gate_proj", "w1"),
            ("up_proj", "w2"),
            ("down_proj", "w3"),
        ):
            assign(
                hf_prefix + f"mlp.{hf_name}.weight",
                meta_prefix + f"mlp.{meta_name}.weight",
            )
            assign(
                hf_prefix + f"mlp.{hf_name}.bias",
                meta_prefix + f"mlp.{meta_name}.bias",
            )

    unused_hf = sorted(set(hf_state) - used_hf)
    constant_meta = sorted(set(target) - replaced_meta)
    expected_constants = [
        "rope_embed.periods",
        *[f"blocks.{i}.attn.qkv.bias_mask" for i in range(32)],
    ]
    if unused_hf:
        raise RuntimeError(f"unconsumed Hugging Face checkpoint tensors: {unused_hf}")
    if constant_meta != sorted(expected_constants):
        raise RuntimeError(f"unexpected Meta-layout constants: {constant_meta}")
    model.load_state_dict(target, strict=True)
    model._transdepth_checkpoint_format = "hf_safetensors_reverse_official_map"
    model._transdepth_hf_tensors_consumed = len(used_hf)


def load_dinov3_h16plus(
    repo_dir: str | Path,
    checkpoint: str | Path,
    expected_sha256: str,
) -> nn.Module:
    repo = Path(repo_dir).expanduser().resolve(strict=True)
    weight = Path(checkpoint).expanduser().resolve(strict=True)
    # Vendored source bytes are authenticated by Stage1Assets.verify_sources().
    if len(expected_sha256) != 64 or _sha256(weight) != expected_sha256.lower():
        raise RuntimeError("DINOv3 checkpoint SHA256 mismatch")
    sys.path.insert(0, str(repo))
    try:
        from dinov3.hub.backbones import dinov3_vith16plus

        model = dinov3_vith16plus(pretrained=False)
    finally:
        if sys.path[0] == str(repo):
            sys.path.pop(0)
    if weight.suffix == ".safetensors":
        _load_hf_safetensors_into_meta_layout(model, weight)
    else:
        state = torch.load(weight, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "blocks.0.attn.qkv.weight" not in state:
            raise RuntimeError("checkpoint is not a native DINOv3 backbone state_dict")
        model.load_state_dict(state, strict=True)
        model._transdepth_checkpoint_format = "meta_pth_state_dict"
        del state
    verify_h16plus_structure(model)
    return model.requires_grad_(False).eval()


def verify_h16plus_structure(model: nn.Module) -> None:
    expected = {
        "embed_dim": 1280,
        "n_blocks": 32,
        "num_heads": 20,
        "patch_size": 16,
        "n_storage_tokens": 4,
    }
    for name, value in expected.items():
        if getattr(model, name, None) != value:
            raise RuntimeError(f"DINOv3 identity mismatch: {name}")
    if len(model.blocks) != 32 or tuple(model.storage_tokens.shape) != (1, 4, 1280):
        raise RuntimeError("DINOv3 block or storage-token identity mismatch")
    if model.rope_embed.dtype != torch.float32 or model.rope_embed.normalize_coords != "separate":
        raise RuntimeError("DINOv3 RoPE identity mismatch")
    rope = model.rope_embed
    if (
        rope.base != 100
        or rope.rescale_coords != 2
        or rope.shift_coords is not None
        or rope.jitter_coords is not None
    ):
        raise RuntimeError("DINOv3 RoPE parameter mismatch")
    for block in model.blocks:
        attention = block.attn
        if attention.num_heads != 20 or tuple(attention.qkv.weight.shape) != (3840, 1280):
            raise RuntimeError("DINOv3 packed QKV shape mismatch")
        if attention.qkv.bias is None or not hasattr(attention.qkv, "bias_mask"):
            raise RuntimeError("DINOv3 masked key-bias projection is missing")
        bias_mask = attention.qkv.bias_mask
        if not (
            bool((bias_mask[:1280] == 1).all())
            and bool((bias_mask[1280:2560] == 0).all())
            and bool((bias_mask[2560:] == 1).all())
        ):
            raise RuntimeError("DINOv3 key-bias mask mismatch")
        if block.norm1.eps != 1e-5 or block.norm2.eps != 1e-5:
            raise RuntimeError("DINOv3 LayerNorm epsilon mismatch")
        mlp = block.mlp
        if not all(hasattr(mlp, name) for name in ("w1", "w2", "w3")):
            raise RuntimeError("DINOv3 H+/16 must use SwiGLU")
        if (
            mlp.w1.out_features != 5120
            or mlp.w2.out_features != 5120
            or mlp.w3.in_features != 5120
        ):
            raise RuntimeError("DINOv3 SwiGLU hidden width mismatch")


class DinoV3Features(nn.Module):
    feature_blocks = (7, 15, 23, 31)

    def __init__(
        self, backbone: nn.Module, feature_blocks: tuple[int, ...] = (7, 15, 23, 31)
    ) -> None:
        super().__init__()
        verify_h16plus_structure(backbone)
        if feature_blocks != self.feature_blocks:
            raise ValueError("this locked H+/16 protocol requires feature blocks 7,15,23,31")
        self.backbone = backbone.requires_grad_(False).eval()
        self.adapter: FARSelfAttention | None = None
        self.activation_checkpoint_suffix = True

    def enable_lora(self, **kwargs):
        raise RuntimeError("the Stage1 expert is frozen; FAR LoRA is not part of this model")

    def train(self, mode: bool = True) -> DinoV3Features:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self, rgb: Tensor, *, return_trace: bool = False
    ) -> tuple[tuple[Tensor, ...], AttentionTrace | None]:
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 384, 512):
            raise ValueError("DINOv3 input must be [B,3,384,512]")
        self.backbone.eval()
        adapter_block = (
            self.adapter.selected_block if self.adapter is not None else len(self.backbone.blocks)
        )
        if return_trace and self.adapter is None:
            raise RuntimeError("attention trace requires an enabled LoRA adapter")
        if self.adapter is not None:
            self.adapter.set_capture_trace(return_trace)
        with torch.no_grad():
            tokens, (grid_h, grid_w) = self.backbone.prepare_tokens_with_masks(rgb, masks=None)
            rope = self.backbone.rope_embed(H=grid_h, W=grid_w)
        features: list[Tensor] = []
        for index, block in enumerate(self.backbone.blocks):
            if index < adapter_block:
                with torch.no_grad():
                    tokens = block(tokens, rope)
            elif (
                self.training
                and self.activation_checkpoint_suffix
                and index > adapter_block
            ):
                def run_block(inputs: Tensor, current_block: nn.Module = block) -> Tensor:
                    return current_block(inputs, rope)

                tokens = activation_checkpoint(run_block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens, rope)
            if index in self.feature_blocks:
                normalized = self.backbone.norm(tokens)
                patches = normalized[:, 1 + self.backbone.n_storage_tokens :]
                patches = patches.transpose(1, 2).reshape(
                    rgb.shape[0], 1280, grid_h, grid_w
                )
                features.append(patches)
        if len(features) != 4 or (grid_h, grid_w) != (24, 32):
            raise RuntimeError("unexpected DINOv3 dense feature geometry")
        trace = self.adapter.consume_trace() if return_trace and self.adapter is not None else None
        return tuple(features), trace
