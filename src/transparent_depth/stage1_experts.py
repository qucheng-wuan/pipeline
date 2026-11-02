"""Arithmetic copied from the authenticated live RGB expert producer.

Path loading and provenance are handled by stage1.py; these functions preserve
letterbox, quantization, per-expert precision and native R5 reconstruction.
"""
from __future__ import annotations
import contextlib
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .stage1 import component_precision
from native_fusion.cache import native_r5, native_to_canvas, descriptor
from native_fusion.data import CANVAS_HW
from native_fusion.runtime import assemble_inputs

def _infer_moge(model: torch.nn.Module, rgb: np.ndarray, device: torch.device) -> np.ndarray:
    image = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    with torch.inference_mode():
        output = model.infer(
            image,
            resolution_level=9,
            refine_steps=3,
            use_fp16=True,
            apply_mask=False,
            fov_x=None,
        )
    depth = output["depth"].detach().float().cpu().numpy()
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"MoGe output shape {depth.shape} != RGB {rgb.shape[:2]}")
    invalid = ~np.isfinite(depth) | (depth <= 0)
    if invalid.any():
        depth = depth.copy()
        depth[invalid] = np.float32(1e-6)
    return np.ascontiguousarray(depth, dtype=np.float32)


def _extract_dino(
    model: torch.nn.Module, normalized_rgb: torch.Tensor, device: torch.device
) -> torch.Tensor:
    image = normalized_rgb.unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        features, trace = model(image, return_trace=False)
    if trace is not None or len(features) != 4:
        raise RuntimeError("unexpected frozen DINOv3 output")
    stacked = torch.stack(
        [item[0].detach().to(dtype=torch.float16, device="cpu") for item in features]
    )
    if tuple(stacked.shape) != (4, 1280, 24, 32):
        raise RuntimeError(f"unexpected DINOv3 cache shape: {tuple(stacked.shape)}")
    return stacked.contiguous()


def _tom_autocast(device: torch.device, amp_dtype: str):
    if amp_dtype == "none" or device.type != "cuda":
        return contextlib.nullcontext()
    if amp_dtype == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if amp_dtype == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    raise ValueError(f"unsupported TOM-Net inference amp dtype: {amp_dtype!r}")


def compact_letterbox(rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Match the existing R5 antialiased bilinear letterbox, then store uint8."""
    height, width = CANVAS_HW
    h0, w0 = rgb.shape[:2]
    scale = min(height / h0, width / w0)
    h1, w1 = min(height, max(1, round(h0 * scale))), min(width, max(1, round(w0 * scale)))
    top, left = (height - h1) // 2, (width - w1) // 2
    padding = (top, height - h1 - top, left, width - w1 - left)
    resized = F.interpolate(
        torch.from_numpy(rgb).permute(2, 0, 1)[None].float(),
        (h1, w1), mode="bilinear", align_corners=False, antialias=True,
    )[0].round().clamp(0, 255).to(torch.uint8)
    canvas = torch.zeros((3, height, width), dtype=torch.uint8)
    content = torch.zeros((1, height, width), dtype=torch.bool)
    canvas[:, top:top + h1, left:left + w1] = resized
    content[:, top:top + h1, left:left + w1] = True
    return canvas, content, padding


@torch.inference_mode()
def live_components(rgb_path: str | Path, experts: SimpleNamespace, *, return_trace=False):
    from moge_transparent_fusion.io_utils import letterbox_rgb,letterbox_rgb_raw,letterbox_scalar
    from moge_transparent_fusion.predicted_mask import probability_to_gate
    from moge_transparent_fusion.tomnet_infer import temperature_scaled_object_probability
    from moge_transparent_fusion.transnormal_teacher_r5 import infer_teacher_normal
    encoded=Path(rgb_path).read_bytes();digest=hashlib.sha256(encoded).hexdigest()
    with Image.open(io.BytesIO(encoded)) as im: rgb=np.asarray(im.convert('RGB')).copy()
    normalized,known,padding=letterbox_rgb(rgb,CANVAS_HW)
    tom_rgb,_,tom_padding=letterbox_rgb_raw(rgb,CANVAS_HW)
    raw,content,own_padding=compact_letterbox(rgb)
    if padding!=tom_padding or padding!=own_padding: raise ValueError('expert letterbox mismatch')
    device=experts.device
    with component_precision('high',cudnn_tf32=True):
        base_np=_infer_moge(experts.moge,rgb,device)
    base,base_padding=letterbox_scalar(base_np,CANVAS_HW,mode='bilinear')
    if padding!=base_padding: raise ValueError('MoGe letterbox mismatch')
    base=base.half().float()[None,None].to(device)
    features=_extract_dino(experts.dino,normalized,device).float()
    with _tom_autocast(device,experts.tom_metadata.get('inference_amp_dtype','none')):
        logits=experts.tomnet(tom_rgb[None].to(device)).final.mask_logits
    p=temperature_scaled_object_probability(logits,float(experts.calibration['temperature']))
    p=p.masked_fill(~known[None,None].to(device),0).half().float()
    gate=probability_to_gate(p,float(experts.calibration['gate_threshold']))
    teacher=infer_teacher_normal(experts.teacher,experts.teacher_contract,rgb,
        sample_id=experts.seed_identities.get(digest,'rgb-sha256/'+digest))
    if tuple(teacher['padding_tblr'])!=padding: raise ValueError('teacher geometry mismatch')
    normal=teacher['normal_cv'].half().float()[None].to(device)
    conf=teacher['confidence'].half().float()[None].to(device)
    with component_precision('highest',cudnn_tf32=True):
        head=experts.r5.model(base,p,known[None,None].float().to(device),
                             features[None].to(device),normal,conf,gate_mask=gate)
        depth=native_r5(torch.from_numpy(base_np).to(device),head['bounded_delta'][0].float(),gate[0])
        context_r5=native_to_canvas(depth)
    with component_precision('highest',cudnn_tf32=False):
        far=experts.anchor(features[None].to(device))[0].cpu()
    compact={'rgb':raw,'content':content,'far_depth':far,'probability':p[0].cpu(),
             'teacher_normal':normal[0].cpu(),'teacher_confidence':conf[0].cpu()}
    inputs = assemble_inputs(rgb=rgb,compact=compact,r5_native=depth.cpu(),
                context_descriptor=descriptor(features,content),context_r5=context_r5.cpu())

    if return_trace:
        return inputs, {"rgb":rgb,"moge_native":torch.from_numpy(base_np.copy()),
            "compact":{**compact,"r5_depth":head["depth"][0].detach().float().cpu()},
            "bounded_delta":head["bounded_delta"][0].detach().float().cpu(),
            "gate":gate[0].detach().float().cpu(),"context_descriptor":inputs["context_descriptor"],
            "context_r5_native":context_r5.cpu(),"r5_native":depth.cpu()}
    return inputs
