"""Exact frozen native R5 assembly arithmetic, independent of dataset location."""
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import CANVAS_HW, target_geometry, sha256_file

def descriptor(features: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    """Mean of normalized layer-31 features weighted by valid token coverage."""
    weights = F.adaptive_avg_pool2d(content.float()[None], features.shape[-2:])[0]
    return (features[-1].float()*weights).sum((-1,-2))/weights.sum().clamp_min(1)


def native_r5(base: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if base.ndim != 2 or delta.shape != (1,*CANVAS_HW) or gate.shape != delta.shape:
        raise ValueError('native R5 component geometry mismatch')
    hw = tuple(base.shape)
    (rh,rw), (top,_,left,_) = target_geometry(hw)
    def restore(v):
        return F.interpolate(v[None,:,top:top+rh,left:left+rw].float(), hw,
                             mode='bilinear', align_corners=False)[0,0]
    d,g = restore(delta), restore(gate)
    return torch.where(g>0, base.float()*torch.exp(d*g), base.float())


def native_to_canvas(value: torch.Tensor) -> torch.Tensor:
    (rh,rw), (top,bottom,left,right) = target_geometry(tuple(value.shape[-2:]))
    resized = F.interpolate(value[None,None].float(), (rh,rw), mode='bilinear',
                            align_corners=False, antialias=True)[0]
    return F.pad(resized, (left,right,top,bottom))


def producer_contract(manifest):
    from transparent_depth.stage1 import Stage1Assets
    assets=Stage1Assets()
    return {"schema_version":"portable_native_rgb_producer_v1", "manifest_sha256":sha256_file(manifest), "stage1_assets_sha256":assets.manifest_sha256, "contains_ground_truth":False}
