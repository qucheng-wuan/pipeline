"""Native V2 assembly/tiling arithmetic and portable runtime factory."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .cache import native_to_canvas
from .data import CANVAS_HW, target_geometry, canvas_to_native_patch
from .model import NativeGeometryFusion
SPATIAL_KEYS=("rgb","far_depth","r5_depth","teacher_normal","teacher_confidence","probability","content")

def assemble_inputs(*, rgb: np.ndarray, compact: dict[str, torch.Tensor],
                    r5_native: torch.Tensor, context_descriptor: torch.Tensor,
                    context_r5: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Assemble only RGB-derived fields; no labels, intrinsics or segmentation GT."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('RGB must be HxWx3 uint8')
    hw = tuple(rgb.shape[:2])
    if tuple(r5_native.shape) != hw:
        raise ValueError('native R5 geometry differs from RGB')
    if not torch.isfinite(r5_native).all() or not (r5_native>0).all():
        raise ValueError('native R5 is not finite-positive')
    result = {'rgb':torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255,
              'r5_depth':r5_native.float()[None].cpu(), 'content':torch.ones(1,*hw,dtype=torch.bool),
              'native_hw':torch.tensor(hw), 'crop_top_left':torch.tensor([0,0]),
              'padding_tblr':torch.tensor(target_geometry(hw)[1]),
              'context_rgb':compact['rgb'].float().cpu()/255,
              'context_far':compact['far_depth'].float().cpu(),
              'context_r5':native_to_canvas(r5_native.cpu()) if context_r5 is None else context_r5.float().cpu(),
              'context_probability':compact['probability'].float().cpu(),
              'context_teacher_normal':compact['teacher_normal'].float().cpu(),
              'context_teacher_confidence':compact['teacher_confidence'].float().cpu(),
              'context_content':compact['content'].float().cpu(),
              'context_descriptor':context_descriptor.float().cpu()}
    for key in ('far_depth','teacher_normal','teacher_confidence','probability'):
        result[key]=canvas_to_native_patch(compact[key].cpu(),hw)
    result['teacher_normal']=F.normalize(result['teacher_normal'],dim=0,eps=1e-6)
    return result


@torch.inference_mode()
def predict_native(model: NativeGeometryFusion, batch: dict[str, torch.Tensor], *,
                   tile_size: int = 384, halo: int = 16, amp: bool = True,
                   disable_detail: bool = False, disable_coarse: bool = False,
                   disable_region: bool = False) -> torch.Tensor:
    """Batch-one native inference with overlap context and one global encoding.

    Returns [1,1,H,W] on the model device. The local head has no spatial
    normalization; halo >= 8 covers its full receptive field including the
    high-pass filter. A conservative 16-pixel halo is the deployment default.
    """
    if batch['r5_depth'].ndim != 4 or batch['r5_depth'].shape[0] != 1:
        raise ValueError('native tiled prediction requires batch one')
    if halo < 8 or tile_size < 1:
        raise ValueError('tile_size must be positive and halo at least 8')
    device=next(model.parameters()).device
    work={k:v.to(device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
    h,w=work['r5_depth'].shape[-2:]
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
        context=model.encode_context(work)
        result=torch.empty(1,1,h,w,device=device,dtype=torch.float32)
        for y in range(0,h,tile_size):
            for x in range(0,w,tile_size):
                y1,x1=min(h,y+tile_size),min(w,x+tile_size)
                ey,ex=max(0,y-halo),max(0,x-halo)
                ey1,ex1=min(h,y1+halo),min(w,x1+halo)
                patch=dict(work)
                for key in SPATIAL_KEYS:
                    patch[key]=work[key][...,ey:ey1,ex:ex1]
                patch['crop_top_left']=work['crop_top_left']+work['crop_top_left'].new_tensor([[ey,ex]])
                pred=model(patch,context=context,disable_detail=disable_detail,
                           disable_coarse=disable_coarse,disable_region=disable_region)['depth']
                result[...,y:y1,x:x1]=pred[...,y-ey:y1-ey,x-ex:x1-ex]
    if not torch.isfinite(result).all() or not (result>0).all():
        raise FloatingPointError('native fusion produced invalid optical-Z depth')
    return result


def runtime_contract(manifest=None,residual_root=None):
    from transparent_depth.stage1 import Stage1Assets
    a=Stage1Assets()
    return {"schema_version":"portable_native_rgb_runtime_v1", "assets_sha256":a.manifest_sha256,"source_sha256":a.verify_sources(),"cache_required":False}

def load_rgb_runtime(checkpoint=None, **kwargs):
    from transparent_depth.stage1 import Stage1Predictor
    if kwargs.get("allow_audit_checkpoint"):
        raise ValueError("portable inference never bypasses weight authentication")
    p=Stage1Predictor(device=kwargs.get("device","cuda:0"))
    if checkpoint is not None and Path(checkpoint).resolve()!=p.assets.path("v2").resolve():
        raise ValueError("unexpected V2 checkpoint; configure the explicit asset manifest")
    return p
