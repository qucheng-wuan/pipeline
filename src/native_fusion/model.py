"""DINO metric calibration of a retained native MoGe/R5 surface.

The native depth is a multiplicative base, never a resized canvas prediction.
Global calibration sees RGB-derived context only. Surface updates are bounded,
and their final depth is explicitly supervised by camera-space normal losses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class Settings:
    context_channels: int = 48
    local_channels: int = 24
    coarse_height: int = 6
    coarse_width: int = 8
    max_scale_residual: float = 0.7
    max_coarse_correction: float = 0.25
    max_detail_raw: float = 0.08
    max_region_correction: float = 0.25
    region_threshold: float = 0.45281050229659175
    region_plateau: float = 0.8
    region_min_pixels: int = 16
    descriptor_channels: int = 1280
    enable_scene_recovery: bool = False
    max_scene_recovery: float = 1.3862943611198906


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    return (x * mask).sum((-2, -1), keepdim=True) / mask.sum((-2, -1), keepdim=True).clamp_min(1)


def _block(cin: int, cout: int, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation),
                         nn.GroupNorm(6, cout), nn.SiLU())


class _ChannelNorm(nn.Module):
    """Per-pixel normalization so native tiling cannot change image statistics."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: Tensor) -> Tensor:
        return self.norm(value.movedim(1, -1)).movedim(-1, 1)


def _local_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), _ChannelNorm(cout), nn.SiLU())


def _rgb(value: Tensor) -> Tensor:
    return value.float() / 255.0 if value.dtype == torch.uint8 else value.float()


def sample_canvas(canvas: Tensor, native_hw: Tensor, crop_top_left: Tensor,
                  padding_tblr: Tensor, output_hw: tuple[int, int]) -> Tensor:
    """Exactly map native pixel centres into the unletterboxed canvas.

    Grid coordinates follow align_corners=False and boundary-coordinate cameras.
    Border replication matches bilinear interpolation of a cropped canvas.
    """
    b, _, ch, cw = canvas.shape
    h, w = output_hw
    dtype = torch.float32
    yy = torch.arange(h, device=canvas.device, dtype=dtype).view(1, h, 1)
    xx = torch.arange(w, device=canvas.device, dtype=dtype).view(1, 1, w)
    native_hw = native_hw.to(device=canvas.device, dtype=dtype).reshape(b, 2)
    crop_top_left = crop_top_left.to(device=canvas.device, dtype=dtype).reshape(b, 2)
    pad = padding_tblr.to(device=canvas.device, dtype=dtype).reshape(b, 4)
    top, bottom, left, right = [pad[:, i].view(b, 1, 1) for i in range(4)]
    nh, nw = native_hw[:, 0].view(b, 1, 1), native_hw[:, 1].view(b, 1, 1)
    sy = (yy + crop_top_left[:, 0].view(b, 1, 1) + .5) * (ch - top - bottom) / nh
    sx = (xx + crop_top_left[:, 1].view(b, 1, 1) + .5) * (cw - left - right) / nw
    # Clamp to CONTENT centres, not the surrounding letterbox zeros.
    sy = torch.minimum(torch.maximum(sy, torch.full_like(sy, .5)), ch - top - bottom - .5) + top
    sx = torch.minimum(torch.maximum(sx, torch.full_like(sx, .5)), cw - left - right - .5) + left
    grid = torch.stack(((sx * (2.0 / cw) - 1).expand(b, h, w),
                        (sy * (2.0 / ch) - 1).expand(b, h, w)), -1)
    return F.grid_sample(canvas.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)


class SceneScaleHead(nn.Module):
    """RGB visual descriptor and metric-anchor context -> log-scale residual."""
    def __init__(self, descriptor_channels: int, context_channels: int):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(descriptor_channels), nn.Linear(descriptor_channels, 128), nn.SiLU())
        self.regressor = nn.Sequential(nn.Linear(128 + context_channels + 4, 128), nn.SiLU(),
                                       nn.Linear(128, 1))
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, descriptor: Tensor, features: Tensor, statistics: Tensor) -> Tensor:
        return self.regressor(torch.cat((self.visual(descriptor), features, statistics), -1))


def component_scale_field(raw: Tensor, probability: Tensor, content: Tensor, settings: Settings) -> Tensor:
    """One bounded log scalar per RGB-predicted connected region.

    Components use the frozen TOM calibration threshold, not labelled masks.
    Their full-context identities are shared by all native crops/tiles. The
    smooth gate has a constant high-confidence plateau; only that interior has
    exact scalar-only normal preservation. Connected regions are not instances.
    """
    from scipy.ndimage import label
    p = probability.detach().float()
    support = (p >= settings.region_threshold) & content.bool()
    raw = F.interpolate(raw.float(),p.shape[-2:],mode='bilinear',align_corners=False)
    transition = ((p-settings.region_threshold)/(settings.region_plateau-settings.region_threshold)).clamp(0,1)
    gate = transition.square()*(3-2*transition)*content.float()
    fields=[]
    for index in range(p.shape[0]):
        labels,n = label(support[index,0].cpu().numpy())
        ids=torch.as_tensor(labels,device=raw.device,dtype=torch.long).flatten()
        counts=torch.bincount(ids,minlength=n+1)
        usable=(counts>=settings.region_min_pixels)
        usable[0]=False
        weight=gate[index,0].flatten()*usable[ids]
        denominator=torch.zeros(n+1,device=raw.device).scatter_add(0,ids,weight)
        numerator=torch.zeros(n+1,device=raw.device).scatter_add(0,ids,raw[index,0].flatten()*weight)
        means=settings.max_region_correction*(numerator/denominator.clamp_min(1e-6)).tanh()
        fields.append((means[ids]*weight).view_as(p[index]))
    return torch.stack(fields)


class NativeGeometryFusion(nn.Module):
    """A geometry-preserving base with learned metric and surface corrections."""
    context_keys = ("context_rgb", "context_far", "context_r5", "context_probability",
                    "context_teacher_normal", "context_teacher_confidence", "context_content",
                    "context_descriptor")

    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        c, lc = self.settings.context_channels, self.settings.local_channels
        self.context_encoder = nn.Sequential(_block(12, c), _block(c, c, 2), _block(c, c, 4))
        self.coarse_head = nn.Conv2d(c, 1, 1)
        self.region_head = nn.Conv2d(c, 1, 1)
        self.scale_head = SceneScaleHead(self.settings.descriptor_channels, c)
        if self.settings.enable_scene_recovery:
            self.scene_recovery_head = SceneScaleHead(self.settings.descriptor_channels, c)
        self.detail_head = nn.Sequential(_local_block(12, lc), _local_block(lc, lc), nn.Conv2d(lc, 1, 3, padding=1))
        nn.init.zeros_(self.coarse_head.weight)
        nn.init.zeros_(self.coarse_head.bias)
        nn.init.zeros_(self.region_head.weight)
        nn.init.zeros_(self.region_head.bias)
        nn.init.zeros_(self.detail_head[-1].weight)
        nn.init.zeros_(self.detail_head[-1].bias)

    def config(self) -> dict:
        return asdict(self.settings)

    def encode_context(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        far = batch["context_far"].float().clamp_min(1e-4)
        base = batch["context_r5"].float().clamp_min(1e-4)
        content = batch["context_content"].float()
        probability = batch["context_probability"].float().clamp(0, 1)
        f, g = far.log(), base.log()
        g_mean = masked_mean(g, content)
        centered_g = (g - g_mean) * content
        f_mean = masked_mean(f, content)
        centered_f = (f - f_mean) * content
        # A soft RGB-derived background weight; never use the GT segmentation.
        anchor_weight = content * (1.0 - .8 * probability)
        anchor_scale = masked_mean(f - g, anchor_weight)
        normal = F.normalize(batch["context_teacher_normal"].float(), dim=1, eps=1e-6)
        inputs = torch.cat((_rgb(batch["context_rgb"]), f * content, centered_g,
                            centered_g - centered_f, probability, normal,
                            batch["context_teacher_confidence"].float(), content), 1)
        hw = (max(4, f.shape[-2] // 4), max(4, f.shape[-1] // 4))
        features = self.context_encoder(F.interpolate(inputs, hw, mode="area"))
        feature_mask = F.interpolate(content, hw, mode="area")
        pooled = masked_mean(features.float(), feature_mask).flatten(1)
        statistics = torch.cat((f_mean, masked_mean(centered_f.square(), content).sqrt(),
                                masked_mean(centered_g.square(), content).sqrt(),
                                masked_mean(probability, content)), 1).flatten(1)
        scale_residual = self.settings.max_scale_residual * self.scale_head(
            batch["context_descriptor"].float(), pooled, statistics).float().tanh().view(-1, 1, 1, 1)
        coarse = F.adaptive_avg_pool2d(self.coarse_head(features),
                                      (self.settings.coarse_height, self.settings.coarse_width))
        coarse = self.settings.max_coarse_correction * F.interpolate(
            coarse.float().tanh(), f.shape[-2:], mode="bilinear", align_corners=False)
        coarse = (coarse - masked_mean(coarse, content)) * content
        # Preserve the zero mean while enforcing the advertised correction bound.
        bound = coarse.abs().flatten(1).amax(1).view(-1, 1, 1, 1)
        coarse = coarse / (bound / self.settings.max_coarse_correction).clamp_min(1.0)
        region = component_scale_field(self.region_head(features),probability,content,self.settings)
        scale = anchor_scale + scale_residual
        context_depth = base * torch.exp(scale + coarse + region)
        recovery = torch.zeros_like(scale)
        if self.settings.enable_scene_recovery:
            initial_log_depth = context_depth.clamp_min(1e-4).log()
            mean_depth = masked_mean(initial_log_depth, content)
            recovery_statistics = torch.cat((mean_depth,
                masked_mean((initial_log_depth-mean_depth).square(),content).sqrt(),
                f_mean,masked_mean(probability,content)),1).flatten(1)
            recovery = self.settings.max_scene_recovery * self.scene_recovery_head(
                batch['context_descriptor'].float(),pooled,recovery_statistics).float().tanh().view(-1,1,1,1)
            scale = scale + recovery
            context_depth = context_depth * recovery.exp()
        return {"context_depth": context_depth, "coarse_canvas": coarse, "scale_log": scale,
                "region_canvas": region,
                "anchor_scale_log": anchor_scale, "scale_residual": scale_residual,
                "scene_recovery_log_scale": recovery,
                "base_log_mean": g_mean, "context_features": pooled}

    def _detail(self, batch: dict[str, Tensor], coarse: Tensor, context: dict[str, Tensor]) -> Tensor:
        log_base = batch["r5_depth"].float().clamp_min(1e-4).log()
        gx = F.pad(log_base[..., 1:] - log_base[..., :-1], (0, 1))
        gy = F.pad(log_base[..., 1:, :] - log_base[..., :-1, :], (0, 0, 0, 1))
        normal = F.normalize(batch["teacher_normal"].float(), dim=1, eps=1e-6)
        inputs = torch.cat((_rgb(batch["rgb"]), log_base - context["base_log_mean"], normal,
                            batch["teacher_confidence"].float(), batch["probability"].float(),
                            coarse, gx, gy), 1)
        raw = self.settings.max_detail_raw * self.detail_head(inputs).float().tanh()
        # Remove the locally constant component; this branch cannot simply learn
        # a uniform scalar instead of surface detail. Boundary means are not a
        # hard image-global gauge constraint and are reported as such.
        smooth = F.avg_pool2d(F.pad(raw, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
        return raw - smooth

    def forward(self, batch: dict[str, Tensor], *, disable_detail: bool = False,
                disable_coarse: bool = False, disable_region: bool = False,
                context: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        context = self.encode_context(batch) if context is None else context
        base = batch["r5_depth"].float().clamp_min(1e-4)
        coarse = sample_canvas(context["coarse_canvas"], batch["native_hw"], batch["crop_top_left"],
                               batch["padding_tblr"], tuple(base.shape[-2:]))
        if disable_coarse:
            coarse = coarse * 0
        region = sample_canvas(context['region_canvas'],batch['native_hw'],batch['crop_top_left'],
                               batch['padding_tblr'],tuple(base.shape[-2:]))
        if disable_region:
            region = region * 0
        detail = self._detail(batch, coarse + region, context)
        if disable_detail:
            detail = detail * 0
        content = batch.get("content", torch.ones_like(base)).float()
        detail = detail * content
        depth = base * torch.exp(context["scale_log"] + coarse + region + detail)
        return {**context, "depth": depth, "base_depth": base,
                "low_correction": coarse, "region_correction": region,"detail_correction": detail}

    def scale_only_parameters(self):
        return self.scale_head.parameters()
