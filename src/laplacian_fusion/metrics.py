"""Efficient selection metrics with the original V2 fixed GT support.

Final evaluation uses the original complete evaluator. This computes only the
three selection/reporting metrics and reuses the reference across candidates.
"""
import sys
from pathlib import Path
import torch

from native_fusion.geometry import surface_normal_reference, depth_to_normals, normal_angles


class SelectionMetrics:
    def __init__(self, batch):
        self.batch = batch
        self.valid = batch['valid'].bool() & batch['content'].bool() & torch.isfinite(batch['target']) & (batch['target'] > 0)
        self.truth = torch.where(self.valid, batch['target'], torch.ones_like(batch['target'])).double()
        self.geometry_valid = self.valid & batch['normal_camera_valid'].bool().view(-1,1,1,1)
        self.ref = surface_normal_reference(self.truth, batch['intrinsics'], self.geometry_valid,
            reference_normal=batch.get('target_normal'), reference_normal_valid=batch.get('normal_valid'),
            reference_available=batch['normal_reference_available'], region_labels=batch['transparent']) if bool(batch['normal_camera_valid'].any()) else None
        self.regions = {'all': self.valid, 'transparent': self.valid & batch['transparent'],
                        'background': self.valid & ~batch['transparent']}

    @torch.no_grad()
    def __call__(self, prediction):
        if bool((self.valid & (~torch.isfinite(prediction) | (prediction <= 0))).any()):
            raise ValueError('invalid prediction')
        pred = torch.where(self.valid, prediction, torch.ones_like(prediction)).double()
        if self.ref is not None:
            normal, _ = depth_to_normals(pred, self.batch['intrinsics'], self.geometry_valid,
                reference_depth=self.truth, region_labels=self.batch['transparent'])
            angle = normal_angles(normal, self.ref.normal)
        error = (pred-self.truth).abs()
        result = {}
        for name, mask in self.regions.items():
            support = mask & self.ref.valid if self.ref is not None else torch.zeros_like(mask)
            count, normal_count = int(mask.sum()), int(support.sum())
            result[name] = {'mae_mm': float(error[mask].mean()*1000) if count else None,
                'abs_rel': float((error/self.truth)[mask].mean()) if count else None,
                'normal_mean_deg': float(angle[support].mean()) if normal_count else None,
                'pixels': count, 'normal_pixels': normal_count}
        return result
