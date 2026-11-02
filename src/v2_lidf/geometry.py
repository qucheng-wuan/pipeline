"""Pinhole geometry using metric Z depth and integer pixel coordinates."""
from typing import Dict

import torch
import torch.nn.functional as F


def pixel_rays(intrinsics: torch.Tensor, height: int, width: int,
               normalize: bool = True) -> torch.Tensor:
    """Return [B,H,W,3] rays; pixels are (u,v)=(0,0),...,(W-1,H-1)."""
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B,3,3]")
    if not torch.isfinite(intrinsics).all() or (intrinsics[:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError("intrinsics must be finite with positive focal lengths")
    # LIDF's original ray model assumes an ordinary zero-skew pinhole camera.
    if (intrinsics[:, 0, 1].abs() > 1e-6).any() or (intrinsics[:, 1, 0].abs() > 1e-6).any():
        raise ValueError("LIDF requires zero-skew intrinsics")
    bottom = intrinsics.new_tensor([0., 0., 1.]).expand(intrinsics.shape[0], -1)
    if not torch.allclose(intrinsics[:, 2], bottom, atol=1e-6, rtol=0.):
        raise ValueError("intrinsics must have homogeneous bottom row [0,0,1]")
    v, u = torch.meshgrid(torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype),
                          torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype),
                          indexing="ij")
    fx, fy = intrinsics[:, 0, 0, None, None], intrinsics[:, 1, 1, None, None]
    cx, cy = intrinsics[:, 0, 2, None, None], intrinsics[:, 1, 2, None, None]
    x, y = (u[None] - cx) / fx, (v[None] - cy) / fy
    rays = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    return F.normalize(rays, dim=-1) if normalize else rays


def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Return [B,3,H,W] XYZ. GT and predicted depth must use the same K."""
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape [B,1,H,W]")
    if depth.shape[0] != intrinsics.shape[0]:
        raise ValueError("depth and intrinsics batch sizes differ")
    rays = pixel_rays(intrinsics, depth.shape[2], depth.shape[3], normalize=False)
    return rays.permute(0, 3, 1, 2) * depth


def surface_normals(xyz: torch.Tensor) -> torch.Tensor:
    """Official LIDF forward-difference normals, shape [B,3,H,W]."""
    dx = F.pad(xyz[:, :, :, 1:] - xyz[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(xyz[:, :, 1:, :] - xyz[:, :, :-1, :], (0, 0, 0, 1))
    normal = torch.cross(dx, dy, dim=1)
    return normal / (torch.linalg.vector_norm(normal, dim=1, keepdim=True) + 1e-8)


def occupied_voxels(points: torch.Tensor, resolution: int = 8) -> Dict[str, torch.Tensor]:
    """Official fixed LIDF volume and half-voxel margin, never GT augmented.

    Base bounds are [-1,-1,0]..[1,1,2] metres. At resolution 8 the cell
    side is .25m and the expanded volume is [-1.125,-1.125,-.125]..
    [1.125,1.125,2.125]. Out-of-volume and nonfinite points are excluded.
    """
    if points.ndim != 2 or points.shape[1] != 3 or resolution <= 0:
        raise ValueError("points must be [N,3] and resolution must be positive")
    side = 2.0 / resolution
    low = points.new_tensor([-1., -1., 0.]) - .5 * side
    high = points.new_tensor([1., 1., 2.]) + .5 * side
    finite = torch.isfinite(points).all(dim=1)
    inside = finite & (points >= low).all(dim=1) & (points < high).all(dim=1)
    point_indices = torch.nonzero(inside, as_tuple=False).flatten()
    selected = points[point_indices]
    coords = torch.floor((selected - low) / side).long()
    unique, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
    bounds_low = low + unique * side
    bounds = torch.cat((bounds_low, bounds_low + side), dim=1).float().contiguous()
    centers = low + (coords + .5) * side
    return {"bounds": bounds, "inverse": inverse, "point_indices": point_indices,
            "relative_points": selected - centers, "side": points.new_tensor(side)}
