"""Native optical-Z data with a separate full-image context and authenticated assets.

All intrinsics use pixel-boundary coordinates: the centre of array pixel (v,u)
is (u+.5,v+.5). Synthetic K is derived from released FOV; real YAML principal
points are shifted by +.5 from integer-centre coordinates. No guessed K is used.
Renderer normals are deliberately not used: geometry targets come from GT Z/K.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import ast
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

CANVAS_HW = (384, 512)
DEFAULT_ROOT = Path(os.environ.get('TD_STAGE1_NATIVE_ROOT', 'artifacts/stage1_training/native')).resolve()
DEFAULT_MANIFEST = DEFAULT_ROOT / 'manifests/dataset.jsonl'
EXPECTED_COUNTS = {'train': 45453, 'val': 704, 'test': 521}
SCHEMA = 'native_metric_normal_record_v2'
OFFICIAL_PROCESSOR = Path(__file__).resolve().parents[2] / 'third_party/cleargrasp/data_processing_script.py'


@lru_cache(maxsize=1)
def official_legacy_fov() -> tuple[float, float, str]:
    """Read the released camera defaults, without importing its processing code."""
    tree = ast.parse(OFFICIAL_PROCESSOR.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate_cos_matrix')
    defaults = dict(zip([a.arg for a in function.args.args][-len(function.args.defaults):],
                        [ast.literal_eval(n) for n in function.args.defaults]))
    return float(defaults['fov_x']), float(defaults['fov_y']), sha256_file(OFFICIAL_PROCESSOR)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_asset(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(os.environ.get('TD_DATA_ROOT', '.')).resolve() / p
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_manifest(path: str | Path = DEFAULT_MANIFEST, split: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if any(r.get('schema_version') != SCHEMA for r in rows):
        raise ValueError('not a native fusion v2 manifest')
    if len({r['cache_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate cache identity')
    return rows if split is None else [r for r in rows if r['split'] == split]


def load_tensor_asset(asset: Mapping[str, Any], *, verify: bool = True) -> dict[str, Any]:
    p = resolve_asset(asset['path'])
    encoded = p.read_bytes()
    if 'bytes' in asset and len(encoded) != asset['bytes']:
        raise ValueError(f'asset size changed: {p}')
    expected = asset.get('sha256', asset.get('cache_sha256'))
    if verify and expected and hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError(f'asset hash changed: {p}')
    return torch.load(io.BytesIO(encoded), map_location='cpu', weights_only=True)


def target_geometry(native_hw: tuple[int, int], canvas_hw: tuple[int, int] = CANVAS_HW):
    h, w = native_hw
    ch, cw = canvas_hw
    scale = min(ch / h, cw / w)
    rh, rw = min(ch, max(1, round(h * scale))), min(cw, max(1, round(w * scale)))
    top, left = (ch-rh)//2, (cw-rw)//2
    return (rh, rw), (top, ch-rh-top, left, cw-rw-left)


def camera_from_rgb(rgb_path: str | Path, native_hw: tuple[int, int]) -> dict[str, Any]:
    """Read camera metadata only; no target image, instance or mask is read."""
    p = resolve_asset(rgb_path)
    h, w = native_hw
    if '-transparent-rgb-img' in p.name:
        source = p.parent / 'camera_intrinsics.yaml'
        fields = {}
        for line in source.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                key, val = line.split(':', 1)
                fields[key.strip()] = val.strip().strip("'\"")
        if (int(fields['yres']), int(fields['xres'])) != native_hw:
            raise ValueError(f'camera YAML resolution mismatch: {p}')
        k = [[float(fields['fx']), 0., float(fields['cx'])+.5],
             [0., float(fields['fy']), float(fields['cy'])+.5], [0., 0., 1.]]
        origin = 'released_real_yaml_integer_centres_plus_half_pixel'
    else:
        frame = p.name.split('-rgb', 1)[0]
        source = p.parent.parent / 'json-files' / f'{frame}-masks.json'
        meta = json.loads(source.read_text())
        if 'image' in meta and (int(meta['image']['height_px']), int(meta['image']['width_px'])) != native_hw:
            raise ValueError(f'synthetic metadata resolution mismatch: {p}')
        fov = meta.get('camera', {}).get('field_of_view')
        if fov is None:
            fx, fy, default_hash = official_legacy_fov()
            origin = 'official_default_fov_for_legacy_missing_metadata'
        else:
            fx, fy = float(fov['x_axis_rads']), float(fov['y_axis_rads'])
            origin = 'released_synthetic_per_frame_fov_boundary_centres'
        if not (0 < fx < math.pi and 0 < fy < math.pi):
            raise ValueError('invalid released camera FOV')
        k = [[w/(2*math.tan(fx/2)), 0., w/2],
             [0., h/(2*math.tan(fy/2)), h/2], [0., 0., 1.]]
    result = {'intrinsics': k, 'source': str(source), 'sha256': sha256_file(source),
              'source_kind': origin, 'pixel_center_offset': .5,
              'normal_camera_valid': origin != 'official_default_fov_for_legacy_missing_metadata'}
    if origin == 'official_default_fov_for_legacy_missing_metadata':
        result.update(default_source=str(OFFICIAL_PROCESSOR), default_source_sha256=default_hash,
                      calibration_caveat='official dataset default; per-frame calibration unavailable')
    return result


def canvas_intrinsics(intrinsics: Tensor, native_hw: tuple[int, int]) -> Tensor:
    (rh, rw), (top, _, left, _) = target_geometry(native_hw)
    h, w = native_hw
    transform = intrinsics.new_tensor([[rw/w, 0, left], [0, rh/h, top], [0, 0, 1]])
    return transform @ intrinsics


def canvas_to_native_patch(value: Tensor, native_hw: tuple[int, int],
                           crop: tuple[int, int, int, int] | None = None) -> Tensor:
    """Exactly sample unletterbox+bilinear native coordinates without full upsample.

    Border replication occurs on the unpadded content, matching interpolate.
    Out-of-image patch padding is zero, and must be masked using `content`.
    """
    if value.ndim != 3 or tuple(value.shape[-2:]) != CANVAS_HW:
        raise ValueError('expected [C,384,512] canvas tensor')
    nh, nw = native_hw
    y, x, ph, pw = crop if crop is not None else (0, 0, nh, nw)
    (rh, rw), (top, _, left, _) = target_geometry(native_hw)
    content = value[:, top:top+rh, left:left+rw].float()
    u = (torch.arange(pw, dtype=torch.float32)+x+.5)/nw*2-1
    v = (torch.arange(ph, dtype=torch.float32)+y+.5)/nh*2-1
    yy, xx = torch.meshgrid(v, u, indexing='ij')
    sampled = F.grid_sample(content[None], torch.stack((xx, yy), -1)[None],
                            align_corners=False, padding_mode='border')[0]
    valid = ((torch.arange(ph)+y < nh)[:,None] & (torch.arange(pw)+x < nw)[None,:]
             & (torch.arange(ph)+y >= 0)[:,None] & (torch.arange(pw)+x >= 0)[None,:])
    return sampled * valid[None]


def crop_native(value: Tensor, crop: tuple[int, int, int, int]) -> Tensor:
    y, x, ph, pw = crop
    if y < 0 or x < 0:
        raise ValueError('negative crop coordinates are unsupported')
    patch = value[..., y:y+ph, x:x+pw]
    return F.pad(patch, (0, pw-patch.shape[-1], 0, ph-patch.shape[-2]))


def label_canvas(value: Tensor, native_hw: tuple[int, int]) -> Tensor:
    (rh, rw), (top, bottom, left, right) = target_geometry(native_hw)
    resized = F.interpolate(value[None].float(), (rh, rw), mode='nearest-exact')[0]
    return F.pad(resized, (left, right, top, bottom))


def read_depth(path: str | Path) -> Tensor:
    import OpenEXR
    with OpenEXR.File(str(resolve_asset(path)), separate_channels=True) as f:
        return torch.from_numpy(np.array(f.channels()['R'].pixels, dtype=np.float32))[None]


def camera_normal_path(row: Mapping[str, Any]) -> Path | None:
    """Trusted synthetic cameraNormals asset; real normal EXRs are not adopted."""
    if row['domain'] != 'synthetic':
        return None
    rgb = resolve_asset(row['rgb_path'])
    frame = rgb.name.split('-rgb', 1)[0]
    path = rgb.parent.parent / 'camera-normals' / f'{frame}-cameraNormals.exr'
    if not path.is_file():
        raise FileNotFoundError(f'expected released synthetic camera normal: {path}')
    return path


def read_camera_normal(path: str | Path) -> tuple[Tensor, Tensor]:
    """Released RGB camera xyz -> OpenCV [R,-G,-B], with explicit support.

    This is a ground-truth label loader, never an RGB inference dependency.
    Sentinel (-1,-1,-1), nonfinite and non-unit-range vectors are invalid.
    """
    import OpenEXR
    with OpenEXR.File(str(resolve_asset(path)), separate_channels=True) as f:
        raw = torch.stack([torch.from_numpy(np.array(f.channels()[c].pixels, dtype=np.float32))
                           for c in ('R','G','B')])
    finite = torch.isfinite(raw).all(0, keepdim=True)
    clean = torch.where(torch.isfinite(raw), raw, torch.zeros_like(raw))
    norm = torch.linalg.vector_norm(clean, dim=0, keepdim=True)
    sentinel = (clean == -1).all(0, keepdim=True)
    valid = finite & ~sentinel & (norm >= .5) & (norm <= 1.5)
    cv = clean * clean.new_tensor([1.,-1.,-1.])[:,None,None]
    return torch.where(valid, F.normalize(cv, dim=0, eps=1e-6), torch.zeros_like(cv)), valid


def camera_normal_canvas(normal: Tensor, valid: Tensor, native_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    (rh,rw),(top,bottom,left,right) = target_geometry(native_hw)
    value = F.interpolate((normal*valid)[None].float(), (rh,rw), mode='bilinear',
                          align_corners=False, antialias=True)[0]
    support = F.interpolate(valid[None].float(), (rh,rw), mode='bilinear',
                            align_corners=False, antialias=True)[0] >= .999
    value = F.normalize(value, dim=0, eps=1e-6) * support
    return F.pad(value,(left,right,top,bottom)), F.pad(support,(left,right,top,bottom))


class NativeFusionDataset(Dataset):
    def __init__(self, manifest: str | Path = DEFAULT_MANIFEST, split: str = 'train', *,
                 patch_size: int | tuple[int, int] | None = 384,
                 include_target: bool = True, seed: int = 20261021,
                 verify_assets: bool = True, residual_root: str | Path | None = None,
                 allow_unsealed_residuals: bool = False):
        self.rows = load_manifest(manifest, split)
        self.patch_size = ((patch_size, patch_size) if isinstance(patch_size, int) else patch_size)
        self.include_target, self.seed, self.epoch = include_target, seed, 0
        self.verify_assets = verify_assets
        self.residual_root = Path(residual_root) if residual_root else DEFAULT_ROOT / 'r5_residuals'
        marker = self.residual_root / 'complete.json'
        self.residual_assets = None
        if marker.is_file():
            complete = json.loads(marker.read_text())
            roster = self.residual_root / 'manifest.jsonl'
            if complete['status'] != 'complete' or sha256_file(roster) != complete['manifest_sha256']:
                raise ValueError('residual manifest is not sealed')
            self.residual_assets = {r['cache_id']:r for r in read_jsonl(roster)}
        elif not allow_unsealed_residuals:
            raise ValueError('residual cache must be sealed before formal data loading')

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        compact = load_tensor_asset(row['compact'], verify=self.verify_assets)
        if compact['sample_id'] != row['sample_id'] or compact['rgb_sha256'] != row['rgb_sha256']:
            raise ValueError('compact identity mismatch')
        residual_asset = (self.residual_assets[row['cache_id']] if self.residual_assets is not None
                          else {'path': str(self.residual_root/f"{row['cache_id']}.pt")})
        residual = load_tensor_asset(residual_asset, verify=self.verify_assets)
        if (residual['sample_id'] != row['sample_id'] or residual['rgb_sha256'] != row['rgb_sha256']
                or residual['contains_ground_truth'] is not False):
            raise ValueError('native R5 residual identity mismatch')
        native_hw = tuple(int(x) for x in row['native_hw'])
        nh, nw = native_hw
        if self.patch_size is None:
            crop = (0, 0, nh, nw)
        else:
            ph, pw = self.patch_size
            gen = torch.Generator().manual_seed(self.seed + self.epoch*1000003 + index)
            y = int(torch.randint(max(1, nh-ph+1), (1,), generator=gen))
            x = int(torch.randint(max(1, nw-pw+1), (1,), generator=gen))
            crop = (y, x, ph, pw)
        y, x, ph, pw = crop
        rgb_path = resolve_asset(row['rgb_path'])
        encoded = rgb_path.read_bytes()
        if self.verify_assets and hashlib.sha256(encoded).hexdigest() != row['rgb_sha256']:
            raise ValueError('RGB source changed')
        with Image.open(io.BytesIO(encoded)) as im:
            rgb = torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float()/255
        if tuple(rgb.shape[-2:]) != native_hw:
            raise ValueError('RGB geometry changed')
        k = torch.tensor(row['camera']['intrinsics'], dtype=torch.float32)
        kp = k.clone(); kp[0,2] -= x; kp[1,2] -= y
        content = crop_native(torch.ones((1, nh, nw), dtype=torch.bool), crop)
        result: dict[str, Any] = {
            'sample_id': row['sample_id'], 'domain': row['domain'],
            'native_hw': torch.tensor(native_hw), 'crop_top_left': torch.tensor([y,x]),
            'padding_tblr': torch.tensor(target_geometry(native_hw)[1]),
            'intrinsics': kp, 'context_intrinsics': canvas_intrinsics(k, native_hw),
            'normal_camera_valid': torch.tensor(row['camera'].get('normal_camera_valid',
                row['camera']['source_kind'] != 'official_default_fov_for_legacy_missing_metadata')),
            'rgb': crop_native(rgb, crop), 'content': content,
            'context_rgb': compact['rgb'].float()/255,
        }
        mapping = {'far_depth':'far', 'r5_depth':'r5', 'probability':'probability',
                   'teacher_normal':'teacher_normal', 'teacher_confidence':'teacher_confidence',
                   'content':'content'}
        for source, dest in mapping.items():
            result['context_'+dest] = compact[source].float()
        result['context_r5_legacy'] = result['context_r5']
        result['context_r5'] = residual['context_r5_native'].float()
        result['context_descriptor'] = residual['context_descriptor'].float()
        for name in ('far_depth','teacher_normal','teacher_confidence','probability'):
            result[name] = canvas_to_native_patch(compact[name], native_hw, crop)
        result['teacher_normal'] = F.normalize(result['teacher_normal'], dim=0, eps=1e-6)
        result['far_depth'] = torch.where(content, result['far_depth'], torch.ones_like(result['far_depth']))
        if row['split'] == 'train':
            native = load_tensor_asset(row['native_moge'], verify=self.verify_assets)
            if native['sample_id'] != row['sample_id'] or native['rgb_sha256'] != row['rgb_sha256']:
                raise ValueError('native MoGe identity mismatch')
            base = crop_native(native['base_depth_native'][None].float(), crop)
            delta = canvas_to_native_patch(residual['bounded_delta'], native_hw, crop)
            gate = canvas_to_native_patch(residual['gate'], native_hw, crop)
            candidate = base * torch.exp(delta*gate)
            result['r5_depth'] = torch.where(gate > 0, candidate, base)
        else:
            asset = row['native_r5']
            path = resolve_asset(asset['path'])
            if self.verify_assets and sha256_file(path) != asset['sha256']:
                raise ValueError('native R5 export changed')
            value = torch.from_numpy(np.load(path, allow_pickle=False).copy())[None].float()
            if tuple(value.shape[-2:]) != native_hw:
                raise ValueError('native R5 geometry mismatch')
            result['r5_depth'] = crop_native(value, crop)
        result['r5_depth'] = torch.where(content, result['r5_depth'], torch.ones_like(result['r5_depth']))
        if self.include_target:
            depth = read_depth(row['target_depth_path'])
            if tuple(depth.shape[-2:]) != native_hw:
                raise ValueError('target geometry mismatch')
            valid = torch.isfinite(depth) & (depth > 0)
            target = torch.where(valid, depth, torch.zeros_like(depth))
            with Image.open(resolve_asset(row['transparent_mask_path'])) as im:
                mask = torch.from_numpy(np.asarray(im.convert('L')).copy())[None] > 0
            result.update(target=crop_native(target, crop), valid=crop_native(valid, crop),
                          transparent=crop_native(mask, crop),
                          context_target=label_canvas(target, native_hw),
                          context_valid=label_canvas(valid, native_hw).bool(),
                          context_transparent=label_canvas(mask, native_hw).bool())
            normal_path = camera_normal_path(row)
            if normal_path is None:
                normal = torch.zeros((3,nh,nw), dtype=torch.float32)
                normal_valid = torch.zeros((1,nh,nw), dtype=torch.bool)
            else:
                normal, normal_valid = read_camera_normal(normal_path)
                if tuple(normal.shape[-2:]) != native_hw:
                    raise ValueError('independent camera normal geometry mismatch')
            context_normal, context_normal_valid = camera_normal_canvas(normal, normal_valid, native_hw)
            result.update(target_normal=crop_native(normal,crop), normal_valid=crop_native(normal_valid,crop),
                          normal_reference_available=torch.tensor(normal_path is not None),
                          context_target_normal=context_normal, context_normal_valid=context_normal_valid)
        return result
