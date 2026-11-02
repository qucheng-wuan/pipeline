"""Original native prediction/label I/O. Labels are used only by evaluation."""
import hashlib, io, os, re
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .data import resolve_asset, sha256_file, read_depth, camera_normal_path, read_camera_normal

def atomic_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('wb') as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_prediction(path: Path, expected_sha256: str, hw: list[int] | tuple[int, int]) -> np.ndarray:
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError(f'prediction checksum changed: {path}')
    value = np.load(io.BytesIO(encoded), allow_pickle=False)
    if value.dtype != np.float32 or tuple(value.shape) != tuple(hw) or not np.isfinite(value).all() or not (value > 0).all():
        raise ValueError('predictions must be finite positive native float32 optical-Z')
    return value


def save_prediction(path: Path, value: torch.Tensor, hw: list[int]) -> dict:
    array = value.detach().float().cpu().reshape(tuple(hw)).numpy()
    if not np.isfinite(array).all() or not (array > 0).all():
        raise ValueError('cannot seal nonfinite/nonpositive prediction')
    buffer = io.BytesIO(); np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    encoded = buffer.getvalue(); atomic_bytes(path, encoded)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(encoded).hexdigest(),
            'bytes': len(encoded), 'dtype': 'float32', 'shape_hw': list(hw)}


def evaluation_label_paths(row: dict) -> tuple[Path, Path]:
    """Eval-only naming rules; test roster intentionally carries no label paths."""
    rgb = Path(row['rgb_path'])
    if row['domain'] == 'real':
        suffix = '-transparent-rgb-img.jpg'
        if not rgb.name.endswith(suffix): raise ValueError('unexpected real RGB naming')
        stem = rgb.name.removesuffix(suffix)
        expected = (rgb.with_name(stem+'-opaque-depth-img.exr'), rgb.with_name(stem+'-mask.png'))
    elif row['domain'] == 'synthetic':
        if rgb.parent.name != 'rgb-imgs' or not rgb.name.endswith('-rgb.jpg'):
            raise ValueError('unexpected synthetic RGB naming')
        stem = rgb.name.removesuffix('-rgb.jpg'); scene = rgb.parent.parent
        expected = (scene/'depth-imgs-rectified'/f'{stem}-depth-rectified.exr',
                    scene/'segmentation-masks'/f'{stem}-segmentation-mask.png')
    else: raise ValueError('unknown ClearGrasp domain')
    result = tuple(resolve_asset(path) for path in expected)
    for key,path in zip(('target_depth_path','transparent_mask_path'),result):
        if key in row and resolve_asset(row[key]) != path:
            raise ValueError('label does not pair with held-out RGB identity')
    return result


def load_evaluation_labels(row: dict) -> tuple[dict, dict]:
    """Call only after all ranks authenticate all sealed predictions."""
    depth_path, mask_path = evaluation_label_paths(row)
    target = read_depth(depth_path)[None]
    with Image.open(mask_path) as image:
        if image.mode != 'L': raise ValueError('transparent mask must be grayscale')
        mask = np.asarray(image).copy()
    if (list(target.shape[-2:]) != row['native_hw'] or mask.shape != tuple(row['native_hw'])
            or mask.dtype != np.uint8 or not set(np.unique(mask)).issubset({0,255})):
        raise ValueError('native target/mask geometry or binary encoding mismatch')
    valid = torch.isfinite(target) & (target > 0)
    camera = row['camera']
    if camera['pixel_center_offset'] != .5 or sha256_file(resolve_asset(camera['source'])) != camera['sha256']:
        raise ValueError('camera calibration contract changed')
    batch = {'target':torch.where(valid,target,torch.zeros_like(target)), 'valid':valid,
        'transparent':torch.from_numpy(mask==255)[None,None], 'content':torch.ones_like(valid),
        'intrinsics':torch.tensor(camera['intrinsics'],dtype=torch.float64)[None],
        'normal_camera_valid':torch.tensor([camera['normal_camera_valid']])}
    assets = {'depth':{'path':str(depth_path),'sha256':sha256_file(depth_path)},
              'transparent_mask':{'path':str(mask_path),'sha256':sha256_file(mask_path)},
              'camera':{'path':camera['source'],'sha256':camera['sha256']}}
    normal_path = camera_normal_path(row)
    batch['normal_reference_available'] = torch.tensor([normal_path is not None])
    if normal_path is not None:
        normal, support = read_camera_normal(normal_path)
        if tuple(normal.shape) != (3,*row['native_hw']): raise ValueError('camera normal geometry mismatch')
        batch.update(target_normal=normal[None],normal_valid=support[None])
        assets['independent_normal'] = {'path':str(normal_path),'sha256':sha256_file(normal_path)}
    return batch, assets

