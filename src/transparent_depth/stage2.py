"""Portable LIDF-SPVNAS loading and RGB/predicted-depth/calibration inference.

The numerical core is preserved. Complete trained states are loaded without
loading ImageNet/SPVNAS initialization weights or opening training datasets.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA = 'transparent_depth.stage2.bundle.v1'
MODEL_HW = (240, 320)
SOURCE_ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY_ROOT = SOURCE_ROOT / 'v2_lidf/assets'


def file_sha(path):
    digest = hashlib.sha256()
    expanded = os.path.expandvars(str(path))
    if '${' in expanded:
        raise ValueError('Unbound path root: ' + str(path))
    with Path(expanded).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    core = SOURCE_ROOT / 'v2_lidf'
    paths = [core / name for name in ('model.py', 'geometry.py', 'spvnas_fusion.py', 'runtime.py', 'data.py', 'data_protocol.py')]
    paths += [Path(__file__)]
    paths += [p for p in (core / 'vendor').rglob('*')
              if p.is_file() and p.suffix in {'.py', '.cpp', '.cu', '.h', '.hpp', '.json'}]
    paths += [core / 'assets/SemanticKITTI_val_SPVNAS@20GMACs/net.config',
              core / 'assets/torchsparse_python_sources.json']
    return {str(p.relative_to(SOURCE_ROOT)): file_sha(p) for p in sorted(paths)}


def portable_model_config(config):
    """Retain numerical settings; discard machine-specific initialization paths."""
    result = copy.deepcopy(config)
    sparse = result.get('spvnas')
    if not isinstance(sparse, dict) or sparse.get('variant') != 'spvnas_fusion_20g':
        raise ValueError('A complete SPVNAS20G model configuration is required')
    for key in ('assets_dir', 'repo_dir', 'audit_dir'):
        sparse.pop(key, None)
    # These flags affect initialization only; every parameter/buffer is restored.
    result['rgb_pretrained'] = False
    sparse['pretrained'] = False
    if result.get('dense_allow_tf32') is not False or sparse.get('allow_tf32') is not False:
        raise ValueError('The released model requires dense/sparse TF32 disabled')
    return result


def export_stage2_bundle(checkpoint_path, output_dir, selection_metadata, expected_sha256=None):
    """CPU-only export; original checkpoint and its history remain unchanged.

    Selection metadata must distinguish validation selection from test-informed
    ranking. No claim of completed training is inferred from an epoch number.
    """
    import torch
    checkpoint_path, output = Path(checkpoint_path), Path(output_dir)
    digest = file_sha(checkpoint_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Original checkpoint SHA256 mismatch')
    if not isinstance(selection_metadata, dict) or not selection_metadata.get('selection_rule'):
        raise ValueError('Explicit selection_rule metadata is required')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve existing model bundle: ' + str(output))
    saved = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    if 'model' not in saved or not saved['model']:
        raise ValueError('Checkpoint does not contain a complete model state')
    cfg = saved.get('model_config') or saved.get('contract', {}).get('config', {}).get('model')
    config = portable_model_config(cfg)
    preprocessing = saved.get('contract', {}).get('config', {}).get('preprocessing')
    if tuple(config.get('input_hw', MODEL_HW)) == (256,256):
        from v2_lidf.data_protocol import validate_protocol
        validate_protocol(preprocessing)
    state = saved['model']
    if any(not torch.is_tensor(v) for v in state.values()):
        raise ValueError('Model state must contain tensors only')
    if not any(k.startswith('spvnas_branch.') for k in state):
        raise ValueError('Checkpoint omits the trained SPVNAS branch')
    if any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('Checkpoint contains nonfinite model tensors')
    output.mkdir(parents=True, exist_ok=True)
    weights = output / 'model.pt'
    temporary = weights.with_suffix('.tmp')
    torch.save({'model': state}, temporary)
    os.replace(temporary, weights)
    record = {'schema': SCHEMA, 'weights': {'file': 'model.pt', 'sha256': file_sha(weights)},
              'model_config': config, 'source_sha256': source_hashes(),
              'origin': {'checkpoint_sha256': digest, 'epoch_one_based': saved.get('epoch', -1) + 1,
                         'updates': saved.get('updates'), 'selection': selection_metadata},
              'input': {'rgb': 'uint8 HWC RGB', 'model_hw': list(config.get('input_hw', MODEL_HW)),
                        'preprocessing': preprocessing,
                        'depth': 'float32 positive optical-Z metres',
                        'intrinsics': 'float32 3x3 integer pixel centres', 'labels_required': False},
              'training_contract': saved.get('contract'),
              'initialization_assets_required': False}
    (output / 'model.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


def validate_bundle(bundle):
    root = Path(bundle).resolve(strict=True)
    metadata = json.loads((root / 'model.json').read_text())
    if metadata.get('schema') != SCHEMA or metadata.get('initialization_assets_required') is not False:
        raise ValueError('Unsupported Stage2 bundle')
    if metadata.get('source_sha256') != source_hashes():
        raise ValueError('Stage2 package source identity differs from the exported bundle')
    ref = metadata['weights']
    if ref.get('file') != 'model.pt':
        raise ValueError('Bundle weights must be the local model.pt file')
    weights = root / ref['file']
    if weights.is_symlink() or file_sha(weights) != ref['sha256']:
        raise ValueError('Stage2 bundle weights changed')
    if portable_model_config(metadata['model_config']) != metadata['model_config']:
        raise ValueError('Bundle contains machine-specific or pretrained initialization settings')
    if tuple(metadata.get('input', {}).get('model_hw', MODEL_HW)) != tuple(metadata['model_config'].get('input_hw', MODEL_HW)):
        raise ValueError('Bundle input/model geometry mismatch')
    if tuple(metadata['model_config'].get('input_hw',MODEL_HW)) == (256,256):
        from v2_lidf.data_protocol import validate_protocol
        validate_protocol(metadata['input'].get('preprocessing'))
    return weights, metadata


def build_stage2(config, device='cuda:0', *, initialization=False, assets_root=None, audit_dir=None):
    """Build the fixed topology, optionally with verified training initialization."""
    import torch
    from v2_lidf.model import LIDFStage1
    from v2_lidf.runtime import validate_device_runtime
    target = torch.device(device)
    if target.type != 'cuda':
        raise ValueError('The actual LIDF/SPVNAS operators require CUDA; CPU is for contract/geometry tests')
    torch.cuda.set_device(target)
    runtime_audit = validate_device_runtime(target)
    cfg = copy.deepcopy(config) if initialization else portable_model_config(config)
    model_hw = tuple(cfg.pop('input_hw', MODEL_HW))
    if model_hw not in {(240,320),(256,256)}:
        raise ValueError('Unsupported model input resolution')
    if initialization:
        if assets_root is None:
            raise ValueError('Training initialization requires an explicit downloaded assets_root')
        cfg['rgb_pretrained'] = True
        cfg['spvnas']['pretrained'] = True
        root = Path(assets_root).resolve(strict=True)
        rgb_path = root / 'resnet34-333f7ec4.pth'
        if file_sha(rgb_path) != '333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef':
            raise ValueError('ImageNet initialization asset SHA256 mismatch')
        # The preserved ResNet loader uses the standard torch.hub cache.
        from unittest.mock import patch
        import v2_lidf.vendor.lidf.resnet_adjusted as resnet
        def local_rgb(url, *args, **kwargs):
            if url != 'https://download.pytorch.org/models/resnet34-333f7ec4.pth':
                raise ValueError('Unexpected initialization URL: ' + url)
            return torch.load(str(rgb_path), map_location='cpu', weights_only=False)
        from contextlib import ExitStack
        rgb_context = ExitStack()
        # Preserve the original backbone loader and its SHA audit, but isolate
        # its standard hub layout in a temporary directory. No network call.
        hub = rgb_context.enter_context(tempfile.TemporaryDirectory(prefix='stage2-hub-'))
        (Path(hub) / 'checkpoints').mkdir()
        (Path(hub) / 'checkpoints' / rgb_path.name).symlink_to(rgb_path)
        rgb_context.enter_context(patch.object(torch.hub, 'get_dir', return_value=hub))
        rgb_context.enter_context(patch.object(resnet.model_zoo, 'load_url', side_effect=local_rgb))
    else:
        root = TOPOLOGY_ROOT
        from contextlib import nullcontext
        rgb_context = nullcontext()
    cfg['spvnas']['assets_dir'] = str(root)
    cfg['spvnas'].pop('repo_dir', None)
    # Initialization audits are useful at construction, but never write into an
    # old training directory from a checkpoint's configuration.
    with tempfile.TemporaryDirectory(prefix='transparent-depth-stage2-') as temp:
        cfg['spvnas']['audit_dir'] = str(Path(audit_dir).resolve()) if audit_dir else temp
        with rgb_context:
            model = LIDFStage1(**cfg).to(target)
    model.input_hw = model_hw
    model.runtime_audit = runtime_audit
    model.register_forward_pre_hook(check_model_grid)
    return model


def check_model_grid(model, inputs):
    """Entry-point assertion; the preserved numerical LIDF core stays generic."""
    if tuple(inputs[0].shape[-2:]) != model.input_hw:
        raise ValueError('RGB grid differs from configured model input_hw')


def load_stage2(bundle, device='cuda:0'):
    import torch
    weights, metadata = validate_bundle(bundle)
    saved = torch.load(str(weights), map_location='cpu', weights_only=True)
    model = build_stage2(metadata['model_config'], device, initialization=False)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    model.bundle_metadata = metadata
    return model


def prepare_stage2_inputs(rgb, depth_input, intrinsics, model_hw=MODEL_HW):
    """Validate before conversion; silently resizing or sanitizing is forbidden."""
    import torch
    image, depth, k = np.asarray(rgb), np.asarray(depth_input), np.asarray(intrinsics)
    model_hw = tuple(model_hw)
    if image.dtype != np.uint8 or image.shape != (*model_hw, 3):
        raise ValueError('rgb must be uint8 RGB at declared model_hw=%s' % (model_hw,))
    if depth.dtype != np.float32 or depth.shape != model_hw or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError('depth_input must be finite positive float32 optical-Z at model_hw')
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('intrinsics must be a calibrated positive-focal-length 3x3 matrix')
    if not np.allclose(k[2], [0, 0, 1], atol=1e-6, rtol=0) or abs(k[0, 1]) > 1e-6 or abs(k[1, 0]) > 1e-6:
        raise ValueError('LIDF requires zero-skew integer-centre pinhole intrinsics')
    image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.
    image = (image - image.new_tensor([.485, .456, .406])[:, None, None]) / image.new_tensor([.229, .224, .225])[:, None, None]
    return image[None], torch.from_numpy(depth.copy())[None, None], torch.from_numpy(k.astype(np.float32, copy=True))[None]


def predict_stage2(model, rgb, depth_input, intrinsics, native_hw=None, seed=0):
    """Full-image pure-P inference; no GT, mask or sensor-depth API exists."""
    import torch
    import torch.nn.functional as F
    if model.training:
        raise ValueError('Deployment requires model.eval()')
    model_hw = tuple(getattr(model, 'input_hw', MODEL_HW))
    image, depth, k = prepare_stage2_inputs(rgb, depth_input, intrinsics, model_hw)
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('Stage2 inference requires CUDA')
    torch.cuda.set_device(device)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    with torch.inference_mode():
        output = model(image.to(device), depth.to(device), k.to(device), compute_loss=False)
        value = output['depth']
        if value.shape != (1, 1, *model_hw) or not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
            raise FloatingPointError('Stage2 output is not finite positive optical-Z at the model resolution')
        if not bool(output['query_mask'].all()):
            raise RuntimeError('Deployment did not query every pixel')
        fallback = ~output['refined_mask']
        if not torch.equal(value[fallback], depth.to(device)[fallback]):
            raise RuntimeError('Fallback changed the supplied RGB-derived depth')
        result = {'depth': value[0, 0].cpu().numpy(),
                  'refined_mask': output['refined_mask'][0, 0].cpu().numpy(),
                  'diagnostics': {key: float(val) if torch.is_tensor(val) else val
                                  for key, val in output['diagnostics'].items()}}
        if native_hw is not None:
            if len(native_hw) != 2 or any(type(x) is not int or x <= 0 for x in native_hw):
                raise ValueError('native_hw must contain two positive integers')
            result['native_depth'] = F.interpolate(value, size=tuple(native_hw), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    return result


def resize_native_inputs(rgb, depth_input, intrinsics, model_hw=MODEL_HW, rgb_interpolation='INTER_LINEAR'):
    """The authenticated export convention: OpenCV linear + half-pixel K."""
    import cv2
    from v2_lidf.data import resize_intrinsics
    image, depth, k = np.asarray(rgb), np.asarray(depth_input), np.asarray(intrinsics)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('native rgb must be uint8 HWC RGB')
    if depth.dtype != np.float32 or depth.shape != image.shape[:2] or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError('native predicted depth must be finite positive float32 HW matching RGB')
    if k.shape != (3, 3):
        raise ValueError('native integer-centre intrinsics must be 3x3')
    if rgb_interpolation not in {'INTER_LINEAR', 'INTER_AREA'}:
        raise ValueError('Unsupported RGB interpolation')
    size = (model_hw[1], model_hw[0])
    image_low = cv2.resize(image, size, interpolation=getattr(cv2, rgb_interpolation))
    depth_low = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    k_low = resize_intrinsics(k, image.shape[:2], model_hw).astype(np.float32)
    prepare_stage2_inputs(image_low, depth_low, k_low, model_hw)
    return image_low, depth_low, k_low


def read_intrinsics(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        if value.get('pixel_convention') != 'integer_pixel_centres':
            raise ValueError('K JSON must explicitly use pixel_convention=integer_pixel_centres')
        value = value['intrinsics']
    # A bare 3x3 array follows the CLI documented integer-centre convention.
    return np.asarray(value, dtype=np.float64)


def predict_stage2_from_files(rgb_path, depth_input_path, intrinsics_path, bundle, output, device='cuda:0'):
    from PIL import Image
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    depth = np.load(depth_input_path, allow_pickle=False)
    k = read_intrinsics(intrinsics_path)
    model = load_stage2(bundle, device)
    protocol = model.bundle_metadata.get('input', {}).get('preprocessing') or {}
    image_low, depth_low, k_low = resize_native_inputs(rgb, depth, k, model.input_hw,
        protocol.get('rgb_interpolation', 'INTER_LINEAR'))
    rgb_sha = file_sha(rgb_path)
    output_native = model.input_hw == MODEL_HW
    result = predict_stage2(model, image_low, depth_low, k_low,
        tuple(rgb.shape[:2]) if output_native else None, int(rgb_sha[:8], 16))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        np.save(stream, result['native_depth'] if output_native else result['depth'], allow_pickle=False)
    metadata = {'schema': 'transparent_depth.stage2.prediction.v1',
                'rgb_sha256': rgb_sha, 'stage1_prediction_sha256': file_sha(depth_input_path),
                'stage2_bundle_sha256': file_sha(Path(bundle) / 'model.json'),
                'output_sha256': file_sha(output), 'native_hw': list(rgb.shape[:2]),
                'output_hw': list(rgb.shape[:2]) if output_native else list(model.input_hw),
                'intrinsics_native': k.tolist(), 'intrinsics_model': k_low.tolist(),
                'pixel_convention': 'integer_pixel_centres', 'depth_semantics': 'optical_z_metres',
                'model_hw': list(model.input_hw),
                'resize': '%s RGB / INTER_LINEAR P to model_hw%s' %
                    (protocol.get('rgb_interpolation', 'INTER_LINEAR'),
                     '; bilinear align_corners=False to native' if output_native else '; output remains on 256 grid'),
                'gt_or_mask_used_as_input': False, 'sensor_depth_used': False,
                'refined_fraction': float(result['refined_mask'].mean()),
                'diagnostics': result['diagnostics']}
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata


def main():
    import argparse
    parser = argparse.ArgumentParser(description='RGB-derived native P + RGB + native K → metric Z; 256 bundles output D256, legacy bundles output native depth')
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--depth-input', required=True, help='Native float32 Stage1 predicted optical-Z .npy, never sensor depth')
    parser.add_argument('--intrinsics', required=True, help='3x3 JSON K; integer pixel centres (top-left centre is 0,0)')
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(predict_stage2_from_files(args.rgb, args.depth_input, args.intrinsics,
                                             args.bundle, args.output, args.device), indent=2))


if __name__ == '__main__':
    main()
