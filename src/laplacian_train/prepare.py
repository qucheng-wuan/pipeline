"""Build exact full-image Laplacian inputs, then cache native training patches.

Only train labels are supported. Pilot and full plans share a producer contract
and per-image paths, so sealed pilot images are reused without recomputation.
Validation/test full-image input construction is RGB-only; evaluation remains a
separate explicit operation in the experiment driver.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .data import (ROOT, SCHEMA, PATCH_SIZE, HALO, LEVELS, GATE_HW, SEED, IDENTITY_KEYS, INPUT_KEYS,
                   digest, read_json, read_jsonl, sha256, load_asset, validate_inputs,
                   validate_labels, verify_record)

PROJECT = Path(__file__).resolve().parents[2]
V2_PROJECT = FIXED_PROJECT = PROJECT
from native_fusion.data import DEFAULT_ROOT as V2_ROOT
MANIFEST = Path(os.environ.get('TD_STAGE1_MANIFEST', V2_ROOT / 'manifests/dataset.jsonl'))
from transparent_depth.stage1 import Stage1Assets
CHECKPOINT = Stage1Assets().path('v2')
CHECKPOINT_SHA = Stage1Assets().entries['v2']['sha256']


def dependencies():
    for project in (V2_PROJECT, FIXED_PROJECT):
        source = str(project / 'src')
        if source not in sys.path:
            sys.path.insert(0, source)
    from native_fusion import data, runtime, cache
    return data, runtime, cache


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def save_asset(path, payload):
    stream = io.BytesIO()
    torch.save(payload, stream)
    encoded = stream.getvalue()
    atomic_bytes(path, encoded)
    return {'path': str(Path(path).resolve()), 'bytes': len(encoded),
            'sha256': hashlib.sha256(encoded).hexdigest()}


@contextmanager
def forbid_label_reads():
    """Tripwire covering Python IO and native OpenEXR during RGB construction."""
    import builtins
    import OpenEXR
    original_open, original_path_open, original_exr = builtins.open, Path.open, OpenEXR.File

    def check(path):
        name = str(path).lower()
        if (name.endswith('.exr') or '-segmentation-mask.' in name
                or '-opaque-depth-img.' in name or name.endswith('-mask.png')):
            raise AssertionError('RGB-only construction attempted to read a label: ' + str(path))

    def guarded_open(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        check(path)
        return original_path_open(path, *args, **kwargs)

    def guarded_exr(path, *args, **kwargs):
        check(path)
        return original_exr(path, *args, **kwargs)

    builtins.open, Path.open, OpenEXR.File = guarded_open, guarded_path_open, guarded_exr
    try:
        yield
    finally:
        builtins.open, Path.open, OpenEXR.File = original_open, original_path_open, original_exr


def cluster(row):
    rgb = Path(row['rgb_path'])
    return rgb.parent.parent.name if rgb.parent.name == 'rgb-imgs' else rgb.parent.name


def hash_order(row):
    return hashlib.sha256(f"{SEED}:{row['rgb_sha256']}".encode()).hexdigest()


def select_roster(rows, scope):
    """Selection depends only on metadata and RGB hashes, never target pixels."""
    if scope not in ('pilot', 'full'):
        raise ValueError('scope must be pilot or full')
    selected = [(index, row) for index, row in enumerate(rows) if row['split'] == 'train']
    if scope == 'pilot':
        groups = defaultdict(list)
        for index, row in selected:
            if row['camera'].get('normal_camera_valid') is True:
                groups[cluster(row)].append((index, row))
        if len(groups) != 5:
            raise ValueError('expected five camera-valid synthetic training groups')
        selected = []
        for name in sorted(groups):
            if len(groups[name]) < 128:
                raise ValueError('fewer than 128 calibrated images in ' + name)
            selected.extend(sorted(groups[name], key=lambda pair: hash_order(pair[1]))[:128])
    if any(row['domain'] != 'synthetic' for _, row in selected):
        raise ValueError('official training roster must be synthetic')
    return [{**{k: row[k] for k in IDENTITY_KEYS}, 'cluster': cluster(row), 'manifest_index': index}
            for index, row in selected]


def producer_contract():
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('locked current V2 checkpoint changed')
    data, _, _ = dependencies()
    rows = data.load_manifest(MANIFEST)
    marker = read_json(MANIFEST.with_name('dataset.complete.json'))
    if (Counter(r['split'] for r in rows) != Counter(data.EXPECTED_COUNTS)
            or marker.get('status') != 'complete' or marker['manifest_sha256'] != sha256(MANIFEST)):
        raise ValueError('original complete training/validation/test roster changed')
    paths = [Path(__file__), Path(__file__).with_name('data.py'), Path(__file__).with_name('model.py')]
    paths += [V2_PROJECT / 'src/native_fusion' / name for name in
              ('data.py', 'model.py', 'runtime.py', 'cache.py')]
    paths += [FIXED_PROJECT / 'src/laplacian_fusion/pyramid.py']
    return {'schema_version': SCHEMA, 'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': CHECKPOINT_SHA,
            'manifest': str(MANIFEST), 'manifest_sha256': sha256(MANIFEST),
            'residual_complete_sha256': sha256(V2_ROOT / 'r5_residuals/complete.json'),
            'source_sha256': {str(p.relative_to(PROJECT)): sha256(p) for p in paths},
            'seed': SEED, 'patch_size': PATCH_SIZE, 'halo': HALO, 'levels': LEVELS, 'gate_hw': list(GATE_HW),
            'pilot_selection': '128 camera-valid images per each of five train objects; SHA256(seed:rgb_sha256)',
            'full_selection': 'all 45453 original train images, one fixed patch each; no calibration exclusion',
            'sampling': 'SHA256(seed:rgb_sha256:crop); 2/3 TOM-weighted centre (0.02+probability), 1/3 uniform crop',
            'baseline': 'frozen V2 exact native tile384 halo16 BF16, followed by full-image levels6 strength0.5 global Laplacian',
            'bands': 'full-image log(MoGe)-log(V2) binomial5 dyadic Laplacian; each band lifted through all preceding shapes',
            'mean_coeff': 'per-band adjoint of full-image bilinear gate interpolation divided by native H*W',
            'context_channels': ['rgb[3]', 'TOM[1]', 'predicted_normal[3]', 'confidence[1]',
                                 'centered_log_v2[1]', 'centered_log_moge[1]', 'abs_lifted_band[6]'],
            'inputs_precision': 'FP32 fixed depth, lifted bands, context and mean coefficients',
            'normal_labels': 'independent synthetic cameraNormals [R,-G,-B]; unreliable K disables camera-dependent losses',
            'input_ground_truth_access': False, 'test_labels_read': False,
            'numerics': {'amp': 'bfloat16', 'tf32': False, 'torch': str(torch.__version__)}}


def create_plan(root=ROOT, scope='pilot'):
    root = Path(root)
    contract = producer_contract()
    roster = select_roster(read_jsonl(MANIFEST), scope)
    expected = {'pilot': 640, 'full': 45453}[scope]
    if len(roster) != expected or len({r['cache_id'] for r in roster}) != expected:
        raise ValueError('selected training count or uniqueness differs from protocol')
    plan = {'scope': scope, 'contract': contract, 'images': roster, 'image_count': expected,
            'counts_by_directory': dict(Counter(r['cluster'] for r in roster)), 'test_labels_read': False}
    path = root / f'cache/plan_{scope}.json'
    if path.exists() and read_json(path) != plan:
        raise ValueError('refusing to replace an existing different training plan')
    for other in ('pilot', 'full'):
        other_path = root / f'cache/plan_{other}.json'
        if other_path.exists() and read_json(other_path)['contract'] != contract:
            raise ValueError('pilot/full producer contracts must remain identical for safe cache reuse')
    if not path.exists():
        atomic_json(path, plan)
    return plan


def patch_origin(probability, rgb_sha256):
    p = probability.detach().cpu().float().reshape(*probability.shape[-2:]).clamp(0, 1)
    h, w = p.shape
    if min(h, w) < PATCH_SIZE or not torch.isfinite(p).all():
        raise ValueError('invalid native TOM map for crop sampling')
    seed = int.from_bytes(hashlib.sha256(f'{SEED}:{rgb_sha256}:crop'.encode()).digest()[:8], 'big')
    generator = torch.Generator().manual_seed(seed)
    kind = 'tom' if int(torch.randint(3, (1,), generator=generator)) < 2 else 'uniform'
    if kind == 'tom':
        index = int(torch.multinomial((.02 + p).flatten(), 1, generator=generator))
        y = max(0, min(h - PATCH_SIZE, index // w - PATCH_SIZE // 2))
        x = max(0, min(w - PATCH_SIZE, index % w - PATCH_SIZE // 2))
    else:
        y = int(torch.randint(h - PATCH_SIZE + 1, (1,), generator=generator))
        x = int(torch.randint(w - PATCH_SIZE + 1, (1,), generator=generator))
    return y, x, kind


def _read_rgb(row, data):
    encoded = data.resolve_asset(row['rgb_path']).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != row['rgb_sha256']:
        raise ValueError('RGB source hash changed')
    with Image.open(io.BytesIO(encoded)) as im:
        rgb = np.asarray(im.convert('RGB')).copy()
    if tuple(rgb.shape[:2]) != tuple(row['native_hw']):
        raise ValueError('RGB native image geometry changed')
    return rgb


def _verify_compact(compact, row):
    if (compact.get('schema_version') != 'far_r5_compact_inputs_v1'
            or compact['sample_id'] != row['sample_id'] or compact['rgb_sha256'] != row['rgb_sha256']
            or compact.get('contains_ground_truth') is not False or compact.get('ground_truth_access') is not False):
        raise ValueError('compact expert inputs are not authenticated RGB-only data')


def _train_components(row, engine):
    """Read native MoGe exactly once and restore the sealed native R5 formula."""
    data, runtime_module, cache = dependencies()
    if row['split'] != 'train':
        raise ValueError('train component loader refuses held-out rows')
    rgb = _read_rgb(row, data)
    compact = data.load_tensor_asset(row['compact'])
    _verify_compact(compact, row)
    residual = data.load_tensor_asset(engine.residuals[row['cache_id']])
    native = data.load_tensor_asset(row['native_moge'])
    for payload in (residual, native):
        if payload['sample_id'] != row['sample_id'] or payload['rgb_sha256'] != row['rgb_sha256']:
            raise ValueError('native expert source identity changed')
    if residual.get('contains_ground_truth') is not False:
        raise ValueError('native R5 residual contains labels')
    moge = native['base_depth_native'].float()
    r5 = cache.native_r5(moge, residual['bounded_delta'], residual['gate'])
    inputs = runtime_module.assemble_inputs(rgb=rgb, compact=compact, r5_native=r5,
        context_descriptor=residual['context_descriptor'], context_r5=residual['context_r5_native'])
    return inputs, moge[None, None]


def load_full_inputs(row, device, runtime=None, sealed_dataset=None, *, include_sampling=False):
    """Rebuild full-image inputs from this project's authenticated expert cache."""
    from .model import fixed_depth_and_bands, mean_coefficients, build_gate_context
    data, runtime_module, cache = dependencies()
    device = torch.device(device)
    if runtime is None:
        raise ValueError('a portable frozen runtime is required')
    with forbid_label_reads(), torch.no_grad():
        rgb_array = _read_rgb(row, data)
        compact = data.load_tensor_asset(row['compact'])
        _verify_compact(compact, row)
        residual = data.load_tensor_asset(runtime.residuals[row['cache_id']])
        if residual.get('contains_ground_truth') is not False or residual['rgb_sha256'] != row['rgb_sha256']:
            raise ValueError('RGB residual identity mismatch')
        if row['split'] == 'train':
            moge = data.load_tensor_asset(row['native_moge'])['base_depth_native'].float()
        else:
            from native_fusion.benchmark import read_prediction
            asset = row['native_moge']
            moge = torch.from_numpy(read_prediction(data.resolve_asset(asset['path']), asset['sha256'], row['native_hw']).copy())
        r5 = cache.native_r5(moge, residual['bounded_delta'], residual['gate'])
        source = runtime_module.assemble_inputs(rgb=rgb_array, compact=compact, r5_native=r5,
            context_descriptor=residual['context_descriptor'], context_r5=residual['context_r5_native'])
        v2 = runtime_module.predict_native(runtime.model, {k:v[None] for k,v in source.items()},tile_size=384,halo=16,amp=True)
        moge = moge[None,None].to(device)
        rgb, probability, normal, confidence = (source[k][None].to(device) for k in
            ('rgb','probability','teacher_normal','teacher_confidence'))
        fixed,bands = fixed_depth_and_bands(v2,moge,levels=LEVELS)
        inputs = {'v2_depth':v2.float(),'fixed_depth':fixed.float(),'bands':bands.float(),
            'context':build_gate_context(rgb,probability,normal,confidence,v2,moge,bands,gate_hw=GATE_HW).float(),
            'mean_coeff':mean_coefficients(bands,gate_hw=GATE_HW).float(),
            'native_hw':torch.tensor([row['native_hw']],dtype=torch.int64,device=device),
            'crop_top_left':torch.zeros((1,2),dtype=torch.int64,device=device)}
    return (inputs, probability[0].detach().cpu()) if include_sampling else inputs


class FullInputBuilder:
    def __init__(self, split, device='cuda:0', runtime=None, sealed_dataset=None):
        from transparent_depth.stage1 import Stage1Predictor
        data, _, _ = dependencies()
        if split not in ('train','val','test'):
            raise ValueError('unknown split')
        self.split,self.device = split,torch.device(device)
        self.runtime=runtime or Stage1Predictor(device=device)
        self.runtime.residuals={r['cache_id']:r for r in read_jsonl(V2_ROOT/'r5_residuals/manifest.jsonl')}
        marker=read_json(V2_ROOT/'r5_residuals/complete.json')
        if marker['status']!='complete' or marker['manifest_sha256']!=sha256(V2_ROOT/'r5_residuals/manifest.jsonl'):
            raise ValueError('portable expert cache is not sealed')
        self.rows=data.load_manifest(MANIFEST,split)

    def __len__(self):return len(self.rows)

    def __call__(self,row_or_index,*,include_sampling=False):
        row=self.rows[row_or_index] if isinstance(row_or_index,int) else row_or_index
        if row['split']!=self.split:raise ValueError('split changed')
        return load_full_inputs(row,self.device,self.runtime,include_sampling=include_sampling)


def native_train_labels(row):
    """Only called after input tensors have been saved in separate files."""
    if row['split'] != 'train' or row['domain'] != 'synthetic':
        raise ValueError('preparation may open only official synthetic training labels')
    data, _, _ = dependencies()
    depth_path = data.resolve_asset(row['target_depth_path'])
    mask_path = data.resolve_asset(row['transparent_mask_path'])
    normal_path = data.camera_normal_path(row)
    if normal_path is None:
        raise ValueError('synthetic training sample lacks independent cameraNormals')
    target = data.read_depth(depth_path)
    valid = torch.isfinite(target) & (target > 0)
    target = torch.where(valid, target, torch.zeros_like(target))
    with Image.open(mask_path) as im:
        transparent = torch.from_numpy(np.asarray(im.convert('L')).copy())[None] > 0
    normal, normal_valid = data.read_camera_normal(normal_path)
    if tuple(target.shape[-2:]) != tuple(row['native_hw']) or transparent.shape != target.shape or normal.shape[-2:] != target.shape[-2:]:
        raise ValueError('training label geometry differs from RGB')
    labels = {'target': target, 'valid': valid, 'transparent': transparent,
              'target_normal': normal, 'normal_valid': normal_valid,
              'K': torch.tensor(row['camera']['intrinsics'], dtype=torch.float32),
              'camera_valid': torch.tensor(row['camera'].get('normal_camera_valid', False), dtype=torch.bool)}
    sources = {name: {'path': str(path), 'sha256': sha256(path)}
               for name, path in (('depth', depth_path), ('mask', mask_path), ('normal', normal_path))}
    return labels, sources


def crop_inputs(full, origin):
    y, x = map(int, origin)
    result = {}
    for key in INPUT_KEYS:
        value = full[key]
        value = value[0].detach().cpu()
        if key in ('fixed_depth', 'bands'):
            value = value[..., y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        result[key] = value.contiguous()
    result['crop_top_left'] = torch.tensor([y, x], dtype=torch.int64)
    validate_inputs(result)
    return result


def crop_labels(full, origin):
    y, x = map(int, origin)
    local = {key: value[..., y:y + PATCH_SIZE, x:x + PATCH_SIZE].contiguous()
             for key, value in full.items() if key not in ('K', 'camera_valid')}
    k = full['K'].clone()
    k[0, 2] -= x
    k[1, 2] -= y
    core = torch.zeros((1, PATCH_SIZE, PATCH_SIZE), dtype=torch.bool)
    core[..., HALO:-HALO, HALO:-HALO] = True
    local.update(K=k, camera_valid=full['camera_valid'].clone(), core=core)
    validate_labels(local)
    return local


def worker(root=ROOT, scope='pilot', *, rank=0, world=1, device='cuda:0', limit=0):
    if world < 1 or not 0 <= rank < world:
        raise ValueError('invalid preparation shard')
    root, device = Path(root), torch.device(device)
    if device.type != 'cuda':
        raise ValueError('frozen V2 preparation requires the deployed CUDA BF16 numerical path')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(device)
    plan = read_json(root / f'cache/plan_{scope}.json')
    if producer_contract() != plan['contract']:
        raise ValueError('producer changed after training data planning')
    fingerprint = digest(plan['contract'])
    data, _, _ = dependencies()
    full_rows = data.load_manifest(MANIFEST)
    selected = plan['images'][rank::world]
    if limit:
        selected = selected[:limit]
    builder = None
    started = time.monotonic()
    for done, identity in enumerate(selected, 1):
        row = full_rows[identity['manifest_index']]
        if any(row[k] != identity[k] for k in IDENTITY_KEYS):
            raise ValueError('planned native source identity changed')
        receipt_path = root / 'cache/receipts/train' / f"{row['cache_id']}.json"
        if receipt_path.exists():
            verify_record(read_json(receipt_path), identity, fingerprint, root)
        else:
            if builder is None:
                builder = FullInputBuilder('train', device=device)
            full, probability = builder(row, include_sampling=True)
            y, x, kind = patch_origin(probability, row['rgb_sha256'])
            inputs = crop_inputs(full, (y, x))
            del full, probability
            input_asset = save_asset(root / 'cache/inputs/train' / f"{row['cache_id']}.pt", inputs)
            labels, label_sources = native_train_labels(row)
            label_asset = save_asset(root / 'cache/labels/train' / f"{row['cache_id']}.pt", crop_labels(labels, (y, x)))
            del labels
            receipt = {**identity, 'contract_sha256': fingerprint, 'crop_top_left': [y, x],
                       'sampling': kind, 'input_asset': input_asset, 'label_asset': label_asset,
                       'input_source_assets': {'compact': row['compact'], 'native_moge': row['native_moge'],
                           'r5_residual': builder.runtime.residuals[row['cache_id']],
                           'rgb': {'path': str(data.resolve_asset(row['rgb_path'])), 'sha256': row['rgb_sha256']}},
                       'label_source_assets': label_sources, 'camera_metadata': row['camera'],
                       'normal_reference': 'independent_synthetic_cameraNormals',
                       'input_ground_truth_access': False, 'test_labels_read': False}
            atomic_json(receipt_path, receipt)
        if done == 1 or done % 10 == 0 or done == len(selected):
            state = {'scope': scope, 'rank': rank, 'world': world, 'done': done, 'total': len(selected),
                     'elapsed_seconds': time.monotonic() - started, 'last_cache_id': row['cache_id']}
            atomic_json(root / f'cache/status_{scope}_rank{rank}.json', state)
            print(json.dumps(state), flush=True)


def seal(root=ROOT, scope='pilot'):
    root = Path(root)
    plan = read_json(root / f'cache/plan_{scope}.json')
    if producer_contract() != plan['contract']:
        raise ValueError('producer changed before cache sealing')
    fingerprint = digest(plan['contract'])
    records = []
    for identity in plan['images']:
        receipt = read_json(root / 'cache/receipts/train' / f"{identity['cache_id']}.json")
        verify_record(receipt, identity, fingerprint, root)
        records.append(receipt)
    expected = {'pilot': 640, 'full': 45453}[scope]
    if len(records) != expected or len({r['cache_id'] for r in records}) != expected:
        raise ValueError('cache coverage contains duplicates or omissions')
    manifest = root / f'cache/manifest_{scope}.jsonl'
    encoded = ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in records).encode()
    marker_path = root / f'cache/complete_{scope}.json'
    marker = {'status': 'complete', 'scope': scope, 'schema_version': SCHEMA, 'images': len(records),
              'manifest_sha256': hashlib.sha256(encoded).hexdigest(), 'contract_sha256': fingerprint,
              'plan_sha256': digest(plan), 'all_asset_sha256_verified': True,
              'input_bytes': sum(r['input_asset']['bytes'] for r in records),
              'label_bytes': sum(r['label_asset']['bytes'] for r in records), 'test_labels_read': False}
    if marker_path.exists():
        if read_json(marker_path) != marker or manifest.read_bytes() != encoded:
            raise ValueError('refusing to replace a different completed cache seal')
        return marker
    atomic_bytes(manifest, encoded)
    atomic_json(marker_path, marker)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'worker', 'seal'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--scope', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'plan':
        plan = create_plan(args.root, args.scope)
        print(json.dumps({'scope': args.scope, 'images': plan['image_count'], 'contract_sha256': digest(plan['contract'])}))
    elif args.action == 'worker':
        worker(args.root, args.scope, rank=args.rank, world=args.world, device=args.device, limit=args.limit)
    else:
        print(json.dumps(seal(args.root, args.scope)))


if __name__ == '__main__':
    main()
