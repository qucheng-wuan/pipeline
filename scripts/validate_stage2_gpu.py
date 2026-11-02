#!/usr/bin/env python3
"""Isolated Stage2 GPU parity worker; parent chooses authorized visible device.

Prepare runs CPU-only, golden/new run sequentially on one explicitly visible
GPU. New uses a relocated source tree/bundle and denies historical source,
checkpoint and input-cache access through Python audit events. These checks are
implementation acceptance, never model-quality results or winner selection.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    import torch
    from transparent_depth.stage2 import export_stage2_bundle
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'parity_inputs.json').exists():
        raise FileExistsError('Preserve existing parity evidence')
    rows = [json.loads(line) for line in Path(args.inputs).open()]
    selected = {}
    for row in rows:
        if row['role'] == 'evaluation':
            key = (row['domain'], row['benchmark_group'], tuple(row['native_hw']))
            selected.setdefault(key, row)
    if len(selected) != 6:
        raise ValueError('Expected six real/synthetic native-format strata')
    samples = []
    (root / 'inputs').mkdir()
    for row in selected.values():
        name = row['cache_id']
        original = Path(args.stage1_root) / 'inputs' / (name + '.npz')
        target = root / 'inputs' / (name + '.npz')
        shutil.copyfile(original, target)
        samples.append({key: row[key] for key in ('cache_id', 'domain', 'benchmark_group', 'native_hw', 'rgb_sha256')}
                       | {'file': str(target.relative_to(root)), 'sha256': sha(target), 'kind': 'production_cached_input'})
    with np.load(root / samples[0]['file'], allow_pickle=False) as data:
        rgb, depth, k = [data[key].copy() for key in ('rgb', 'depth_input', 'intrinsics')]
    for name, value in (('empty_support', 10.0), ('volume_boundary', 2.1249)):
        path = root / 'inputs' / (name + '.npz')
        np.savez(path, rgb=rgb, depth_input=np.full_like(depth, value), intrinsics=k)
        samples.append({'cache_id': name, 'file': str(path.relative_to(root)), 'sha256': sha(path),
                        'native_hw': [480, 848], 'domain': 'controlled_geometry',
                        'rgb_sha256': samples[0]['rgb_sha256'], 'kind': name})
    exported = export_stage2_bundle(args.checkpoint, root / 'bundle',
        {'selection_rule': 'E63 interim implementation parity only; not release winner',
         'quality_result': False}, expected_sha256=args.checkpoint_sha256)
    relocated = root / 'relocated'
    shutil.copytree(Path(args.project) / 'src', relocated / 'src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.move(root / 'bundle', relocated / 'bundle')
    # Both workers receive only copied, label-free arrays.
    cfg = json.loads((relocated / 'bundle/model.json').read_text())['model_config']
    record = {'samples': samples, 'checkpoint_sha256': sha(args.checkpoint),
        'model_config': cfg, 'original_checkpoint': str(Path(args.checkpoint).resolve()),
        'golden_source': str(Path(args.golden_source).resolve()),
        'relocated_source': str((relocated / 'src').resolve()),
        'bundle': 'relocated/bundle', 'stage2_bundle_sha256': sha(relocated / 'bundle/model.json'),
        'historical_forbidden_prefixes': [str(Path(args.golden_source).resolve().parent),
            str(Path(args.checkpoint).resolve().parents[1]), str(Path(args.stage1_root).resolve())],
        'declared_dependency_prefixes': [str(Path(sys.prefix).resolve()), str(Path(os.environ['CUDA_HOME']).resolve())],
        'purpose': 'implementation_parity_only_not_final_selection'}
    (root / 'parity_inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': 'prepared', 'samples': len(samples), 'bundle': record['bundle']}))


def deny_history(record, root):
    forbidden = [Path(p).resolve() for p in record['historical_forbidden_prefixes']]
    forbidden += [Path(record['original_checkpoint']).resolve().parents[2],
                  Path(__file__).resolve().parents[1] / 'src']
    allowed = [Path(p).resolve() for p in record['declared_dependency_prefixes']]
    allowed += [Path(root).resolve()]
    denied = []
    def audit(event, args):
        if event not in ('open', 'ctypes.dlopen') or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        # Do not resolve here: resolving would itself trigger more filesystem events.
        if any(path == p or p in path.parents for p in allowed):
            return
        if any(path == p or p in path.parents for p in forbidden):
            denied.append(str(path))
            raise PermissionError('Portable acceptance forbids historical asset/source: ' + str(path))
    sys.addaudithook(audit)
    return denied


def run_worker(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '4' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise ValueError('This acceptance is authorized only on physical GPU4')
    root = Path(args.output).resolve()
    record = json.loads((root / 'parity_inputs.json').read_text())
    denied = []
    if args.phase == 'portable':
        sys.path.insert(0, record['relocated_source'])
        denied = deny_history(record, root)
        try:
            with open(record['original_checkpoint'], 'rb'):
                raise AssertionError('Historical checkpoint guard did not reject access')
        except PermissionError:
            pass
        if len(denied) != 1:
            raise AssertionError('Historical path guard self-test failed')
        denied.clear()
    else:
        sys.path.insert(0, record['golden_source'])
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Only one authorized logical GPU may be visible')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    began = time.monotonic()
    if args.phase == 'portable':
        from transparent_depth.stage2 import load_stage2, predict_stage2
        model = load_stage2(root / record['bundle'], device)
    else:
        from v2_lidf.model import LIDFStage1
        cfg = copy.deepcopy(record['model_config'])
        cfg['spvnas']['assets_dir'] = str(root / 'relocated/src/v2_lidf/assets')
        cfg['spvnas']['audit_dir'] = str(root / 'golden_initialization')
        model = LIDFStage1(**cfg).to(device)
        state = torch.load(record['original_checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        del state
        model.eval()
    destination = root / args.phase
    destination.mkdir(exist_ok=False)
    receipts = []
    for row in record['samples']:
        path = root / row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Acceptance input changed')
        with np.load(path, allow_pickle=False) as data:
            rgb, depth, k = [data[key].copy() for key in ('rgb', 'depth_input', 'intrinsics')]
        seed = int(row['rgb_sha256'][:8], 16)
        if args.phase == 'portable':
            result = predict_stage2(model, rgb, depth, k, tuple(row['native_hw']), seed)
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.
            image = (image-image.new_tensor([.485,.456,.406])[:,None,None])/image.new_tensor([.229,.224,.225])[:,None,None]
            p = torch.from_numpy(depth)[None,None].to(device)
            with torch.inference_mode():
                out = model(image[None].to(device), p, torch.from_numpy(k)[None].to(device), compute_loss=False)
                if not bool(out['query_mask'].all()) or not torch.equal(out['depth'][~out['refined_mask']], p[~out['refined_mask']]):
                    raise AssertionError('Golden full-query/P fallback contract failed')
                result = {'depth': out['depth'][0,0].cpu().numpy(),
                    'refined_mask': out['refined_mask'][0,0].cpu().numpy(),
                    'native_depth': F.interpolate(out['depth'], size=row['native_hw'], mode='bilinear', align_corners=False)[0,0].cpu().numpy()}
        output = destination / (row['cache_id'] + '.npz')
        np.savez(output, depth=result['depth'], refined_mask=result['refined_mask'], native_depth=result['native_depth'])
        receipts.append({'cache_id': row['cache_id'], 'sha256': sha(output), 'refined_fraction': float(result['refined_mask'].mean())})
        print(json.dumps({'phase': args.phase, 'sample': row['cache_id'], 'elapsed': time.monotonic()-began}), flush=True)
    if denied:
        raise AssertionError('Historical file access was attempted')
    marker = {'status': 'complete', 'phase': args.phase, 'pid': os.getpid(),
              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
              'physical_gpu4_uuid': 'GPU-9c9f5c94-e2c6-4f01-6419-8bd0dfc4ba37',
              'cuda_device': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
              'samples': receipts, 'seconds': time.monotonic()-began,
              'relocated_source': args.phase == 'portable', 'historical_access_denied': denied,
              'historical_access_guard_enabled': args.phase == 'portable',
              'guard_scope': 'Python open/dlopen audit events; relocated tree has no historical links',
              'historical_checkpoint_guard_self_test': args.phase == 'portable'}
    (destination / 'complete.json').write_text(json.dumps(marker, indent=2)+'\n')


def compare(args):
    root = Path(args.output)
    record = json.loads((root / 'parity_inputs.json').read_text())
    checks = []
    border = np.zeros((240,320), dtype=bool)
    border[:4] = border[-4:] = True
    border[:, :4] = border[:, -4:] = True
    for row in record['samples']:
        arrays = []
        for phase in ('golden', 'portable'):
            marker = json.loads((root / phase / 'complete.json').read_text())
            path = root / phase / (row['cache_id']+'.npz')
            receipt = next(r for r in marker['samples'] if r['cache_id']==row['cache_id'])
            if marker['status'] != 'complete' or sha(path) != receipt['sha256']:
                raise ValueError('Parity output seal changed')
            with np.load(path, allow_pickle=False) as data:
                arrays.append({key: data[key].copy() for key in data.files})
        old, new = arrays
        if not np.array_equal(old['refined_mask'], new['refined_mask']):
            raise AssertionError('Refined mask differs: '+row['cache_id'])
        errors = {}
        for key in ('depth', 'native_depth'):
            a, b = old[key].astype(np.float64), new[key].astype(np.float64)
            if not np.allclose(a, b, atol=1e-6, rtol=0):
                raise AssertionError('Depth parity exceeds 1e-6m: '+row['cache_id'])
            d = np.abs(a-b)
            errors[key] = {'max_abs_m': float(d.max()), 'mae_m': float(d.mean()),
                           'bitwise_equal_fraction': float((old[key] == new[key]).mean())}
        with np.load(root / row['file'], allow_pickle=False) as data:
            p = data['depth_input'].copy()
        fallback = ~new['refined_mask']
        if not np.array_equal(new['depth'][fallback], p[fallback]):
            raise AssertionError('Fallback differs from P')
        if row['kind'] == 'empty_support' and (new['refined_mask'].any() or not np.array_equal(new['depth'], p)):
            raise AssertionError('Empty support must return exactly all P')
        checks.append({'cache_id': row['cache_id'], 'domain': row['domain'], 'kind': row['kind'],
            'errors': errors, 'refined_mask_exact': True, 'fallback_P_exact': True,
            'fallback_fraction': float(fallback.mean()),
            'border4_max_abs_m': float(np.abs(old['depth']-new['depth'])[border].max())})
    result = {'status': 'passed', 'purpose': 'E63 implementation parity only, not winner selection',
        'samples': checks, 'checkpoint_sha256': record['checkpoint_sha256'],
        'bundle_metadata_sha256': record['stage2_bundle_sha256'],
        'relocated_portable_sources': True, 'historical_path_access_guard': True,
        'test_tolerance_abs_m': 1e-6, 'gpu': 4}
    (root / 'parity.complete.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['prepare','golden','portable','compare'])
    p.add_argument('--output', required=True)
    for name in ('inputs','stage1-root','checkpoint','checkpoint-sha256','project','golden-source'):
        p.add_argument('--'+name)
    args = p.parse_args()
    if args.phase == 'prepare': prepare(args)
    elif args.phase == 'compare': compare(args)
    else: run_worker(args)
