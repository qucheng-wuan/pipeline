#!/usr/bin/env python3
"""Full-roster portable Stage2 acceptance, with separate prediction/GT processes.

``cached`` compares the same sealed RGB/P/K against the finally selected sweep
checkpoint. ``live`` uses a complete portable Stage1 RGB-only export. Neither
prediction mode accepts labels. Only metric-worker opens the label manifest,
after all predictions have been sealed. Paths are explicit; no old checkout is
imported. Running ``run`` requires an explicit authorized --gpus list.
"""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from transparent_depth.input_guard import RGBInputGuard
SCHEMA = 'portable_stage2_full_roster_v1'
IMAGES = 1226
TEST_IMAGES = 521
ABS_TOL_M = 1e-6
METRIC_TOL_MM = 2e-5
METHODS = ('stage1_native', 'stage1_resized', 'stage1_lidf')
REGIONS = ('all', 'transparent', 'background')
IDENTITY = ('cache_id', 'sample_id', 'rgb_sha256', 'native_hw', 'domain',
            'official_split', 'benchmark_group')


class Stage2InputGuard(RGBInputGuard):
    """The same allowlist with both legacy and current OpenEXR reader APIs.

    Stage1 uses OpenEXR 3.5; the pinned Stage2 3.2.4 has InputFile but no File.
    Keep this adaptation local instead of changing the frozen Stage1 guard.
    """
    def __enter__(self):
        from unittest.mock import patch
        import cv2
        import OpenEXR
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        def wrapper(original):
            def read_path(path, *args, **kwargs):
                self.check(path)
                return original(path, *args, **kwargs)
            return read_path
        self.stack.enter_context(patch.object(cv2, 'imread', wrapper(cv2.imread)))
        for name in ('File', 'InputFile', 'TiledInputFile', 'MultiPartInputFile'):
            if hasattr(OpenEXR, name):
                self.stack.enter_context(patch.object(OpenEXR, name, wrapper(getattr(OpenEXR, name))))
        sys.addaudithook(self._audit)
        self.active = True
        return self


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def check_ref(value):
    require(sha(value['path']) == value['sha256'], 'Referenced file changed: ' + value['path'])


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.%d' % os.getpid())
    with temporary.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sources():
    from transparent_depth.stage2 import source_hashes
    result = {'src/' + name: value for name, value in source_hashes().items()}
    names = ('scripts/validate_stage2_full.py', 'src/transparent_depth/input_guard.py',
             'src/transparent_depth/evaluation.py', 'src/transparent_depth/statistics.py',
             'src/native_fusion/geometry.py', 'src/native_fusion/evaluation.py')
    result.update({name: sha(PROJECT / name) for name in names})
    return result


def validate_roster(actual, expected, expected_count=IMAGES):
    require(len(actual) == len(expected) == expected_count, 'Incomplete full image coverage')
    indexed = {row['cache_id']: row for row in actual}
    require(len(indexed) == len(actual) and len({row['sample_id'] for row in actual}) == len(actual),
            'Duplicate image identity')
    require(set(indexed) == {row['cache_id'] for row in expected}, 'Image roster differs')
    for row in expected:
        require(all(indexed[row['cache_id']].get(key) == row[key] for key in IDENTITY),
                'RGB/native geometry/domain identity differs: ' + row['cache_id'])


def selection_evidence(root, metadata):
    """Consume completed all-80 verification; do not rescan all 98,080 arrays."""
    root = Path(root)
    selected = read(root / 'selection.complete.json')
    verified = read(root / 'publication_verification.complete.json')
    resolution = read(root / 'selected_geometry/resolved_selection.json')
    require(selected.get('status') == 'complete' and selected.get('completed_epochs') == 80
            and len(selected['ranking']) == 80
            and {r['epoch'] for r in selected['ranking']} == set(range(1, 81)),
            'Release acceptance requires all eighty checkpoints')
    require(verified.get('status') == 'passed' and verified.get('epochs') == 80
            and verified.get('predictions_verified') == 80 * IMAGES
            and verified.get('selection') == ref(root / 'selection.complete.json'),
            'The full sweep lacks completed independent verification')
    require(resolution.get('status') == 'complete'
            and resolution.get('original_selection') == ref(root / 'selection.complete.json')
            and resolution.get('verification') == ref(root / 'publication_verification.complete.json'),
            'Final near-tie selection has not been authenticated')
    epoch = resolution['selected_epoch']
    ranked = next(r for r in selected['ranking'] if r['epoch'] == epoch)
    require(metadata['origin']['epoch_one_based'] == epoch
            and metadata['origin']['checkpoint_sha256'] == ranked['checkpoint_sha256']
            and resolution['selected_checkpoint']['sha256'] == ranked['checkpoint_sha256'],
            'Portable bundle is not the finally resolved sweep checkpoint')
    policy = selected['selection_policy']
    require(policy.get('uses_test_labels_for_checkpoint_selection') is True
            and policy.get('independent_test_performance_claimed') is False,
            'Test-informed selection disclosure is required')
    seal_path = root / ('epoch%03d' % epoch) / 'predictions.complete.json'
    seal = read(seal_path)
    require(seal.get('images') == IMAGES and seal.get('smoke') is False
            and seal.get('label_access') is False and seal.get('input_alpha') == 0.0
            and seal.get('compute_loss') is False
            and seal.get('checkpoint_sha256') == ranked['checkpoint_sha256'],
            'Reference predictions are incomplete or from a different checkpoint')
    epoch_marker = read(seal_path.parent / 'complete.json')
    require(epoch_marker.get('status') == 'complete'
            and epoch_marker.get('prediction_seal') == ref(seal_path),
            'Reference prediction seal differs from the verified epoch')
    sweep_contract = read(root / 'contract.json')
    require(selected['contract'] == ref(root / 'contract.json'), 'Selected sweep contract changed')
    return {'selected_epoch': epoch, 'checkpoint_sha256': ranked['checkpoint_sha256'],
            'selection': ref(root / 'selection.complete.json'),
            'verification': ref(root / 'publication_verification.complete.json'),
            'resolved_selection': ref(root / 'selected_geometry/resolved_selection.json'),
            'reference_prediction_seal': ref(seal_path), 'policy': policy,
            'sweep_contract': ref(root / 'contract.json'),
            'input_manifest_sha256': sweep_contract['input_manifest']['sha256'],
            'label_manifest_sha256': sweep_contract['label_manifest']['sha256'],
            'historical_transparent_mae_mm': resolution['scores'][str(epoch)]['transparent_mae_mm']}, seal


def portable_prediction_map(path):
    path = Path(path)
    marker = read(path.parent / 'complete.json')
    contract = read(path.parent / 'contract.json')
    require(marker.get('schema') == 'portable_stage1_predictions_seal_v1'
            and marker.get('status') == 'complete' and marker.get('images') == IMAGES
            and marker.get('ground_truth_access') is False
            and marker.get('all_arrays_sha256_verified') is True
            and marker.get('manifest_sha256') == sha(path)
            and marker.get('contract_sha256') == fingerprint(contract)
            and contract.get('ground_truth_inputs') is False,
            'A complete portable RGB-only Stage1 export is required')
    registry = read(PROJECT / 'provenance/stage1_assets.json')
    expected_weights = {r['id']: r['sha256'] for r in registry['assets']}
    require(contract.get('assets_sha256') == sha(PROJECT / 'provenance/stage1_assets.json')
            and contract.get('sources_sha256') == sha(PROJECT / 'provenance/stage1_runtime_sources.json'),
            'Live Stage1 did not use this release source/asset identity')
    require(set(marker.get('worker_receipts', {})) == {str(rank) for rank in range(contract['world_size'])},
            'Live Stage1 requires a completed receipt for every actual producer rank')
    for rank, value in marker['worker_receipts'].items():
        worker_path = inside(path.parent, value['path'])
        require(sha(worker_path) == value['sha256'], 'Live Stage1 worker receipt changed')
        audit = read(worker_path)
        require(audit.get('status') == 'complete' and audit['rank'] == int(rank)
                and audit['contract_sha256'] == marker['contract_sha256']
                and audit['file_access_guard']['ground_truth_read'] is False
                and not audit['file_access_guard']['denied_paths'], 'Live Stage1 producer isolation failed')
    rows = lines(path)
    indexed = {r['sample_id']: r for r in rows}
    require(len(indexed) == len(rows) == IMAGES, 'Portable Stage1 roster is incomplete or duplicate')
    for row in rows:
        metadata = row.get('metadata', {})
        require(row.get('status') == 'complete' and row.get('ground_truth_access') is False
                and row.get('legacy_cache_used') is False
                and row.get('contract_sha256') == marker['contract_sha256']
                and metadata.get('weight_sha256') == expected_weights
                and metadata.get('assets_sha256') == contract['assets_sha256']
                and metadata.get('source_manifest_sha256') == contract['sources_sha256'],
                'Portable Stage1 receipt does not belong to this producer')
    return indexed, marker


def inside(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Artifact path must stay relative to its declared root')
    root = Path(root).resolve()
    result = (root / relative).resolve()
    require(result.is_relative_to(root), 'Artifact path escapes its declared root')
    return result


def prepare(args):
    from v2_lidf.data import load_input_manifest
    from transparent_depth.stage2 import validate_bundle
    weights, metadata = validate_bundle(args.bundle)
    selection, reference = selection_evidence(args.selection_root, metadata)
    require(sha(args.input_manifest) == selection['input_manifest_sha256'],
            'Both acceptance modes require the same authenticated RGB/K roster as the sweep')
    rows = load_input_manifest(args.input_manifest, 'evaluation')
    validate_roster(rows, reference['predictions'])
    require(sum(r['official_split'].endswith('test') for r in rows) == TEST_IMAGES,
            'The original 521 test images must remain identifiable')
    golden = {r['cache_id']: r for r in reference['predictions']}
    dependencies = {'input_manifest': ref(args.input_manifest),
                    'bundle_metadata': ref(Path(args.bundle) / 'model.json'), 'bundle_weights': ref(weights)}
    indexed = None
    if args.mode == 'cached':
        require(args.stage1_root is not None, 'cached mode needs --stage1-root')
        source_root = Path(args.stage1_root)
        marker = read(source_root / 'complete.json')
        original = read(Path(args.selection_root) / 'contract.json')
        require(marker.get('status') == 'complete' and marker.get('ground_truth_access') is False
                and marker.get('all_arrays_sha256_verified') is True
                and marker['input_manifest_sha256'] == sha(args.input_manifest)
                and marker['sealed_inputs_manifest_sha256'] == sha(source_root / 'inputs.sealed.jsonl')
                and sha(source_root / 'complete.json') == original['stage1_seal']['sha256']
                and sha(args.input_manifest) == original['input_manifest']['sha256'],
                'Cached P must be the exact sealed input used by the winning sweep')
        indexed = {r['cache_id']: r for r in lines(source_root / 'inputs.sealed.jsonl')}
        dependencies.update(stage1_seal=ref(source_root / 'complete.json'),
                            stage1_receipts=ref(source_root / 'inputs.sealed.jsonl'))
    else:
        require(args.stage1_predictions is not None, 'live mode needs --stage1-predictions')
        indexed, marker = portable_prediction_map(args.stage1_predictions)
        source_root = Path(args.stage1_predictions).parent
        dependencies.update(stage1_seal=ref(source_root / 'complete.json'),
                            stage1_contract=ref(source_root / 'contract.json'),
                            stage1_receipts=ref(args.stage1_predictions))
        require(set(indexed) == {r['sample_id'] for r in rows}, 'Live Stage1 sample roster differs')
    planned = []
    for row in rows:
        item = dict(row)
        require(sha(row['rgb_path']) == row['rgb_sha256'], 'RGB bytes changed: ' + row['cache_id'])
        if args.mode == 'cached':
            receipt = indexed[row['cache_id']]
            require(receipt['rgb_sha256'] == row['rgb_sha256'], 'Cached RGB identity differs')
            low = source_root / 'inputs' / (row['cache_id'] + '.npz')
            native = source_root / 'native' / (row['cache_id'] + '.npy')
            require(sha(low) == receipt['input_sha256'] and sha(native) == receipt['native_sha256'],
                    'Frozen input/native cache changed')
            item['input_cache'] = ref(low)
        else:
            receipt = indexed[row['sample_id']]
            require(receipt['rgb_sha256'] == row['rgb_sha256'] and receipt['native_hw'] == row['native_hw'],
                    'Live Stage1 RGB/native identity differs')
            native = inside(source_root, receipt['prediction']['path'])
            require(sha(native) == receipt['prediction']['sha256'], 'Live Stage1 prediction changed')
        item['stage1_native'] = ref(native)
        golden_path = Path(selection['reference_prediction_seal']['path']).parent / 'predictions' / (row['cache_id'] + '.npz')
        require(sha(golden_path) == golden[row['cache_id']]['prediction_sha256'], 'Selected sweep prediction changed')
        item['golden_prediction'] = ref(golden_path)
        planned.append(item)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    roster_text = ''.join(canonical(row) + '\n' for row in planned)
    contract = {'schema': SCHEMA, 'images': IMAGES, 'test_images': TEST_IMAGES, 'mode': args.mode,
        'world_size': args.world, 'sources': sources(), 'selection': selection,
        'dependencies': dependencies, 'bundle': str(Path(args.bundle).resolve()),
        'roster_sha256': hashlib.sha256(roster_text.encode()).hexdigest(),
        'prediction_inputs': ['RGB', 'frozen_RGB_predicted_P', 'known_intrinsics'],
        'labels_used_as_prediction_input': False, 'input_alpha': 0.0,
        'cached_parity_abs_tolerance_m': ABS_TOL_M, 'cached_parity_rtol': 0.0,
        'native_resize': 'portable GPU bilinear align_corners=False; reference CPU resize compared separately',
        'live_quality_comparison': 'new live-P end-to-end measurement; no numerical equivalence inferred'}
    path = output / 'contract.json'
    if path.exists():
        require(read(path) == contract and sha(output / 'roster.jsonl') == contract['roster_sha256'],
                'Existing output belongs to different inputs/model/source/protocol')
    else:
        atomic_text(output / 'roster.jsonl', roster_text)
        atomic_json(path, contract)
    return contract


def authenticate(output):
    output = Path(output)
    contract = read(output / 'contract.json')
    require(contract.get('schema') == SCHEMA and contract.get('images') == IMAGES,
            'Unsupported or partial acceptance contract')
    require(contract['sources'] == sources(), 'Portable acceptance source changed')
    require(sha(output / 'roster.jsonl') == contract['roster_sha256'], 'Acceptance roster changed')
    for value in contract['dependencies'].values():
        check_ref(value)
    for name in ('selection', 'verification', 'resolved_selection', 'reference_prediction_seal', 'sweep_contract'):
        check_ref(contract['selection'][name])
    return contract, lines(output / 'roster.jsonl')


def resize_cpu(depth, native_hw):
    import torch
    return torch.nn.functional.interpolate(torch.from_numpy(np.asarray(depth).copy())[None, None],
        size=tuple(native_hw), mode='bilinear', align_corners=False)[0, 0].numpy()


def input_arrays(row, mode):
    from PIL import Image
    from transparent_depth.stage2 import resize_native_inputs, prepare_stage2_inputs
    check_ref(row['stage1_native'])
    native = np.load(row['stage1_native']['path'], allow_pickle=False)
    require(native.dtype == np.float32 and list(native.shape) == row['native_hw']
            and np.isfinite(native).all() and bool((native > 0).all()), 'Invalid native Stage1 P')
    if mode == 'cached':
        check_ref(row['input_cache'])
        with np.load(row['input_cache']['path'], allow_pickle=False) as data:
            require(set(data.files) <= {'rgb', 'depth_input', 'intrinsics', 'native_hw', 'metadata_json'},
                    'A prediction input archive may not contain labels')
            rgb, depth, k = (data[key].copy() for key in ('rgb', 'depth_input', 'intrinsics'))
            metadata = json.loads(str(data['metadata_json'].item()))
        require(metadata.get('rgb_sha256') == row['rgb_sha256']
                and metadata.get('ground_truth_access') is False, 'Input archive RGB identity/purity differs')
    else:
        require(sha(row['rgb_path']) == row['rgb_sha256'], 'Live RGB changed after prepare')
        with Image.open(row['rgb_path']) as image:
            rgb_native = np.asarray(image.convert('RGB')).copy()
        require(list(rgb_native.shape[:2]) == row['native_hw'], 'Live RGB geometry differs')
        rgb, depth, k = resize_native_inputs(rgb_native, native, np.asarray(row['intrinsics_native']))
    prepare_stage2_inputs(rgb, depth, k)
    require(np.allclose(k, row['intrinsics'], atol=1e-5, rtol=1e-6), 'Model camera calibration differs')
    return rgb, depth, k, native


def difference(actual, expected, tolerance=ABS_TOL_M):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            'Comparison requires matching finite arrays')
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {'max_abs_m': float(delta.max()), 'mae_m': float(delta.mean()),
            'rmse_m': float(np.square(delta).mean() ** .5),
            'bitwise_equal_fraction': float((a == b).mean()),
            'pixels_over_tolerance': int((delta > tolerance).sum()),
            'abs_tolerance_m': tolerance, 'rtol': 0.0}


def compare_golden(result, row):
    check_ref(row['golden_prediction'])
    with np.load(row['golden_prediction']['path'], allow_pickle=False) as archive:
        low, mask = archive['depth'].copy(), archive['refined_mask'].copy()
    require(mask.shape == result['refined_mask'].shape, 'Reference refinement mask shape differs')
    values = {'low': difference(result['depth'], low),
              'native_vs_reference_cpu_resize': difference(result['native_depth'], resize_cpu(low, row['native_hw'])),
              'refined_mask_disagree_pixels': int((mask != result['refined_mask']).sum())}
    values['passed'] = values['low']['pixels_over_tolerance'] == 0 and values['refined_mask_disagree_pixels'] == 0
    # Native CPU/GPU interpolation differences are reported separately. The
    # low-resolution neural output and exact support are the equivalence gate.
    return values


def validate_arrays(values, row):
    required = {'depth', 'native_depth', 'refined_mask', 'stage1_native', 'stage1_resized'}
    require(set(values) == required, 'Prediction archive fields differ')
    for key in required - {'refined_mask'}:
        array = values[key]
        shape = [240, 320] if key == 'depth' else row['native_hw']
        require(array.dtype == np.float32 and list(array.shape) == shape
                and np.isfinite(array).all() and bool((array > 0).all()), 'Invalid prediction array: ' + key)
    require(values['refined_mask'].dtype == np.bool_ and values['refined_mask'].shape == (240, 320),
            'Refinement mask must be boolean at model resolution')


def read_output(output, row, receipt):
    path = inside(output, receipt['prediction']['path'])
    require(sha(path) == receipt['prediction']['sha256'], 'Prediction changed after sealing')
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    validate_arrays(arrays, row)
    return arrays


def project_modules():
    result = {}
    for name, module in list(sys.modules.items()):
        if not any(name == p or name.startswith(p + '.') for p in ('transparent_depth', 'v2_lidf', 'native_fusion')):
            continue
        if not getattr(module, '__file__', None):
            continue
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(PROJECT), 'Historical/foreign numerical package imported: ' + name)
        result[name] = str(path.relative_to(PROJECT))
    return result


def worker(args):
    import torch
    from transparent_depth.stage2 import load_stage2, predict_stage2
    from transparent_depth.input_guard import configure_jit_cache
    output = Path(args.output)
    contract, rows = authenticate(output)
    require(0 <= args.rank < contract['world_size'], 'Invalid prediction rank')
    require(os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
            and len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) == 1
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES'))
            and torch.cuda.device_count() == 1, 'Prediction worker requires one explicitly visible authorized GPU')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    selected = rows[args.rank::contract['world_size']]
    cache_root = configure_jit_cache(output / 'runtime' / ('rank%02d' % args.rank))
    metadata = [r['stage1_native']['path'] for r in selected]
    metadata += [r['golden_prediction']['path'] for r in selected]
    metadata += [r['input_cache']['path'] for r in selected if 'input_cache' in r]
    runtime = [os.environ['CUDA_HOME']] if os.environ.get('CUDA_HOME') else []
    guard = Stage2InputGuard(project=PROJECT, asset_root=contract['bundle'],
        rgb_paths=[r['rgb_path'] for r in selected], metadata_paths=metadata,
        output_root=output, cache_root=cache_root, extra_runtime_roots=runtime)
    began = time.monotonic()
    audit = {'schema': SCHEMA, 'rank': args.rank, 'world_size': contract['world_size'],
        'contract_sha256': sha(output / 'contract.json'), 'status': 'running',
        'pid': os.getpid(), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'prediction_inputs': ['rgb', 'depth_input', 'intrinsics'], 'input_alpha': 0.0,
        'compute_loss': False, 'labels_used_as_prediction_input': False, 'completed_ids': []}
    attempt = 'workers/attempts/rank%02d_%d_%d.json' % (args.rank, time.time_ns(), os.getpid())
    audit['attempt'] = attempt
    audit['new_ids'] = []
    audit['reused_ids'] = []
    model = None
    try:
        with guard:
            for index, row in enumerate(selected):
                record_path = output / 'receipts' / (row['cache_id'] + '.json')
                if record_path.exists():
                    receipt = read(record_path)
                    require(receipt['contract_sha256'] == audit['contract_sha256']
                            and receipt['rank'] == args.rank, 'Resume receipt belongs to a different producer')
                    validate_roster([receipt], [row], 1)
                    read_output(output, row, receipt)
                    audit['reused_ids'].append(row['cache_id'])
                else:
                    rgb, depth, k, baseline = input_arrays(row, contract['mode'])
                    if model is None:
                        model = load_stage2(contract['bundle'], 'cuda:0')
                    result = predict_stage2(model, rgb, depth, k, tuple(row['native_hw']), int(row['rgb_sha256'][:8], 16))
                    parity = compare_golden(result, row)
                    arrays = {key: result[key] for key in ('depth', 'native_depth', 'refined_mask')}
                    arrays.update(stage1_native=baseline, stage1_resized=resize_cpu(depth, row['native_hw']))
                    validate_arrays(arrays, row)
                    target = output / 'predictions' / (row['cache_id'] + '.npz')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + '.tmp.%d' % os.getpid())
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, **arrays)
                    os.replace(temporary, target)
                    receipt = {key: row[key] for key in IDENTITY}
                    receipt.update(schema=SCHEMA, rank=args.rank, status='complete',
                        contract_sha256=audit['contract_sha256'], labels_used_as_prediction_input=False,
                        producer_attempt=attempt, producer_pid=os.getpid(), created_unix=time.time(),
                        prediction={'path': str(target.relative_to(output)), 'sha256': sha(target)},
                        input_sha256={key: row[key]['sha256'] for key in ('stage1_native', 'golden_prediction', 'input_cache') if key in row},
                        parity_against_cached_winner=parity, refined_fraction=float(result['refined_mask'].mean()),
                        model_diagnostics=result['diagnostics'])
                    atomic_json(record_path, receipt)
                    audit['new_ids'].append(row['cache_id'])
                audit['completed_ids'].append(row['cache_id'])
                if index % 50 == 0:
                    print(json.dumps({'phase': 'predict', 'rank': args.rank, 'images': index + 1,
                                      'assigned': len(selected), 'seconds': time.monotonic() - began}), flush=True)
        require(not guard.denials, 'Prediction file-access guard recorded a denied read')
        audit.update(status='complete', predictor_instances=int(model is not None), modules=project_modules(),
                     torch=str(torch.__version__), device=torch.cuda.get_device_name(0),
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        audit.update(status='failed', error=repr(error))
        raise
    finally:
        # A new attempt may resume completed arrays, but never pretends that a
        # failed attempt completed the roster. The latest audit must be complete.
        receipt = guard.receipt()
        receipt['scope'] = 'Python audit open events + OpenCV/current-and-legacy OpenEXR readers; not OS sandbox'
        receipt.pop('legacy_cache_read', None)
        receipt['historical_P_cache_is_declared_input'] = contract['mode'] == 'cached'
        audit.update(file_access_guard=receipt, seconds=time.monotonic() - began)
        atomic_json(output / attempt, audit)
        atomic_json(output / 'workers' / ('rank%02d.json' % args.rank), audit)


def seal(args):
    output = Path(args.output)
    contract, rows = authenticate(output)
    digest = sha(output / 'contract.json')
    records, workers, producers = [], {}, {}
    for rank in range(contract['world_size']):
        path = output / 'workers' / ('rank%02d.json' % rank)
        record = read(path)
        require(record.get('status') == 'complete' and record['contract_sha256'] == digest
                and record['rank'] == rank and record['world_size'] == contract['world_size']
                and record['completed_ids'] == [r['cache_id'] for r in rows[rank::contract['world_size']]]
                and record['labels_used_as_prediction_input'] is False
                and record['file_access_guard']['ground_truth_read'] is False
                and record['file_access_guard']['strict_read_allowlist'] is True
                and not record['file_access_guard']['denied_paths'], 'A prediction rank lacks complete clean execution evidence')
        workers[str(rank)] = ref(path)
    for row_index, row in enumerate(rows):
        receipt = read(output / 'receipts' / (row['cache_id'] + '.json'))
        require(receipt.get('status') == 'complete' and receipt['contract_sha256'] == digest
                and receipt['rank'] == row_index % contract['world_size']
                and receipt['labels_used_as_prediction_input'] is False, 'Incomplete/mixed prediction receipt')
        attempt_path = inside(output, receipt['producer_attempt'])
        if str(attempt_path) not in producers:
            produced = read(attempt_path)
            require(produced['contract_sha256'] == digest
                    and produced['labels_used_as_prediction_input'] is False
                    and not produced['file_access_guard']['denied_paths']
                    and produced['file_access_guard']['ground_truth_read'] is False,
                    'Original producer attempt does not have clean input-isolation evidence')
            producers[str(attempt_path)] = (ref(attempt_path), produced)
        produced = producers[str(attempt_path)][1]
        require(receipt['producer_pid'] == produced['pid']
                and row['cache_id'] in produced['new_ids']
                and receipt['rank'] == produced['rank'], 'Prediction lacks original forward-execution provenance')
        read_output(output, row, receipt)
        records.append(receipt)
    validate_roster(records, rows)
    require({p.stem for p in (output / 'receipts').glob('*.json')} == {r['cache_id'] for r in rows},
            'Foreign prediction receipt found')
    failed = [r['cache_id'] for r in records if not r['parity_against_cached_winner']['passed']]
    atomic_text(output / 'predictions.jsonl', ''.join(canonical(r) + '\n' for r in records))
    result = {'schema': SCHEMA, 'status': 'complete', 'images': IMAGES, 'test_images': TEST_IMAGES,
        'mode': contract['mode'], 'contract_sha256': digest, 'all_arrays_sha256_verified': True,
        'predictions': ref(output / 'predictions.jsonl'), 'workers': workers,
        'original_producer_attempts': {path: value[0] for path, value in producers.items()},
        'labels_used_as_prediction_input': False, 'selected_checkpoint_sha256': contract['selection']['checkpoint_sha256'],
        'cached_input_parity_required': contract['mode'] == 'cached',
        'cached_input_parity_passed': not failed if contract['mode'] == 'cached' else None,
        'different_from_historical_cached_input_images': len(failed), 'different_image_ids': failed,
        'scope': 'all 1226 predictions complete; quality requires separate labelled evaluation',
        'completed_unix': time.time()}
    atomic_json(output / 'predictions.complete.json', result)
    return result


def prediction_seal(output, verify_arrays=False):
    output = Path(output)
    contract, rows = authenticate(output)
    marker = read(output / 'predictions.complete.json')
    require(marker.get('status') == 'complete' and marker.get('images') == IMAGES
            and marker.get('labels_used_as_prediction_input') is False
            and marker.get('all_arrays_sha256_verified') is True
            and marker['contract_sha256'] == sha(output / 'contract.json'),
            'All predictions must be sealed before labels are opened')
    check_ref(marker['predictions'])
    for value in marker['workers'].values():
        check_ref(value)
    require(set(marker['workers']) == {str(rank) for rank in range(contract['world_size'])},
            'Prediction seal omits a worker')
    require(bool(marker.get('original_producer_attempts')), 'Prediction seal omits actual forward-execution evidence')
    for value in marker['original_producer_attempts'].values():
        check_ref(value)
    records = lines(marker['predictions']['path'])
    validate_roster(records, rows)
    require(all(r.get('status') == 'complete' and r['contract_sha256'] == marker['contract_sha256']
                and r['rank'] == index % contract['world_size']
                and r['labels_used_as_prediction_input'] is False for index, r in enumerate(records)),
            'Prediction seal has mixed source/rank/purity receipts')
    if verify_arrays:
        for row, receipt in zip(rows, records):
            require(row['cache_id'] == receipt['cache_id'], 'Sealed ordering differs')
            read_output(output, row, receipt)
    return contract, rows, records, marker


def metric_worker(args):
    # No model is constructed in this process. The CLI only passes labels to
    # this phase, after the complete prediction seal exists.
    import torch
    from v2_lidf.data import load_label_manifest
    from native_fusion.evaluation import evaluate_prediction
    from transparent_depth.evaluation import load_evaluation_labels
    output = Path(args.output)
    contract, rows, records, marker = prediction_seal(output)
    require(args.label_manifest is not None, 'Only metric-worker requires --label-manifest')
    require(0 <= args.rank < contract['world_size'], 'Invalid metric rank')
    assigned = list(zip(rows, records))[args.rank::contract['world_size']]
    # Reauthenticate every assigned prediction before opening the label roster.
    for row, receipt in assigned:
        read_output(output, row, receipt)
    label_ref = ref(args.label_manifest)
    require(label_ref['sha256'] == contract['selection']['label_manifest_sha256'],
            'Use the same authenticated supervision manifest as the selected full1226 sweep')
    labels = {row['cache_id']: row for row in load_label_manifest(args.label_manifest, 'evaluation')}
    validate_roster(list(labels.values()), rows)
    device = args.metric_device
    torch.set_num_threads(2)
    if device != 'cpu':
        require(torch.cuda.device_count() == 1 and bool(os.environ.get('CUDA_VISIBLE_DEVICES')),
                'Metric worker requires one explicitly visible GPU')
        torch.cuda.set_device(0)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    begun, evaluated = time.monotonic(), []
    for index, (row, receipt) in enumerate(assigned):
        arrays = read_output(output, row, receipt)
        label = labels[row['cache_id']]
        label_assets = {}
        for key in ('target_depth_path', 'transparent_mask_path', 'normal_path'):
            if label.get(key):
                label_assets[key] = ref(label[key])
        for key, field in (('depth', 'target_depth_path'), ('transparent_mask', 'transparent_mask_path'), ('normal', 'normal_path')):
            if label.get('source_sha256', {}).get(key):
                require(label_assets[field]['sha256'] == label['source_sha256'][key], 'Evaluation label asset changed')
        batch = load_evaluation_labels(label, device)
        mapping = {'stage1_native': 'stage1_native', 'stage1_resized': 'stage1_resized', 'stage1_lidf': 'native_depth'}
        metrics = {name: evaluate_prediction(torch.from_numpy(arrays[key])[None, None].to(device), batch)
                   for name, key in mapping.items()}
        for region in REGIONS:
            require(len({metrics[name][region]['pixels'] for name in METHODS}) == 1,
                    'Evaluated methods do not share identical GT support')
        record = {key: row[key] for key in IDENTITY}
        record.update(metrics=metrics, prediction_sha256=receipt['prediction']['sha256'],
            normal_reference='synthetic_released_camera_normals' if label.get('normal_path') else 'real_depth_induced_proxy',
            label_assets=label_assets)
        evaluated.append(record)
        if index % 30 == 0:
            print(json.dumps({'phase': 'metric', 'rank': args.rank, 'images': index + 1,
                              'assigned': len(assigned), 'seconds': time.monotonic() - begun}), flush=True)
    path = output / 'metric_shards' / ('rank%02d.jsonl' % args.rank)
    atomic_text(path, ''.join(canonical(r) + '\n' for r in evaluated))
    atomic_json(path.with_suffix('.complete.json'), {'schema': SCHEMA, 'status': 'complete',
        'rank': args.rank, 'world_size': contract['world_size'], 'images': len(evaluated),
        'predictions_seal': ref(output / 'predictions.complete.json'), 'labels': label_ref,
        'records': ref(path), 'model_executed': False, 'labelled_process': True,
        'device': device, 'seconds': time.monotonic() - begun, 'pid': os.getpid()})


def metrics_merge(args):
    from native_fusion.evaluation import aggregate
    from transparent_depth.statistics import paired_difference
    output = Path(args.output)
    contract, rows, predictions, marker = prediction_seal(output)
    combined, label_refs, evidence = [], [], {}
    for rank in range(contract['world_size']):
        path = output / 'metric_shards' / ('rank%02d.complete.json' % rank)
        shard = read(path)
        require(shard.get('status') == 'complete' and shard['rank'] == rank
                and shard['world_size'] == contract['world_size']
                and shard['predictions_seal'] == ref(output / 'predictions.complete.json')
                and shard['model_executed'] is False and shard['labelled_process'] is True,
                'Metric shard lacks completed separate-process evidence')
        check_ref(shard['records'])
        check_ref(shard['labels'])
        part = lines(shard['records']['path'])
        validate_roster(part, rows[rank::contract['world_size']], len(rows[rank::contract['world_size']]))
        combined.extend(part)
        label_refs.append(shard['labels'])
        evidence[str(rank)] = ref(path)
    require(all(value == label_refs[0] for value in label_refs), 'Metric workers used different supervision manifests')
    indexed = {r['cache_id']: r for r in combined}
    combined = [indexed[r['cache_id']] for r in rows]
    validate_roster(combined, rows)
    expected = {r['cache_id']: r['prediction']['sha256'] for r in predictions}
    require(all(r['prediction_sha256'] == expected[r['cache_id']] for r in combined),
            'Metric record refers to different prediction bytes')
    subsets = {'full1226': combined, 'test521': [r for r in combined if r['official_split'].endswith('test')]}
    require(len(subsets['test521']) == TEST_IMAGES, 'Original test subset coverage differs')
    for group in sorted({r['benchmark_group'] for r in combined}):
        subsets[group] = [r for r in combined if r['benchmark_group'] == group]
    metrics = {name: aggregate(part) for name, part in subsets.items()}
    paired, rng = {}, np.random.default_rng(23)
    for subset in ('full1226', 'test521'):
        paired[subset] = {}
        for domain in ('combined', 'real', 'synthetic'):
            part = [r for r in subsets[subset] if domain == 'combined' or r['domain'] == domain]
            for region in REGIONS:
                # Preserve the registered paired-error CI protocol. Edge F1
                # can be null for only one prediction when GT has no edges, so
                # geometry is reported with its own metric-image support.
                for metric in ('mae_mm', 'abs_rel'):
                    paired[subset]['%s/%s/%s' % (domain, region, metric)] = paired_difference(part, region, metric, rng)
    reports = output / 'reports'
    atomic_text(reports / 'per_image.jsonl', ''.join(canonical(r) + '\n' for r in combined))
    atomic_json(reports / 'metrics.json', metrics)
    atomic_json(reports / 'paired_differences.json', paired)
    with (reports / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['subset', 'domain', 'method', 'region', 'averaging', 'metric', 'value', 'images', 'pixels'])
        for subset, domains in metrics.items():
            for domain, methods in domains.items():
                for method, regions in methods.items():
                    for region, item in regions.items():
                        for averaging in ('image_macro', 'pixel_pooled'):
                            for metric, value in item[averaging].items():
                                writer.writerow([subset, domain, method, region, averaging, metric, value, item['images'], item['pixels']])
    actual = metrics['full1226']['combined']['stage1_lidf']['transparent']['image_macro']['mae_mm']
    historical = contract['selection']['historical_transparent_mae_mm']
    metric_delta = actual - historical
    result = {'schema': SCHEMA, 'status': 'complete', 'images': IMAGES, 'test_images': TEST_IMAGES,
        'mode': contract['mode'], 'selected_epoch': contract['selection']['selected_epoch'],
        'selected_checkpoint_sha256': contract['selection']['checkpoint_sha256'],
        'predictions_seal': ref(output / 'predictions.complete.json'), 'labels': label_refs[0],
        'metric_shards': evidence, 'transparent_mae_mm': actual,
        'historical_cached_winner_transparent_mae_mm': historical,
        'transparent_mae_change_vs_historical_mm': metric_delta,
        'reports': {name: ref(reports / name) for name in ('metrics.json', 'per_image.jsonl', 'paired_differences.json', 'comparison.csv')},
        'cached_input_parity_passed': marker['cached_input_parity_passed'],
        'test_informed_selection': True, 'independent_test_performance_claimed': False,
        'bootstrap_scope': 'paired images; excludes scene correlation and 80-checkpoint selection uncertainty',
        'model_inputs_included_gt_or_mask': False, 'finished_unix': time.time()}
    if contract['mode'] == 'cached' and marker['cached_input_parity_passed'] is not True:
        result['status'] = 'failed_cached_input_numerical_parity'
    if contract['mode'] == 'cached':
        result['cached_transparent_mae_tolerance_mm'] = METRIC_TOL_MM
        result['cached_transparent_mae_agrees'] = math.isfinite(metric_delta) and abs(metric_delta) <= METRIC_TOL_MM
        if not result['cached_transparent_mae_agrees']:
            result['status'] = 'failed_cached_input_metric_parity'
    atomic_json(output / 'evaluation.complete.json', result)
    return result


def child_command(args, phase, rank):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), phase,
               '--output', str(args.output), '--rank', str(rank)]
    if phase == 'metric-worker':
        command += ['--label-manifest', str(args.label_manifest), '--metric-device', args.metric_device]
    return command


def launch_group(args, phase, gpus):
    output = Path(args.output)
    workers, started = [], time.time()
    for rank, gpu in enumerate(gpus):
        command = child_command(args, phase, rank)
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           PYTHONPATH=str(PROJECT / 'src'), PYTHONDONTWRITEBYTECODE='1')
        if phase == 'metric-worker' and args.metric_device == 'cpu':
            environment['CUDA_VISIBLE_DEVICES'] = ''
        log = output / 'logs' / ('%s_rank%02d.log' % (phase, rank))
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open('a')
        process = subprocess.Popen(command, env=environment, cwd=PROJECT,
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        workers.append((process, stream, {'rank': rank, 'physical_gpu': gpu, 'pid': process.pid,
                                        'command': command, 'log': str(log)}))
    record = {'phase': phase, 'started_unix': started, 'workers': [x[2] for x in workers]}
    atomic_json(output / ('%s.execution.json' % phase), record)
    # Every launched child is allowed to finish. No other process is stopped.
    pending = set(range(len(workers)))
    while pending:
        for index in list(pending):
            process, stream, item = workers[index]
            code = process.poll()
            if code is not None:
                item.update(returncode=code, completion_observed_unix=time.time())
                stream.close()
                item['log_sha256'] = sha(item['log'])
                pending.remove(index)
                atomic_json(output / ('%s.execution.json' % phase), record)
        if pending:
            time.sleep(5)
    record.update(status='passed' if all(r['returncode'] == 0 for r in record['workers']) else 'failed',
                  finished_unix=time.time())
    atomic_json(output / ('%s.execution.json' % phase), record)
    require(record['status'] == 'passed', phase + ' worker failed; preserve logs and inspect before resuming')


def run(args):
    import fcntl
    require(args.gpus is not None, 'run requires an explicitly authorized --gpus list')
    gpus = args.gpus.split(',')
    require(gpus and all(g.isdecimal() for g in gpus) and len(set(gpus)) == len(gpus), 'Invalid/duplicate GPU list')
    require(args.label_manifest is not None, 'run requires an explicit separate --label-manifest')
    args.world = len(gpus)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args)
        launch_group(args, 'worker', gpus)
        seal(args)
        launch_group(args, 'metric-worker', gpus)
        result = metrics_merge(args)
        require(result['status'] == 'complete', 'All-image measurement completed but cached-input parity failed')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'worker', 'seal', 'metric-worker', 'metrics-merge', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('cached', 'live'))
    for name in ('input-manifest', 'bundle', 'selection-root', 'stage1-root', 'stage1-predictions', 'label-manifest'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--gpus', help='Explicitly authorized physical IDs, e.g. 0,1,2,3,4,5,6,7')
    parser.add_argument('--metric-device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args()
    if args.phase in ('prepare', 'run'):
        for name in ('mode', 'input_manifest', 'bundle', 'selection_root'):
            require(getattr(args, name) is not None, '--' + name.replace('_', '-') + ' is required')
        require(args.world > 0, 'world must be positive')
    functions = {'prepare': prepare, 'worker': worker, 'seal': seal, 'metric-worker': metric_worker,
                 'metrics-merge': metrics_merge, 'run': run}
    result = functions[args.phase](args)
    if result is not None:
        print(json.dumps({key: result[key] for key in ('status', 'images', 'mode', 'transparent_mae_mm') if key in result}), flush=True)


if __name__ == '__main__':
    main()
