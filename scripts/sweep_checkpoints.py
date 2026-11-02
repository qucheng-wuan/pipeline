"""Authenticated 80-checkpoint, full-1226 *test-informed* selection.

Prediction workers never load labels. A distinct persistent spawned process
opens labels only after an epoch's complete prediction seal exists. Epochs are
ranked by image-macro transparent MAE; empty supports follow the original
evaluator (null, excluded from metric image mean), ties select earlier epochs.
This is a historical production-source audit tool, not the portable inference
entrypoint. No training file or checkpoint is changed.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
TRAIN_PROJECT = Path('/home/hengxianli/FusionV2_LIDF_SPVNAS_GPU567_20261031')
V2_SOURCE = Path('/home/hengxianli/FAR_MoGe_TransNormal_V2_20261021/src')
ROOT = Path('/mnt/gluster_dcpu/hengxianli/fusion_v2_lidf_spvnas_20261030/continuation_gpu567_20261031')
OUTPUT = ROOT / 'selection_full1226_transparent_mae_20261101'
REFERENCE = ROOT / 'interim_evaluations/e063_full1226_20261101'
SCHEMA = 'full1226_transparent_mae_checkpoint_sweep_v1'
REGIONS = ('all', 'transparent', 'background')
METRICS = ('mae_mm', 'rmse_mm', 'abs_rel', 'delta_1.05', 'delta_1.10', 'delta_1.25')
POLICY = {'epochs': list(range(1, 81)), 'selection_images': 1226,
          'primary_metric': 'full1226/combined/transparent/image_macro/mae_mm',
          'tie_break': 'smaller_epoch', 'empty_region': 'null_per_image_excluded_from_macro',
          'uses_test_labels_for_checkpoint_selection': True,
          'independent_test_performance_claimed': False,
          'original_val704_selection_is_preserved_separately': True,
          'gt_or_mask_used_as_prediction_input': False, 'input_alpha': 0.0}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.%d' % os.getpid())
    with tmp.open('w') as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_lines(path, rows):
    tmp = Path(path).with_name(Path(path).name + '.tmp.%d' % os.getpid())
    with tmp.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def reference_rows(args):
    return lines(args.reference / 'reports/per_image.jsonl')


def install_paths(args):
    sys.path.insert(0, str(args.train_project / 'src'))
    sys.path.insert(0, str(args.v2_source))


def execution_sources(args):
    files = [Path(__file__).resolve()]
    files += [p for p in (args.train_project / 'src/v2_lidf').rglob('*')
              if p.is_file() and p.suffix in ('.py', '.cpp', '.cu', '.h', '.hpp')]
    files += [args.v2_source / 'native_fusion' / name for name in ('geometry.py', 'data.py', 'evaluation.py')]
    return {str(p.resolve()): sha(p) for p in sorted(set(files))}


def image_identity(row):
    return {k: row[k] for k in ('cache_id', 'sample_id', 'rgb_sha256', 'domain',
                                'official_split', 'benchmark_group', 'native_hw')}


def validate_roster(rows, expected):
    require(len(rows) == len(expected) and len({r['cache_id'] for r in rows}) == len(rows),
            'Incomplete or duplicate image coverage')
    by_id = {r['cache_id']: r for r in rows}
    require(set(by_id) == {r['cache_id'] for r in expected}, 'Image identities differ')
    for row in expected:
        require(all(by_id[row['cache_id']].get(k) == v for k, v in image_identity(row).items()),
                'RGB/split/native identity differs')


def depth_only(metrics):
    return {region: {k: metrics[region][k] for k in ('pixels', *METRICS)}
            for region in REGIONS}


def aggregate(rows):
    result = {}
    parts = {'full1226': rows, 'test521': [r for r in rows if r['official_split'].endswith('test')]}
    for group in sorted({r['benchmark_group'] for r in rows}):
        parts[group] = [r for r in rows if r['benchmark_group'] == group]
    for subset, part in parts.items():
        result[subset] = {}
        for domain in ('combined', 'real', 'synthetic'):
            selected = [r for r in part if domain == 'combined' or r['domain'] == domain]
            if not selected:
                continue
            result[subset][domain] = {}
            for region in REGIONS:
                macro, counts = {}, {}
                for metric in METRICS:
                    values = [r['metrics'][region][metric] for r in selected
                              if r['metrics'][region][metric] is not None]
                    require(all(math.isfinite(v) and v >= 0 for v in values), 'Invalid depth metric')
                    macro[metric] = math.fsum(values) / len(values) if values else None
                    counts[metric] = len(values)
                result[subset][domain][region] = {'images': len(selected),
                    'pixels': sum(r['metrics'][region]['pixels'] for r in selected),
                    'metric_image_counts': counts, 'image_macro': macro}
    return result


def ranking(entries, require_complete=True):
    require(len({r['epoch'] for r in entries}) == len(entries), 'Duplicate epoch')
    if require_complete:
        require(sorted(r['epoch'] for r in entries) == list(range(1, 81)), 'All 80 epochs must finish before selection')
    for row in entries:
        require(row['images'] == 1226 and row['smoke'] is False, 'Incomplete/probe epoch cannot rank')
        score = row['transparent_mae_mm']
        require(score is not None and math.isfinite(score), 'Empty or nonfinite selection score')
    return sorted(entries, key=lambda r: (r['transparent_mae_mm'], r['epoch']))


def prepare(args):
    install_paths(args)
    from v2_lidf.data import load_input_manifest
    from v2_lidf.artifact_contract import authenticate_stage1
    marker = args.output / 'contract.json'
    require(not marker.exists(), 'Prepared sweep is immutable; use verify/worker, do not overwrite')
    roster = load_input_manifest(args.root / 'manifests/input_manifest.jsonl', 'evaluation')
    require(len(roster) == 1226 and len({r['cache_id'] for r in roster}) == 1226, 'Need complete1226')
    stage1 = authenticate_stage1(args.root / 'stage1', args.root / 'manifests/input_manifest.jsonl')
    complete = read(args.reference / 'evaluation.complete.json')
    require(complete['images'] == 1226 and complete['test_images'] == 521 and complete['selected_epoch'] == 63,
            'Baseline reference is not the authenticated complete E63 evaluation')
    for name, digest in complete['reports_sha256'].items():
        require(sha(args.reference / 'reports' / name) == digest, 'Reference report changed')
    seal = read(args.reference / 'predictions.complete.json')
    require(sha(args.reference / 'predictions.complete.json') == complete['predictions_seal_sha256']
            and seal['stage1_seal_sha256'] == stage1['marker_sha256'], 'Baseline inputs differ')
    reference = reference_rows(args)
    original = {r['cache_id']: r for r in roster}
    baseline = []
    for row in reference:
        baseline.append({**image_identity(original[row['cache_id']]),
            'metrics': depth_only(row['metrics']['stage1_native']),
            'resized_metrics': depth_only(row['metrics']['stage1_resized'])})
    validate_roster(baseline, roster)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_lines(args.output / 'roster.jsonl', roster)
    atomic_lines(args.output / 'baseline.per_image.jsonl', baseline)
    atomic_json(args.output / 'baseline.metrics.json', aggregate(baseline))
    contract = {'schema_version': SCHEMA, 'selection_policy': POLICY,
        'training_root': str(args.root), 'training_project': str(args.train_project),
        'roster': ref(args.output / 'roster.jsonl'), 'baseline': ref(args.output / 'baseline.per_image.jsonl'),
        'baseline_metrics': ref(args.output / 'baseline.metrics.json'),
        'reference_evaluation': ref(args.reference / 'evaluation.complete.json'),
        'reference_per_image': ref(args.reference / 'reports/per_image.jsonl'),
        'reference_predictions_seal': ref(args.reference / 'predictions.complete.json'),
        'input_manifest': ref(args.root / 'manifests/input_manifest.jsonl'),
        'label_manifest': ref(args.root / 'manifests/label_manifest.jsonl'),
        'stage1_seal': ref(args.root / 'stage1/complete.json'),
        'source_sha256': execution_sources(args), 'created_unix': time.time(),
        'depth_metric_implementation': 'original native_fusion.geometry.evaluate_depth; intrinsics=None omits normals',
        'prediction_gt_isolation': 'separate OS process; all epoch predictions sealed before metric process request',
        'baseline_policy': 'reuse identical sealed E08 native/resized reference, no new baseline model execution'}
    atomic_json(marker, contract)
    print(json.dumps({'status': 'prepared', 'contract': ref(marker)}), flush=True)


def verify_contract(args):
    contract = read(args.output / 'contract.json')
    require(contract['schema_version'] == SCHEMA and contract['selection_policy'] == POLICY, 'Selection policy changed')
    require(contract['training_root'] == str(args.root) and contract['training_project'] == str(args.train_project),
            'Wrong original production experiment')
    require(contract['source_sha256'] == execution_sources(args), 'Sweep/production source changed')
    for key in ('roster', 'baseline', 'baseline_metrics', 'reference_evaluation', 'reference_per_image',
                'reference_predictions_seal', 'input_manifest', 'label_manifest', 'stage1_seal'):
        require(sha(contract[key]['path']) == contract[key]['sha256'], 'Changed sealed dependency: ' + key)
    return contract


def checkpoint_for_epoch(args, epoch):
    receipts = lines(args.root / 'training/epoch_checkpoints.jsonl')
    records = lines(args.root / 'training/epochs.jsonl')
    receipt = next((r for r in receipts if r['epoch'] == epoch), None)
    record = next((r for r in records if r['epoch'] == epoch), None)
    if receipt is None or record is None:
        return None
    path = args.root / 'training' / ('epoch%03d.pt' % epoch)
    require(path.resolve() == Path(receipt['path']).resolve(), 'Archived checkpoint pointer differs')
    require(sha(path) == receipt['sha256'] and record['updates'] == receipt['updates'], 'Checkpoint not committed/changed')
    return path, receipt, record


def authenticate_checkpoint(args, checkpoint, receipt, record):
    c = checkpoint['contract']
    require(checkpoint['epoch'] + 1 == record['epoch'] and checkpoint['epoch_record'] == record
            and checkpoint['updates'] == receipt['updates'] and c.get('smoke_steps') == 0,
            'Checkpoint does not match immutable committed epoch history')
    current = read(args.root / 'training/contract.json')
    for key in ('input_manifest_sha256', 'label_manifest_sha256', 'stage1_seal_sha256', 'stage1_source_sha256'):
        require(c[key] == current[key], 'Checkpoint uses different data/stage one')
    def normalized(cfg):
        cfg = copy.deepcopy(cfg['model'])
        cfg['spvnas'].pop('audit_dir', None)
        return cfg
    require(normalized(c['config']) == normalized(current['config']), 'Model numerical configuration changed')
    for path, digest in c['sources'].items():
        require(sha(path) == digest, 'Archived checkpoint source changed: ' + path)
        if '/src/v2_lidf/' in path:
            relative = path.split('/src/v2_lidf/', 1)[1]
            if relative not in ('train.py', 'curriculum.py', 'artifact_contract.py', 'migration.py'):
                require(sha(args.train_project / 'src/v2_lidf' / relative) == digest,
                        'Executing different numerical source: ' + relative)
    return c


def predict_epoch(args, model, items, rows, epoch, checkpoint_sha, receipt, record):
    import numpy as np
    import torch
    from v2_lidf.predict import PredictionInputGuard, prediction_forward
    directory = args.output / ('probe_epoch%03d' % epoch if args.limit else 'epoch%03d' % epoch)
    directory.mkdir(parents=True, exist_ok=True)
    pred_dir = directory / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    contract_sha = sha(args.output / 'contract.json')
    marker = directory / 'predictions.complete.json'
    if marker.exists():
        sealed = read(marker)
        require(sealed['checkpoint_sha256'] == checkpoint_sha and sealed['contract_sha256'] == contract_sha,
                'Existing predictions have a different checkpoint/contract')
        validate_roster(sealed['predictions'], rows)
        for row in sealed['predictions']:
            require(sha(pred_dir / (row['cache_id'] + '.npz')) == row['prediction_sha256'], 'Sealed prediction changed')
        return directory
    started = time.monotonic()
    predictions = []
    with PredictionInputGuard(args.root / 'stage1', pred_dir) as guard, torch.inference_mode():
        for index, (row, item) in enumerate(zip(rows, items)):
            torch.manual_seed(int(row['rgb_sha256'][:8], 16))
            torch.cuda.manual_seed_all(int(row['rgb_sha256'][:8], 16))
            result = prediction_forward(model, item, 'cuda:0')
            path = pred_dir / (row['cache_id'] + '.npz')
            temporary = path.with_name(path.name + '.tmp.%d' % os.getpid())
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, depth=result['depth'][0, 0].cpu().numpy().astype(np.float32),
                                    refined_mask=result['refined_mask'][0, 0].cpu().numpy())
            os.replace(temporary, path)
            predictions.append({**image_identity(row), 'prediction_sha256': sha(path)})
            if index % 100 == 0:
                print(json.dumps({'phase': 'predict', 'epoch': epoch, 'index': index,
                    'images': len(rows), 'seconds': time.monotonic()-started}), flush=True)
        guard.assert_clean()
    validate_roster(predictions, rows)
    atomic_json(marker, {'schema_version': SCHEMA, 'epoch': epoch, 'images': len(rows),
        'smoke': bool(args.limit), 'checkpoint_sha256': checkpoint_sha,
        'checkpoint_path': receipt['path'], 'checkpoint_receipt': receipt,
        'epoch_record': record, 'contract_sha256': contract_sha, 'predictions': predictions,
        'input_alpha': 0.0, 'label_access': False, 'forward_inputs': ['rgb', 'depth_input', 'intrinsics'],
        'compute_loss': False, 'input_read_guard': guard.stats,
        'prediction_seconds': time.monotonic()-started, 'finished_unix': time.time()})
    return directory


def metric_process(args, connection):
    """No model object is created here; GT arrays never cross this pipe."""
    try:
        install_paths(args)
        import numpy as np
        import torch
        import torch.nn.functional as F
        from v2_lidf.data import load_label_manifest, load_native_labels
        from native_fusion.geometry import evaluate_depth
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        cache, sources, label_rows = {}, {}, None
        expected = {r['cache_id']: r for r in lines(args.output / 'baseline.per_image.jsonl')}
        reference = {r['cache_id']: r for r in reference_rows(args)}
        while True:
            message = connection.recv()
            if message is None:
                return
            directory = Path(message)
            seal = read(directory / 'predictions.complete.json')
            require(seal['label_access'] is False and seal['contract_sha256'] == sha(args.output/'contract.json'),
                    'Metrics require sealed independent predictions')
            require(seal['images'] == len(seal['predictions']) and
                    (seal['images'] == 1226 or seal['smoke']), 'Incomplete prediction seal')
            started = time.monotonic()
            if label_rows is None:
                label_rows = {r['cache_id']: r for r in load_label_manifest(args.root/'manifests/label_manifest.jsonl', 'evaluation')}
            records = []
            for index, row in enumerate(seal['predictions']):
                cid = row['cache_id']
                pred = directory/'predictions'/(cid+'.npz')
                require(sha(pred) == row['prediction_sha256'], 'Prediction changed before GT evaluation')
                with np.load(str(pred), allow_pickle=False) as payload:
                    low = torch.from_numpy(payload['depth'].copy())[None, None]
                if cid not in cache:
                    lr = label_rows[cid]
                    assets = {key: ref(lr[pathkey]) for key, pathkey in
                              (('depth', 'target_depth_path'), ('transparent_mask', 'transparent_mask_path'))}
                    for key, item in assets.items():
                        require(item['sha256'] == reference[cid]['label_assets'][key]['sha256'],
                                'Native supervision differs from authenticated E63 baseline')
                    native = load_native_labels(lr)
                    cache[cid] = {key: torch.from_numpy(native[key])[None, None]
                                  for key in ('target', 'valid', 'transparent')}
                    sources[cid] = assets
                batch = {key: value.to('cuda:0') for key, value in cache[cid].items()}
                # Original production interpolates the CPU float32 array first.
                native = F.interpolate(low, size=row['native_hw'], mode='bilinear', align_corners=False).to('cuda:0')
                metrics = depth_only(evaluate_depth(native, batch['target'], batch['valid'], batch['transparent'], intrinsics=None))
                for region in REGIONS:
                    require(metrics[region]['pixels'] == expected[cid]['metrics'][region]['pixels'], 'GT support differs from baseline')
                records.append({**image_identity(row), 'metrics': metrics,
                    'prediction_sha256': row['prediction_sha256'], 'label_assets': sources[cid]})
            validate_roster(records, seal['predictions'])
            atomic_lines(directory/'per_image.jsonl', records)
            summary = aggregate(records)
            atomic_json(directory/'metrics.json', summary)
            score = summary['full1226']['combined']['transparent']['image_macro']['mae_mm']
            complete = {'schema_version': SCHEMA, 'status': 'complete', 'epoch': seal['epoch'],
                'images': len(records), 'smoke': seal['smoke'], 'transparent_mae_mm': score,
                'selection_policy': POLICY, 'checkpoint_sha256': seal['checkpoint_sha256'],
                'contract_sha256': seal['contract_sha256'], 'prediction_seal': ref(directory/'predictions.complete.json'),
                'per_image': ref(directory/'per_image.jsonl'), 'metrics': ref(directory/'metrics.json'),
                'metric_seconds': time.monotonic()-started, 'finished_unix': time.time(),
                'gt_isolation': 'separate process, only started evaluation after sealed prediction request',
                'geometry': 'not part of sweep; winner requires separate full geometry evaluation'}
            if args.limit and seal['epoch'] == 63:
                differences = []
                for row in records:
                    for region in REGIONS:
                        for metric in METRICS:
                            a, b = row['metrics'][region][metric], reference[row['cache_id']]['metrics']['stage1_lidf'][region][metric]
                            require((a is None) == (b is None), 'E63 reference support differs')
                            if a is not None:
                                differences.append({'cache_id': row['cache_id'], 'region': region, 'metric': metric,
                                                    'absolute_difference': abs(a-b)})
                                require(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-5), 'E63 probe differs from existing native metrics')
                complete['e63_reference_parity'] = {'status': 'passed', 'rtol': 1e-6, 'atol': 1e-5,
                    'comparisons': len(differences), 'maximum_absolute_difference': max(x['absolute_difference'] for x in differences),
                    'differences': differences}
            atomic_json(directory/'complete.json', complete)
            connection.send({'status': 'passed', 'complete': complete})
    except BaseException:
        connection.send({'status': 'failed', 'traceback': traceback.format_exc()})
        raise


def worker(args):
    require(os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID', 'Explicit physical GPU ordering required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    require(len(visible) == 1 and visible[0] in {str(i) for i in range(8)}, 'Each worker needs exactly one allowed physical GPU')
    require(0 <= args.rank < args.world <= 8 and (not args.limit or args.epoch == 63), 'Invalid worker/probe assignment')
    install_paths(args)
    import torch
    from v2_lidf.data import V2LIDFDataset
    from v2_lidf.predict import PredictionInputGuard, validate_input_archive
    from v2_lidf.model import LIDFStage1
    verify_contract(args)
    if not args.limit:
        probe = read(args.output/'probe_epoch063/complete.json')
        require(probe.get('smoke') is True and probe['images'] >= 8
                and probe.get('e63_reference_parity', {}).get('status') == 'passed'
                and probe['contract_sha256'] == sha(args.output/'contract.json'),
                'Full sweep requires the successful same-source real E63 parity probe')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    dataset = V2LIDFDataset(args.root/'manifests/input_manifest.jsonl', None, args.root/'stage1', None,
                            role='evaluation', include_target=False)
    rows = dataset.rows
    if args.limit:
        # Include real/synthetic and known/novel, rather than the first domain only.
        selected = []
        for group in sorted({r['benchmark_group'] for r in rows}):
            selected += [i for i,r in enumerate(rows) if r['benchmark_group'] == group][:args.limit]
    else:
        selected = list(range(len(rows)))
    with PredictionInputGuard(args.root/'stage1', args.output/'input_guard') as guard:
        items = []
        for i in selected:
            validate_input_archive(args.root/'stage1/inputs'/(rows[i]['cache_id']+'.npz'))
            items.append(dataset[i])
        guard.assert_clean()
    rows = [rows[i] for i in selected]
    cfg = copy.deepcopy(read(args.root/'training/contract.json')['config']['model'])
    cfg['spvnas']['audit_dir'] = str(args.output/'initialization_audit'/('rank%d' % args.rank))
    model = LIDFStage1(**cfg).to('cuda:0').eval()
    connection, remote = multiprocessing.get_context('spawn').Pipe()
    evaluator = None
    epochs = [args.epoch] if args.epoch else list(range(args.rank+1,81,args.world))
    try:
        for epoch in epochs:
            directory = args.output / ('probe_epoch%03d' % epoch if args.limit else 'epoch%03d' % epoch)
            if (directory/'complete.json').exists():
                validate_epoch(args, directory, rows, verify_predictions=True)
                continue
            while True:
                committed = checkpoint_for_epoch(args, epoch)
                if committed is not None:
                    break
                require(args.wait_for_checkpoints, 'Requested checkpoint has not been committed')
                print(json.dumps({'phase': 'waiting_for_committed_checkpoint', 'epoch': epoch}), flush=True)
                time.sleep(30)
            path, receipt, record = committed
            saved = torch.load(str(path), map_location='cpu', weights_only=False)
            authenticate_checkpoint(args, saved, receipt, record)
            model.load_state_dict(saved['model'], strict=True)
            del saved
            directory = predict_epoch(args, model, items, rows, epoch, receipt['sha256'], receipt, record)
            if evaluator is None:
                evaluator = multiprocessing.get_context('spawn').Process(target=metric_process, args=(args, remote))
                evaluator.start()
                remote.close()
            connection.send(str(directory))
            while not connection.poll(10):
                require(evaluator.is_alive(), 'Independent evaluator exited without completion')
            result = connection.recv()
            require(result['status'] == 'passed', result.get('traceback', 'Evaluation failed'))
            print(json.dumps({'phase': 'epoch_complete', 'epoch': epoch,
                'transparent_mae_mm': result['complete']['transparent_mae_mm'],
                'peak_cuda_gb': torch.cuda.max_memory_allocated()/1e9}), flush=True)
    finally:
        if evaluator is not None and evaluator.is_alive():
            connection.send(None)
            evaluator.join(timeout=30)


def validate_epoch(args, directory, roster, verify_predictions=False):
    marker = read(directory/'complete.json')
    require(marker['status'] == 'complete' and marker['contract_sha256'] == sha(args.output/'contract.json')
            and marker['selection_policy'] == POLICY, 'Epoch completion identity differs')
    for key in ('prediction_seal', 'per_image', 'metrics'):
        require(sha(marker[key]['path']) == marker[key]['sha256'], 'Completed artifact changed')
    rows = lines(directory/'per_image.jsonl')
    validate_roster(rows, roster)
    seal = read(directory/'predictions.complete.json')
    validate_roster(seal['predictions'], roster)
    require(seal['checkpoint_sha256'] == marker['checkpoint_sha256'] and seal['label_access'] is False,
            'Prediction/metric checkpoint identity differs')
    if verify_predictions:
        for row in seal['predictions']:
            require(sha(directory/'predictions'/(row['cache_id']+'.npz')) == row['prediction_sha256'], 'Prediction changed')
    require(aggregate(rows) == read(directory/'metrics.json'), 'Stored summary differs from per-image metrics')
    require(marker['transparent_mae_mm'] == read(directory/'metrics.json')['full1226']['combined']['transparent']['image_macro']['mae_mm'],
            'Selection score differs from full transparent metric')
    return marker


def merge(args):
    verify_contract(args)
    roster = lines(args.output/'roster.jsonl')
    entries = []
    for epoch in range(1,81):
        directory = args.output/('epoch%03d' % epoch)
        if (directory/'complete.json').exists():
            entries.append(validate_epoch(args, directory, roster, verify_predictions=True))
    ranked = ranking(entries, require_complete=not args.partial)
    result = {'schema_version': SCHEMA, 'status': 'partial_not_selected' if args.partial else 'complete',
        'selection_policy': POLICY, 'completed_epochs': len(entries), 'contract': ref(args.output/'contract.json'),
        'ranking': [{k:r[k] for k in ('epoch','checkpoint_sha256','images','smoke','transparent_mae_mm','metrics','per_image')}
                    for r in ranked], 'finished_unix':time.time()}
    if not args.partial:
        result['selected_epoch'] = ranked[0]['epoch']
        result['selected_checkpoint'] = ref(args.root/'training'/('epoch%03d.pt' % ranked[0]['epoch']))
        result['selected_metrics'] = ranked[0]['metrics']
        result['full_geometry_required'] = True
    atomic_json(args.output/('ranking.partial.json' if args.partial else 'selection.complete.json'), result)
    with (args.output/'ranking.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank','epoch','transparent_mae_mm','full_mae_mm','full_rmse_mm','full_abs_rel','checkpoint_sha256'])
        for index,row in enumerate(ranked,1):
            full = read(row['metrics']['path'])['full1226']['combined']['all']['image_macro']
            writer.writerow([index,row['epoch'],row['transparent_mae_mm'],full['mae_mm'],full['rmse_mm'],full['abs_rel'],row['checkpoint_sha256']])
    print(json.dumps({k:v for k,v in result.items() if k!='ranking'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','worker','merge','verify'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--train-project',type=Path,default=TRAIN_PROJECT)
    parser.add_argument('--v2-source',type=Path,default=V2_SOURCE)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world',type=int,default=1)
    parser.add_argument('--epoch',type=int,default=0)
    parser.add_argument('--limit',type=int,default=0,help='Probe only: E63 images per each of four domain/novelty groups')
    parser.add_argument('--wait-for-checkpoints',action='store_true')
    parser.add_argument('--partial',action='store_true')
    args = parser.parse_args()
    require(args.limit>=0 and 0<=args.epoch<=80, 'Invalid epoch/limit')
    if args.phase=='verify':
        verify_contract(args)
        print('contract verified')
    else:
        globals()[args.phase](args)


if __name__ == '__main__':
    main()
