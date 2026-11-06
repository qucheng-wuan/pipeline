"""Rescore saved 256 predictions/GT into a NEW report without running a model.

Legacy reports omit cache_id: supply the original, hash-matching input manifest
to recover identities. Neither file order nor a guessed sample-id hash is used.
"""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path

import numpy as np

from v2_lidf.data import atomic_json, read_jsonl
from v2_lidf.data_protocol import validate_protocol
from . import evaluate256
from .evaluate256 import EVALUATION_SCHEMA, aggregate, metric_metadata, score_predictions
from .stage2 import file_sha


ERROR_KEYS = ('mae_m', 'rmse_m', 'abs_rel')
COUNT_KEYS = ('valid_pixels', 'empty', 'images', 'nonempty_images', 'empty_images')
IDENTITY_KEYS = ('sample_id', 'dataset', 'scene', 'camera')


def _cache_id(value):
    if (not isinstance(value, str) or not value or value in {'.', '..'}
            or '/' in value or '\\' in value or '\x00' in value):
        raise ValueError('Unsafe cache_id')
    return value


def _identities(report, input_manifest):
    rows = report['images']
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Report must contain unique nonempty sample identities')
    manifest_rows = None
    if input_manifest is not None:
        if file_sha(input_manifest) != report['input_manifest_sha256']:
            raise ValueError('Original input manifest SHA256 mismatch')
        inputs = read_jsonl(input_manifest)
        manifest_rows = {r['sample_id']: r for r in inputs}
        if (len(manifest_rows) != len(inputs)
                or len({r['cache_id'] for r in inputs}) != len(inputs)):
            raise ValueError('Input manifest has duplicate sample/cache identities')
    if manifest_rows is None and any('cache_id' not in r for r in rows):
        raise ValueError('Legacy report requires --input-manifest for authenticated cache_id mapping')
    identities = []
    for row in rows:
        cache_id = row.get('cache_id')
        if manifest_rows is not None:
            original = manifest_rows.get(row['sample_id'])
            if original is None or any(original.get(k) != row[k] for k in IDENTITY_KEYS):
                raise ValueError('Report/input manifest image identity mismatch')
            if cache_id is not None and cache_id != original['cache_id']:
                raise ValueError('Report/input manifest cache_id mismatch')
            cache_id = original['cache_id']
        identities.append(_cache_id(cache_id))
    if len(set(identities)) != len(identities):
        raise ValueError('Report has duplicate cache identities')
    return identities


def _check_unchanged(old, new, context):
    """Only delta definitions may change; detect wrong pairs or changed support."""
    if set(old) != set(new):
        raise ValueError('Model set changed: ' + context)
    for model, regions in new.items():
        if set(old[model]) != set(regions):
            raise ValueError('Region set changed: ' + context)
        for region, metrics in regions.items():
            previous = old[model][region]
            where = '/'.join((context, model, region))
            for key in COUNT_KEYS:
                if key in metrics and previous.get(key) != metrics[key]:
                    raise ValueError('Historical count/support mismatch: ' + where + '/' + key)
            for key in ERROR_KEYS:
                if key not in metrics:  # Empty per-image regions contain only counts.
                    continue
                before, after = previous.get(key), metrics[key]
                if before is None and after is None:
                    continue
                if (before is None or after is None or not math.isfinite(before)
                        or not math.isclose(before, after, rel_tol=1e-10, abs_tol=1e-12)):
                    raise ValueError('Historical error metric mismatch: ' + where + '/' + key)
            # The legacy first threshold was exactly the current final one.
            if 'delta_1' in previous and metrics.get('delta_1.25') is not None:
                if not math.isclose(previous['delta_1'], metrics['delta_1.25'],
                                    rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Historical delta_1/1.25 mismatch: ' + where)


def _artifact(root, subdirectory, cache_id):
    path = root / subdirectory / (cache_id + '.npz')
    if path.resolve().parent != (root / subdirectory).resolve():
        raise ValueError('Saved array path escapes its directory')
    if not path.is_file():
        raise FileNotFoundError('Missing saved prediction/supervision; cannot rescore: ' + str(path))
    return path


def rescore(input_dir, output_dir, input_manifest=None):
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError('Output must be separate from the historical input directory')
    if output.exists():
        raise FileExistsError('Rescoring output must be a new directory: ' + str(output))
    source_path = root / 'metrics.json'
    source_hash = file_sha(source_path)
    report = json.loads(source_path.read_text())
    if report.get('schema') not in ('fullfield256_evaluation_v1', EVALUATION_SCHEMA):
        raise ValueError('Unsupported 256 evaluation report schema')
    validate_protocol(report['preprocessing'])
    for key, unit in (('MAE', 'm'), ('RMSE', 'm'), ('AbsRel', 'dimensionless')):
        if report['units'].get(key) != unit:
            raise ValueError('Unsupported historical metric unit: ' + key)
    identities = _identities(report, input_manifest)
    expected_names = {cache_id + '.npz' for cache_id in identities}
    for subdirectory in ('predictions', 'supervision'):
        names = {p.name for p in (root / subdirectory).glob('*.npz')}
        if names != expected_names:
            raise ValueError('Saved array coverage differs from report: ' + subdirectory)
    records, artifacts = [], []
    for previous, cache_id in zip(report['images'], identities):
        prediction_path = _artifact(root, 'predictions', cache_id)
        supervision_path = _artifact(root, 'supervision', cache_id)
        pred_hash, gt_hash = file_sha(prediction_path), file_sha(supervision_path)
        with np.load(prediction_path, allow_pickle=False) as payload:
            baseline, prediction = payload['P256'].copy(), payload['D256'].copy()
        with np.load(supervision_path, allow_pickle=False) as payload:
            labels = {k: payload[k].copy() for k in ('target', 'valid', 'transparent')}
        if any(array.shape != (256, 256) for array in (baseline, prediction, *labels.values())):
            raise ValueError('Saved arrays must retain the 256x256 metric grid')
        if labels['valid'].dtype != np.bool_ or labels['transparent'].dtype != np.bool_:
            raise ValueError('Saved evaluation masks must be boolean')
        metrics = score_predictions(baseline, prediction, labels)
        _check_unchanged(previous['metrics'], metrics, previous['sample_id'])
        if file_sha(prediction_path) != pred_hash or file_sha(supervision_path) != gt_hash:
            raise ValueError('Saved arrays changed during rescoring')
        records.append({**previous, 'cache_id': cache_id, 'metrics': metrics})
        artifacts.append({'sample_id': previous['sample_id'], 'cache_id': cache_id,
                          'prediction': {'path': str(prediction_path), 'sha256': pred_hash},
                          'supervision': {'path': str(supervision_path), 'sha256': gt_hash}})
    overall = aggregate(records)
    _check_unchanged(report['overall'], overall, 'overall')
    grouped = defaultdict(list)
    for row in records:
        grouped['/'.join(row[k] for k in ('dataset', 'scene', 'camera'))].append(row)
    groups = {key: aggregate(rows) for key, rows in grouped.items()}
    if set(groups) != set(report['groups']):
        raise ValueError('Historical report group coverage differs')
    for key in groups:
        _check_unchanged(report['groups'][key], groups[key], key)
    if file_sha(source_path) != source_hash:
        raise ValueError('Historical report changed during rescoring')
    result = copy.deepcopy(report)
    result.update(schema=EVALUATION_SCHEMA, **metric_metadata(),
                  images=records, overall=overall, groups=groups)
    result['units']['delta'] = ('fraction; max(P/G,G/P)<t; t=1.05,1.10,1.25; '
                                'multiply by 100 only for percent display')
    result['rescoring'] = {
        'source_report': {'path': str(source_path), 'sha256': source_hash,
                          'schema': report['schema'], 'units': report['units']},
        'input_manifest': ({'path': str(Path(input_manifest).resolve()),
                            'sha256': file_sha(input_manifest)} if input_manifest else None),
        'evaluator_sha256': file_sha(evaluate256.__file__),
        'rescore_source_sha256': file_sha(__file__), 'arrays': artifacts,
        'array_hash_scope': ('Hashes observed at rescoring; legacy arrays were not hash-sealed '
                             'by the original evaluator. Original non-delta metrics verified.'),
        'unchanged_error_metrics_and_counts_verified': True,
        'error_comparison_tolerance': {'rtol': 1e-10, 'atol': 1e-12},
        'model_executed': False, 'source_modified': False,
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'metrics.json', result)
    return {'images': len(records), 'report': str(output / 'metrics.json'),
            'source_report_sha256': source_hash, 'model_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Original evaluate256 output directory')
    parser.add_argument('--output', required=True, help='New output directory; historical files are preserved')
    parser.add_argument('--input-manifest', help='Original hash-matching input manifest, required for legacy reports')
    args = parser.parse_args()
    print(json.dumps(rescore(args.input, args.output, args.input_manifest), indent=2))


if __name__ == '__main__':
    main()
