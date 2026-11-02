#!/usr/bin/env python3
"""CPU parity against existing sealed production metric JSON; no inference."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from transparent_depth.evaluation import evaluate_native_prediction
from transparent_depth.stage2 import file_sha


def compare(actual, expected, prefix=''):
    differences = []
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError('Metric keys differ at ' + prefix)
        for key in expected:
            differences.extend(compare(actual[key], expected[key], prefix + '/' + key))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(actual, expected, abs_tol=1e-5, rel_tol=1e-6):
            raise AssertionError('%s: %r != %r' % (prefix, actual, expected))
        if isinstance(expected, int) and actual != expected:
            raise AssertionError('Metric support count differs: ' + prefix)
        differences.append({'field': prefix, 'absolute': abs(actual-expected),
                            'relative': abs(actual-expected)/max(abs(expected), 1e-12)})
    elif actual != expected:
        raise AssertionError('Metric type/value differs at ' + prefix)
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--production-evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    chosen = {}
    for line in Path(args.labels).open():
        row = json.loads(line)
        if row['role'] == 'evaluation' and row['domain'] not in chosen:
            chosen[row['domain']] = row
    if set(chosen) != {'real', 'synthetic'}:
        raise ValueError('One sample of each domain is required')
    root = Path(args.production_evaluation)
    records = []
    for domain, row in chosen.items():
        pred = root / 'predictions' / (row['cache_id'] + '.npz')
        metric = root / 'metrics' / (row['cache_id'] + '.json')
        old = json.loads(metric.read_text())
        if old['source']['prediction_sha256'] != file_sha(pred):
            raise ValueError('Production prediction identity mismatch')
        with np.load(pred, allow_pickle=False) as data:
            low = torch.from_numpy(data['depth'].copy())[None, None]
        value = torch.nn.functional.interpolate(low, size=row['native_hw'], mode='bilinear', align_corners=False)[0, 0].numpy()
        result = evaluate_native_prediction(value, row)
        differences = compare(result, old['metrics']['stage1_lidf'])
        records.append({'domain': domain, 'cache_id': row['cache_id'], 'passed': True,
            'prediction_sha256': file_sha(pred), 'metric_sha256': file_sha(metric),
            'largest_absolute': sorted(differences, key=lambda r:r['absolute'], reverse=True)[:5],
            'max_relative_difference': max(r['relative'] for r in differences),
            'largest_reported_metric_absolute': sorted([r for r in differences if '/_sums/' not in r['field']],
                key=lambda r:r['absolute'], reverse=True)[:5],
            'exact_support_counts': True})
    report = {'status': 'passed', 'device': 'cpu', 'gpu_inference_performed': False,
              'labels_sha256': file_sha(args.labels), 'samples': records,
              'interpretation': 'Native evaluator/label adapter parity on one real and one synthetic sealed prediction; not full-model GPU parity.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
