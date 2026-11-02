"""Paired image-macro uncertainty, preserving the experiment's bootstrap."""
import math

import numpy as np


def paired_difference(rows, region, metric, random, draws=2000):
    differences = []
    for row in rows:
        current = row['metrics']['stage1_lidf'][region][metric]
        baseline = row['metrics']['stage1_native'][region][metric]
        if current is None and baseline is None:
            continue
        if current is None or baseline is None:
            raise ValueError('Paired methods do not share the same metric support')
        if not math.isfinite(current) or not math.isfinite(baseline):
            raise ValueError('Paired metric contains nonfinite values')
        differences.append(current - baseline)
    values = np.asarray(differences, dtype=float)
    if not len(values):
        return {'paired_images': 0, 'difference_lidf_minus_stage1': None, 'image_bootstrap_95pct': None}
    samples = random.integers(0, len(values), size=(draws, len(values)))
    return {'paired_images': len(values), 'difference_lidf_minus_stage1': float(values.mean()),
            'image_bootstrap_95pct': np.quantile(values[samples].mean(axis=1), [.025, .975]).tolist()}
