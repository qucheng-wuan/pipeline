"""Shared native-resolution validation, aggregation and predeclared selection.

No per-image GT alignment is applied to reported predictions. The oracle scale
diagnostic is explicitly separate and is never a deployed input.
"""
from __future__ import annotations

import math
from collections import Counter
import numpy as np
import torch

from .geometry import evaluate_depth

GROUPS = ('combined', 'real', 'synthetic')
REGIONS = ('all', 'background', 'transparent')
ERRORS = ('mae_mm', 'rmse_mm', 'abs_rel')
ACCURACIES = ('delta_1.05', 'delta_1.10', 'delta_1.25')
SELECTION_RULE = {
    'primary': 'lowest combined image-macro whole-scene AbsRel among eligible trained checkpoints',
    'eligibility': '36 depth gates: all/background x combined/real/synthetic x MAE/RMSE/AbsRel/delta1.05/delta1.10/delta1.25 versus the same cached FAR runtime',
    'numerical_tolerance': 1e-7,
    'geometry': 'report independently; synthetic official camera normals and real depth-induced proxy remain separate',
    'no_eligible': 'report failure to meet preservation criterion; best diagnostic checkpoint is not promoted as eligible',
    'scale_stage_trigger': 'in any of combined/real/synthetic validation: median abs oracle log-scale > log(1.10) OR 90th percentile > log(1.25)',
    'test_use': 'no test labels for optimization, scale-stage trigger or checkpoint selection',
}


def evaluate_prediction(prediction, batch):
    kwargs = {'normal_camera_valid': bool(batch['normal_camera_valid'].item())}
    independent = bool(batch.get('normal_reference_available', torch.tensor(False)).item())
    if independent:
        kwargs.update(reference_normal=batch['target_normal'], reference_normal_valid=batch['normal_valid'])
    return evaluate_depth(prediction, batch['target'], batch['valid'] & batch['content'].bool(),
                          batch['transparent'], batch['intrinsics'], **kwargs)


def scale_diagnostic(prediction, batch):
    valid = batch['valid'].bool() & batch['content'].bool()
    # This is a labelled diagnostic of the existing prediction, not a transform
    # applied to any reported prediction or an inference-time scale fitting step.
    log_scale = (batch['target'][valid].double().log() - prediction[valid].double().log()).mean().item()
    return {'oracle_log_scale': log_scale, 'oracle_scale': math.exp(log_scale)}


def image_record(prediction, batch, *, include_far=False):
    metrics = {'fusion': evaluate_prediction(prediction, batch)}
    if include_far:
        metrics['far'] = evaluate_prediction(batch['far_depth'], batch)
    independent = bool(batch.get('normal_reference_available', torch.tensor(False)).item())
    return {'sample_id': batch['sample_id'][0], 'domain': batch['domain'][0],
            'normal_reference': 'synthetic_released_camera_normals' if independent else 'real_depth_induced_proxy',
            'metrics': metrics, 'scale_diagnostic': scale_diagnostic(prediction, batch)}


def aggregate(rows):
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('evaluation must contain unique nonempty samples')
    experts = tuple(rows[0]['metrics'])
    if any(tuple(r['metrics']) != experts for r in rows):
        raise ValueError('inconsistent evaluated experts')
    result = {}
    for group in GROUPS:
        selected = [r for r in rows if group == 'combined' or r['domain'] == group]
        if not selected:
            continue
        result[group] = {}
        for expert in experts:
            result[group][expert] = {}
            for region in REGIONS:
                values = [r['metrics'][expert][region] for r in selected]
                sums = {k: math.fsum(v['_sums'][k] for v in values) for k in values[0]['_sums']}
                macro, counts = {}, {}
                count_keys = [key for key in values[0] if key.endswith(('_pixels', '_images')) or key == 'pixels']
                for key in values[0]:
                    if key in count_keys or key in ('_sums', 'normal_reference_kind'):
                        continue
                    entries = [v[key] for v in values if v[key] is not None]
                    macro[key] = math.fsum(entries)/len(entries) if entries else None
                    counts[key] = len(entries)
                p, n = sums['pixels'], sums['normal_pixels']
                pooled = {'mae_mm': 1000*sums['absolute_m']/p if p else None,
                          'rmse_mm': 1000*math.sqrt(sums['squared_m']/p) if p else None,
                          'abs_rel': sums['relative']/p if p else None,
                          **{k: sums[k]/p if p else None for k in ACCURACIES}}
                for prefix in ('normal', 'depth_induced_normal'):
                    support_count = sums[f'{prefix}_pixels']
                    pooled.update({f'{prefix}_mean_deg': sums[f'{prefix}_angle_sum_deg']/support_count if support_count else None,
                                   f'{prefix}_median_deg': None,
                                   **{f'{prefix}_{x:g}': sums[f'{prefix}_{x:g}']/support_count if support_count else None for x in (11.25,22.5,30)}})
                pe, ge = sums['pred_edges'], sums['gt_edges']
                precision = sums['matched_pred_edges']/pe if pe else (0. if ge else None)
                recall = sums['matched_gt_edges']/ge if ge else (0. if pe else None)
                f1 = 2*precision*recall/(precision+recall) if precision is not None and precision+recall > 0 else (0. if pe or ge else None)
                pooled.update(edge_precision=precision, edge_recall=recall, edge_f1=f1)
                result[group][expert][region] = {'images': len(selected), 'pixels': int(p),
                    'normal_pixels': int(n), 'normal_reference_counts': dict(Counter(r['normal_reference'] for r in selected)),
                    'support_counts': {key:int(sums[key]) for key in count_keys},
                    'metric_image_counts': counts, 'image_macro': macro, 'pixel_pooled': pooled, '_sums': sums}
    return result


def qualification(summary, baseline):
    checks = []
    for group in GROUPS:
        for region in ('all', 'background'):
            current = summary[group]['fusion'][region]['image_macro']
            reference = baseline[group]['far'][region]['image_macro']
            for metric in (*ERRORS, *ACCURACIES):
                delta = current[metric] - reference[metric]
                passed = delta <= 1e-7 if metric in ERRORS else delta >= -1e-7
                checks.append({'group': group, 'region': region, 'metric': metric,
                               'current': current[metric], 'reference': reference[metric], 'passed': passed})
    return {'eligible': all(c['passed'] for c in checks), 'passed': sum(c['passed'] for c in checks),
            'total': len(checks), 'checks': checks,
            'primary_score': summary['combined']['fusion']['all']['image_macro']['abs_rel']}


def summarize_scale(rows):
    result = {}
    for group in GROUPS:
        values = np.array([r['scale_diagnostic']['oracle_log_scale'] for r in rows
                           if group == 'combined' or r['domain'] == group], dtype=np.float64)
        if not len(values):
            continue
        absolute = np.abs(values)
        median, p90 = float(np.median(absolute)), float(np.quantile(absolute, .9))
        result[group] = {'images': len(values), 'signed_mean_log_scale': float(values.mean()),
            'median_abs_log_scale': median, 'p90_abs_log_scale': p90,
            'median_multiplicative_scale_error': math.exp(median)-1,
            'p90_multiplicative_scale_error': math.exp(p90)-1,
            'severe': median > math.log(1.10) or p90 > math.log(1.25)}
    return result
