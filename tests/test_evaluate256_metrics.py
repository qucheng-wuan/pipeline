"""Numerical protocol regression tests, independent of model/checkpoint assets."""
import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from native_fusion.evaluation import aggregate as aggregate_native
from native_fusion.geometry import evaluate_depth
from transparent_depth import evaluate256, training_utils
from v2_lidf.data_protocol import PROTOCOL


# Intentionally literal: changing production constants must not change the oracle.
DEPTH_KEYS = ('mae_m', 'rmse_m', 'abs_rel',
              'delta_1.05', 'delta_1.10', 'delta_1.25')
REGIONS = ('all', 'transparent', 'background')


def scalar_reference(prediction, target, support):
    """Python scalar/math reference; no calls to either project evaluator."""
    pairs = [(float(p), float(g)) for p, g, use in
             zip(prediction.flat, target.flat, support.flat) if use]
    count = len(pairs)
    errors = [abs(p - g) for p, g in pairs]
    ratios = [max(p / g, g / p) for p, g in pairs]
    return {'mae_m': math.fsum(errors) / count,
            'rmse_m': math.sqrt(math.fsum(e * e for e in errors) / count),
            'abs_rel': math.fsum(e / g for e, (_, g) in zip(errors, pairs)) / count,
            'delta_1.05': sum(r < 1.05 for r in ratios) / count,
            'delta_1.10': sum(r < 1.10 for r in ratios) / count,
            'delta_1.25': sum(r < 1.25 for r in ratios) / count}


def assert_depth_values(actual, expected):
    assert not {'delta_1', 'delta_2', 'delta_3'} & actual.keys()
    for key in DEPTH_KEYS:
        assert actual[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-14)


def test_perfect_and_hand_computed_six_metrics():
    target = np.array([[1., 2.], [4., 8.]], dtype=np.float64)
    support = np.ones_like(target, dtype=bool)
    perfect = evaluate256.region_metrics(target, target, support)
    assert_depth_values(perfect, dict(zip(DEPTH_KEYS, (0., 0., 0., 1., 1., 1.))))
    # Errors: .02, .16, .8, 4 metres; ratios: 1.02, 1.08, 1.2, 2.
    prediction = np.array([[1.02, 2.16], [4.8, 4.]], dtype=np.float64)
    expected = dict(zip(DEPTH_KEYS, (1.245, math.sqrt(4.1665), .2, .25, .5, .75)))
    result = evaluate256.region_metrics(prediction, target, support)
    assert_depth_values(result, expected)
    assert_depth_values(result, scalar_reference(prediction, target, support))
    assert result['valid_pixels'] == 4 and result['empty'] is False


@pytest.mark.parametrize('threshold,key', [(1.05, 'delta_1.05'),
                                         (1.10, 'delta_1.10'), (1.25, 'delta_1.25')])
@pytest.mark.parametrize('underprediction', [False, True])
def test_delta_exact_boundary_and_adjacent_float64_values(threshold, key, underprediction):
    ratios = np.array([[np.nextafter(threshold, -np.inf), threshold,
                        np.nextafter(threshold, np.inf)]], dtype=np.float64)
    ones = np.ones_like(ratios)
    # Swap p/g for the reciprocal direction. Division by 1 preserves all three
    # adjacent float64 values, including the exact threshold, without rounding.
    prediction, target = (ones, ratios) if underprediction else (ratios, ones)
    result = evaluate256.region_metrics(prediction, target, np.ones_like(ones, dtype=bool))
    assert result[key] == 1 / 3  # Exactly equal MUST fail; <= would yield 2/3.
    for other_threshold, other_key in [(1.05, 'delta_1.05'), (1.10, 'delta_1.10'),
                                       (1.25, 'delta_1.25')]:
        if other_threshold < threshold:
            assert result[other_key] == 0.
        elif other_threshold > threshold:
            assert result[other_key] == 1.


def test_invalid_gt_outside_support_is_excluded_without_changing_denominator():
    target = np.array([[1., 0., -1., np.nan, np.inf]])
    support = np.array([[True, False, False, False, False]])
    result = evaluate256.region_metrics(np.ones_like(target), target, support)
    assert result['valid_pixels'] == 1
    assert_depth_values(result, dict(zip(DEPTH_KEYS, (0., 0., 0., 1., 1., 1.))))


@pytest.mark.parametrize('bad_gt', [0., -1., np.nan, np.inf, -np.inf])
def test_inconsistent_support_rejects_invalid_gt_instead_of_dropping_pixels(bad_gt):
    with pytest.raises(ValueError, match='Invalid GT inside evaluation support'):
        evaluate256.region_metrics(np.ones((1, 2)), np.array([[1., bad_gt]]),
                                   np.ones((1, 2), dtype=bool))


@pytest.mark.parametrize('bad_prediction', [0., -1., np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('support', [[[True, True]], [[True, False]], [[False, False]]])
def test_invalid_prediction_is_rejected_even_outside_support(bad_prediction, support):
    # Preserve the existing 256 whole-prediction validity contract. In particular,
    # a model cannot gain accuracy by silently dropping its failed pixels.
    with pytest.raises(FloatingPointError, match='model prediction'):
        evaluate256.region_metrics(np.array([[1., bad_prediction]]), np.ones((1, 2)),
                                   np.array(support, dtype=bool))


def test_empty_regions_are_counted_and_never_zero_filled():
    labels = {'target': np.zeros((2, 2)), 'valid': np.zeros((2, 2), dtype=bool),
              'transparent': np.ones((2, 2), dtype=bool)}
    scored = evaluate256.score_predictions(np.ones((2, 2)), np.ones((2, 2)), labels)
    summary = evaluate256.aggregate([{'metrics': scored}])
    for model in ('P256', 'Stage2'):
        for region in REGIONS:
            assert scored[model][region] == {'valid_pixels': 0, 'empty': True}
            value = summary[model][region]
            assert value['images'] == value['empty_images'] == 1
            assert value['nonempty_images'] == value['valid_pixels'] == 0
            assert all(value[key] is None for key in DEPTH_KEYS)


def test_unequal_image_support_preserves_image_macro_rmse_and_delta_weights():
    target = np.ones((2, 2), dtype=np.float64)
    records = []
    # Image A: one pixel, error=1. Image B: four pixels, error=0.
    for prediction, valid in [(target * 2, np.array([[1, 0], [0, 0]], dtype=bool)),
                              (target, np.ones((2, 2), dtype=bool))]:
        labels = {'target': target, 'valid': valid, 'transparent': valid}
        records.append({'metrics': evaluate256.score_predictions(target, prediction, labels)})
    result = evaluate256.aggregate(records)
    expected = dict(zip(DEPTH_KEYS, (.5, .5, .5, .5, .5, .5)))
    for region in ('all', 'transparent'):
        values = result['Stage2'][region]
        assert_depth_values(values, expected)
        assert values['images'] == values['nonempty_images'] == 2
        assert values['valid_pixels'] == 5
        assert values['rmse_m'] != pytest.approx(math.sqrt(1 / 5))  # Pooled RMSE.
        assert values['rmse_m'] != pytest.approx(math.sqrt((1. + 0.) / 2))
        assert values['mae_m'] != pytest.approx(1 / 5)  # Pixel weighted MAE.
        assert values['delta_1.05'] != pytest.approx(4 / 5)
        assert_depth_values(result['P256'][region], dict(zip(DEPTH_KEYS, (0., 0., 0., 1., 1., 1.))))
    assert result['Stage2']['background']['empty_images'] == 2


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_per_image_and_macro_match_original_native_evaluator_after_unit_conversion(dtype):
    rng = np.random.default_rng(713)
    grid_records, native_records = [], []
    for index, shape in enumerate([(5, 7), (4, 9)]):
        target = rng.uniform(.3, 1.5, shape).astype(dtype)
        valid = rng.random(shape) > .3
        transparent = rng.random(shape) > .4
        valid[0, :] = False
        target[0, :] = np.nan
        # Distinct baseline and Stage2 predictions; no GT alignment or resizing.
        base = rng.uniform(.2, 1.6, shape).astype(dtype)
        pred = rng.uniform(.2, 1.6, shape).astype(dtype)
        labels = {'target': target, 'valid': valid, 'transparent': transparent}
        grid = evaluate256.score_predictions(base, pred, labels)
        native = {}
        for name, prediction in [('P256', base), ('Stage2', pred)]:
            tensors = [torch.from_numpy(value)[None, None]
                       for value in (prediction, target, valid, transparent)]
            native[name] = evaluate_depth(*tensors, intrinsics=None, normal_camera_valid=False)
            masks = {'all': valid, 'transparent': valid & transparent,
                     'background': valid & ~transparent}
            for region, support in masks.items():
                assert support.any()
                assert grid[name][region]['valid_pixels'] == native[name][region]['pixels']
                assert_depth_values(grid[name][region], scalar_reference(prediction, target, support))
                converted = {**native[name][region],
                             'mae_m': native[name][region]['mae_mm'] / 1000,
                             'rmse_m': native[name][region]['rmse_mm'] / 1000}
                assert_depth_values(grid[name][region], converted)
        grid_records.append({'metrics': grid})
        native_records.append({'sample_id': str(index), 'domain': 'real',
                               'normal_reference': 'none', 'metrics': native})
    actual = evaluate256.aggregate(grid_records)
    original = aggregate_native(native_records)['combined']
    for name in ('P256', 'Stage2'):
        for region in REGIONS:
            macro = original[name][region]['image_macro']
            converted = {**macro, 'mae_m': macro['mae_mm'] / 1000,
                         'rmse_m': macro['rmse_mm'] / 1000}
            assert_depth_values(actual[name][region], converted)


def test_training_validation_emits_correct_deltas_and_keeps_mae_selection(monkeypatch):
    class Dataset:
        def __init__(self):
            self.rows = [{'cache_id': str(i), 'sample_id': str(i), 'dataset': 'transcg',
                          'role': 'val', 'official_split': 'train', 'scene': 'scene1',
                          'camera': 'D435', 'preprocessing': copy.deepcopy(PROTOCOL)}
                         for i in range(2)]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {'rgb': torch.zeros(3, 256, 256), 'intrinsics': torch.eye(3),
                    'depth_input': torch.full((1, 256, 256), (1.08, 1.2)[index])}

    class Predictor(torch.nn.Module):
        calls = 0

        def forward(self, rgb, depth, intrinsics, compute_loss):
            assert compute_loss is False
            self.calls += 1
            return {'depth': depth, 'query_mask': torch.ones_like(depth, dtype=torch.bool)}

    model, dataset = Predictor(), Dataset()
    label_rows = {row['cache_id']: row for row in dataset.rows}

    def load_labels(row):
        assert model.calls == int(row['cache_id']) + 1  # Prediction precedes GT reads.
        return {'target': np.ones((256, 256), dtype=np.float32),
                'valid': np.ones((256, 256), dtype=bool),
                'transparent': np.ones((256, 256), dtype=bool)}

    monkeypatch.setattr(evaluate256, 'load_authenticated_native_labels', load_labels)
    result = training_utils.validate(model, dataset, label_rows, torch.device('cpu'), 0, 1)
    assert model.training  # validate restores the incoming mode.
    assert result['delta_thresholds'] == [1.05, 1.10, 1.25]
    assert result['delta_comparison'] == 'strict_less_than'
    assert result['delta_storage'] == 'fraction_0_to_1'
    expected = dict(zip(DEPTH_KEYS, (.14, .14, .14, 0., .5, 1.)))
    for model_name in ('P256', 'Stage2'):
        metrics = result['metrics256'][model_name]['transparent']
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value, abs=1e-7)
    assert result['transparent_mae_m'] == result['metrics256']['Stage2']['transparent']['mae_m']
    assert result['transparent_mae_m'] == pytest.approx(.14, abs=1e-7)
    assert result['mae_mm'] == pytest.approx(140., abs=1e-4)
    assert result['rmse_mm'] == pytest.approx(140., abs=1e-4)
    assert result['input_alpha'] == 0. and result['gt_or_mask_used_as_input'] is False


def test_evaluation_report_serializes_explicit_protocol_and_saved_array_identity(tmp_path, monkeypatch):
    row = {'cache_id': 'sample', 'sample_id': 'transcg/sample', 'dataset': 'transcg',
           'role': 'val', 'official_split': 'train', 'scene': 'scene1', 'camera': 'D435',
           'rgb_sha256': '1' * 64, 'preprocessing': copy.deepcopy(PROTOCOL)}

    class Dataset:
        rows = [row]

        def __getitem__(self, index):
            return {'rgb': torch.zeros(3, 256, 256), 'depth_input': torch.ones(1, 256, 256),
                    'intrinsics': torch.eye(3)}

    class Predictor(torch.nn.Module):
        called = False

        def forward(self, rgb, depth, intrinsics, compute_loss):
            assert not self.training and compute_loss is False
            self.called = True
            return {'depth': depth * 1.2, 'query_mask': torch.ones_like(depth, dtype=torch.bool)}

    model = Predictor()

    def load_labels(label_row):
        assert model.called
        assert label_row['cache_id'] == row['cache_id']
        return {'target': np.ones((256, 256), dtype=np.float32),
                'valid': np.ones((256, 256), dtype=bool),
                'transparent': np.ones((256, 256), dtype=bool)}

    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'contract': {'sources': {}, 'config': {'data_protocol': 'transcg256',
                'preprocessing': PROTOCOL}}, 'model_config': {}, 'model': {}, 'epoch': 0}, checkpoint)
    inputs, labels, stage1 = tmp_path / 'inputs.jsonl', tmp_path / 'labels.jsonl', tmp_path / 'stage1'
    inputs.write_text(json.dumps(row) + '\n')
    labels.write_text(json.dumps(row) + '\n')
    stage1.mkdir()
    (stage1 / 'complete.json').write_text('{}\n')
    cache_checks = []
    monkeypatch.setattr(evaluate256, 'source_hashes', lambda: {})
    monkeypatch.setattr(evaluate256, 'validate_prediction_caches', lambda *args: cache_checks.append(args))
    monkeypatch.setattr(evaluate256, 'build_stage2', lambda *args, **kwargs: model)
    monkeypatch.setattr(evaluate256, 'V2LIDFDataset', lambda *args, **kwargs: Dataset())
    monkeypatch.setattr(evaluate256, 'load_label_manifest', lambda *args: [row])
    monkeypatch.setattr(evaluate256, 'load_authenticated_native_labels', load_labels)
    args = SimpleNamespace(checkpoint=checkpoint, input_manifest=inputs, label_manifest=labels,
                           stage1_root=stage1, output=tmp_path / 'evaluation', device='cpu',
                           dataset='transcg', role='val', limit=0, visualizations=0, smoke=True)
    receipt = evaluate256.evaluate(args)
    assert receipt['images'] == 1 and len(cache_checks) == 1
    report = json.loads((args.output / 'metrics.json').read_text())
    assert report['schema'] == 'fullfield256_evaluation_v2'
    assert report['delta_thresholds'] == [1.05, 1.10, 1.25]
    assert report['delta_comparison'] == 'strict_less_than'
    assert report['delta_storage'] == 'fraction_0_to_1'
    assert report['units']['MAE'] == report['units']['RMSE'] == 'm'
    assert report['units']['AbsRel'] == 'dimensionless'
    assert '1.25**' not in report['units']['delta']
    assert report['images'][0]['cache_id'] == 'sample'
    for scope in [report['overall'], report['groups']['transcg/scene1/D435'],
                  report['images'][0]['metrics']]:
        values = scope['Stage2']['transparent']
        assert [values[key] for key in DEPTH_KEYS[3:]] == [0., 0., 1.]
        assert not {'delta_1', 'delta_2', 'delta_3'} & values.keys()
    with np.load(args.output / 'predictions/sample.npz', allow_pickle=False) as predictions:
        np.testing.assert_array_equal(predictions['P256'], 1.)
        np.testing.assert_allclose(predictions['D256'], 1.2)
    with np.load(args.output / 'supervision/sample.npz', allow_pickle=False) as supervision:
        assert supervision['valid'].all() and supervision['transparent'].all()
