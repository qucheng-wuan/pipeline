"""Synthetic CPU evidence tests; fixtures are not measured release results."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quality_report', PROJECT / 'scripts/build_release_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, value):
    path.write_text(json.dumps(value) + '\n')
    return report.reference(path)


def rows_fixture():
    predictions, old, new = [], [], []
    for index in range(1226):
        identity = dict(cache_id=f'c{index}', sample_id=f's{index}', rgb_sha256=f'{index:064d}',
            native_hw=[480, 640], domain='real' if index % 2 else 'synthetic',
            official_split='real-test' if index < 521 else 'real-val', benchmark_group='fixture')
        differs = index < 5
        low = dict(max_abs_m=.056 if differs else 1e-7, mae_m=.056/76800 if differs else 1e-9,
            rmse_m=.00021 if differs else 1e-8, bitwise_equal_fraction=.9,
            pixels_over_tolerance=int(differs), abs_tolerance_m=1e-6, rtol=0.)
        native = {**low, 'mae_m': 2e-7 if differs else 1e-9}
        predictions.append({**identity, 'prediction': {'sha256': f'{index:064d}'},
            'parity_against_cached_winner': {'low': low, 'native_vs_reference_cpu_resize': native,
                'refined_mask_disagree_pixels': 0, 'passed': not differs}})
        metrics = {method: {region: {'pixels': 100, 'mae_mm': 20.} for region in report.REGIONS}
                   for method in report.METHODS}
        common = {**identity, 'metrics': metrics, 'normal_reference': 'fixture'}
        old.append({**copy.deepcopy(common), 'label_assets': {'depth': {'sha256': 'depth'},
                    'transparent_mask': {'sha256': 'mask'}}})
        new.append({**copy.deepcopy(common), 'prediction_sha256': f'{index:064d}',
                    'label_assets': {'target_depth_path': {'sha256': 'depth'},
                                     'transparent_mask_path': {'sha256': 'mask'}}})
        for region in report.REGIONS:
            new[-1]['metrics']['stage1_lidf'][region]['mae_mm'] += .001
    return predictions, old, new


def test_full_statistics_disclose_large_single_pixel_without_claiming_strict_pass():
    result = report.cached_quality_statistics(*rows_fixture())
    assert len(result['per_image']) == 1226 and len(result['partitions']) == 18
    assert result['summary']['max_low_abs_m'] == .056
    assert result['summary']['low_images_over_1e6'] == 5
    assert result['summary']['low_pixels_over_1e6'] == 5
    assert result['summary']['native_comparison_definition'] == 'portable_native_vs_cpu_resize_of_historical_low'
    assert all(report.quality_limit_checks(result).values())


def test_partition_degradation_cannot_hide_in_combined_average():
    predictions, old, new = rows_fixture()
    for row in new:
        row['metrics']['stage1_lidf']['background']['mae_mm'] = 20. + (.02 if row['domain'] == 'real' else -.02)
    stats = report.cached_quality_statistics(predictions, old, new)
    assert abs(stats['partitions']['full1226/combined/background']['signed_mae_change_mm']) < 1e-12
    assert not report.quality_limit_checks(stats)['all_partition_mae']


@pytest.mark.parametrize('field,value,check', [
    ('max_per_image_low_mean_abs_m', 1.001e-6, 'per_image_low_mean'),
    ('max_per_image_native_mean_abs_m', 1.001e-6, 'per_image_native_mean'),
    ('max_per_image_low_over_1e6_fraction', 1.001e-4, 'per_image_low_fraction'),
    ('total_low_over_1e6_fraction', 1.001e-6, 'total_low_fraction'),
    ('refined_mask_disagree_pixels', 1, 'mask_exact')])
def test_each_fixed_operational_limit_is_enforced(field, value, check):
    stats = report.cached_quality_statistics(*rows_fixture())
    stats['summary'][field] = value
    assert not report.quality_limit_checks(stats)[check]


def test_prediction_identity_labels_and_strict_receipt_inconsistency_fail():
    for mutation, message in (
        (lambda p, o, n: n[0].update(rgb_sha256='changed'), 'identity'),
        (lambda p, o, n: n[0]['label_assets']['target_depth_path'].update(sha256='changed'), 'GT assets'),
        (lambda p, o, n: n[0].update(prediction_sha256='changed'), 'prediction bytes'),
        (lambda p, o, n: p[0]['parity_against_cached_winner'].update(passed=True), 'inconsistent')):
        values = rows_fixture(); mutation(*values)
        with pytest.raises(ValueError, match=message):
            report.cached_quality_statistics(*values)


def acceptance_fixture(tmp_path, monkeypatch):
    failure = write(tmp_path / 'strict.json', {'status': 'failed_cached_input_numerical_parity'})
    stats = report.cached_quality_statistics(*rows_fixture())
    proposed = dict(schema='transparent_depth_cached_quality_statistics_v1', status='proposed',
        root_reviewed=False, statistics=stats, candidate_limit_checks=report.quality_limit_checks(stats),
        strict_failure=failure, evidence_sha256={failure['path']: failure['sha256']})
    proposed_ref = write(tmp_path / 'proposed.json', proposed)
    monkeypatch.setattr(report, 'assess_cached_quality', lambda *args: (copy.deepcopy(proposed), {'fixture': True}))
    diagnostic = write(tmp_path / 'repeatability.json', {'status': 'completed_diagnostic_not_acceptance',
                       'original_tolerance_m': 1e-6, 'original_failed_acceptance_not_overridden': True})
    record = dict(schema='transparent_depth_cached_quality_acceptance_v1', status='accepted', root_reviewed=True,
        verification_profile='training_release', strict_numerical_parity_passed=False,
        authorization=copy.deepcopy(report.QUALITY_AUTHORIZATION), limits=copy.deepcopy(report.QUALITY_LIMITS),
        root_review={'reviewer': '/root', 'decision': 'accept_training_release_approximate_quality',
                     'reviewed_unix': 100., 'note': 'CPU fixture only', 'repeatability_evidence_reviewed': True},
        proposed_statistics=proposed_ref, strict_failure=failure, repeatability_evidence=[diagnostic])
    path = tmp_path / 'accepted.json'; write(path, record)
    return path, record, proposed


def test_explicit_acceptance_required_and_strict_failure_preserved(tmp_path, monkeypatch):
    path, record, _ = acceptance_fixture(tmp_path, monkeypatch)
    result = report.validate_cached_quality_acceptance(path, tmp_path, tmp_path)
    assert result['status'] == 'accepted_approximate_quality'
    assert result['strict_numerical_parity_passed'] is False
    for key, value in (('status', 'proposed'), ('root_reviewed', False),
                       ('strict_numerical_parity_passed', True), ('verification_profile', 'full_live')):
        changed = {**record, key: value}; write(path, changed)
        with pytest.raises(ValueError, match='explicit'):
            report.validate_cached_quality_acceptance(path, tmp_path, tmp_path)


def test_acceptance_cannot_relax_limits_or_hide_changed_statistics(tmp_path, monkeypatch):
    path, record, proposed = acceptance_fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(record); changed['limits']['partition_mae_abs_change_mm'] = .1; write(path, changed)
    with pytest.raises(ValueError, match='explicit'):
        report.validate_cached_quality_acceptance(path, tmp_path, tmp_path)
    changed = copy.deepcopy(proposed); changed['statistics']['summary']['max_low_abs_m'] = 1e-7
    record['proposed_statistics'] = write(tmp_path / 'proposed.json', changed); write(path, record)
    with pytest.raises(ValueError, match='statistics/inputs changed'):
        report.validate_cached_quality_acceptance(path, tmp_path, tmp_path)


def test_repeatability_evidence_required_and_rehashed(tmp_path, monkeypatch):
    path, record, _ = acceptance_fixture(tmp_path, monkeypatch)
    record['repeatability_evidence'] = []; write(path, record)
    with pytest.raises(ValueError, match='self-repeatability'):
        report.validate_cached_quality_acceptance(path, tmp_path, tmp_path)


def test_orchestration_recovery_is_part_of_final_evidence_and_rehashed(tmp_path, monkeypatch):
    path, record, _ = acceptance_fixture(tmp_path, monkeypatch)
    recovery = write(tmp_path / 'recovery.json', {'fixture': 'real recovery evidence is separate from the old failure'})
    record['orchestration_recovery_evidence'] = [recovery]; write(path, record)
    evidence = report.Evidence()
    report.validate_cached_quality_acceptance(path, tmp_path, tmp_path, evidence)
    assert evidence.files[recovery['path']] == recovery['sha256']
    write(tmp_path / 'recovery.json', {'fixture': 'changed'})
    with pytest.raises(ValueError, match='SHA/path differs'):
        report.validate_cached_quality_acceptance(path, tmp_path, tmp_path)


def test_quality_option_cannot_be_used_in_full_live_profile():
    args = argparse.Namespace(verification_profile='full_live', live_validation=Path('/live'),
                              cached_quality_acceptance=Path('/quality'))
    with pytest.raises(ValueError, match='only available'):
        report.verification_profile(args)


def test_quality_measurement_does_not_skip_checkpoint_identity(tmp_path):
    marker = dict(status='failed_cached_input_numerical_parity', mode='cached', images=1226, test_images=521,
        selected_epoch=2, selected_checkpoint_sha256='wrong', model_inputs_included_gt_or_mask=False,
        test_informed_selection=True, independent_test_performance_claimed=False)
    write(tmp_path / 'evaluation.complete.json', marker)
    with pytest.raises(ValueError, match='another checkpoint'):
        report.portable_payload(tmp_path, 'cached', dict(epoch=2, checkpoint_sha256='right'), {}, {}, {},
                                report.Evidence(), quality_measurement=True)
