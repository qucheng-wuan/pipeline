"""Saved-array rescoring: identities, provenance, and unchanged metric checks."""
import copy
import json

import numpy as np
import pytest

from transparent_depth.evaluate256 import EVALUATION_SCHEMA
from transparent_depth.rescore256 import rescore
from transparent_depth.stage2 import file_sha
from v2_lidf.data_protocol import PROTOCOL


def legacy_metrics(pred, gt, valid, transparent):
    """Independent reference for the OLD report, never the corrected evaluator."""
    result = {}
    for name, mask in [('all', valid), ('transparent', valid & transparent),
                       ('background', valid & ~transparent)]:
        p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
        error = p - g
        ratio = np.maximum(p / g, g / p)
        result[name] = {'valid_pixels': int(mask.sum()), 'empty': False,
                        'mae_m': float(np.abs(error).mean()),
                        'rmse_m': float(np.sqrt(np.square(error).mean())),
                        'abs_rel': float((np.abs(error) / g).mean()),
                        'delta_1': float((ratio < 1.25).mean()),
                        'delta_2': float((ratio < 1.5625).mean()),
                        'delta_3': float((ratio < 1.953125).mean())}
    return result


def legacy_aggregate(rows):
    result = {}
    for model in ('P256', 'Stage2'):
        result[model] = {}
        for region in ('all', 'transparent', 'background'):
            values = [r['metrics'][model][region] for r in rows]
            metrics = {'images': len(rows), 'nonempty_images': len(rows), 'empty_images': 0,
                       'valid_pixels': sum(v['valid_pixels'] for v in values)}
            for key in ('mae_m', 'rmse_m', 'abs_rel', 'delta_1', 'delta_2', 'delta_3'):
                metrics[key] = sum(v[key] for v in values) / len(values)
            result[model][region] = metrics
    return result


def history(tmp_path):
    root = tmp_path / 'old'
    (root / 'predictions').mkdir(parents=True)
    (root / 'supervision').mkdir()
    gt = np.ones((256, 256), np.float32)
    valid = np.ones_like(gt, bool)
    transparent = np.zeros_like(valid)
    transparent[:, :128] = True
    rows, manifest_rows = [], []
    # Deliberately reverse alphabetic file order relative to report order.
    for sample_id, cache_id, depth in [('sample-a', 'cache-z', 1.08), ('sample-b', 'cache-a', 1.2)]:
        prediction = np.full_like(gt, depth)
        identity = {'sample_id': sample_id, 'dataset': 'transcg', 'scene': 'scene1', 'camera': 'D435'}
        manifest_rows.append({**identity, 'cache_id': cache_id})
        rows.append({**identity, 'metrics': {
            'P256': legacy_metrics(gt, gt, valid, transparent),
            'Stage2': legacy_metrics(prediction, gt, valid, transparent)}})
        np.savez_compressed(root / 'predictions' / (cache_id + '.npz'), P256=gt, D256=prediction)
        np.savez_compressed(root / 'supervision' / (cache_id + '.npz'),
                            target=gt, valid=valid, transparent=transparent,
                            coverage=np.ones_like(gt))
    manifest = tmp_path / 'inputs.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in manifest_rows))
    report = {'schema': 'fullfield256_evaluation_v1', 'preprocessing': copy.deepcopy(PROTOCOL),
              'units': {'MAE': 'm', 'RMSE': 'm', 'AbsRel': 'dimensionless', 'delta': 'fraction; 1.25**k'},
              'input_manifest_sha256': file_sha(manifest), 'checkpoint_sha256': 'a' * 64,
              'test_set_selection': True, 'accuracy_claim': 'test-set-selected result',
              'aggregation': 'unweighted mean of per-image metrics over nonempty regions; RMSE averaged per image',
              'overall': legacy_aggregate(rows), 'groups': {'transcg/scene1/D435': legacy_aggregate(rows)},
              'images': rows}
    (root / 'metrics.json').write_text(json.dumps(report))
    return root, manifest, report


def test_rescore_recomputes_thresholds_preserves_history_and_exposure(tmp_path):
    root, manifest, old = history(tmp_path)
    original_files = {str(p): file_sha(p) for p in root.rglob('*') if p.is_file()}
    out = tmp_path / 'rescored'
    result = rescore(root, out, manifest)
    new = json.loads((out / 'metrics.json').read_text())
    assert result['model_executed'] is False and result['images'] == 2
    assert new['schema'] == EVALUATION_SCHEMA
    assert new['delta_thresholds'] == [1.05, 1.10, 1.25]
    assert new['delta_storage'] == 'fraction_0_to_1'
    metrics = new['overall']['Stage2']['all']
    assert metrics['delta_1.05'] == 0
    assert metrics['delta_1.10'] == .5
    assert metrics['delta_1.25'] == 1
    assert 'delta_1' not in metrics
    assert [r['cache_id'] for r in new['images']] == ['cache-z', 'cache-a']
    assert new['images'][0]['metrics']['Stage2']['all']['delta_1.10'] == 1
    for key in ('mae_m', 'rmse_m', 'abs_rel', 'valid_pixels'):
        assert metrics[key] == old['overall']['Stage2']['all'][key]
    for key in ('test_set_selection', 'accuracy_claim', 'checkpoint_sha256', 'preprocessing', 'aggregation'):
        assert new[key] == old[key]
    assert new['rescoring']['source_report']['sha256'] == original_files[str(root / 'metrics.json')]
    assert len(new['rescoring']['arrays']) == 2
    assert {str(p): file_sha(p) for p in root.rglob('*') if p.is_file()} == original_files
    with pytest.raises(FileExistsError):
        rescore(root, out, manifest)


def test_legacy_mapping_requires_original_manifest_and_hash(tmp_path):
    root, manifest, _ = history(tmp_path)
    with pytest.raises(ValueError, match='--input-manifest'):
        rescore(root, tmp_path / 'new')
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='SHA256'):
        rescore(root, tmp_path / 'new', manifest)
    assert not (tmp_path / 'new').exists()


def test_embedded_cache_ids_support_manifest_free_rescoring(tmp_path):
    root, _, report = history(tmp_path)
    for row, cache_id in zip(report['images'], ['cache-z', 'cache-a']):
        row['cache_id'] = cache_id
    (root / 'metrics.json').write_text(json.dumps(report))
    rescore(root, tmp_path / 'new')
    assert json.loads((tmp_path / 'new/metrics.json').read_text())['overall']['Stage2']['all']['delta_1.10'] == .5


@pytest.mark.parametrize('change,match', [
    ('duplicate', 'unique'), ('unsafe', 'Unsafe'), ('wrong_identity', 'identity mismatch'),
    ('missing', 'coverage'), ('extra', 'coverage'), ('error', 'error metric mismatch'),
    ('count', 'count/support mismatch'), ('group', 'group coverage'), ('shape', '256x256'),
])
def test_rejects_ambiguous_or_changed_historical_evidence(tmp_path, change, match):
    root, manifest, report = history(tmp_path)
    if change == 'duplicate':
        report['images'][1]['sample_id'] = report['images'][0]['sample_id']
    elif change == 'unsafe':
        report['images'][0]['cache_id'] = '../outside'
        report['images'][1]['cache_id'] = 'cache-a'
        manifest = None
    elif change == 'wrong_identity':
        report['images'][0]['scene'] = 'wrong-scene'
    elif change == 'missing':
        (root / 'supervision/cache-z.npz').unlink()
    elif change == 'extra':
        np.savez(root / 'predictions/unlisted.npz', D256=np.ones((256, 256)))
    elif change == 'error':
        report['images'][0]['metrics']['Stage2']['all']['mae_m'] = 0
    elif change == 'count':
        report['overall']['Stage2']['all']['valid_pixels'] += 1
    elif change == 'group':
        report['groups'] = {}
    elif change == 'shape':
        np.savez(root / 'predictions/cache-z.npz', P256=np.ones((2, 2)), D256=np.ones((2, 2)))
    (root / 'metrics.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        rescore(root, tmp_path / 'new', manifest)
    assert not (tmp_path / 'new').exists()


def test_source_directory_cannot_be_output(tmp_path):
    root, manifest, _ = history(tmp_path)
    for output in (root, root / 'nested'):
        with pytest.raises(ValueError, match='separate'):
            rescore(root, output, manifest)
