import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
import torch

from transparent_depth import stage2
from transparent_depth.train_stage2 import validate_training_config
from transparent_depth.training_utils import accumulation_window_size, repair_selected_checkpoint
from v2_lidf.curriculum import CurriculumSchedule, mix_background_depth
from v2_lidf.data import resize_intrinsics

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / 'configs/stage2_train.json').read_text())


def test_numerical_sources_preserved():
    origin = json.loads((ROOT / 'provenance/stage2_source_origin.json').read_text())
    adapters = json.loads((ROOT / 'provenance/transcg256_source_amendment.json').read_text())['changes']
    changed = {'src/v2_lidf/__init__.py', 'src/v2_lidf/distributed.py'}
    for relative, expected in origin['numerical_source_sha256'].items():
        if relative in adapters:
            assert adapters[relative]['original_sha256'] == expected
            expected = adapters[relative]['sha256']
        if relative not in changed:
            assert stage2.file_sha(ROOT / relative) == expected, relative


def test_no_machine_paths_in_published_training_config():
    text = json.dumps(config())
    assert '/home/' not in text and '/mnt/' not in text
    schedule = validate_training_config(config(), 3)
    assert schedule.total_steps == 80 * 1516
    with pytest.raises(ValueError, match='WORLD_SIZE'):
        validate_training_config(config(), 8)


def test_native_preprocessing_matches_production():
    import cv2
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, (521, 777, 3), dtype=np.uint8)
    depth = rng.uniform(.1, 2, (521, 777)).astype(np.float32)
    k = np.array([[600., 0., 375.2], [0., 601., 251.7], [0., 0., 1.]])
    a, b, c = stage2.resize_native_inputs(rgb, depth, k)
    assert np.array_equal(a, cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_LINEAR))
    assert np.array_equal(b, cv2.resize(depth, (320, 240), interpolation=cv2.INTER_LINEAR))
    assert np.array_equal(c, resize_intrinsics(k, (521, 777)).astype(np.float32))
    # Projection of a native pixel centre after resizing is exactly half-pixel transformed.
    uv = np.array([163., 101., 1.])
    expected_uv = np.array([(uv[0]+.5)*320/777-.5, (uv[1]+.5)*240/521-.5, 1.])
    assert np.allclose(c @ np.linalg.solve(k, uv), expected_uv, atol=2e-5, rtol=0)


def test_input_contract_rejects_sensor_holes_and_bad_calibration():
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.ones((240, 320), dtype=np.float32)
    k = np.array([[100., 0., 150.], [0., 100., 100.], [0., 0., 1.]])
    tensor, d, _ = stage2.prepare_stage2_inputs(rgb, depth, k)
    assert tensor.shape == (1, 3, 240, 320) and torch.equal(d[0, 0], torch.from_numpy(depth))
    depth[0, 0] = 0
    with pytest.raises(ValueError, match='finite positive'):
        stage2.prepare_stage2_inputs(rgb, depth, k)
    depth[0, 0] = 1
    k[0, 1] = 1
    with pytest.raises(ValueError, match='zero-skew'):
        stage2.prepare_stage2_inputs(rgb, depth, k)


def test_bundle_relocates_and_loads_complete_state_without_initialization(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spvnas_branch = torch.nn.Linear(3, 2)
    original = Model()
    ckpt = tmp_path / 'trusted.pt'
    cfg = config()['model']
    cfg['spvnas']['assets_dir'] = '/removed/history'
    cfg['spvnas']['audit_dir'] = '/removed/logs'
    torch.save({'model': original.state_dict(), 'model_config': cfg, 'epoch': 62, 'updates': 100}, ckpt)
    bundle = tmp_path / 'bundle'
    exported = stage2.export_stage2_bundle(ckpt, bundle, {'selection_rule': 'test-informed transparent MAE'})
    assert exported['origin']['epoch_one_based'] == 63
    assert 'assets_dir' not in exported['model_config']['spvnas']
    assert not exported['model_config']['rgb_pretrained']
    relocated = tmp_path / 'other' / 'bundle'
    relocated.parent.mkdir()
    shutil.move(str(bundle), relocated)
    ckpt.unlink()
    calls = []
    def build(cfg, device, initialization):
        calls.append(initialization)
        assert not cfg['spvnas']['pretrained'] and not cfg['rgb_pretrained']
        return Model()
    monkeypatch.setattr(stage2, 'build_stage2', build)
    restored = stage2.load_stage2(relocated)
    assert calls == [False] and not restored.training
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in original.state_dict().items())
    (relocated / 'model.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='weights changed'):
        stage2.validate_bundle(relocated)


def test_bundle_source_and_shape_fail_closed(tmp_path, monkeypatch):
    ckpt = tmp_path / 'checkpoint.pt'
    torch.save({'model': {'spvnas_branch.weight': torch.ones(2)}, 'model_config': config()['model'], 'epoch': 0}, ckpt)
    bundle = tmp_path / 'bundle'
    stage2.export_stage2_bundle(ckpt, bundle, {'selection_rule': 'validation-only'})
    monkeypatch.setattr(stage2, 'source_hashes', lambda: {})
    with pytest.raises(ValueError, match='source identity'):
        stage2.validate_bundle(bundle)


def test_curriculum_keeps_transparent_gt_private_and_zero_alpha_exact():
    p = torch.tensor([[[[.2, .4, .6, .8]]]])
    g = torch.tensor([[[[1., 2., 3., float('nan')]]]])
    valid = torch.tensor([[[[True, True, False, False]]]])
    mask = torch.tensor([[[[False, True, False, False]]]])
    a = mix_background_depth(p, g, valid, mask, .5)
    g[mask] = 1e9
    b = mix_background_depth(p, g, valid, mask, .5)
    assert torch.equal(a, b)
    assert torch.equal(a[..., 1:], p[..., 1:])
    assert mix_background_depth(p, object(), object(), object(), 0.0) is p


def test_schedule_and_partial_accumulation_resume():
    s = CurriculumSchedule(1516, (20, 30, 30))
    assert s.alpha(20 * 1516 - 1) == s.alpha(20 * 1516) == 1
    assert s.alpha(50 * 1516 - 1) == s.alpha(50 * 1516) == 0
    updates = 34567
    state = json.loads(json.dumps({'updates': updates}))
    assert s.alpha(state['updates']) == s.alpha(updates)
    assert [accumulation_window_size(i, 7, 5) for i in range(7)] == [5]*5+[2]*2


def test_recover_crash_after_last_before_best(tmp_path):
    saved = {'best_score': .1, 'validation': {'abs_rel': .1}, 'epoch_record': {'selected': True},
             'model': {'x': torch.ones(1)}, 'contract': {}, 'epoch': 55}
    torch.save({'stale': True}, tmp_path / 'best.pt')
    repair_selected_checkpoint(tmp_path, saved)
    recovered = torch.load(tmp_path / 'best.pt', weights_only=False)
    assert recovered['epoch'] == 55 and torch.equal(recovered['model']['x'], saved['model']['x'])


def test_runtime_inspection_does_not_initialize_cuda(monkeypatch):
    from v2_lidf.runtime import validate_runtime
    def forbidden(*args, **kwargs):
        raise AssertionError('Runtime source inspection attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    record = validate_runtime()
    assert record['torchsparse_commit'] == '385f5ce8718fcae93540511b7f5832f4e71fd835'
