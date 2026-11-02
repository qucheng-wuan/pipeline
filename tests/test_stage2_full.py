"""CPU boundary tests for all-roster acceptance; no CUDA/quality claims."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage2_full', PROJECT / 'scripts/validate_stage2_full.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


def identity(cid='one', split='real-test'):
    return dict(cache_id=cid, sample_id=cid, rgb_sha256='a'*64,
                native_hw=[6, 8], domain='real', official_split=split, benchmark_group='real_novel')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def selection_fixture(tmp_path):
    root = tmp_path / 'selection'
    write(root/'contract.json', {'label_manifest':{'sha256':'b'*64}, 'input_manifest':{'sha256':'c'*64}})
    selected = {'status': 'complete', 'completed_epochs': 80,
        'contract':full.ref(root/'contract.json'),
        'ranking': [{'epoch': i, 'checkpoint_sha256': str(i).zfill(64)} for i in range(1, 81)],
        'selection_policy': {'uses_test_labels_for_checkpoint_selection': True,
                             'independent_test_performance_claimed': False}}
    write(root/'selection.complete.json', selected)
    write(root/'publication_verification.complete.json', {'status':'passed','epochs':80,
        'predictions_verified':98080,'selection':full.ref(root/'selection.complete.json')})
    checkpoint = {'path': '/evidence/unchanged/epoch004.pt', 'sha256':str(4).zfill(64)}
    write(root/'selected_geometry/resolved_selection.json', {'status':'complete','selected_epoch':4,
        'selected_checkpoint':checkpoint,'original_selection':full.ref(root/'selection.complete.json'),
        'verification':full.ref(root/'publication_verification.complete.json'),
        'scores':{'4':{'transparent_mae_mm':52.5}}})
    seal = root/'epoch004/predictions.complete.json'
    write(seal, {'images':1226,'smoke':False,'label_access':False,'input_alpha':0.,
                 'compute_loss':False,'checkpoint_sha256':checkpoint['sha256']})
    write(seal.parent/'complete.json', {'status':'complete','prediction_seal':full.ref(seal)})
    metadata = {'origin':{'epoch_one_based':4,'checkpoint_sha256':checkpoint['sha256']}}
    return root, metadata


def test_all_eighty_resolution_required_and_wrong_bundle_rejected(tmp_path):
    root, metadata = selection_fixture(tmp_path)
    result, _ = full.selection_evidence(root, metadata)
    assert result['selected_epoch'] == 4 and result['historical_transparent_mae_mm'] == 52.5
    wrong = copy.deepcopy(metadata)
    wrong['origin']['epoch_one_based'] = 63
    with pytest.raises(ValueError, match='finally resolved'):
        full.selection_evidence(root, wrong)
    selected = full.read(root/'selection.complete.json')
    selected['completed_epochs'] = 79
    write(root/'selection.complete.json', selected)
    with pytest.raises(ValueError, match='eighty'):
        full.selection_evidence(root, metadata)


def test_unfinished_resolution_and_changed_epoch_seal_are_rejected(tmp_path):
    root, metadata = selection_fixture(tmp_path)
    resolution = full.read(root/'selected_geometry/resolved_selection.json')
    resolution['status'] = 'running'
    write(root/'selected_geometry/resolved_selection.json', resolution)
    with pytest.raises(ValueError, match='near-tie'):
        full.selection_evidence(root, metadata)
    resolution['status'] = 'complete'
    write(root/'selected_geometry/resolved_selection.json', resolution)
    seal = root/'epoch004/predictions.complete.json'
    changed = full.read(seal)
    changed['unused_extra'] = True
    write(seal, changed)
    with pytest.raises(ValueError, match='seal differs'):
        full.selection_evidence(root, metadata)


def test_full_identity_rejects_missing_duplicate_and_changed_rgb():
    rows = [identity('a'), identity('b')]
    full.validate_roster(rows, rows, 2)
    with pytest.raises(ValueError, match='coverage'):
        full.validate_roster(rows[:1], rows, 2)
    with pytest.raises(ValueError, match='Duplicate'):
        full.validate_roster([rows[0], rows[0]], rows, 2)
    changed = copy.deepcopy(rows)
    changed[0]['rgb_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='identity'):
        full.validate_roster(changed, rows, 2)


def test_native_cpu_resize_and_explicit_numeric_tolerance():
    import torch
    low = np.arange(12, dtype=np.float32).reshape(3,4) + .5
    wanted = torch.nn.functional.interpolate(torch.from_numpy(low)[None,None],
        size=(6,8), mode='bilinear', align_corners=False)[0,0].numpy()
    assert np.array_equal(full.resize_cpu(low, [6,8]), wanted)
    a = np.ones((2,2), np.float32)
    assert full.difference(a+5e-7, a)['pixels_over_tolerance'] == 0
    assert full.difference(a+2e-6, a)['pixels_over_tolerance'] == 4
    with pytest.raises(ValueError, match='finite'):
        full.difference(a*np.nan, a)


def test_cached_inputs_accept_original_native_hw_and_reject_label_arrays(tmp_path):
    from v2_lidf.data import resize_intrinsics
    row = identity()
    native = np.full(row['native_hw'], .7, dtype=np.float32)
    path = tmp_path/'native.npy'
    np.save(path, native)
    row['stage1_native'] = full.ref(path)
    k_native = np.array([[50.,0,3.],[0,50.,2.],[0,0,1.]])
    k = resize_intrinsics(k_native, row['native_hw']).astype(np.float32)
    row.update(intrinsics=k.tolist(), intrinsics_native=k_native.tolist())
    values = {'rgb':np.full((240,320,3), 50, dtype=np.uint8),
        'depth_input':np.full((240,320), .7, dtype=np.float32), 'intrinsics':k,
        'native_hw':np.array(row['native_hw'], dtype=np.int32),
        'metadata_json':np.array(json.dumps({'rgb_sha256':row['rgb_sha256'],'ground_truth_access':False}))}
    archive = tmp_path/'input.npz'
    np.savez(archive, **values)
    row['input_cache'] = full.ref(archive)
    rgb, p, camera, original = full.input_arrays(row, 'cached')
    assert np.array_equal(rgb, values['rgb']) and np.array_equal(p, values['depth_input'])
    assert np.array_equal(camera, k) and np.array_equal(original, native)
    np.savez(archive, **values, depth_gt=np.ones((240,320), np.float32))
    row['input_cache'] = full.ref(archive)
    with pytest.raises(ValueError, match='may not contain labels'):
        full.input_arrays(row, 'cached')


def test_prediction_archive_shapes_and_masks_are_authenticated(tmp_path):
    row = identity()
    values = {'depth':np.ones((240,320),np.float32),
        'refined_mask':np.zeros((240,320),bool),
        **{k:np.ones((6,8),np.float32) for k in ('native_depth','stage1_native','stage1_resized')}}
    full.validate_arrays(values, row)
    path = tmp_path/'predictions/one.npz'
    path.parent.mkdir()
    np.savez(path, **values)
    receipt = {'prediction':{'path':'predictions/one.npz','sha256':full.sha(path)}}
    assert np.array_equal(full.read_output(tmp_path,row,receipt)['depth'],values['depth'])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='changed after sealing'):
        full.read_output(tmp_path,row,receipt)
    values['refined_mask'] = values['refined_mask'].astype(np.uint8)
    with pytest.raises(ValueError, match='boolean'):
        full.validate_arrays(values,row)


def test_prediction_process_command_cannot_receive_labels(tmp_path):
    args = argparse.Namespace(output=tmp_path/'run', label_manifest=tmp_path/'secret.jsonl', metric_device='cpu')
    command = full.child_command(args, 'worker', 2)
    assert '--label-manifest' not in command and str(args.label_manifest) not in command
    labelled = full.child_command(args, 'metric-worker', 2)
    assert labelled[labelled.index('--label-manifest')+1] == str(args.label_manifest)
    assert '--bundle' not in labelled and '--stage1-predictions' not in labelled


def test_incomplete_predictions_fail_before_any_labels_are_opened(tmp_path, monkeypatch):
    import v2_lidf.data
    calls = []
    monkeypatch.setattr(v2_lidf.data, 'load_label_manifest', lambda *a,**kw:calls.append('GT'))
    args = argparse.Namespace(output=tmp_path, label_manifest=tmp_path/'GT.jsonl', rank=0, metric_device='cpu')
    with pytest.raises(FileNotFoundError):
        full.metric_worker(args)
    assert calls == []


def test_cache_vs_live_difference_never_relabels_changed_output_as_numeric_pass(tmp_path):
    row = identity()
    low = np.full((240,320), .7, np.float32)
    mask = np.ones((240,320), bool)
    path = tmp_path/'golden.npz'
    np.savez(path,depth=low,refined_mask=mask)
    row['golden_prediction'] = full.ref(path)
    result = dict(depth=low.copy(), native_depth=full.resize_cpu(low,row['native_hw']),refined_mask=mask.copy())
    assert full.compare_golden(result,row)['passed'] is True
    result['depth'][0,0] += .01
    assert full.compare_golden(result,row)['passed'] is False
    result['depth'] = low.copy()
    result['refined_mask'][0,0] = False
    comparison = full.compare_golden(result,row)
    assert comparison['refined_mask_disagree_pixels'] == 1 and comparison['passed'] is False


def test_artifact_paths_cannot_escape_via_traversal_or_symlink(tmp_path):
    output = tmp_path/'output'
    outside = tmp_path/'outside'
    output.mkdir(); outside.mkdir()
    (output/'link').symlink_to(outside, target_is_directory=True)
    for relative in ('../outside/file.npy','link/file.npy',str(outside/'file.npy')):
        with pytest.raises(ValueError,match='relative|escapes'):
            full.inside(output,relative)


def test_actual_stage2_openexr_legacy_guard_blocks_label_reader(tmp_path):
    import OpenEXR
    target = tmp_path/'ground_truth.exr'
    target.write_bytes(b'not opened')
    original = OpenEXR.InputFile
    guard = full.Stage2InputGuard(project=tmp_path,asset_root=tmp_path,rgb_paths=[],
        output_root=tmp_path/'out',cache_root=tmp_path/'cache')
    with guard:
        with pytest.raises(PermissionError,match='RGB-only'):
            OpenEXR.InputFile(str(target))
    assert OpenEXR.InputFile is original
    assert guard.denials == [str(target.resolve())]
