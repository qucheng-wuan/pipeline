import argparse
import hashlib
import json
from pathlib import Path

import pytest

from transparent_depth import cli
from transparent_depth.assets import check_file, destination, fetch_asset, validate_manifest


def test_release_paths_reject_traversal_and_symlink_escape(tmp_path):
    outside = tmp_path / 'outside'
    project = tmp_path / 'project'
    outside.mkdir()
    project.mkdir()
    (project / 'escape').symlink_to(outside, target_is_directory=True)
    for name in ('../outside/weight.pt', '/tmp/weight.pt', 'escape/weight.pt'):
        with pytest.raises(ValueError):
            destination(project, name)
    assert destination(project, 'nested/weight.pt') == project / 'nested/weight.pt'


def test_pipeline_keeps_calibration_in_stage2_and_only_model_predictions_as_depth(tmp_path):
    args = argparse.Namespace(asset_root=None, stage1_python='/env/one/python',
        stage2_python='/env/two/python', rgb=tmp_path/'rgb.png',
        intrinsics=tmp_path/'K.json', output=tmp_path/'depth.npy', device='cuda:0',
        runtime_cache=tmp_path/'writable_runtime')
    cfg = {'asset_root':'artifacts', 'stage1':{'assets_manifest':'provenance/weights.json'},
           'stage2':{'bundle':'stage2'}}
    intermediate = tmp_path / 'predicted.npy'
    one, two = cli.inference_commands(args, tmp_path, cfg, intermediate)
    assert one[0] == '/env/one/python' and two[0] == '/env/two/python'
    assert '--intrinsics' not in one
    assert two[two.index('--intrinsics')+1] == str(args.intrinsics)
    assert two[two.index('--depth-input')+1] == str(intermediate)
    assert one[one.index('--runtime-cache')+1] == str(args.runtime_cache)
    assert '--runtime-cache' not in two
    assert not any('mask' in value or 'target' in value for value in one + two)


def test_unverified_selection_cannot_be_used_for_release_inference(tmp_path):
    (tmp_path/'configs').mkdir()
    (tmp_path/'src/transparent_depth').mkdir(parents=True)
    (tmp_path/'configs/release.json').write_text(json.dumps({'status':'selection_pending'}))
    args = argparse.Namespace(project_root=tmp_path, config=None)
    with pytest.raises(ValueError, match='not complete'):
        cli.infer(args)


def test_asset_hash_is_required_before_reusing_or_overwriting_a_file(tmp_path):
    root, work = tmp_path/'assets', tmp_path/'work'
    root.mkdir()
    work.mkdir()
    path = root/'model.bin'
    path.write_bytes(b'expected')
    entry = {'path':'model.bin','sha256':hashlib.sha256(b'expected').hexdigest(),
             'bytes':8,'download':{'kind':'https','url':'https://example.invalid/model'}}
    assert check_file(path, entry)
    assert fetch_asset(entry, root, work)['status'] == 'already_verified'
    path.write_bytes(b'corrupted')
    with pytest.raises(FileExistsError):
        fetch_asset(entry, root, work)


def test_release_manifest_rejects_duplicate_or_untrusted_paths(tmp_path):
    asset = {'path':'model.pt','sha256':'0'*64,'bytes':1,
             'download':{'kind':'https','url':'https://example.invalid/file'}}
    with pytest.raises(ValueError, match='Duplicate'):
        validate_manifest({'schema':'transparent_depth_release_assets_v1','assets':[asset,asset]})
    asset['path'] = '../outside'
    with pytest.raises(ValueError):
        validate_manifest({'schema':'transparent_depth_release_assets_v1','assets':[asset]})


def test_legitimate_but_wrong_bundle_cannot_be_reported_as_selected_winner(tmp_path):
    (tmp_path/'provenance').mkdir()
    manifest = tmp_path/'provenance/assets.json'
    manifest.write_text('{}')
    bundle = tmp_path/'artifacts/stage2'
    bundle.mkdir(parents=True)
    weights = bundle/'model.pt'
    weights.write_bytes(b'complete trained state fixture')
    selection = {'checkpoint_epoch':12, 'checkpoint_sha256':'a'*64,
                 'metric':'native_image_macro_transparent_mae_mm'}
    metadata = {'origin':{'epoch_one_based':12,'checkpoint_sha256':'a'*64,
                          'selection':selection},
                'weights':{'file':'model.pt','sha256':cli.sha256(weights)}}
    model_json = bundle/'model.json'
    model_json.write_text(json.dumps(metadata))
    cfg = {'stage1':{'assets_manifest':'provenance/assets.json',
                     'assets_manifest_sha256':cli.sha256(manifest)},
           'stage2':{'bundle':'stage2','bundle_manifest_sha256':cli.sha256(model_json),
                     'checkpoint_epoch':12,'source_checkpoint_sha256':'a'*64,
                     'weights_sha256':cli.sha256(weights)},
           'selection':selection}
    assert cli.validate_release_assets(tmp_path,cfg,tmp_path/'artifacts')
    metadata['origin']['epoch_one_based'] = 63
    model_json.write_text(json.dumps(metadata))
    # Even if another bundle has internally valid weights and JSON, a release
    # cannot label it as the chosen epoch merely by changing one file checksum.
    cfg['stage2']['bundle_manifest_sha256'] = cli.sha256(model_json)
    with pytest.raises(ValueError, match='not the declared'):
        cli.validate_release_assets(tmp_path,cfg,tmp_path/'artifacts')


def test_changed_stage1_registry_is_rejected_before_any_model_starts(tmp_path):
    (tmp_path/'provenance').mkdir()
    (tmp_path/'provenance/assets.json').write_text('{"different":"expert"}')
    cfg = {'stage1':{'assets_manifest':'provenance/assets.json','assets_manifest_sha256':'0'*64}}
    with pytest.raises(ValueError, match='Stage1 asset registry'):
        cli.validate_release_assets(tmp_path,cfg,tmp_path/'artifacts')
