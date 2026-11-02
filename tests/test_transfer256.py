"""CPU protocol tests. Synthetic fixtures here are NOT an end-to-end receipt."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from v2_lidf.data import (load_native_labels, prepare_label_cache, resize_intrinsics,
                         nearest_exact_resize, resolve_data_path, V2LIDFDataset)
from v2_lidf.data_protocol import PROTOCOL, weighted_area_depth, resize_supervision
from v2_lidf.curriculum import CurriculumSchedule, mix_background_depth
from v2_lidf.geometry import pixel_rays, unproject_depth, surface_normals
from transparent_depth.prepare_transfer_data import scene_split, cleargrasp_records, transcg_records
from transparent_depth.stage2 import resize_native_inputs, prepare_stage2_inputs, file_sha, export_stage2_bundle, validate_bundle
from transparent_depth.evaluate256 import region_metrics, aggregate, score_predictions
from transparent_depth.train_stage2 import resolve_manifest_counts, validate_training_config
from transparent_depth.prepare_stage2_data import prepare
from transparent_depth.training_cache import validate_training_caches
from transparent_depth import training_utils

ROOT=Path(__file__).resolve().parents[1]


def test_weighted_area_holes_coverage_threshold_and_range():
    # Each 2x2 block maps to one output; valid depths are never diluted by 0.
    x=np.tile(np.array([[1.,0.],[np.nan,1.]],np.float32),(256,256))
    d,v,c=weighted_area_depth(x,PROTOCOL)
    assert d.dtype==c.dtype==np.float32 and v.dtype==np.bool_
    np.testing.assert_array_equal(c,.5);np.testing.assert_array_equal(d,1.)
    assert v.all()
    x[1::2,1::2]=0
    d,v,c=weighted_area_depth(x,PROTOCOL)
    assert not v.any() and np.all(c==.25) and np.all(d==1.)
    x[:]=np.inf
    d,v,c=weighted_area_depth(x,PROTOCOL)
    assert not v.any() and np.all(d==0) and np.all(c==0)
    d,v,c=weighted_area_depth(np.full((720,1280),3,np.float32),PROTOCOL)
    assert np.all(c>.999) and not v.any() and np.allclose(d,3)


def test_full_fov_rgb_area_prior_linear_and_half_pixel_camera():
    rng=np.random.default_rng(23)
    rgb=rng.integers(0,256,(720,1280,3),dtype=np.uint8)
    p=np.linspace(.2,1.2,720*1280,dtype=np.float32).reshape(720,1280)
    k=np.array([[927.17,0,651.32],[0,927.37,349.62],[0,0,1.]])
    r,d,low=resize_native_inputs(rgb,p,k,[256,256],'INTER_AREA')
    np.testing.assert_array_equal(r,cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA))
    np.testing.assert_array_equal(d,cv2.resize(p,(256,256),interpolation=cv2.INTER_LINEAR))
    assert d.min()>.2 and d.max()>1.19  # no unit/scale/min-max operation
    expected=resize_intrinsics(k,(720,1280),(256,256))
    np.testing.assert_allclose(low,expected,rtol=1e-6)
    u,v=133.,87.
    xn=(u+.5)*1280/256-.5;yn=(v+.5)*720/256-.5
    np.testing.assert_allclose([(xn-k[0,2])/k[0,0],(yn-k[1,2])/k[1,1]],
        [(u-low[0,2])/low[0,0],(v-low[1,2])/low[1,1]],atol=1e-7)
    assert prepare_stage2_inputs(r,d,low,(256,256))[0].shape==(1,3,256,256)
    with pytest.raises(ValueError):prepare_stage2_inputs(r,d,low)


@pytest.mark.parametrize('camera,divisor',[('D435',1000),('L515',4000)])
def test_png_units_masks_and_no_sensor_read(tmp_path,camera,divisor):
    Image.fromarray(np.full((8,12),divisor,np.uint16)).save(tmp_path/'gt.png')
    Image.fromarray(np.tile(np.array([0,1],np.uint8),(8,6))).save(tmp_path/'mask.png')
    row=dict(depth_semantics='transcg_uint16_optical_z',camera=camera,depth_divisor=divisor,
        target_depth_path=str(tmp_path/'gt.png'),transparent_mask_path=str(tmp_path/'mask.png'),
        native_hw=[8,12],intrinsics_native=np.eye(3).tolist())
    labels=load_native_labels(row)
    assert np.all(labels['target']==1) and labels['transparent'].dtype==bool
    row['depth_divisor']=4000 if divisor==1000 else 1000
    with pytest.raises(ValueError,match='camera/unit'):load_native_labels(row)
    row['depth_divisor']=divisor
    Image.fromarray(np.full((8,12),255,np.uint8)).save(tmp_path/'mask.png')
    with pytest.raises(ValueError,match='encoding'):load_native_labels(row)


@pytest.mark.parametrize('prefix', ['', 'transcg-1'])
def test_official_flat_and_archive_layout_keep_identity_and_reject_duplicates(tmp_path, prefix):
    info=tmp_path/'info'; info.mkdir()
    official={'train':[1], 'test':[], 'perspective_num':1,
              'train_D435_samples':1, 'train_L515_samples':1,
              'test_D435_samples':0, 'test_L515_samples':0}
    (info/'metadata.json').write_text(json.dumps(official))
    (info/'camera_intrinsics').mkdir()
    root=tmp_path/'data'; scene=root/prefix/'scene1'; (scene/'0').mkdir(parents=True)
    meta={'split':'train','D435_valid_perspective_list':[0],'D435_valid_perspective_num':1,
          'L515_valid_perspective_list':[0],'L515_valid_perspective_num':1}
    (scene/'metadata.json').write_text(json.dumps(meta))
    for c,camera in ((1,'D435'),(2,'L515')):
        np.save(info/'camera_intrinsics'/('%d-camIntrinsics-%s.npy'%(c,camera)),np.eye(3))
        for name in ('rgb%d.png','depth%d-gt.png','depth%d-gt-mask.png'):
            (scene/'0'/(name%c)).touch()
    records,_=transcg_records(root,info,{'train':[1],'val':[]})
    assert [r['sample_id'] for r in records]==['transcg/scene1/perspective0/camera1','transcg/scene1/perspective0/camera2']
    duplicate=root/('transcg-1' if not prefix else '')/'scene1'
    duplicate.mkdir(parents=True); (duplicate/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Duplicate scene identity'):
        transcg_records(root,info,{'train':[1],'val':[]})


def test_scene_split_deterministic_camera_independent():
    official=json.loads((ROOT/'data/transcg_info/metadata.json').read_text())
    split=scene_split(official)
    assert split==scene_split(official) and len(split['val'])==8
    assert not set(split['train']) & set(split['val'])
    assert set(split['train']+split['val'])==set(official['train'])
    assert split['official_test']==sorted(official['test'])
    assert scene_split(official,24)['val']!=split['val']


def test_cleargrasp_frame_pairing_never_file_sorting(tmp_path):
    d=tmp_path/'scene'/'d435';d.mkdir(parents=True)
    (d/'camera_intrinsics.yaml').write_text('camera_name: D435\nxres: 1280\nyres: 720\nfx: 921\nfy: 921\ncx: 642\ncy: 359\n')
    for i in [12,3]:
        for suffix in ['transparent-rgb-img.jpg','opaque-depth-img.exr','mask.png']:
            (d/('%09d-%s'%(i,suffix))).touch()
    # Forbidden sources are never paired, opened or listed as inputs.
    (d/'000000003-transparent-depth-img.exr').touch()
    (d/'000000003-output-depth.exr').touch()
    rows=cleargrasp_records(tmp_path)
    assert [r['sample_id'].split('/')[-1] for r in rows]==['000000003','000000012']
    assert all(r['gt'].name.endswith('opaque-depth-img.exr') for r in rows)
    (d/'000000004-mask.png').touch()
    with pytest.raises(ValueError,match='match exactly'):cleargrasp_records(tmp_path)


def test_metric_support_macro_empty_and_baseline_fairness():
    g=np.ones((2,2),np.float32);p=g*1.1;d=g*1.2
    labels={'target':g,'valid':np.array([[1,1],[1,0]],bool),
            'transparent':np.array([[1,0],[1,1]],bool)}
    scored=score_predictions(p,d,labels)
    for region,count in [('all',3),('transparent',2),('background',1)]:
        assert scored['P256'][region]['valid_pixels']==scored['Stage2'][region]['valid_pixels']==count
    assert scored['Stage2']['all']['mae_m']==pytest.approx(.2)
    empty=score_predictions(p,d,{**labels,'transparent':np.zeros((2,2),bool)})
    result=aggregate([{'metrics':scored},{'metrics':empty}])
    assert result['Stage2']['transparent']['nonempty_images']==1
    assert result['Stage2']['transparent']['empty_images']==1
    with pytest.raises(FloatingPointError):region_metrics(np.full((2,2),np.nan),g,labels['valid'])


def test_optical_z_rays_normals_and_invalid_neighbour_semantics():
    k=torch.tensor([[[256.,0.,127.5],[0.,256.,127.5],[0.,0.,1.]]])
    xyz=unproject_depth(torch.ones((1,1,256,256)),k)
    assert torch.all(xyz[:,2]==1)
    normals=surface_normals(xyz)
    assert torch.all(normals[:,2,:-1,:-1]>.999)
    # Existing model masks current, right and lower GT plus image boundaries.
    source=(ROOT/'src/v2_lidf/model.py').read_text()
    assert 'normal_valid[:, :, :, :-1] &= supervision_valid[:, :, :, 1:]' in source
    assert 'normal_valid[:, :, :-1, :] &= supervision_valid[:, :, 1:, :]' in source


def test_curriculum_transparent_and_holes_never_leak():
    p=torch.tensor([1.,2.,3.,4.]);g=torch.tensor([2.,99.,float('nan'),5.])
    v=torch.tensor([1,1,0,1],dtype=torch.bool);m=torch.tensor([0,1,0,0],dtype=torch.bool)
    for alpha in (1.,.5,0.):
        mixed=mix_background_depth(p,g,v,m,alpha)
        assert torch.equal(mixed[1:3],p[1:3])
        assert mixed[0]==1+alpha
    assert mix_background_depth(p,object(),object(),object(),0) is p


def test_stage1_guard_rejects_transcg_labels_and_sensor_even_under_project(tmp_path):
    from transparent_depth.input_guard import RGBInputGuard
    from transparent_depth.stage1 import forbid_label_reads
    guard=RGBInputGuard(project=tmp_path,asset_root=tmp_path/'assets',rgb_paths=[],
        output_root=tmp_path/'out',cache_root=tmp_path/'jit')
    names=['depth1.png','depth2.png','depth1-gt.png','depth2-gt.png','depth1-gt-mask.png']
    guard.active=True
    for name in names:
        with pytest.raises(PermissionError):guard.check(tmp_path/name)
    guard.active=False
    with forbid_label_reads():
        for name in names:
            with pytest.raises(AssertionError):open(tmp_path/name,'rb')


@pytest.mark.parametrize('steps',[2,13,101])
def test_formal_and_smoke_schedule_boundaries(steps):
    s=CurriculumSchedule(steps,(20,20,20))
    assert s.alpha(20*steps-1)==s.alpha(20*steps)==1
    assert 0<s.alpha(30*steps)<1
    assert s.alpha(40*steps-1)==s.alpha(40*steps)==s.alpha(60*steps-1)==0
    smoke=CurriculumSchedule(2,(1,2,1))
    assert [smoke.alpha(i) for i in range(8)]==pytest.approx([1,1,1,2/3,1/3,0,0,0])


def fixture256(tmp_path):
    # Unit-test receipt only. It never enters work/, artifacts/, or GPU smoke.
    from test_stage2_preparation import fixture_data, canonical
    old,inputs,labels,predictions=fixture_data(tmp_path)
    from transparent_depth.prepare_stage2_data import fingerprint
    contract=json.loads((predictions.parent/'contract.json').read_text())
    contract.update(assets_sha256=file_sha(ROOT/'provenance/stage1_assets.json'),
                    sources_sha256=file_sha(ROOT/'provenance/stage1_runtime_sources.json'))
    (predictions.parent/'contract.json').write_text(json.dumps(contract))
    records=[json.loads(x) for x in predictions.read_text().splitlines()]
    for r in records:
        r['metadata'].update(assets_sha256=contract['assets_sha256'],source_manifest_sha256=contract['sources_sha256'])
        r['contract_sha256']=fingerprint(contract)
    predictions.write_text(''.join(canonical(r)+'\n' for r in records))
    seal=json.loads((predictions.parent/'complete.json').read_text())
    seal.update(manifest_sha256=file_sha(predictions),contract_sha256=fingerprint(contract))
    (predictions.parent/'complete.json').write_text(json.dumps(seal))
    raw=[json.loads(x) for x in inputs.read_text().splitlines()]
    targets=[json.loads(x) for x in labels.read_text().splitlines()]
    for i,(r,label) in enumerate(zip(raw,targets)):
        r.update(dataset='transcg',scene='scene%d'%i,camera='D435',official_split='train',
            split_sha256='a'*64,intrinsics_source='official-test-fixture',intrinsics_source_sha256='b'*64)
        r['roles']=['train'] if i==0 else ['val']
        name='gt%d.png'%i;Image.fromarray(np.full((8,12),800,np.uint16)).save(inputs.parent/name)
        label.update(depth_semantics='transcg_uint16_optical_z',depth_divisor=1000,camera='D435',target_depth_path=name)
        Image.fromarray(np.pad(np.ones((4,4),np.uint8),((2,2),(4,4)))).save(inputs.parent/label['transparent_mask_path'])
    inputs.write_text(''.join(canonical(r)+'\n' for r in raw));labels.write_text(''.join(canonical(r)+'\n' for r in targets))
    out=tmp_path/'prepared';prepare(inputs,labels,predictions,out,PROTOCOL)
    cfg=json.loads((ROOT/'configs/transcg_stage2_60.json').read_text())
    return cfg,inputs,labels,predictions,out


def test_256_cache_invalidation_and_manifest_counts(tmp_path):
    cfg,inputs,labels,pred,out=fixture256(tmp_path)
    im=out/'manifests/input_manifest.jsonl';lm=out/'manifests/label_manifest.jsonl'
    cfg=resolve_manifest_counts(cfg,im,1)
    assert cfg['counts']=={'train':1,'val':1,'evaluation':0}
    assert validate_training_config(cfg,1).steps_per_epoch==1
    validate_training_caches(cfg,im,lm,out/'stage1',out/'labels')
    ds=V2LIDFDataset(im,lm,out/'stage1',out/'labels',curriculum=True)
    item=ds[0]
    assert item['coverage'].shape==(1,256,256) and item['rgb'].shape==(3,256,256)
    val=V2LIDFDataset(im,None,out/'stage1',None,role='val',include_target=False)[0]
    assert not {'depth_gt','gt_valid','transparent_mask','coverage'} & set(val)
    changed=copy.deepcopy(PROTOCOL);changed['depth_range_m']=[.2,2.]
    with pytest.raises(ValueError,match='different source'):prepare(inputs,labels,pred,out,changed)
    cfg['preprocessing']=changed
    with pytest.raises(ValueError,match='preprocessing'):validate_training_caches(cfg,im,lm,out/'stage1',out/'labels')


def test_val_pure_p_label_order_and_metric(tmp_path,monkeypatch):
    cfg,inputs,labels,pred,out=fixture256(tmp_path)
    ds=V2LIDFDataset(out/'manifests/input_manifest.jsonl',None,out/'stage1',None,role='val',include_target=False)
    rows={r['cache_id']:r for r in map(json.loads,(out/'manifests/label_manifest.jsonl').read_text().splitlines()) if r['role']=='val'}
    class Predictor(torch.nn.Module):
        def forward(self,rgb,p,k,compute_loss):
            assert compute_loss is False
            self.called=True
            return {'depth':p,'query_mask':torch.ones_like(p,dtype=torch.bool)}
    model=Predictor();model.called=False
    import transparent_depth.evaluate256 as ev
    original=ev.load_authenticated_native_labels
    def after_forward(row):
        assert model.called
        return original(row)
    monkeypatch.setattr(ev,'load_authenticated_native_labels',after_forward)
    result=training_utils.validate(model,ds,rows,torch.device('cpu'),0,1)
    assert result['gt_or_mask_used_as_input'] is False and result['transparent_mae_m']>0
    with pytest.raises(ValueError,match='always uses pure P'):
        training_utils.validate(model,ds,rows,torch.device('cpu'),0,1,alpha=.5)


def test_portable_root_binding_and_missing_root(monkeypatch,tmp_path):
    monkeypatch.setenv('TRANSCG_ROOT',str(tmp_path))
    assert resolve_data_path('${TRANSCG_ROOT}/a.png')==str(tmp_path/'a.png')
    monkeypatch.delenv('TRANSCG_ROOT')
    with pytest.raises(ValueError,match='Unbound'):resolve_data_path('${TRANSCG_ROOT}/a.png')


def test_cache_relocation_preserves_seals_without_old_dataset_path(tmp_path,monkeypatch):
    import shutil
    from transparent_depth.prepare_stage2_data import canonical
    cfg,inputs,labels,pred,out=fixture256(tmp_path)
    source=inputs.parent
    monkeypatch.setenv('TRANSCG_ROOT',str(source))
    rgb_rows=[json.loads(x) for x in inputs.read_text().splitlines()]
    label_rows=[json.loads(x) for x in labels.read_text().splitlines()]
    for row in rgb_rows:row['rgb_path']='${TRANSCG_ROOT}/'+row['rgb_path']
    for row in label_rows:
        for field in ('target_depth_path','transparent_mask_path'):
            row[field]='${TRANSCG_ROOT}/'+row[field]
    inputs.write_text(''.join(canonical(r)+'\n' for r in rgb_rows))
    labels.write_text(''.join(canonical(r)+'\n' for r in label_rows))
    portable=tmp_path/'portable';prepare(inputs,labels,pred,portable,PROTOCOL)
    moved=tmp_path/'moved';shutil.copytree(portable,moved)
    relocated=tmp_path/'new-dataset';shutil.copytree(source,relocated)
    monkeypatch.setenv('TRANSCG_ROOT',str(relocated))
    im=moved/'manifests/input_manifest.jsonl';lm=moved/'manifests/label_manifest.jsonl'
    validate_training_caches(cfg,im,lm,moved/'stage1',moved/'labels')
    ds=V2LIDFDataset(im,lm,moved/'stage1',moved/'labels',curriculum=True)
    assert ds[0]['depth_gt'].shape==(1,256,256)
    row=next(r for r in map(json.loads,lm.read_text().splitlines()) if r['role']=='train')
    assert prepare_label_cache(row,moved/'labels')['status']=='existing'
    assert file_sha(im)==file_sha(portable/'manifests/input_manifest.jsonl')


def test_formal_config_a100_legacy_ddp_guard_and_lr():
    cfg=json.loads((ROOT/'configs/transcg_stage2_60.json').read_text())
    cfg['curriculum']['steps_per_epoch']=3
    s=validate_training_config(cfg,1)
    assert s.total_steps==180 and cfg['model']['spvnas']['voxel_size']==.05
    assert cfg['scheduler']=={'step_size':20,'gamma':.1}
    p=torch.nn.Parameter(torch.tensor(1.));o=torch.optim.Adam([p],lr=cfg['learning_rate'])
    lr=torch.optim.lr_scheduler.StepLR(o,**cfg['scheduler']);values=[]
    for i in range(60):values.append(o.param_groups[0]['lr']);o.step();lr.step()
    assert values[19]==pytest.approx(.001) and values[20]==pytest.approx(.0001) and values[40]==pytest.approx(.00001)
    cfg['distributed']['torch_version']='2.11.0'
    with pytest.raises(ValueError,match='pinned'):validate_training_config(cfg,1)


def test_bundle_preserves_256_protocol_and_resolution(tmp_path):
    cfg=json.loads((ROOT/'configs/transcg_stage2_60.json').read_text())
    path=tmp_path/'checkpoint.pt'
    torch.save({'model':{'spvnas_branch.example':torch.ones(1)},'model_config':cfg['model'],
        'contract':{'config':cfg},'epoch':0,'updates':1},path)
    out=tmp_path/'bundle'
    meta=export_stage2_bundle(path,out,{'selection_rule':'CPU serialization fixture only'})
    assert meta['input']['model_hw']==[256,256] and meta['input']['preprocessing']==PROTOCOL
    validate_bundle(out)
    meta['input']['model_hw']=[240,320]
    (out/'model.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='geometry'):validate_bundle(out)
