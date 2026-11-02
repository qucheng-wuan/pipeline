import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from transparent_depth.prepare_stage2_data import prepare, canonical, fingerprint
from transparent_depth.stage2 import file_sha
from transparent_depth.training_cache import (validate_training_caches,
    AuthenticatedTrainingDataset, load_authenticated_native_labels)
from transparent_depth.train_stage2 import establish_contract, validate_training_config
from transparent_depth import train_stage2, training_utils
from v2_lidf.data import read_jsonl, V2LIDFDataset

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fixture_data(tmp):
    import OpenEXR
    import Imath
    cfg=json.loads((ROOT/'configs/stage2_train.json').read_text())
    source=tmp/'source';source.mkdir()
    producer=tmp/'producer';producer.mkdir()
    contract={'assets_sha256':'a'*64,'sources_sha256':'b'*64,'ground_truth_inputs':False}
    write_json(producer/'contract.json',contract)
    records=[];roster=[];labels=[]
    for number, roles in enumerate((['train'],['val','evaluation'])):
        sample='sample-%d'%number
        rgb=np.full((8,12,3),20+number,dtype=np.uint8)
        Image.fromarray(rgb).save(source/(sample+'.png'))
        Image.fromarray(np.pad(np.full((4,4),255,dtype=np.uint8),((2,2),(4,4)))).save(source/(sample+'-mask.png'))
        depth=np.linspace(.2,.8,96,dtype=np.float32).reshape(8,12)
        native=producer/(sample+'.npy');np.save(native,depth)
        header=OpenEXR.Header(12,8);header['channels']={'R':Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        f=OpenEXR.OutputFile(str(source/(sample+'.exr')),header);f.writePixels({'R':depth.tobytes()});f.close()
        rgb_sha=file_sha(source/(sample+'.png'))
        metadata={'ground_truth_access':False,'assets_sha256':contract['assets_sha256'],
                  'source_manifest_sha256':contract['sources_sha256'],
                  'weight_sha256':cfg['stage1']['expected_asset_sha256']}
        records.append({'sample_id':sample,'native_hw':[8,12],'rgb_sha256':rgb_sha,'status':'complete',
            'ground_truth_access':False,'contract_sha256':fingerprint(contract),'metadata':metadata,
            'prediction':{'path':native.name,'sha256':file_sha(native),'bytes':native.stat().st_size,
                'dtype':'float32','unit':'m','coordinate':'optical_z'}})
        roster.append({'sample_id':sample,'rgb_path':sample+'.png','intrinsics':[[20,0,5.5],[0,20,3.5],[0,0,1]],
            'pixel_convention':'integer_pixel_centres','roles':roles,'domain':'real','official_split':'real-val' if number else 'real-train'})
        labels.append({'sample_id':sample,'target_depth_path':sample+'.exr','transparent_mask_path':sample+'-mask.png',
            'depth_semantics':'already_rectified_optical_z_m','normal_camera_valid':True})
    prediction=producer/'predictions.jsonl';prediction.write_text(''.join(canonical(r)+'\n' for r in records))
    write_json(producer/'complete.json',{'schema':'portable_stage1_predictions_seal_v1','status':'complete',
        'ground_truth_access':False,'all_arrays_sha256_verified':True,'manifest_sha256':file_sha(prediction),
        'contract_sha256':fingerprint(contract),'images':2})
    inputs=source/'inputs.jsonl';inputs.write_text(''.join(canonical(r)+'\n' for r in roster))
    target=source/'labels.jsonl';target.write_text(''.join(canonical(r)+'\n' for r in labels))
    return cfg,inputs,target,prediction


def test_complete_fresh_data_boundary_and_authentication(tmp_path):
    cfg,inputs,labels,pred=fixture_data(tmp_path)
    out=tmp_path/'prepared'
    summary=prepare(inputs,labels,pred,out)
    assert summary['roles']=={'train':1,'val':1,'evaluation':1}
    im=out/'manifests/input_manifest.jsonl';lm=out/'manifests/label_manifest.jsonl'
    assert 'target_depth_path' not in im.read_text() and 'transparent_mask_path' not in im.read_text()
    info=validate_training_caches(cfg,im,lm,out/'stage1',out/'labels')
    ds=AuthenticatedTrainingDataset(im,lm,out/'stage1',out/'labels',curriculum=True,
        label_sha256=info['label_file_sha256'])
    sample=ds[0]
    assert sample['rgb'].shape==(3,240,320) and sample['transparent_mask'].dtype==torch.bool
    val=V2LIDFDataset(im,None,out/'stage1',None,role='val',include_target=False)[0]
    assert 'depth_gt' not in val and 'transparent_mask' not in val
    assert len(list((out/'stage1/native').glob('*.npy')))==1
    # The same inputs/producer can be resumed without recomputing or changing seals.
    prior=file_sha(out/'stage1/complete.json')
    assert prepare(inputs,labels,pred,out)==summary
    assert file_sha(out/'stage1/complete.json')==prior
    # Swapping an unrecognized P producer fails even with otherwise valid arrays.
    cfg['stage1']['adapter_checkpoint_sha256']='0'*64
    with pytest.raises(ValueError,match='producer/weights'):
        validate_training_caches(cfg,im,lm,out/'stage1',out/'labels')
    cfg['stage1']['adapter_checkpoint_sha256']=cfg['stage1']['expected_asset_sha256']['laplacian_e08']
    # Modifying a label cache with the same RGB/item metadata cannot resume/train.
    key=ds.rows[0]['cache_id'];path=out/'labels'/(key+'.npz')
    raw=path.read_bytes();path.write_bytes(raw+b'tamper')
    with pytest.raises(ValueError,match='Supervision cache changed'):
        validate_training_caches(cfg,im,lm,out/'stage1',out/'labels')
    with pytest.raises(ValueError,match='Supervision changed'):
        ds[0]


def test_native_validation_labels_are_authenticated_after_prediction(tmp_path):
    cfg,inputs,labels,pred=fixture_data(tmp_path);out=tmp_path/'prepared'
    prepare(inputs,labels,pred,out)
    row=next(r for r in read_jsonl(out/'manifests/label_manifest.jsonl') if r['role']=='val')
    assert load_authenticated_native_labels(row)['target'].shape==(8,12)
    Path(row['target_depth_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Native validation supervision changed'):
        load_authenticated_native_labels(row)


def test_csv_and_label_field_rejection(tmp_path):
    import csv
    cfg,inputs,labels,pred=fixture_data(tmp_path)
    rows=read_jsonl(inputs)
    path=inputs.with_suffix('.csv')
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    prepare(path,labels,pred,tmp_path/'csv_prepared')
    rows[0]['target_depth_path']='secret.exr'
    inputs.write_text(''.join(canonical(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='unapproved fields'):
        prepare(inputs,labels,pred,tmp_path/'bad')


def test_late_rank_does_not_treat_fresh_contract_as_resume(tmp_path,monkeypatch):
    message=[]
    def broadcast(payload,src):
        if payload[0] is not None:message[:]=[payload[0]]
        else:payload[0]=message[0]
    monkeypatch.setattr(train_stage2.dist,'broadcast_object_list',broadcast)
    contract={'fake_test_contract':True}
    establish_contract(tmp_path,contract,False,rank=0,world=2)
    # Simulates a rank that reaches initialization after rank0 has already written.
    establish_contract(tmp_path,contract,False,rank=1,world=2)
    with pytest.raises(RuntimeError,match='requires --resume'):
        establish_contract(tmp_path,contract,False,rank=0,world=2)


def test_progress_failure_is_observational(monkeypatch):
    observed=[]
    def fail(*args,**kwargs):raise OSError(107,'Transport endpoint is not connected')
    monkeypatch.setattr(training_utils,'atomic_json',fail)
    monkeypatch.setattr(training_utils,'_telemetry_failure',lambda error,op,payload:observed.append(op))
    training_utils.emit_progress({'updates':3},'unused.json')
    assert observed==['progress_json']


def test_eight_gpu_profile_full_pass_math():
    import math
    cfg=json.loads((ROOT/'configs/stage2_train_8gpu.json').read_text())
    s=validate_training_config(cfg,8)
    per_rank=math.ceil(45453/8)
    assert per_rank*8==45456
    assert math.ceil(math.ceil(per_rank/2)/2)==s.steps_per_epoch==1421


def test_exact_resume_tree_comparison():
    import copy
    state={'tensor':torch.tensor([1.,2.]),'array':np.array([3,4],dtype=np.uint32),
           'groups':[{'lr':.001}],'rng':(1,None)}
    training_utils.assert_state_tree_equal(state,copy.deepcopy(state))
    bad=copy.deepcopy(state);bad['tensor'][0]+=1
    with pytest.raises(RuntimeError,match='Resume tensor differs'):
        training_utils.assert_state_tree_equal(state,bad)
    bad=copy.deepcopy(state);bad['array'][0]+=1
    with pytest.raises(RuntimeError,match='Resume NumPy state differs'):
        training_utils.assert_state_tree_equal(state,bad)


def test_diagnostic_smoke_schedule_and_wait_gate(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('smoke_runner',ROOT/'scripts/run_stage2_smoke.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    base=json.loads((ROOT/'configs/stage2_train.json').read_text())
    rows=[{'role':'train'}]*10+[{'role':'val'}]*2
    cfg=mod.smoke_config(base,rows,1)
    schedule=validate_training_config(cfg,1)
    assert [schedule.alpha(i) for i in range(9)]==[1,1,1,1,.5,0,0,0,0]
    assert cfg['gradient_accumulation_steps']==2
    assert cfg['sampler_protocol']['presentations_per_epoch']==10
    cfg['diagnostic_smoke']=False
    with pytest.raises(ValueError,match='diagnostic_smoke'):
        validate_training_config(cfg,1)
    with pytest.raises(TimeoutError,match='no training was started'):
        mod.wait_for_predictions(tmp_path/'missing.jsonl',.01,.005)
    cfg2=mod.smoke_config(base,rows,2)
    assert validate_training_config(cfg2,2).total_steps==9
    assert cfg2['sampler_protocol']['presentations_per_epoch']==10


def test_smoke_requires_authenticated_policy_and_fresh_batch(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('smoke_policy_runner',ROOT/'scripts/run_stage2_smoke.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    predictions=tmp_path/'predictions.jsonl';predictions.write_text('test fixture\n')
    write_json(tmp_path/'complete.json',{'fixture':True})
    comparisons=[]
    for index in range(4):
        path=tmp_path/('comparison%d.json'%index)
        write_json(path,{'status':'passed_user_policy','numeric_tolerance_user_authorized':True,
            'numeric_tolerance_automatically_relaxed':False,'fields':{'depth':{'accepted':True}}})
        comparisons.append(mod.file_ref(path))
    marker=tmp_path/'policy_complete.json'
    record={'schema':'portable_stage1_validation_complete_v2','status':'passed_user_policy',
        'numeric_tolerance_user_authorized':True,'numeric_tolerance_automatically_relaxed':False,
        'fresh_predictions':mod.file_ref(predictions),'fresh_predictions_seal':mod.file_ref(tmp_path/'complete.json'),
        'bindings':{'source_inventory':mod.file_ref(ROOT/'provenance/stage1_runtime_sources.json'),
                    'asset_manifest':mod.file_ref(ROOT/'provenance/stage1_assets.json')},
        'policy_comparisons':comparisons}
    write_json(marker,record)
    assert mod.authenticate_stage1_acceptance(marker,predictions)==mod.file_ref(marker)
    record['numeric_tolerance_automatically_relaxed']=True;write_json(marker,record)
    with pytest.raises(ValueError,match='not complete'):
        mod.authenticate_stage1_acceptance(marker,predictions)
    record['numeric_tolerance_automatically_relaxed']=False;write_json(marker,record)
    predictions.write_text('changed producer output\n')
    with pytest.raises(ValueError,match='binding changed'):
        mod.authenticate_stage1_acceptance(marker,predictions)
