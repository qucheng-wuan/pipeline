"""CPU fixtures test contracts only; these are never GPU acceptance evidence."""
import copy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from transparent_depth.prepare_cleargrasp256 import (synthetic_calibration, synthetic_records,
    check_record, duplicates, roster, historical_comparison)
from transparent_depth.stage2 import file_sha, export_stage2_bundle, validate_bundle
from transparent_depth.train_stage2 import resolve_manifest_counts, validate_training_config
from transparent_depth.training_utils import (atomic_save,
    repair_epoch_archive, repair_epoch_history, repair_selected_checkpoint)
from transparent_depth.prepare_stage2_data import prepare, canonical
from transparent_depth.training_cache import validate_training_caches
from v2_lidf.data import V2LIDFDataset
from v2_lidf.data_protocol import PROTOCOL, CLEARGRASP_PROTOCOL, validate_partition

ROOT = Path(__file__).resolve().parents[1]


def config(smoke=False):
    return json.loads((ROOT/('configs/cleargrasp256_stage2_%s.json' % ('smoke' if smoke else '60'))).read_text())


def exr(path, depth):
    import OpenEXR
    import Imath
    h,w = depth.shape
    header = OpenEXR.Header(w,h)
    header['channels'] = {'R':Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    output = OpenEXR.OutputFile(str(path),header)
    output.writePixels({'R':np.asarray(depth,dtype=np.float32).tobytes()});output.close()


def synthetic_fixture(root, scene='scene-train', frame='000000003', hw=(24,40)):
    paths = {}
    for kind,folder,suffix in [('rgb','rgb-imgs','-rgb.jpg'),('gt','depth-imgs-rectified','-depth-rectified.exr'),
        ('mask','segmentation-masks','-segmentation-mask.png'),('calibration','json-files','-masks.json')]:
        path=root/scene/folder/(frame+suffix);path.parent.mkdir(parents=True,exist_ok=True);paths[kind]=path
    h,w=hw
    Image.fromarray(np.full((h,w,3),90,np.uint8)).save(paths['rgb'])
    depth=np.ones(hw,np.float32)*.8;depth[:2,:2]=0
    exr(paths['gt'],depth)
    mask=np.zeros(hw,np.uint8);mask[4:10,5:20]=255;Image.fromarray(mask).save(paths['mask'])
    paths['calibration'].write_text(json.dumps({'image':{'width_px':w,'height_px':h},
        'camera':{'field_of_view':{'x_axis_rads':1.2,'y_axis_rads':.75}}}))
    return paths


@pytest.mark.parametrize('hw',[(1080,1920),(576,1024)])
def test_per_frame_fov_half_pixel_centres(tmp_path,hw):
    h,w=hw;p=tmp_path/'frame.json'
    p.write_text(json.dumps({'image':{'width_px':w,'height_px':h},
        'camera':{'field_of_view':{'x_axis_rads':1.2,'y_axis_rads':.75}}}))
    size,k=synthetic_calibration(p)
    assert size==list(hw) and k[0][2]==(w-1)/2 and k[1][2]==(h-1)/2
    assert k[0][0]==pytest.approx(w/(2*math.tan(.6)))
    from v2_lidf.data import resize_intrinsics
    low=resize_intrinsics(k,hw,[256,256])
    assert low[0,2]==low[1,2]==127.5
    assert low[0,0]==pytest.approx(256/(2*math.tan(.6)))
    p.write_text('{"image":{"width_px":1920,"height_px":1080},"camera":{}}')
    with pytest.raises(KeyError):synthetic_calibration(p)  # no guessed K


def test_strict_directory_frame_pairing_no_silent_drop(tmp_path):
    first=synthetic_fixture(tmp_path,'first-train')
    synthetic_fixture(tmp_path,'second-train')
    rows,errors,counts=synthetic_records(tmp_path,'train')
    assert len(rows)==2 and not errors and counts==dict(rgb=2,gt=2,mask=2,calibration=2)
    assert len({r['sample_id'] for r in rows})==2
    first['calibration'].unlink()
    rows,errors,counts=synthetic_records(tmp_path,'train')
    assert len(rows)==1 and errors[0]['missing']==['calibration']
    assert counts['rgb']==2 and counts['calibration']==1


def test_exr_units_mask_dimensions_corruption_and_full_gt(tmp_path):
    paths=synthetic_fixture(tmp_path)
    row=synthetic_records(tmp_path,'train')[0][0]
    result=check_record((row,CLEARGRASP_PROTOCOL))
    assert result['status']=='ok' and result['gt_pixel_decode']
    assert result['gt_native_positive_range_m']==pytest.approx([.8,.8])
    assert result['zero_coverage256']>0 and result['mask_encoding']==[0,255]
    structural=check_record((row,CLEARGRASP_PROTOCOL,False))
    assert structural['status']=='ok' and not structural['gt_pixel_decode']
    assert 'valid256' not in structural
    Image.fromarray(np.ones((24,40),np.uint8)).save(paths['mask'])
    assert 'encoding' in check_record((row,CLEARGRASP_PROTOCOL))['error']
    Image.fromarray(np.zeros((12,20),np.uint8)).save(paths['mask'])
    assert 'dimensions' in check_record((row,CLEARGRASP_PROTOCOL))['error']
    paths['rgb'].write_bytes(b'corrupt')
    assert check_record((row,CLEARGRASP_PROTOCOL))['status']=='error'


def test_missing_calibration_still_audits_rgb_and_gt_without_inventing_k(tmp_path):
    paths=synthetic_fixture(tmp_path)
    paths['calibration'].write_text('{"camera":{"world_pose":{}}}')
    row=synthetic_records(tmp_path,'train')[0][0]
    result=check_record((row,CLEARGRASP_PROTOCOL))
    assert result['status']=='error' and 'calibration:' in result['error']
    assert result['native_hw']==[24,40] and result['gt_pixel_decode']
    assert 'intrinsics_native' not in result and 'intrinsics256' not in result
    assert result['source_sha256']['rgb']==file_sha(paths['rgb'])
    assert result['gt_native_positive_range_m']==pytest.approx([.8,.8])


def test_full_roster_refuses_missing_calibration_audit(tmp_path):
    from transparent_depth import prepare_cleargrasp256 as module
    frames=tmp_path/'frames.jsonl';frames.write_text('')
    (tmp_path/'audit.json').write_text(json.dumps({'preprocessing':CLEARGRASP_PROTOCOL,
        'source_sha256':file_sha(module.__file__),'frames_sha256':file_sha(frames),
        'status':'blocked'}))
    with pytest.raises(ValueError,match='no automatic sample exclusion'):
        roster({},tmp_path,tmp_path/'out',CLEARGRASP_PROTOCOL)


def test_encoded_and_reencoded_rgb_duplicates_detect_cross_split():
    a=dict(status='ok',sample_id='a',role='train',source_sha256={'rgb':'a'*64},decoded_rgb_sha256='x'*64)
    b=dict(a,sample_id='b',role='evaluation',source_sha256={'rgb':'b'*64})
    assert not duplicates([a,b],'encoded')
    assert duplicates([a,b],'decoded')[0]['cross_split']
    assert set(duplicates([a,b],'decoded')[0]['roles'])=={'train','evaluation'}


def test_explicit_test_selection_policy_without_synthetic_val():
    base={'preprocessing':CLEARGRASP_PROTOCOL,'dataset':'cleargrasp','role':'train','domain':'synthetic','official_split':'train'}
    assert validate_partition(base,'cleargrasp256')=='cleargrasp256'
    assert validate_partition(dict(base,role='evaluation',domain='real',official_split='real-val'))=='cleargrasp256'
    for bad in [dict(base,role='val',official_split='synthetic-val'), dict(base,role='val',domain='real',official_split='real-val'),
                dict(base,domain='real'),dict(base,dataset='transcg')]:
        with pytest.raises(ValueError):validate_partition(bad,'cleargrasp256')
    with pytest.raises(ValueError):validate_partition(base,'transcg256')
    cfg=config();cfg['curriculum']['steps_per_epoch']=2
    assert validate_training_config(cfg,1).total_steps==120
    for key,value in [('validation_mode','enabled'),('selection_policy','best'),('selection_metric','abs_rel'),('test_set_selection',False)]:
        changed=copy.deepcopy(cfg);changed[key]=value
        with pytest.raises(ValueError):validate_training_config(changed,1)
    changed=copy.deepcopy(cfg);changed['preprocessing']=PROTOCOL
    with pytest.raises(ValueError):validate_training_config(changed,1)


def cleargrasp_cache_fixture(tmp_path):
    # Reuse authentication unit fixtures, never create a production seal.
    from test_transfer256 import fixture256
    _,inputs,labels,predictions,old=fixture256(tmp_path)
    rows=[json.loads(line) for line in inputs.read_text().splitlines()]
    labels_rows=[json.loads(line) for line in labels.read_text().splitlines()]
    for index,(row,label) in enumerate(zip(rows,labels_rows)):
        row.update(dataset='cleargrasp',domain='synthetic' if index==0 else 'real',
            roles=['train'] if index==0 else ['evaluation'],official_split='train' if index==0 else 'real-val')
        label.pop('depth_divisor');label.update(depth_semantics='already_rectified_optical_z_m',target_depth_path='gt%d.exr'%index)
        exr(inputs.parent/label['target_depth_path'],np.full((8,12),.8,np.float32))
        Image.fromarray(np.pad(np.ones((4,4),np.uint8)*255,((2,2),(4,4)))).save(inputs.parent/label['transparent_mask_path'])
    inputs.write_text(''.join(canonical(r)+'\n' for r in rows))
    labels.write_text(''.join(canonical(r)+'\n' for r in labels_rows))
    output=tmp_path/'cleargrasp-prepared'
    prepare(inputs,labels,predictions,output,CLEARGRASP_PROTOCOL)
    return output,inputs,labels,predictions


def test_train_only_cache_counts_coverage_and_experiment_isolation(tmp_path):
    output,inputs,labels,pred=cleargrasp_cache_fixture(tmp_path)
    im=output/'manifests/input_manifest.jsonl';lm=output/'manifests/label_manifest.jsonl'
    cfg=resolve_manifest_counts(config(),im,1)
    assert cfg['counts']=={'train':1,'val':0,'evaluation':1}
    validate_training_config(cfg,1)
    validate_training_caches(cfg,im,lm,output/'stage1',output/'labels')
    train=V2LIDFDataset(im,lm,output/'stage1',output/'labels',curriculum=True)
    assert train[0]['coverage'].shape==(1,256,256)
    test=V2LIDFDataset(im,None,output/'stage1',None,role='evaluation',include_target=False)
    assert not {'depth_gt','gt_valid','transparent_mask','coverage'} & set(test[0])
    with pytest.raises(ValueError):prepare(inputs,labels,pred,output,PROTOCOL)
    wrong=copy.deepcopy(cfg);wrong['data_protocol']='transcg256';wrong['preprocessing']=PROTOCOL
    with pytest.raises(ValueError):validate_training_caches(wrong,im,lm,output/'stage1',output/'labels')
    rows=[json.loads(l) for l in im.read_text().splitlines()];rows[1]['rgb_sha256']=rows[0]['rgb_sha256']
    im.write_text(''.join(canonical(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Duplicate RGB'):resolve_manifest_counts(config(),im,1)


def test_metric_best_retains_earlier_epoch_and_repairs_resume(tmp_path):
    cfg=config(True)
    def checkpoint(epoch,score,best,selected):
        return {'model':{'p':torch.tensor(float(epoch))},'contract':{'config':cfg},
            'epoch':epoch-1,'updates':2*epoch,'best_score':best,
            'validation':{'transparent_mae_m':score,'abs_rel':score},
            'validation_curriculum':{'abs_rel':score}, 'selection_metric':'transparent_mae_m',
            'epoch_record':{'epoch':epoch,'selected':selected,'selection_eligible':True,'input_alpha_last':0}}
    early=checkpoint(1,.08,.08,True)
    atomic_save(tmp_path/'last.pt',early)
    repair_selected_checkpoint(tmp_path,early)
    assert torch.load(tmp_path/'best.pt',weights_only=False)['epoch']==0
    later=checkpoint(4,.10,.08,False)
    atomic_save(tmp_path/'last.pt',later)
    repair_selected_checkpoint(tmp_path,later)
    assert torch.load(tmp_path/'best.pt',weights_only=False)['epoch']==0
    assert not (tmp_path/'final.pt').exists()
    better=checkpoint(4,.05,.05,True)
    atomic_save(tmp_path/'last.pt',better)
    repair_selected_checkpoint(tmp_path,better)
    assert torch.load(tmp_path/'best.pt',weights_only=False)['best_score']==.05
    receipt=repair_epoch_archive(tmp_path,better)
    assert receipt['validation_abs_rel']==.05 and receipt['selection_eligible']


def test_real_val_selection_is_pure_p_explicit_and_labels_after_prediction(tmp_path,monkeypatch):
    from transparent_depth import training_utils, evaluate256
    output,*_=cleargrasp_cache_fixture(tmp_path)
    im=output/'manifests/input_manifest.jsonl';lm=output/'manifests/label_manifest.jsonl'
    ds=V2LIDFDataset(im,None,output/'stage1',None,role='evaluation',include_target=False)
    rows={r['cache_id']:r for r in map(json.loads,lm.read_text().splitlines()) if r['role']=='evaluation'}
    class Predictor(torch.nn.Module):
        called=False
        def forward(self,rgb,p,k,compute_loss):
            assert compute_loss is False
            self.called=True
            return {'depth':p,'query_mask':torch.ones_like(p,dtype=torch.bool)}
    model=Predictor();original=evaluate256.load_authenticated_native_labels
    def after(row):
        assert model.called
        return original(row)
    monkeypatch.setattr(evaluate256,'load_authenticated_native_labels',after)
    with pytest.raises(ValueError,match='Selection role'):
        training_utils.validate(model,ds,rows,torch.device('cpu'),0,1)
    with pytest.raises(ValueError,match='pure P'):
        training_utils.validate(model,ds,rows,torch.device('cpu'),0,1,alpha=.5,selection_policy='real_val_best')
    result=training_utils.validate(model,ds,rows,torch.device('cpu'),0,1,selection_policy='real_val_best')
    assert result['test_set_selection'] and result['selection_policy']=='real_val_best'
    assert result['input_alpha']==0 and not result['gt_or_mask_used_as_input']
    assert result['transparent_mae_m']>0


def test_cleargrasp_bundle_preserves_protocol_and_test_selection_policy(tmp_path):
    cfg=config();path=tmp_path/'checkpoint.pt'
    torch.save({'model':{'spvnas_branch.fixture':torch.ones(1)},'model_config':cfg['model'],
                'contract':{'config':cfg},'epoch':59,'updates':60},path)
    output=tmp_path/'bundle'
    meta=export_stage2_bundle(path,output,{'selection_rule':'explicit real-val best; CPU serialization unit fixture'})
    assert meta['input']['model_hw']==[256,256] and meta['input']['preprocessing']==CLEARGRASP_PROTOCOL
    validate_bundle(output)
