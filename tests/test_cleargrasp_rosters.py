import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('cleargrasp_rosters',ROOT/'scripts/prepare_cleargrasp_rosters.py')
rosters=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(rosters)


def fixture(tmp_path):
    data=tmp_path/'data';inputs=[];labels=[]
    native=[[1000.,0.,959.5],[0.,1000.,539.5],[0.,0.,1.]]
    for sid,role,directory in [('train-a','train','cleargrasp-dataset-train'),
                               ('test-b','val','cleargrasp-dataset-test-val'),
                               ('test-b','evaluation','cleargrasp-dataset-test-val')]:
        folder=data/directory/'object';folder.mkdir(parents=True,exist_ok=True)
        for name in ('rgb.jpg','depth-rectified.exr','mask.png','normal.exr'):
            (folder/name).write_bytes(b'metadata-only fixture, intentionally not an image')
        inputs.append({'sample_id':sid,'role':role,'rgb_path':str(folder/'rgb.jpg'),
            'rgb_sha256':rosters.sha(folder/'rgb.jpg'),'intrinsics_native':native,
            'intrinsics':[[100.,0.,159.5],[0.,100.,119.5],[0.,0.,1.]],
            'intrinsics_pixel_convention':'integer_pixel_centres','native_hw':[1080,1920],
            'intrinsics_source_kind':'official_lidf_default' if sid=='train-a' else 'released_real_yaml',
            'intrinsics_source':'/historical/LIDF/camera_defaults.py' if sid=='train-a' else str(folder/'camera.yaml'),
            'intrinsics_source_sha256':'a'*64,'domain':'synthetic' if sid=='train-a' else 'real'})
        labels.append({'sample_id':sid,'role':role,'target_depth_path':str(folder/'depth-rectified.exr'),
            'transparent_mask_path':str(folder/'mask.png'),'normal_path':str(folder/'normal.exr'),
            'depth_semantics':'already_rectified_optical_z_m','depth_scale':1.,
            'camera':{'normal_camera_valid':sid!='train-a'}})
    ip=tmp_path/'input.jsonl';lp=tmp_path/'labels.jsonl'
    rosters.write_rows(ip,inputs);rosters.write_rows(lp,labels)
    return data,ip,lp,native


def test_export_and_materialize_preserve_protocol_without_reading_arrays(tmp_path,monkeypatch):
    from transparent_depth.prepare_stage2_data import INPUT_FIELDS,LABEL_FIELDS
    assert rosters.INPUT_FIELDS==INPUT_FIELDS and rosters.LABEL_FIELDS==LABEL_FIELDS
    data,ip,lp,native=fixture(tmp_path);reference=tmp_path/'reference'
    summary=rosters.export_reference(ip,lp,reference)
    assert summary['roles']=={'train':1,'val':1,'evaluation':1} and summary['unique_samples']==2
    raw=rosters.rows(reference/'rgb_k.jsonl.gz')
    assert [r['sample_id'] for r in raw]==['train-a','test-b']
    assert raw[1]['roles']==['val','evaluation'] and raw[0]['intrinsics']==native
    provenance=rosters.rows(reference/'calibration_provenance.jsonl.gz')
    assert provenance[0]['source_kind']=='official_lidf_default'
    assert provenance[0]['source_sha256']=='a'*64 and not provenance[0]['reference_required_at_runtime']
    assert all('/historical/' not in json.dumps(r) for r in provenance)
    real_open=Path.open
    def no_array_read(path,*args,**kwargs):
        if path.suffix in ('.jpg','.png','.exr'):raise AssertionError('Arrays must not be opened')
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',no_array_read)
    result=rosters.materialize(reference,data,tmp_path/'out')
    assert not result['gt_arrays_opened'] and not result['rgb_bytes_sha256_verified']
    rgb=rosters.rows(tmp_path/'out/rgb_only.jsonl')
    assert all(set(r)=={'sample_id','rgb_path','rgb_sha256','native_hw'} for r in rgb)
    mapped=rosters.rows(tmp_path/'out/input_roster.jsonl')
    assert all(set(r)<=INPUT_FIELDS for r in mapped) and mapped[0]['intrinsics']==native
    labels=rosters.rows(tmp_path/'out/label_roster.jsonl')
    assert all(set(r)<=LABEL_FIELDS for r in labels)
    assert [r['normal_camera_valid'] for r in labels]==[False,True]


def test_export_rejects_cross_split_and_conflicting_camera_flag(tmp_path):
    data,ip,lp,native=fixture(tmp_path)
    rows=rosters.rows(ip);rows[-1]['sample_id']='train-a';rows[-1]={**rows[0],'role':'evaluation'}
    rosters.write_rows(ip,rows)
    with pytest.raises(ValueError,match='Training sample'):
        rosters.export_reference(ip,lp,tmp_path/'bad')
    with pytest.raises(ValueError,match='Conflicting'):
        rosters.normal_flag({'normal_camera_valid':True,'camera':{'normal_camera_valid':False}})


def test_data_root_rejects_parent_traversal_absolute_and_symlink_escape(tmp_path):
    data=tmp_path/'data';data.mkdir();outside=tmp_path/'outside';outside.write_bytes(b'outside')
    with pytest.raises(ValueError,match='relative'):
        rosters.resolve_inside(data,'cleargrasp-dataset-train/../outside')
    with pytest.raises(ValueError,match='relative'):
        rosters.resolve_inside(data,str(outside))
    inside=data/'cleargrasp-dataset-train';inside.mkdir();(inside/'link').symlink_to(outside)
    with pytest.raises(ValueError,match='escapes'):
        rosters.resolve_inside(data,'cleargrasp-dataset-train/link')


def test_optional_rgb_verification_and_compressed_tamper_detection(tmp_path):
    data,ip,lp,native=fixture(tmp_path);reference=tmp_path/'reference'
    rosters.export_reference(ip,lp,reference)
    report=rosters.materialize(reference,data,tmp_path/'verified',verify_rgb_sha=True)
    assert report['rgb_bytes_sha256_verified'] is True
    rgb=Path(rosters.rows(ip)[0]['rgb_path']);rgb.write_bytes(b'changed')
    with pytest.raises(ValueError,match='RGB content changed'):
        rosters.materialize(reference,data,tmp_path/'bad_rgb',verify_rgb_sha=True)
    (reference/'labels.jsonl.gz').write_bytes(b'changed compressed metadata')
    with pytest.raises(ValueError,match='reference identity changed'):
        rosters.materialize(reference,data,tmp_path/'bad_reference')
