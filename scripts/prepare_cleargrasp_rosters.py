#!/usr/bin/env python3
"""Materialize the audited ClearGrasp split using metadata and a new data root.

No torch, image decoder, sensor depth, or GT array is loaded. The optional
maintainer export subcommand creates the compressed references once; normal
materialization has no dependency on a historical project or its manifests.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = PROJECT/'data/cleargrasp_reference'
ROOT_NAMES = ('cleargrasp-dataset-train', 'cleargrasp-dataset-test-val')
INPUT_FIELDS = {'sample_id','rgb_path','rgb_sha256','intrinsics','intrinsics_json',
    'pixel_convention','role','roles','domain','official_split','benchmark_group'}
LABEL_FIELDS = {'sample_id','target_depth_path','transparent_mask_path','normal_path',
    'depth_semantics','normal_camera_valid'}
FILES = ('rgb_k.jsonl.gz','labels.jsonl.gz','calibration_provenance.jsonl.gz')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, values):
    path=Path(path)
    if path.suffix=='.gz':
        with path.open('wb') as stream, gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
            for value in values:compressed.write((canonical(value)+'\n').encode())
    else:
        with path.open('w',encoding='utf8') as stream:
            for value in values:stream.write(canonical(value)+'\n')


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def relative_data_path(value):
    parts=Path(value).parts
    matches=[i for i,p in enumerate(parts) if p in ROOT_NAMES]
    if len(matches)!=1:raise ValueError('Path must identify one declared ClearGrasp dataset root: '+str(value))
    path=Path(*parts[matches[0]:])
    if '..' in path.parts:raise ValueError('Dataset path traversal is forbidden')
    return path.as_posix()


def native_k(row):
    k=row['intrinsics_native']
    if (len(k)!=3 or any(len(r)!=3 for r in k)
        or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for r in k for v in r)
        or k[0][0]<=0 or k[1][1]<=0 or k[2]!=[0.,0.,1.]
        or row['intrinsics_pixel_convention']!='integer_pixel_centres'):
        raise ValueError('Invalid native integer-centre camera calibration')
    return k


def normal_flag(row):
    direct=row.get('normal_camera_valid')
    camera=row.get('camera',{}).get('normal_camera_valid')
    if direct is not None and camera is not None and direct!=camera:
        raise ValueError('Conflicting normal-camera validity flags')
    value=direct if direct is not None else camera
    if type(value) is not bool:raise ValueError('Missing explicit normal-camera validity flag')
    return value


def export_reference(input_manifest,label_manifest,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/p).exists() for p in (*FILES,'manifest.json')):
        raise FileExistsError('Reference output already exists; preserve the audited version')
    inputs={};calibration={};original=rows(input_manifest)
    for row in original:
        sid=row['sample_id'];role=row['role']
        if role not in ('train','val','evaluation'):raise ValueError('Invalid split role')
        current={'sample_id':sid,'rgb_path':relative_data_path(row['rgb_path']),
            'rgb_sha256':row['rgb_sha256'],'intrinsics':native_k(row),
            'pixel_convention':'integer_pixel_centres','native_hw':row['native_hw'],
            **{k:row[k] for k in ('domain','official_split','benchmark_group') if k in row}}
        source={'sample_id':sid,'source_kind':row['intrinsics_source_kind'],
            'source_sha256':row.get('intrinsics_source_sha256')}
        try:source['source_path']=relative_data_path(row['intrinsics_source'])
        except ValueError:
            source['non_dataset_reference_name']=Path(row['intrinsics_source']).name
            source['reference_required_at_runtime']=False
        if sid in inputs:
            prior={k:v for k,v in inputs[sid].items() if k!='roles'}
            if prior!=current or calibration[sid]!=source:raise ValueError('One sample has conflicting identity/calibration: '+sid)
            if role in inputs[sid]['roles']:raise ValueError('Duplicate role for sample: '+sid)
            inputs[sid]['roles'].append(role)
        else:inputs[sid]={**current,'roles':[role]};calibration[sid]=source
        if 'train' in inputs[sid]['roles'] and len(inputs[sid]['roles'])!=1:
            raise ValueError('Training sample may not cross into validation/evaluation')
    labels={}
    for row in rows(label_manifest):
        sid=row['sample_id']
        if sid not in inputs:raise ValueError('Label has no RGB identity')
        if row.get('depth_semantics')!='already_rectified_optical_z_m' or row.get('depth_scale',1.)!=1.:
            raise ValueError('Only existing rectified optical-Z metres are supported; no implicit depth conversion')
        current={'sample_id':sid,'depth_semantics':'already_rectified_optical_z_m',
            'normal_camera_valid':normal_flag(row),
            **{k:relative_data_path(row[k]) for k in ('target_depth_path','transparent_mask_path','normal_path') if row.get(k)}}
        if sid in labels and labels[sid]!=current:raise ValueError('Conflicting supervision identity: '+sid)
        labels[sid]=current
    if set(labels)!=set(inputs):raise ValueError('Input/label sample coverage differs')
    ordered=list(inputs)
    write_rows(output/FILES[0],inputs.values())
    write_rows(output/FILES[1],(labels[sid] for sid in ordered))
    write_rows(output/FILES[2],(calibration[sid] for sid in ordered))
    role_counts=dict(Counter(role for row in inputs.values() for role in row['roles']))
    calibration_counts=dict(Counter(row['source_kind'] for row in calibration.values()))
    train_rgb={row['rgb_sha256'] for row in inputs.values() if 'train' in row['roles']}
    eval_rgb={row['rgb_sha256'] for row in inputs.values() if 'evaluation' in row['roles']}
    manifest={'schema':'cleargrasp_audited_metadata_reference_v1','unique_samples':len(inputs),
        'original_input_rows':len(original),'roles':role_counts,'calibration_source_counts_unique':calibration_counts,
        'dataset_directory_names':list(ROOT_NAMES),'native_intrinsics_pixel_convention':'integer_pixel_centres',
        'sample_order':'first occurrence in authenticated original input manifest',
        'sample_order_sha256':hashlib.sha256(('\n'.join(ordered)+'\n').encode()).hexdigest(),
        'source_manifest_sha256':{'input':sha(input_manifest),'labels':sha(label_manifest)},
        'arrays_included':False,'rgb_sha_fields_recomputed_during_metadata_export':False,
        'gt_arrays_opened':False,'gt_depth_semantics':'already_rectified_optical_z_m',
        'training_sample_ids_overlap_other_roles':False,
        'distinct_rgb_sha_intersection_train_evaluation':len(train_rgb&eval_rgb),
        'files':{name:{'sha256':sha(output/name),'bytes':(output/name).stat().st_size} for name in FILES}}
    write_json(output/'manifest.json',manifest)
    return manifest


def resolve_inside(root,relative):
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] not in ROOT_NAMES:
        raise ValueError('Dataset paths must remain relative to the declared data root')
    result=(root/path).resolve()
    if not result.is_relative_to(root):raise ValueError('Dataset symlink escapes the declared data root')
    if not result.is_file():raise FileNotFoundError('Required existing ClearGrasp file is missing: '+str(result))
    return str(result)


def materialize(reference,data_root,output,verify_rgb_sha=False):
    reference,data_root,output=Path(reference),Path(data_root).resolve(),Path(output)
    manifest=json.loads((reference/'manifest.json').read_text())
    if manifest['schema']!='cleargrasp_audited_metadata_reference_v1' or manifest['arrays_included'] is not False:
        raise ValueError('Unrecognized reference metadata')
    for name in FILES:
        if manifest['files'][name]!={'sha256':sha(reference/name),'bytes':(reference/name).stat().st_size}:
            raise ValueError('Compressed reference identity changed: '+name)
    raw,labels,calibration=[rows(reference/name) for name in FILES]
    ids=[r['sample_id'] for r in raw]
    if (len(ids)!=len(set(ids)) or len(ids)!=manifest['unique_samples']
        or any([r['sample_id'] for r in group]!=ids for group in (labels,calibration))
        or hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()!=manifest['sample_order_sha256']):
        raise ValueError('Reference sample identity/order differs')
    output.mkdir(parents=True,exist_ok=True)
    names=('rgb_only.jsonl','input_roster.jsonl','label_roster.jsonl','calibration_provenance.jsonl','complete.json')
    if any((output/name).exists() for name in names):raise FileExistsError('Use an empty roster output directory')
    rgb_only=[];inputs=[];targets=[]
    for row in raw:
        if set(row)-INPUT_FIELDS-{'native_hw'}:raise ValueError('Unexpected fields in reference RGB/K record')
        roles=row['roles']
        if (not roles or len(roles)!=len(set(roles)) or set(roles)-{'train','val','evaluation'}
            or ('train' in roles and len(roles)>1)):
            raise ValueError('Training sample may not cross into validation/evaluation')
        if not re.fullmatch('[0-9a-f]{64}',row['rgb_sha256']):raise ValueError('Invalid RGB SHA256 field')
        value={k:v for k,v in row.items() if k!='native_hw'}
        value['rgb_path']=resolve_inside(data_root,row['rgb_path'])
        if verify_rgb_sha and sha(value['rgb_path'])!=row['rgb_sha256']:raise ValueError('RGB content changed: '+row['sample_id'])
        inputs.append(value)
        rgb_only.append({k:(value[k] if k!='native_hw' else row[k]) for k in ('sample_id','rgb_path','rgb_sha256','native_hw')})
    for row in labels:
        if set(row)-LABEL_FIELDS:raise ValueError('Unexpected fields in reference label record')
        if row['depth_semantics']!='already_rectified_optical_z_m' or type(row['normal_camera_valid']) is not bool:
            raise ValueError('Invalid supervision semantics')
        targets.append({k:resolve_inside(data_root,v) if k.endswith('_path') else v for k,v in row.items()})
    roles=dict(Counter(role for row in inputs for role in row['roles']))
    if roles!=manifest['roles']:raise ValueError('Role counts differ from audited metadata')
    for name,values in zip(names[:4],(rgb_only,inputs,targets,calibration)):write_rows(output/name,values)
    complete={'schema':'cleargrasp_materialized_rosters_v1','status':'complete','unique_samples':len(ids),'roles':roles,
        'reference_manifest_sha256':sha(reference/'manifest.json'),'sample_order_sha256':manifest['sample_order_sha256'],
        'data_root':str(data_root),'required_file_paths_checked':True,'rgb_bytes_sha256_verified':verify_rgb_sha,
        'gt_arrays_opened':False,'image_dimensions_decoded':False,
        'depth_semantics':'already_rectified_optical_z_m; supplied existing files, not converted by this command',
        'files':{name:{'sha256':sha(output/name),'bytes':(output/name).stat().st_size} for name in names[:4]}}
    write_json(output/'complete.json',complete)
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,help='Parent of cleargrasp-dataset-train and cleargrasp-dataset-test-val')
    parser.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-rgb-sha',action='store_true',help='Additionally read every RGB file and verify its recorded SHA256')
    parser.add_argument('--export-from-audited-manifests',nargs=2,type=Path,metavar=('INPUT','LABEL'),
                        help='Maintainer-only reference export; unnecessary for normal dataset setup')
    args=parser.parse_args()
    if args.export_from_audited_manifests:
        result=export_reference(*args.export_from_audited_manifests,args.output)
    else:
        if args.data_root is None:parser.error('--data-root is required for normal materialization')
        result=materialize(args.reference,args.data_root,args.output,args.verify_rgb_sha)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
