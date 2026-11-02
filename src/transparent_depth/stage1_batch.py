"""Persistent-worker RGB-only Stage1 export and exact full-roster sealing."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from .stage1 import PROJECT, DEFAULT_MANIFEST, Stage1Assets, Stage1Predictor, sha256
from .input_guard import RGBInputGuard, assert_project_modules, configure_jit_cache


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_bytes(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        stream.write(value);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def atomic_json(path,value):
    atomic_bytes(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


@contextmanager
def file_lock(path,*,blocking=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def read_rgb_manifest(path,expected_count=None):
    path=Path(path).resolve()
    rows=[];seen=set();paths={}
    allowed={'sample_id','rgb_path','rgb_sha256','native_hw'}
    required={'sample_id','rgb_path','rgb_sha256'}
    for line in path.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if not required <= set(row) or set(row)-allowed:
            raise ValueError('Stage1 manifest accepts only sample_id/rgb_path/rgb_sha256/optional native_hw')
        if not isinstance(row['sample_id'],str) or not row['sample_id'] or row['sample_id'] in seen:
            raise ValueError('empty or duplicate sample ID')
        seen.add(row['sample_id'])
        digest=row['rgb_sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid RGB SHA256')
        value=os.path.expandvars(row['rgb_path'])
        if '${' in value:raise ValueError('Unbound RGB dataset root')
        rgb=Path(value).expanduser()
        if not rgb.is_absolute():rgb=path.parent/rgb
        if rgb.suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}:
            raise ValueError('RGB manifest must name an ordinary color image')
        row={**row,'rgb_path':str(rgb.resolve())}
        if 'native_hw' in row and (len(row['native_hw'])!=2 or any(type(n) is not int or n<1 for n in row['native_hw'])):
            raise ValueError('invalid optional native geometry')
        key=hashlib.sha256(row['sample_id'].encode()).hexdigest()[:24]
        if key in paths:raise ValueError('output identity hash collision')
        paths[key]=row['sample_id']
        row['output_id']=key
        rows.append(row)
    if not rows or (expected_count is not None and len(rows)!=expected_count):
        raise ValueError(f'expected {expected_count} images, found {len(rows)}')
    return rows


def make_contract(args,rows,assets):
    return {'schema':'portable_stage1_batch_contract_v1','images':len(rows),'world_size':args.world,
        'rgb_manifest_sha256':sha256(args.manifest),
        'roster_sha256':fingerprint([{k:r[k] for k in ('sample_id','rgb_sha256')} for r in rows]),
        'assets_sha256':assets.manifest_sha256,'sources_sha256':assets.verify_sources(),
        'prediction':'persistent live RGB experts -> native V2 tile384 halo16 -> learned E08; FP32 optical-Z metres',
        'ground_truth_inputs':False,'camera_intrinsics_input':False,'expert_cache_read':False}


def read_prediction(output,record):
    relative=Path(record['prediction']['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('prediction path must be output-relative')
    path=output/relative
    if path.stat().st_size!=record['prediction']['bytes'] or sha256(path)!=record['prediction']['sha256']:
        raise ValueError('prediction bytes changed')
    value=np.load(path,allow_pickle=False)
    if value.dtype!=np.float32 or list(value.shape)!=record['native_hw'] or not np.isfinite(value).all() or not (value>0).all():
        raise ValueError('prediction must be finite-positive native FP32 depth')
    return value


def verify_receipt(output,row,contract,record):
    if (record.get('schema')!='portable_stage1_rgb_receipt_v1' or record.get('status')!='complete'
        or record.get('sample_id')!=row['sample_id'] or record.get('rgb_sha256')!=row['rgb_sha256']
        or record.get('contract_sha256')!=fingerprint(contract)
        or record.get('ground_truth_access') is not False or record.get('legacy_cache_used') is not False
        or record.get('camera_intrinsics_input') is not False
        or record['prediction']['path']!=f"predictions/{row['output_id']}.npy"):
        raise ValueError('prediction receipt identity/producer mismatch')
    if row.get('native_hw',record['native_hw'])!=record['native_hw']:raise ValueError('native geometry differs')
    read_prediction(output,record)


def predict(args):
    if not 0<=args.rank<args.world:raise ValueError('invalid worker rank/world')
    rows=read_rgb_manifest(args.manifest,args.expected_count)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    assets=Stage1Assets(args.assets,args.asset_root)
    contract=make_contract(args,rows,assets)
    with file_lock(output/'locks/contract.lock',blocking=True):
        path=output/'contract.json'
        if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('existing output has a different contract')
        atomic_json(path,contract)
    with file_lock(output/f'locks/rank_{args.rank:03d}.lock'):
        selected=rows[args.rank::args.world]
        cache=configure_jit_cache(args.cache_root or output/'runtime_cache'/f'rank_{args.rank:03d}')
        started=time.time();predictor=None;new=0;reused=0
        audit={'schema':'portable_stage1_worker_v1','status':'running','rank':args.rank,'world_size':args.world,
            'pid':os.getpid(),'started_unix':started,'assigned_images':len(selected),'contract_sha256':fingerprint(contract),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':args.device}
        attempt=output/'workers'/f"rank_{args.rank:03d}_{os.getpid()}_{int(started)}.json"
        guard=RGBInputGuard(project=PROJECT,asset_root=assets.root,rgb_paths=[r['rgb_path'] for r in selected],
            output_root=output,cache_root=cache,metadata_paths=[assets.manifest],deny_prefixes=args.deny_prefix)
        try:
            with guard:
                for row in selected:
                    if sha256(row['rgb_path'])!=row['rgb_sha256']:raise ValueError('RGB source changed: '+row['sample_id'])
                    receipt_path=output/'receipts'/(row['output_id']+'.json')
                    if receipt_path.exists():
                        verify_receipt(output,row,contract,json.loads(receipt_path.read_text()));reused+=1;continue
                    if predictor is None:predictor=Stage1Predictor(args.assets,args.asset_root,device=args.device,runtime_cache=cache)
                    depth,metadata=predictor.predict(row['rgb_path'])
                    if metadata['rgb_sha256']!=row['rgb_sha256']:raise ValueError('RGB changed during prediction')
                    if row.get('native_hw',list(depth.shape))!=list(depth.shape):raise ValueError('native RGB shape changed')
                    relative=f"predictions/{row['output_id']}.npy"
                    stream=io.BytesIO();np.save(stream,np.ascontiguousarray(depth),allow_pickle=False)
                    raw=stream.getvalue();atomic_bytes(output/relative,raw)
                    record={'schema':'portable_stage1_rgb_receipt_v1','status':'complete',
                        'sample_id':row['sample_id'],'rgb_sha256':row['rgb_sha256'],'native_hw':list(depth.shape),
                        'contract_sha256':fingerprint(contract),'ground_truth_access':False,'legacy_cache_used':False,
                        'camera_intrinsics_input':False,'rank':args.rank,'completed_unix':time.time(),
                        'prediction':{'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                                      'dtype':'float32','unit':'m','coordinate':'optical_z'},'metadata':metadata}
                    atomic_json(receipt_path,record);new+=1
                    print(json.dumps({'phase':'stage1_predict','rank':args.rank,'done':new+reused,'total':len(selected),
                        'sample_id':row['sample_id'],'elapsed_seconds':time.time()-started}),flush=True)
            audit.update(status='complete',new_images=new,reused_images=reused,
                predictor_instances=0 if predictor is None else 1,modules=assert_project_modules(PROJECT))
            if predictor is not None and predictor.device.type=='cuda':
                import torch
                audit['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(predictor.device)
                audit['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved(predictor.device)
        except BaseException as error:
            audit.update(status='failed',error=repr(error),new_images=new,reused_images=reused)
            raise
        finally:
            audit.update(finished_unix=time.time(),seconds=time.time()-started,file_access_guard=guard.receipt())
            atomic_json(attempt,audit)
        return audit


def seal(args):
    rows=read_rgb_manifest(args.manifest,args.expected_count)
    output=args.output.resolve();assets=Stage1Assets(args.assets,args.asset_root)
    contract=json.loads((output/'contract.json').read_text())
    # The worker world size is a property of the actual run, not this CPU seal process.
    args.world=contract['world_size']
    if contract!=make_contract(args,rows,assets):raise ValueError('producer changed before sealing')
    records=[]
    for row in rows:
        record=json.loads((output/'receipts'/(row['output_id']+'.json')).read_text())
        verify_receipt(output,row,contract,record);records.append(record)
    expected={r['output_id']+'.json' for r in rows}
    if {p.name for p in (output/'receipts').glob('*.json')}!=expected:raise ValueError('foreign or missing receipt in output')
    workers={}
    for path in sorted((output/'workers').glob('*.json')):
        worker=json.loads(path.read_text())
        if worker.get('status')!='complete' or worker.get('contract_sha256')!=fingerprint(contract):continue
        rank=worker['rank'];guard=worker['file_access_guard']
        if (worker['world_size']!=contract['world_size'] or not 0<=rank<contract['world_size']
            or worker['assigned_images']!=len(rows[rank::contract['world_size']])
            or worker['new_images']+worker['reused_images']!=worker['assigned_images']
            or guard['denied_paths'] or guard['ground_truth_read'] is not False
            or guard['legacy_cache_read'] is not False or guard['strict_read_allowlist'] is not True):
            raise ValueError('worker isolation/completion receipt is inconsistent')
        workers[rank]={'path':str(path.relative_to(output)),'sha256':sha256(path)}
    if set(workers)!=set(range(contract['world_size'])):raise ValueError('each rank must have a successful worker receipt')
    raw=''.join(canonical(r)+'\n' for r in records).encode()
    marker={'schema':'portable_stage1_predictions_seal_v1','status':'complete','images':len(records),
        'unique_sample_ids':len({r['sample_id'] for r in records}),
        'contract_sha256':fingerprint(contract),'manifest_sha256':hashlib.sha256(raw).hexdigest(),
        'rgb_manifest_sha256':sha256(args.manifest),'all_arrays_sha256_verified':True,
        'worker_receipts':{str(k):v for k,v in sorted(workers.items())},
        'ground_truth_access':False,'expert_cache_read':False,'camera_intrinsics_input':False}
    with file_lock(output/'locks/seal.lock'):
        path=output/'complete.json'
        if path.exists() and json.loads(path.read_text())!=marker:raise ValueError('refusing to overwrite different sealed export')
        atomic_bytes(output/'predictions.jsonl',raw);atomic_json(path,marker)
    return marker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('predict','seal'))
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--asset-root',type=Path)
    parser.add_argument('--expected-count',type=int,default=None,help='Optional extra assertion; otherwise count comes from manifest')
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--world',type=int,default=1)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--cache-root',type=Path)
    parser.add_argument('--deny-prefix',action='append',default=[])
    args=parser.parse_args()
    result=predict(args) if args.action=='predict' else seal(args)
    print(json.dumps({k:result[k] for k in ('status','images','new_images','reused_images') if k in result}))


if __name__=='__main__':main()
