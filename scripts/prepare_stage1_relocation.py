"""Copy portable Stage1 to a new root and validate imports without CUDA.

The output also contains an explicit relocated GPU trace command for later use.
This script never launches GPU inference. Assets remain in an explicitly supplied
portable asset root; historical directories and the original project must be
forbidden in the subsequent relocated trace.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--rgb',type=Path,required=True);p.add_argument('--rgb-sha256',required=True)
    p.add_argument('--trace-output',type=Path,required=True)
    p.add_argument('--deny-prefix',action='append',default=[])
    args=p.parse_args();source=Path(__file__).resolve().parents[1];target=args.destination.resolve()
    if target.exists():raise ValueError('choose a new relocation directory')
    if target.is_relative_to(source):raise ValueError('relocation must be outside original project')
    target.mkdir(parents=True)
    for directory in ('src','third_party','provenance','scripts'):
        shutil.copytree(source/directory,target/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc','stage1_local_bindings.json'))
    for name in ('requirements-stage1.txt',):shutil.copyfile(source/name,target/name)
    inventory=json.loads((target/'provenance/stage1_runtime_sources.json').read_text())
    for relative,digest in inventory['sha256'].items():
        if sha(target/relative)!=digest:raise ValueError('relocated source differs: '+relative)
    command=[str(args.python),'-c',
        'import json,torch; from transparent_depth.stage1 import PROJECT,Stage1Assets; '
        'print(json.dumps({"project":str(PROJECT),"sources_sha256":Stage1Assets().verify_sources(),"cuda_initialized":torch.cuda.is_initialized()}))']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(target/'src'),PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run(command,cwd=target,env=env,text=True,capture_output=True)
    report={'schema':'stage1_relocation_preparation_v1','status':'failed','source':str(source),'destination':str(target),
        'source_inventory_sha256':sha(target/'provenance/stage1_runtime_sources.json'),'cpu_command':command,
        'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'gpu_inference_executed':False}
    if result.returncode==0:
        observed=json.loads(result.stdout.strip().splitlines()[-1]);report['observed']=observed
        if observed['project']==str(target) and observed['cuda_initialized'] is False:report['status']='cpu_relocation_passed'
    gpu_command=[str(args.python),str(target/'scripts/trace_stage1_live.py'),'--mode','portable','--project',str(target),
        '--rgb',str(args.rgb.resolve()),'--rgb-sha256',args.rgb_sha256,'--asset-root',str(args.asset_root.resolve()),
        '--output',str(args.trace_output.resolve()),'--device','cuda:0','--deny-prefix',str(source)]
    for prefix in args.deny_prefix:gpu_command.extend(['--deny-prefix',prefix])
    report['pending_gpu_trace_command']=gpu_command
    (target/'relocation_preparation.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'report':str(target/'relocation_preparation.json')}))
    if report['status']!='cpu_relocation_passed':raise SystemExit(1)


if __name__=='__main__':main()
