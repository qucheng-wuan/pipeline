"""CPU-only source/asset verification before moving or starting real smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def verify(root,init_root):
    manifest=json.loads((ROOT/'provenance/stage1_assets.json').read_text())
    entries=[]
    for asset in manifest['assets']:
        path=root/asset['path']
        present=path.is_file()
        good=present and path.stat().st_size==asset['bytes'] and sha(path)==asset['sha256']
        entries.append({'id':asset['id'],'path':asset['path'],'sha256':asset['sha256'],
                        'status':'verified' if good else ('missing' if not present else 'changed')})
    sys.path.insert(0,str(ROOT/'scripts'))
    from fetch_stage2_initialization import ASSETS
    initial=[]
    for name,url,digest in ASSETS:
        path=init_root/name
        good=path.is_file() and sha(path)==digest
        initial.append({'path':name,'sha256':digest,'status':'verified' if good else 'missing_or_changed'})
    inventory=json.loads((ROOT/'provenance/stage1_runtime_sources.json').read_text())
    source_good=all((ROOT/name).is_file() and sha(ROOT/name)==digest for name,digest in inventory['sha256'].items())
    return {'status':'ready' if source_good and all(r['status']=='verified' for r in entries+initial) else 'blocked',
        'stage1_sources_verified':source_good,'stage1_assets':entries,'stage2_initialization':initial,
        'gpu_execution_attempted':False,'end_to_end_passed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--init-root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();report=verify(a.asset_root,a.init_root)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));sys.exit(0 if report['status']=='ready' else 2)
