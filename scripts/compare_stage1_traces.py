"""CPU comparison of SHA-bound live traces; tolerances are declared, never inferred."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def compare(first,second,*,atol=0.,rtol=0.):
    first=Path(first);second=Path(second)
    a=json.loads((first/'complete.json').read_text());b=json.loads((second/'complete.json').read_text())
    if a['status']!='trace_complete' or b['status']!='trace_complete' or a['rgb_sha256']!=b['rgb_sha256']:
        raise ValueError('traces are incomplete or use different RGB bytes')
    if set(a['fields'])!=set(b['fields']):raise ValueError('trace field sets differ')
    if a['calls']!=b['calls']:raise ValueError('trace execution call counts differ')
    fields={}
    for name,left in a['fields'].items():
        right=b['fields'][name]
        if left['shape']!=right['shape'] or left['source_dtype']!=right['source_dtype']:
            raise ValueError('trace shape/source dtype differs: '+name)
        arrays=[]
        for folder,record in ((first,left),(second,right)):
            path=folder/record['path']
            if sha(path)!=record['sha256']:raise ValueError('trace array changed: '+str(path))
            arrays.append(np.load(path,allow_pickle=False,mmap_mode='r').reshape(-1))
        exact=0;bad=0;total=0.;maximum=0.;finite=True
        for start in range(0,arrays[0].size,1_000_000):
            x=arrays[0][start:start+1_000_000].astype(np.float64);y=arrays[1][start:start+1_000_000].astype(np.float64)
            valid=np.isfinite(x)&np.isfinite(y);finite=finite and bool(valid.all())
            delta=np.abs(x-y);exact+=int(np.count_nonzero(x!=y))
            bad+=int(np.count_nonzero(~valid|(delta>atol+rtol*np.abs(x))))
            total+=float(delta[valid].sum());maximum=max(maximum,float(delta[valid].max(initial=0)))
        fields[name]={'shape':left['shape'],'source_dtype':left['source_dtype'],'elements':arrays[0].size,
            'exact_equal':exact==0,'unequal_elements':exact,'outside_tolerance':bad,'finite':finite,
            'max_abs':maximum,'mean_abs':total/max(1,arrays[0].size),'within_declared_tolerance':bad==0}
    passed=all(r['within_declared_tolerance'] for r in fields.values())
    return {'schema':'stage1_live_trace_comparison_v1','status':'passed' if passed else 'failed_numeric_comparison',
        'reference':{'path':str(first/'complete.json'),'sha256':sha(first/'complete.json')},
        'candidate':{'path':str(second/'complete.json'),'sha256':sha(second/'complete.json')},
        'rgb_sha256':a['rgb_sha256'],'declared_tolerance':{'atol':atol,'rtol':rtol},
        'tolerance_automatically_relaxed':False,'fields':fields}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--atol',type=float,default=0.);p.add_argument('--rtol',type=float,default=0.)
    args=p.parse_args()
    if args.atol<0 or args.rtol<0:raise ValueError('negative tolerance')
    if args.output.exists():raise ValueError('preserve previous comparison; choose a new output')
    report=compare(args.reference,args.candidate,atol=args.atol,rtol=args.rtol)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'fields':len(report['fields'])}))
    if report['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
