import hashlib,itertools,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compare(a,b):
 d=np.abs(a.astype(np.float64)-b.astype(np.float64))
 return {'max_abs_m':float(d.max()),'mean_abs_m':float(d.mean()),'pixels_over_original_1e_6m':int((d>1e-6).sum()),'bitwise_equal':bool(np.array_equal(a,b))}
def main():
 complete={m:json.loads((ROOT/m/'complete.json').read_text()) for m in ('old','portable')}
 result={'status':'completed_diagnostic_not_acceptance','source':{m:{'path':str(ROOT/m/'complete.json'),'sha256':sha(ROOT/m/'complete.json')} for m in complete},'original_tolerance_m':1e-6,'original_failed_acceptance_not_overridden':True,'images':[]}
 ids=[r['cache_id'] for r in complete['old']['records']]
 for cid in ids:
  records={m:next(r for r in complete[m]['records'] if r['cache_id']==cid) for m in complete};arrays={};runs={}
  for m,rec in records.items():
   for r in rec['runs']:
    label=m+('_observed' if r['observed'] else '_plain')+str(r['repeat']);path=r['output']['path'];assert sha(path)==r['output']['sha256']
    with np.load(path) as z:arrays[label]=z['depth'].copy()
    runs[label]={'target_z':r['target_z'],'vs_golden':r['vs_golden'],'mask_vs_golden_exact':r['mask_vs_golden_exact']}
  pairs=[{'left':x,'right':y,**compare(arrays[x],arrays[y])} for x,y in itertools.combinations(arrays,2)]
  classes={}
  for m in complete:
   classes[m+'_plain_self']=[p for p in pairs if p['left'].startswith(m+'_plain') and p['right'].startswith(m+'_plain')]
   classes[m+'_all_self']=[p for p in pairs if p['left'].startswith(m+'_') and p['right'].startswith(m+'_')]
  classes['cross_plain']=[p for p in pairs if p['left'].startswith('old_plain') and p['right'].startswith('portable_plain')]
  summary={k:{'max_abs_m':max(p['max_abs_m'] for p in v),'max_pixels_over_1e_6m':max(p['pixels_over_original_1e_6m'] for p in v)} for k,v in classes.items()}
  result['images'].append({'cache_id':cid,'target_flat':records['old']['target_flat'],'runs':runs,'pairwise_summary':summary,'pairwise':pairs,'argmax_observations':{m:[q for q in complete[m]['argmax_observations'] if q['cache_id']==cid] for m in complete}})
 (ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps([{k:v for k,v in r.items() if k in ('cache_id','pairwise_summary','runs','argmax_observations')} for r in result['images']],indent=2))
if __name__=='__main__':main()
