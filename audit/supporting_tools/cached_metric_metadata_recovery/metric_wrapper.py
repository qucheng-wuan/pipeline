"""Metadata-only compatibility for original sealed supervision manifest, no metric/math edits."""
import argparse,copy,hashlib,importlib.util,json,os
from pathlib import Path
from types import SimpleNamespace

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--script',required=True);p.add_argument('--output',required=True);p.add_argument('--label-manifest',required=True);p.add_argument('--audit-dir',required=True);p.add_argument('--rank',type=int);a=p.parse_args()
 root=Path(a.output);contract=json.loads((root/'contract.json').read_text());roster=lines(root/'roster.jsonl')
 assert sha(root/'roster.jsonl')==contract['roster_sha256'];assert sha(a.script)==contract['sources']['scripts/validate_stage2_full.py']
 assert sha(a.label_manifest)==contract['selection']['label_manifest_sha256']
 from v2_lidf import data
 original=data.load_label_manifest;labels=original(a.label_manifest,'evaluation');idx={r['cache_id']:r for r in roster}
 assert len(labels)==len(idx)==1226 and len({r['cache_id'] for r in labels})==1226
 normalized=[];mapping=[]
 for row in labels:
  r=idx[row['cache_id']];origin=row['original_row'];out=copy.deepcopy(row)
  for k in ('cache_id','sample_id','rgb_sha256','native_hw','domain'):
   assert row[k]==r[k]==origin[k],(row['cache_id'],k)
  assert row['intrinsics_native']==r['intrinsics_native']
  split=origin['official_split'];assert split in ('real-test','real-val','synthetic-test','synthetic-val')
  group=row['domain']+('_known' if split.endswith('-val') else '_novel')
  for k,v in [('official_split',split),('benchmark_group',group)]:
   assert k not in row or row[k]==v;assert v==r[k];out[k]=v
  assert out['original_row']==row['original_row']
  normalized.append(out);mapping.append({'cache_id':row['cache_id'],'official_split':split,'benchmark_group':group,'original_row_sha256':hashlib.sha256(json.dumps(origin,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
 audit=Path(a.audit_dir);audit.mkdir(parents=True,exist_ok=True)
 rec={'status':'passed_metadata_identity_only','images':1226,'script':ref(__file__),'frozen_validator':ref(a.script),'label_manifest':ref(a.label_manifest),'roster':ref(root/'roster.jsonl'),'contract':ref(root/'contract.json'),'mapping_rule':"official_split=label.original_row.official_split; benchmark_group=label.domain+('_known' if split.endswith('-val') else '_novel')",'mapping':mapping,'labels_or_arrays_modified':False,'prediction_model_executed':False,'metric_math_modified':False,'rank':a.rank}
 if a.rank is None:
  (audit/'cpu_metadata_identity.complete.json').write_text(json.dumps(rec,indent=2)+'\n');return
 assert 0<=a.rank<8
 spec=importlib.util.spec_from_file_location('frozen_full_validation',a.script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.validate_roster(normalized,roster)
 def adapted(path,role=None):
  assert str(Path(path).resolve())==str(Path(a.label_manifest).resolve()) and role=='evaluation'
  assert sha(path)==contract['selection']['label_manifest_sha256']
  return normalized
 data.load_label_manifest=adapted
 import torch
 assert os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID' and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.4,0)
 (audit/('rank%02d.metadata.json'%a.rank)).write_text(json.dumps(rec,indent=2)+'\n')
 try:module.metric_worker(SimpleNamespace(output=root,label_manifest=Path(a.label_manifest),rank=a.rank,metric_device='cuda:0'))
 finally:data.load_label_manifest=original
if __name__=='__main__':main()
