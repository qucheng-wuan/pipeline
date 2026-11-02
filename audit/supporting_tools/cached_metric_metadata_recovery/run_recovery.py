import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
A=Path('/mnt/gluster_dcpu/hengxianli/transparent_depth_release_20261101');P=Path('/home/hengxianli/RGBOnly_Fusion_Laplacian_LIDF_SPVNAS');R=Path('/mnt/gluster_dcpu/hengxianli/fusion_v2_lidf_spvnas_20261030');B=Path(__file__).resolve().parent;O=A/'audit/stage2_full_cached';PY=A/'runtime/stage2_fresh/bin/python';LABEL=Path('/mnt/gluster_dcpu/hengxianli/fusion_v2_lidf_curriculum_20261027/manifests/label_manifest.jsonl')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 cpu=json.loads((B/'cpu_metadata_identity.complete.json').read_text());assert cpu['status']=='passed_metadata_identity_only' and cpu['images']==1226
 assert sha(B/'metric_wrapper.py')==cpu['script']['sha256'];assert sha(P/'scripts/validate_stage2_full.py')==cpu['frozen_validator']['sha256']
 failed=json.loads((O/'metric-worker.execution.json').read_text());assert failed['status']=='failed' and len(failed['workers'])==8 and all(w['returncode']==1 for w in failed['workers'])
 snap=B/'failure_snapshot';snap.mkdir(exist_ok=False);preserved=[]
 paths=[O/'metric-worker.execution.json',A/'runtime/cached_final_validation.status.json',A/'runtime/cached_final_validation.log']+[Path(w['log']) for w in failed['workers']]
 for i,p in enumerate(paths):
  dst=snap/('%02d_'%i+p.name);shutil.copyfile(p,dst);assert sha(p)==sha(dst);preserved.append({'original':ref(p),'snapshot':ref(dst)})
 write(snap/'manifest.json',{'files':preserved})
 for p in (O/'metric_shards').glob('*') if (O/'metric_shards').exists() else []:raise RuntimeError('Unexpected prior metric shard: '+str(p))
 env=dict(os.environ,PYTHONPATH=str(P/'src'),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUDA_HOME=str(R/'runtime/cuda-12.1'),CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',PATH=str(PY.parent)+':'+str(R/'runtime/cuda-12.1/bin')+':'+os.environ['PATH'],TMPDIR=str(A/'runtime/tmp'),PYTHONDONTWRITEBYTECODE='1')
 stable=None
 while True:
  observed=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True);rows=[s.split(',') for s in observed.splitlines()];free={int(s[0]):int(s[2]) for s in rows};assert set(free)==set(range(8))
  if min(free.values())<8192:stable=None
  elif stable is None:stable=time.monotonic()
  write(B/'status.json',{'status':'waiting_8gpus_free_8192MiB_30seconds','free_mib':free,'unix':time.time()})
  if stable is not None and time.monotonic()-stable>=30:break
  time.sleep(5)
 record={'status':'running_metrics_only','started_unix':time.time(),'wrapper':ref(B/'metric_wrapper.py'),'runner':ref(__file__),'cpu_identity':ref(B/'cpu_metadata_identity.complete.json'),'failure_snapshot':ref(snap/'manifest.json'),'prediction_seal':ref(O/'predictions.complete.json'),'gpu_observation':observed,'model_rerun':False,'workers':[]};workers=[]
 for rank in range(8):
  cmd=[str(PY),'-u',str(B/'metric_wrapper.py'),'--script',str(P/'scripts/validate_stage2_full.py'),'--output',str(O),'--label-manifest',str(LABEL),'--audit-dir',str(B),'--rank',str(rank)];log=B/('rank%02d.log'%rank);stream=log.open('x');child=subprocess.Popen(cmd,cwd=P,env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  item={'rank':rank,'pid':child.pid,'physical_gpu':rank,'command':cmd,'log':str(log)};record['workers'].append(item);workers.append((child,stream,item))
 write(B/'execution.json',record)
 for child,stream,item in workers:
  item['returncode']=child.wait();item['completion_observed_unix']=time.time();stream.close();item['log_sha256']=sha(item['log']);write(B/'execution.json',record)
 assert all(w['returncode']==0 for w in record['workers']),'Metric recovery failed; logs preserved'
 cmd=[str(PY),'-u',str(P/'scripts/validate_stage2_full.py'),'metrics-merge','--output',str(O)];log=B/'merge.log'
 with log.open('x') as stream:r=subprocess.run(cmd,cwd=P,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT)
 record['merge']={'command':cmd,'returncode':r.returncode,'log':ref(log)};write(B/'execution.json',record);assert r.returncode==0
 result=json.loads((O/'evaluation.complete.json').read_text());assert result['images']==1226 and result['test_images']==521
 assert result['cached_input_parity_passed'] is False and result['status'].startswith('failed_cached_input_')
 record.update(status='completed_metadata_recovery_original_numerical_failure_preserved',evaluation_complete=ref(O/'evaluation.complete.json'),numerical_acceptance_status=result['status'],transparent_mae_mm=result['transparent_mae_mm'],finished_unix=time.time());write(B/'complete.json',record);print(json.dumps({'status':record['status'],'metric_status':result['status'],'transparent_mae_mm':result['transparent_mae_mm']}),flush=True)
if __name__=='__main__':main()
