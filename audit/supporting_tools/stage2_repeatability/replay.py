"""External observation only: real E58, sealed RGB/P/K, no labels or modified math."""
import argparse, copy, hashlib, inspect, json, os, sys, time
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()

def ref(p): return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def diff(a,b):
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'max_abs_m':float(d.max()),'mean_abs_m':float(d.mean()),'pixels_over_1e_6m':int((d>1e-6).sum()),'bitwise_equal':bool(np.array_equal(a,b))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['old','portable'],required=True)
    p.add_argument('--acceptance-root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();o=Path(a.output);o.mkdir(parents=True,exist_ok=False)
    c=read(Path(a.acceptance_root)/'contract.json');rp=Path(a.acceptance_root)/'roster.jsonl'
    assert sha(rp)==c['roster_sha256']
    targets={'evaluation_a55778db2d9d498258c58162':45385,'evaluation_c414bc76776ccdf4f55fc375':40995}
    rows=[json.loads(s) for s in rp.read_text().splitlines() if s.strip()];rows=[r for r in rows if r['cache_id'] in targets]
    assert {r['cache_id'] for r in rows}==set(targets)
    for v in c['dependencies'].values(): assert sha(v['path'])==v['sha256']
    bundle=Path(c['bundle']);meta=read(bundle/'model.json')
    assert sha(a.checkpoint)==meta['origin']['checkpoint_sha256'] and meta['origin']['epoch_one_based']==58
    assert sha(bundle/'model.pt')==meta['weights']['sha256']
    import torch
    torch.set_num_threads(2);torch.cuda.set_device(0)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state=torch.load(str(bundle/'model.pt'),map_location='cpu',weights_only=True)['model']
    assert state.keys()==saved['model'].keys() and all(torch.equal(state[k],saved['model'][k]) for k in state)
    if a.mode=='old':
        from v2_lidf.model import LIDFStage1
        from v2_lidf.predict import prediction_forward
        cfg=copy.deepcopy(saved['contract']['config']['model']);cfg['spvnas']['audit_dir']=str(o/'initialization')
        model=LIDFStage1(**cfg).to('cuda:0');model.load_state_dict(saved['model'],strict=True);model.eval()
        def predict(rgb,depth,k,seed):
            image=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255.
            image=(image-image.new_tensor([.485,.456,.406])[:,None,None])/image.new_tensor([.229,.224,.225])[:,None,None]
            item={'rgb':image,'depth_input':torch.from_numpy(depth.copy())[None],'intrinsics':torch.from_numpy(k.copy()),'item_id':'diagnostic','index':0}
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with torch.inference_mode(): result=prediction_forward(model,item,'cuda:0')
            return {'depth':result['depth'][0,0].cpu().numpy(),'refined_mask':result['refined_mask'][0,0].cpu().numpy()}
    else:
        from transparent_depth.stage2 import load_stage2,predict_stage2
        model=load_stage2(bundle,'cuda:0')
        def predict(rgb,depth,k,seed): return predict_stage2(model,rgb,depth,k,seed=seed)
    actual=model.state_dict();assert actual.keys()==state.keys()
    assert all(torch.equal(actual[k].cpu(),state[k]) for k in state)
    del actual,state,saved
    import v2_lidf.model as numerical
    original=numerical.stable_group_argmax;observations=[];active={}
    def observe(scores,groups,n):
        selected=original(scores,groups,n)
        f=inspect.currentframe().f_back;v=f.f_locals;q=v['query_ids'];target=active['target']
        local=torch.nonzero(q==target,as_tuple=False).flatten()
        if local.numel():
            ix=int(local.item());ids=torch.nonzero(groups==ix,as_tuple=False).flatten()
            vals=scores[ids].detach().cpu().numpy();order=np.argsort(-vals,kind='stable');picked=int(selected[ix].item())
            observations.append({'cache_id':active['id'],'repeat':active['repeat'],'query':target,'chunk_query':ix,
                'selected_pair':picked,'selected_z':float(v['pair_positions'][picked,2].item()),
                'top2_margin':float(vals[order[0]]-vals[order[1]]) if len(order)>1 else None,
                'candidates':[{'pair_index':int(ids[j].item()),'voxel_index':int(v['vi'][ids[j]].item()),
                    'score':float(vals[j]),'candidate_z':float(v['pair_positions'][ids[j],2].item()),
                    'offset':v['offsets'][ids[j]].detach().cpu().tolist(),'enter_z':float(v['enter'][ids[j],2].item())} for j in order]})
        return selected
    records=[]
    for row in rows:
        for key in ('input_cache','golden_prediction'): assert sha(row[key]['path'])==row[key]['sha256']
        with np.load(row['input_cache']['path'],allow_pickle=False) as z:
            assert set(z.files)=={'rgb','depth_input','intrinsics','native_hw','metadata_json'}
            rgb,depth,k=[z[key].copy() for key in ('rgb','depth_input','intrinsics')]
            md=json.loads(str(z['metadata_json'].item()));assert md['rgb_sha256']==row['rgb_sha256'] and md['ground_truth_access'] is False
        with np.load(row['golden_prediction']['path'],allow_pickle=False) as z: golden=z['depth'].copy();goldmask=z['refined_mask'].copy()
        seed=int(row['rgb_sha256'][:8],16);runs=[]
        for observed in (False,True):
            numerical.stable_group_argmax=observe if observed else original
            for repeat in range(a.repeats):
                active.update(id=row['cache_id'],target=targets[row['cache_id']],repeat=repeat)
                t=time.time();result=predict(rgb,depth,k,seed);torch.cuda.synchronize()
                path=o/(row['cache_id']+('_observed' if observed else '_plain')+str(repeat)+'.npz')
                np.savez(path,depth=result['depth'],refined_mask=result['refined_mask'])
                item={'observed':observed,'repeat':repeat,'seconds':time.time()-t,'output':ref(path),
                    'vs_golden':diff(result['depth'],golden),'mask_vs_golden_exact':bool(np.array_equal(result['refined_mask'],goldmask)),
                    'target_z':float(result['depth'].reshape(-1)[targets[row['cache_id']]])}
                runs.append(item);print(json.dumps({'mode':a.mode,'cache_id':row['cache_id'],**item}),flush=True)
        records.append({'cache_id':row['cache_id'],'input':row['input_cache'],'golden':row['golden_prediction'],'rgb_sha256':row['rgb_sha256'],'seed':seed,'target_flat':targets[row['cache_id']],'runs':runs})
    numerical.stable_group_argmax=original
    modules={n:ref(m.__file__) for n,m in list(sys.modules.items()) if (n=='v2_lidf' or n.startswith('v2_lidf.') or n=='transparent_depth' or n.startswith('transparent_depth.')) and getattr(m,'__file__',None)}
    write(o/'complete.json',{'status':'completed_observation_not_acceptance','mode':a.mode,'script':ref(__file__),'argv':sys.argv,
        'checkpoint':ref(a.checkpoint),'model_state_exact_to_original_checkpoint':True,'bundle_metadata':ref(bundle/'model.json'),
        'contract':ref(Path(a.acceptance_root)/'contract.json'),'roster':ref(rp),'records':records,'argmax_observations':observations,
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0),
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'modules':modules,
        'labels_read':False,'mathematical_operators_changed':False,'observed_pass_cpu_copies_synchronize_cuda':True,
        'original_tolerance_m':1e-6,'original_failed_acceptance_not_overridden':True})
if __name__=='__main__':main()
