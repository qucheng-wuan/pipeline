"""Isolated legacy/portable live RGB tracing; run each mode in a fresh process.

Hooks only copy actual intermediate values to CPU. They do not replace expert
results or run an alternative reconstruction. Trace copies synchronize CUDA and
therefore change timing, so this is not a performance benchmark. A trace receipt
means execution completed, not numerical equivalence; compare separately.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path)


class Recorder:
    def __init__(self,output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.fields={};self.calls={}

    def put(self,name,value):
        import numpy as np
        import torch
        if isinstance(value,dict):
            for key,item in value.items():self.put(name+'/'+str(key),item)
            return
        if isinstance(value,(list,tuple)):
            for i,item in enumerate(value):self.put(name+'/'+str(i),item)
            return
        if not isinstance(value,(np.ndarray,torch.Tensor)):return
        if name in self.fields:raise RuntimeError('duplicate trace field: '+name)
        dtype=str(value.dtype).removeprefix('torch.')
        if isinstance(value,torch.Tensor):
            value=value.detach().cpu()
            if value.dtype==torch.bfloat16:value=value.float()
            value=value.numpy()
        value=np.ascontiguousarray(value)
        relative='arrays/'+hashlib.sha256(name.encode()).hexdigest()[:24]+'.npy'
        path=self.output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as stream:np.save(stream,value,allow_pickle=False)
        self.fields[name]={'path':relative,'sha256':sha(path),'bytes':path.stat().st_size,
            'source_dtype':dtype,'stored_dtype':str(value.dtype),'shape':list(value.shape),
            'finite':bool(np.isfinite(value).all())}

    def wrap(self,stack,module,function,label,*,input_names=()):
        original=getattr(module,function)
        def call(*args,**kwargs):
            index=self.calls.get(label,0);self.calls[label]=index+1
            prefix=label+'/'+str(index)
            for i,name in input_names:self.put(prefix+'/'+name,args[i])
            result=original(*args,**kwargs);self.put(prefix+'/output',result);return result
        stack.enter_context(patch.object(module,function,call))


def install_common_trace(stack,record,runtime,lapmodule,expertmath,teacher_module,tom_module):
    record.wrap(stack,expertmath,'_infer_moge','moge')
    record.wrap(stack,expertmath,'_extract_dino','dino')
    record.wrap(stack,teacher_module,'infer_teacher_normal','teacher')
    record.wrap(stack,tom_module,'temperature_scaled_object_probability','tom',input_names=((0,'logits'),))
    # live_components resolves these functions at call time in both implementations.
    record.wrap(stack,runtime,'assemble_inputs','v2_inputs')
    record.wrap(stack,runtime,'predict_native','v2')
    for name,label in [('fixed_depth_and_bands','lap/fixed_bands'),('build_gate_context','lap/context'),
                       ('mean_coefficients','lap/mean_coeff'),('apply_gates_full','lap/applied')]:
        record.wrap(stack,lapmodule,name,label)


def attach_expert_hooks(stack,record,experts):
    def capture(name):
        def hook(module,args,result):
            index=record.calls.get(name,0);record.calls[name]=index+1
            record.put(name+'/'+str(index)+'/output',result)
        return hook
    for name,module in [('r5',experts.r5.model),('far',experts.anchor)]:
        handle=module.register_forward_hook(capture(name));stack.callback(handle.remove)


def run_portable(args,record):
    project=args.project.resolve();sys.path.insert(0,str(project/'src'))
    from transparent_depth.stage1 import Stage1Assets,Stage1Predictor,PROJECT
    if PROJECT!=project:raise RuntimeError('portable module imported from another checkout')
    from transparent_depth.input_guard import RGBInputGuard,configure_jit_cache,assert_project_modules
    assets=Stage1Assets(args.assets,args.asset_root)
    cache=configure_jit_cache(args.cache_root or args.output/'jit_cache')
    guard=RGBInputGuard(project=project,asset_root=assets.root,rgb_paths=[args.rgb],
        output_root=args.output,cache_root=cache,metadata_paths=[assets.manifest],deny_prefixes=args.deny_prefix)
    metadata={}
    try:
        with guard,ExitStack() as stack:
            predictor=Stage1Predictor(args.assets,args.asset_root,device=args.device,runtime_cache=cache)
            experts=predictor.load_experts()
            from transparent_depth import stage1_experts
            from native_fusion import runtime
            from laplacian_train import model as lapmodule
            from moge_transparent_fusion import transnormal_teacher_r5,tomnet_infer
            install_common_trace(stack,record,runtime,lapmodule,stage1_experts,transnormal_teacher_r5,tomnet_infer)
            # The portable expert module bound assemble_inputs when imported.
            stack.enter_context(patch.object(stage1_experts,'assemble_inputs',runtime.assemble_inputs))
            attach_expert_hooks(stack,record,experts)
            handle=predictor.adapter.register_forward_hook(lambda module,inputs,result:record.put('lap/gates',result))
            stack.callback(handle.remove)
            depth,metadata=predictor.predict(args.rgb);record.put('final_depth',depth)
        metadata.update(module_sources=assert_project_modules(project),file_access_guard=guard.receipt())
        return metadata
    finally:
        write_json(args.output/'file_access_guard.json',guard.receipt())


def run_legacy(args,record):
    if not args.legacy_project or args.checkpoint is None or args.checkpoint_sha256 is None:
        raise ValueError('legacy mode requires explicit --legacy-project(s), --checkpoint and --checkpoint-sha256')
    for root in reversed(args.legacy_project):sys.path.insert(0,str(root.resolve()/'src'))
    from laplacian_train import predict_rgb as lapmodule
    from native_fusion import runtime
    if Path(lapmodule.__file__).resolve().is_relative_to(args.project.resolve()):
        raise RuntimeError('reference accidentally imported portable source')
    # Load dependencies using the legacy loader, including its authenticated overlay.
    # Cache loaders are patched before construction; live=True remains mandatory.
    from native_fusion import data
    from laplacian_train import prepare
    def no_cache(*args,**kwargs):raise AssertionError('live reference attempted to read an expert cache')
    with ExitStack() as stack:
        stack.enter_context(patch.object(data,'load_tensor_asset',no_cache))
        stack.enter_context(patch.object(runtime,'load_tensor_asset',no_cache))
        stack.enter_context(patch.object(prepare.FullInputBuilder,'__init__',no_cache))
        with prepare.forbid_label_reads():
            lapmodule.numerics()
            predictor=lapmodule.RGBLearnedLaplacianPredictor(args.checkpoint,
                expected_sha256=args.checkpoint_sha256,device=args.device,live=True)
            predictor.runtime=runtime.load_rgb_runtime(prepare.CHECKPOINT,device=predictor.device,prefer_cache=False)
            predictor.runtime.experts=runtime.load_live_experts(predictor.device)
            from moge_transparent_fusion import full_cache,transnormal_teacher_r5,tomnet_infer
            install_common_trace(stack,record,runtime,lapmodule,full_cache,transnormal_teacher_r5,tomnet_infer)
            attach_expert_hooks(stack,record,predictor.runtime.experts)
            handle=predictor.model.register_forward_hook(lambda module,inputs,result:record.put('lap/gates',result))
            stack.callback(handle.remove)
            depth,metadata=predictor.predict(args.rgb);record.put('final_depth',depth)
        if metadata.get('mode')!='live_rgb_experts' or metadata.get('force_live') is not True:
            raise RuntimeError('reference did not execute full live RGB inference')
        metadata['cache_loaders_blocked']=True
        prefixes=('native_fusion','laplacian_train','laplacian_fusion','far_r5','moge_transparent_fusion','moge','dinov3','transnormal')
        metadata['module_sources']={name:{'path':str(Path(module.__file__).resolve()),'sha256':sha(module.__file__)}
            for name,module in list(sys.modules.items()) if getattr(module,'__file__',None)
            and any(name==p or name.startswith(p+'.') for p in prefixes)}
        return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('legacy','portable'),required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--rgb',type=Path,required=True);p.add_argument('--rgb-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--assets',type=Path);p.add_argument('--asset-root',type=Path);p.add_argument('--cache-root',type=Path)
    p.add_argument('--deny-prefix',action='append',default=[])
    p.add_argument('--legacy-project',type=Path,action='append',default=[])
    p.add_argument('--checkpoint',type=Path);p.add_argument('--checkpoint-sha256')
    args=p.parse_args();args.output=args.output.resolve();args.rgb=args.rgb.resolve()
    if sha(args.rgb)!=args.rgb_sha256:raise ValueError('RGB changed before trace')
    if (args.output/'complete.json').exists() or (args.output/'attempt.json').exists():
        raise ValueError('use a new trace directory; preserve previous attempts')
    record=Recorder(args.output);started=time.time()
    audit={'schema':'stage1_live_trace_v1','mode':args.mode,'status':'running','rgb_sha256':args.rgb_sha256,
        'script_sha256':sha(__file__),'started_unix':started,'command':sys.argv,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'observational_cpu_copies_and_synchronization':True,'numerical_equivalence_claimed':False}
    write_json(args.output/'attempt.json',audit)
    try:
        import torch
        audit['runtime']={'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda}
        metadata=run_portable(args,record) if args.mode=='portable' else run_legacy(args,record)
        audit.update(status='trace_complete',metadata=metadata,fields=record.fields,calls=record.calls)
        audit['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
        audit['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved(args.device)
        if record.calls.get('moge')!=2 or record.calls.get('v2')!=1 or record.calls.get('dino')!=1:
            raise RuntimeError('trace did not execute the expected full live expert calls')
        if sha(args.rgb)!=args.rgb_sha256:raise ValueError('RGB changed during trace')
    except BaseException as error:
        audit.update(status='failed',error=repr(error),fields=record.fields,calls=record.calls)
        raise
    finally:
        audit.update(finished_unix=time.time(),seconds=time.time()-started)
        write_json(args.output/'attempt.json',audit)
    write_json(args.output/'complete.json',audit)
    print(json.dumps({'status':audit['status'],'mode':args.mode,'fields':len(record.fields)}),flush=True)


if __name__=='__main__':main()
