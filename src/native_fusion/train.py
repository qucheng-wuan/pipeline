"""Eight-rank training of native geometry fusion on every training image."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .data import DEFAULT_MANIFEST, DEFAULT_ROOT, EXPECTED_COUNTS, NativeFusionDataset, sha256_file
from .cache import producer_contract
from .evaluation import SELECTION_RULE, aggregate, image_record, qualification, summarize_scale
from .losses import LossConfig, objective
from .model import NativeGeometryFusion, Settings
from .runtime import predict_native, runtime_contract

SCHEMA = 'native_geometry_fusion_v2'


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(tmp, path)


def write_checkpoint(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed); np.random.seed(seed)


@torch.inference_mode()
def validate(model, dataset, rank, world, device, workers, output, epoch, include_far):
    model.eval()
    subset = Subset(dataset, list(range(rank, len(dataset), world)))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=False, worker_init_fn=seed_worker)
    rows = []
    start = time.time()
    for index, cpu_batch in enumerate(loader):
        batch = move(cpu_batch, device)
        depth = predict_native(model, batch, tile_size=384, halo=16, amp=True)
        rows.append(image_record(depth, batch, include_far=include_far))
        if rank == 0 and (index+1) % 16 == 0:
            print(json.dumps({'phase':'validation','epoch':epoch,'rank0_images':index+1,
                              'rank0_total':len(subset),'seconds':time.time()-start}),flush=True)
    gathered = [None]*world if rank == 0 else None
    dist.gather_object(rows, gathered, dst=0)
    if rank != 0:
        return None
    rows = sorted([r for part in gathered for r in part], key=lambda r:r['sample_id'])
    if len(rows) != EXPECTED_COUNTS['val'] or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('validation coverage must be exactly 704 unique images')
    per_image = output/f'validation/epoch_{epoch:02d}.jsonl'
    per_image.parent.mkdir(parents=True,exist_ok=True)
    with per_image.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
    result = {'epoch':epoch, 'images':len(rows), 'summary':aggregate(rows),
              'scale_diagnostic':summarize_scale(rows), 'seconds':time.time()-start,
              'per_image_sha256':sha256_file(per_image)}
    write_json(output/f'validation/epoch_{epoch:02d}.json',result)
    return result


def provenance(args, config):
    source = Path(__file__).parent
    if not args.smoke:
        complete = json.loads((DEFAULT_ROOT/'r5_residuals/complete.json').read_text())
        if complete['status'] != 'complete' or complete['contract'] != producer_contract(args.manifest):
            raise ValueError('residual producer contract does not match the current source and dataset')
        if complete['samples'] != sum(EXPECTED_COUNTS.values()):
            raise ValueError('incomplete residual population')
    return {'schema_version':SCHEMA,'config':config,'manifest':str(args.manifest),
            'manifest_sha256':sha256_file(args.manifest),
            'source_sha256':{'src/native_fusion/'+p.name:sha256_file(p) for p in sorted(source.glob('*.py'))},
            'residual_cache_complete_sha256':sha256_file(DEFAULT_ROOT/'r5_residuals/complete.json') if not args.smoke else None,
            'rgb_runtime_contract':runtime_contract(args.manifest) if not args.smoke else None,
            'selection_rule':SELECTION_RULE,'training_images':EXPECTED_COUNTS['train'],
            'validation_images':EXPECTED_COUNTS['val'],'test_labels_read':False,
            'frozen_experts':['DINOv3+FAR','native MoGe','R5','TOM','TransNormal'],
            'training_parameters':'context encoder, scene scale head, coarse zero-mean correction, predicted-region scale head, native detail head',
            'native_training':'one deterministic uniform native crop per image per epoch plus full-scene context metric supervision',
            'precision':'BF16 neural layers; FP32 depth and differentiable surface construction; FP64 evaluation accumulation',
            'camera_use':'K is used for labelled geometry supervision/evaluation only; RGB prediction never reads K',
            'normal_reference':'synthetic independent cameraNormals; real depth-induced proxy; missing reliable camera disables normal losses',
            'torch_version':str(torch.__version__),'started_unix':time.time()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--output',type=Path,default=DEFAULT_ROOT/'training')
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--val-workers',type=int,default=2)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--seed',type=int,default=20261021)
    parser.add_argument('--smoke',type=int,default=0,help='separate fixed-sample optimization check, no formal results')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    rank,world,local=map(int,(os.environ['RANK'],os.environ['WORLD_SIZE'],os.environ['LOCAL_RANK']))
    if world != 8:
        raise ValueError('formal and smoke launches require exactly eight ranks')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    # Validation can involve substantial I/O; avoid the default 10-minute NCCL timeout.
    from datetime import timedelta
    dist.init_process_group('nccl',timeout=timedelta(hours=2),device_id=device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config={'epochs':args.epochs,'batch_size_per_rank':args.batch_size,'world_size':world,
            'workers_per_rank':args.workers,'val_workers_per_rank':args.val_workers,
            'learning_rate':args.lr,'weight_decay':.01,'gradient_clip':1.,'seed':args.seed,
            'patch_size':384,'optimizer':'AdamW','scheduler':'linear warmup one tenth epoch then cosine to 0.1 x initial lr',
            'model_settings':asdict(Settings()),'loss':asdict(LossConfig()),
            'sampler_padding_per_epoch':math.ceil(EXPECTED_COUNTS['train']/world)*world-EXPECTED_COUNTS['train']}
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'complete.json').exists():
        raise FileExistsError('completed training output is immutable')
    if not args.resume and (args.output/'checkpoints/last.pt').exists():
        raise FileExistsError('use explicit --resume for an interrupted run')
    if args.smoke:
        dataset=NativeFusionDataset(args.manifest,'train',patch_size=384,
            residual_root=DEFAULT_ROOT/'r5_residuals_preview_initial',allow_unsealed_residuals=True)
        dataset.rows=dataset.rows[:1]
        dataset.set_epoch(1)  # Predefined crop intersects the transparent training object.
    else:
        dataset=NativeFusionDataset(args.manifest,'train',patch_size=384,seed=args.seed)
        if len(dataset)!=EXPECTED_COUNTS['train']:
            raise ValueError('full training split changed')
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=args.seed,drop_last=False)
    generator=torch.Generator().manual_seed(args.seed+rank)
    loader=DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,
                      pin_memory=True,persistent_workers=False,drop_last=False,worker_init_fn=seed_worker,generator=generator)
    model=NativeGeometryFusion(Settings()).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    steps_per_epoch=len(loader)
    warmup=max(1,steps_per_epoch//10)
    total_steps=steps_per_epoch*args.epochs
    def schedule(step):
        if step < warmup:
            return (step+1)/warmup
        progress=min(1.,(step-warmup)/max(1,total_steps-warmup))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    start_epoch=0;best_score=float('inf');diagnostic_score=float('inf');baseline=None
    run_provenance=provenance(args,config)
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint['schema_version']!=SCHEMA or checkpoint['provenance']['config']!=config:
            raise ValueError('resume configuration mismatch')
        for key in ('manifest_sha256', 'residual_cache_complete_sha256', 'rgb_runtime_contract'):
            if checkpoint['provenance'][key] != run_provenance[key]:
                raise ValueError(f'resume data binding changed: {key}')
        for path,digest in checkpoint['provenance']['source_sha256'].items():
            if sha256_file(Path(__file__).resolve().parents[2]/path)!=digest:raise ValueError(f'resume source changed: {path}')
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch=checkpoint['epoch'];best_score=checkpoint['best_score'];diagnostic_score=checkpoint['diagnostic_score']
        run_provenance=checkpoint['provenance']
        baseline=json.loads((args.output/'validation/epoch_00.json').read_text())['summary']
    if rank==0:
        write_json(args.output/'protocol.json',run_provenance)
        print(json.dumps({'phase':'start','world_size':world,'config':config,
                          'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
    ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
    if args.smoke:
        batch=move(next(iter(loader)),device)
        losses=[];gradient_branches={}
        # Fixed learning rate for this optimization test, separate from warmup.
        for group in optimizer.param_groups:group['lr']=args.lr
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        for step in range(args.smoke):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=ddp(batch)
            loss,stats=objective(output,batch)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for name,parameter in model.named_parameters():
                if parameter.grad is not None:
                    branch=name.split('.')[0]
                    gradient_branches[branch]=max(gradient_branches.get(branch,0.),float(parameter.grad.float().abs().max()))
            optimizer.step();losses.append(float(loss.detach()))
        changed={branch:any(not torch.equal(value,initial[name]) for name,value in model.state_dict().items()
                            if name.startswith(branch+'.')) for branch in ('context_encoder','coarse_head','region_head','scale_head','detail_head')}
        result={'rank':rank,'losses':losses,'gradient_branches':gradient_branches,'changed':changed,
                'finite':all(math.isfinite(v) for v in losses),'last_loss_less_than_first':losses[-1]<losses[0],
                'normal_pixels':float(stats['normal_pixels']),'trusted_teacher_pixels':float(stats['trusted_teacher_pixels']),
                'peak_gpu_bytes':torch.cuda.max_memory_allocated(device)}
        gathered=[None]*world if rank==0 else None
        dist.gather_object(result,gathered,dst=0)
        if rank==0:
            write_json(args.output/'smoke.json',{'schema_version':SCHEMA,'ranks':gathered,
                       'passed':all(r['finite'] and r['last_loss_less_than_first'] and all(r['changed'].values()) for r in gathered)})
            print(json.dumps({'phase':'smoke_complete','ranks':gathered}),flush=True)
        dist.barrier();dist.destroy_process_group();return
    validation=NativeFusionDataset(args.manifest,'val',patch_size=None)
    if baseline is None:
        initial_result=validate(model,validation,rank,world,device,args.val_workers,args.output,0,True)
        if rank==0:
            baseline=initial_result['summary']
            write_json(args.output/'initial_diagnostic.json',{'selection':qualification(initial_result['summary'],baseline),
                        'scale_diagnostic':initial_result['scale_diagnostic']})
        dist.barrier()
    history=[]
    for epoch in range(start_epoch+1,args.epochs+1):
        dataset.set_epoch(epoch);sampler.set_epoch(epoch);ddp.train()
        epoch_start=time.time();sums=None;stat_keys=None;sample_count=0
        for index,cpu_batch in enumerate(loader):
            batch=move(cpu_batch,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=ddp(batch)
            loss,stats=objective(output,batch)
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            count=batch['rgb'].shape[0];sample_count+=count
            if sums is None:
                stat_keys=list(stats);sums=torch.zeros(len(stat_keys),device=device,dtype=torch.float64)
            sums+=torch.stack([stats[k].double() for k in stat_keys])*count
            if rank==0 and ((index+1)%100==0 or index+1==steps_per_epoch):
                state={'phase':'training','epoch':epoch,'step':index+1,'steps':steps_per_epoch,
                       'rank0_loss':float(loss.detach()),'lr':optimizer.param_groups[0]['lr'],
                       'gradient_norm':float(grad_norm),'elapsed_seconds':time.time()-epoch_start}
                write_json(args.output/'status.json',state);print(json.dumps(state),flush=True)
        counts=torch.tensor(sample_count,device=device,dtype=torch.float64)
        dist.all_reduce(sums);dist.all_reduce(counts)
        training_stats={k:float(sums[i]/counts) for i,k in enumerate(stat_keys)}
        expected_presentations=math.ceil(len(dataset)/world)*world
        if int(counts)!=expected_presentations:raise ValueError('incomplete training epoch')
        result=validate(model,validation,rank,world,device,args.val_workers,args.output,epoch,False)
        if rank==0:
            selection=qualification(result['summary'],baseline);score=selection['primary_score']
            is_best=selection['eligible'] and score<best_score
            diagnostic_best=score<diagnostic_score
            if is_best:best_score=score
            if diagnostic_best:diagnostic_score=score
            payload={'schema_version':SCHEMA,'epoch':epoch,'model_settings':model.config(),
                     'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                     'selection':selection,'best_score':best_score,'diagnostic_score':diagnostic_score,
                     'provenance':run_provenance,'validation_scale_diagnostic':result['scale_diagnostic']}
            write_checkpoint(args.output/'checkpoints/last.pt',payload)
            write_checkpoint(args.output/f'checkpoints/epoch_{epoch:02d}.pt',payload)
            if is_best:write_checkpoint(args.output/'checkpoints/best_eligible.pt',payload)
            if diagnostic_best:write_checkpoint(args.output/'checkpoints/best_diagnostic.pt',payload)
            record={'epoch':epoch,'presentations':int(counts),'unique_training_images':len(dataset),
                    'training_statistics':training_stats,'selection':selection,'scale_diagnostic':result['scale_diagnostic'],
                    'epoch_seconds':time.time()-epoch_start}
            history.append(record);write_json(args.output/f'epochs/epoch_{epoch:02d}.json',record)
            print(json.dumps({'phase':'epoch_complete','epoch':epoch,'eligible':selection['eligible'],
                    'gates_passed':selection['passed'],'whole_scene_abs_rel':score,
                    'scale':result['scale_diagnostic']}),flush=True)
        dist.barrier()
    if rank==0:
        for path,digest in run_provenance['source_sha256'].items():
            if sha256_file(path)!=digest:raise RuntimeError(f'training source changed during run: {path}')
        chosen=args.output/'checkpoints/best_eligible.pt'
        status='eligible' if chosen.exists() else 'no_checkpoint_passes_all_preservation_gates'
        if not chosen.exists():chosen=args.output/'checkpoints/best_diagnostic.pt'
        selected=torch.load(chosen,map_location='cpu',weights_only=False)
        complete={'status':'complete','qualification_status':status,'completed_epochs':args.epochs,
                  'world_size':world,'selected_checkpoint':str(chosen),'selected_sha256':sha256_file(chosen),
                  'selected_epoch':selected['epoch'],'selection':selected['selection'],
                  'scale_diagnostic':selected['validation_scale_diagnostic'],'test_labels_read':False,
                  'protocol_sha256':sha256_file(args.output/'protocol.json'),'finished_unix':time.time()}
        write_json(args.output/'complete.json',complete);print(json.dumps(complete),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    main()
