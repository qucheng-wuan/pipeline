"""Train the added native multiscale layer while both experts and V2 stay frozen."""
import argparse
from datetime import timedelta
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from .common import ROOT, check_protocol, move, numerics, read, save_checkpoint, sha, write
from .data import NativePatchDataset
from .model import LearnableLaplacianFusion, apply_gates_patch
from .objective import objective


def train(args):
    numerics()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rank=int(os.environ.get('RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    distributed=world>1
    if world<1 or not 0<=rank<world:raise ValueError('invalid distributed rank/world')
    device=torch.device('cuda',local) if distributed else torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group('nccl',timeout=timedelta(hours=2),device_id=device)
    protocol=check_protocol(args.root)
    if args.scope=='full':
        promotion=read(args.root/'promotion.json')
        if not promotion['promoted'] or promotion['pyramid_weight']!=args.pyramid_weight:
            raise ValueError('formal training must follow the frozen pilot promotion')
    ds=NativePatchDataset(args.root,scope=args.scope,include_target=True)
    presentations=math.ceil(len(ds)/world)*world
    padding=presentations-len(ds)
    if not len(ds):raise ValueError('training dataset is empty')
    model=LearnableLaplacianFusion().to(device)
    config={'scope':args.scope,'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,
        'batch_size_per_rank':args.batch_size,'world_size':world,
        'effective_batch_size':args.batch_size*world,
        'learning_rate':args.learning_rate,'pyramid_weight':args.pyramid_weight,
        'samples_per_epoch':len(ds),'protocol_sha256':sha(args.root/'protocol.json'),
        'sample_presentations_per_epoch':presentations,'sampler_padding_per_epoch':padding,
        'sampler':'DistributedSampler shuffle seed+epoch drop_last=False' if distributed else 'seeded shuffled DataLoader',
        'model_settings':model.config(),
        'optimizer':'AdamW','weight_decay':1e-4,'gradient_clip':1.,'autocast':False,
        'init':'exact fixed L6 alpha0.5 global fusion','frozen_backbones_and_v2':True}
    run=args.root/'runs'/args.run
    if rank==0:
        run.mkdir(parents=True,exist_ok=True)
        if (run/'config.json').exists() and read(run/'config.json')!=config:raise ValueError('run config changed')
        write(run/'config.json',config)
    if distributed:dist.barrier()
    if read(run/'config.json')!=config:raise ValueError('distributed run config mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    first=0
    if (run/'last.pt').exists():
        state=torch.load(run/'last.pt',map_location=device,weights_only=False)
        if state['config']!=config:raise ValueError('resume config changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']);first=state['epoch']
    elif rank==0:
        save_checkpoint(run/'epoch00.pt',{'model':model.state_dict(),'epoch':0,'config':config})
    train_model=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False) if distributed else model
    sampler=(DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,
                                seed=args.seed,drop_last=False) if distributed else None)
    start=time.time()
    if rank==0:
        write(run/'status.json',{'status':'running','pid':os.getpid(),'device':str(device),
            'world_size':world,'start_time':start,'start_epoch':first})
    log=(run/'progress.jsonl').open('a',buffering=1) if rank==0 else None
    try:
        for epoch in range(first+1,args.epochs+1):
            generator=torch.Generator().manual_seed(args.seed+epoch)
            if sampler is not None:sampler.set_epoch(epoch)
            loader=DataLoader(ds,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,drop_last=False,
                num_workers=args.workers,pin_memory=device.type=='cuda',generator=generator,
                persistent_workers=False)
            epoch_start=time.time(); seen=0;totals={}; gate_sum=torch.zeros(6,device=device,dtype=torch.float64)
            train_model.train()
            # The frozen per-stage schedule also fixes resumed epoch rates.
            rate=args.learning_rate*(.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
            for group in optimizer.param_groups:group['lr']=rate
            for step,batch in enumerate(loader,1):
                inputs=move(batch['inputs'],device);labels=move(batch['labels'],device)
                optimizer.zero_grad(set_to_none=True)
                output=train_model(inputs['context'])
                gates=output['gate_delta']
                depth=apply_gates_patch(inputs,gates)['depth']
                loss,stats=objective(depth,gates,labels,args.pyramid_weight)
                if not torch.isfinite(loss):raise ValueError(f'nonfinite loss epoch {epoch} step {step}')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                batch_count=len(depth);seen+=batch_count
                for key,value in stats.items():totals[key]=totals.get(key,0.)+float(value)*batch_count
                gate_sum+=gates.detach().double().mean((0,2,3))*batch_count
                if rank==0 and (step==1 or step%20==0 or step==len(loader)):
                    record={'epoch':epoch,'step':step,'seen':seen,'seconds':time.time()-epoch_start,
                        'rank':rank,'world_size':world,'loss':float(loss.detach()),'gradient_norm':float(norm),'lr':float(rate)}
                    log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
            expected_local=math.ceil(len(ds)/world)
            if seen!=expected_local:raise ValueError('incomplete rank training epoch')
            stat_keys=sorted(totals)
            aggregates=torch.tensor([seen]+[totals[k] for k in stat_keys],device=device,dtype=torch.float64)
            if distributed:
                dist.all_reduce(aggregates)
                dist.all_reduce(gate_sum)
            global_seen=int(aggregates[0])
            if global_seen!=presentations:raise ValueError('incomplete global training epoch')
            if rank==0:
                record={'epoch':epoch,'samples':global_seen,'unique_training_images':len(ds),
                    'sampler_padding':padding,'world_size':world,'seconds':time.time()-epoch_start,
                    'metrics':{k:float(aggregates[i+1]/global_seen) for i,k in enumerate(stat_keys)},
                    'mean_alpha':(.5+gate_sum/global_seen).tolist()}
                write(run/f'epoch{epoch:02d}.json',record)
                state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'config':config,
                    'metrics':record,'source_sha256':protocol['source_sha256']}
                save_checkpoint(run/f'epoch{epoch:02d}.pt',state)
                save_checkpoint(run/'last.pt',state)
                write(run/'status.json',{'status':'running','epoch':epoch,'device':str(device),
                    'world_size':world,'elapsed_seconds':time.time()-start})
            if distributed:dist.barrier()
        if rank==0:
            checkpoints={p.name:sha(p) for p in sorted(run.glob('epoch*.pt'))}
            write(run/'complete.json',{'status':'complete','epochs':args.epochs,'samples_per_epoch':len(ds),
                'sample_presentations_per_epoch':presentations,'sampler_padding_per_epoch':padding,
                'training_sample_presentations':args.epochs*presentations,'checkpoint_sha256':checkpoints,
                'config':config,'seconds':time.time()-start})
            write(run/'status.json',{'status':'complete','epoch':args.epochs,
                'world_size':world,'elapsed_seconds':time.time()-start})
        if distributed:dist.barrier()
    except Exception as exc:
        if rank==0:write(run/'status.json',{'status':'failed','error':repr(exc),'elapsed_seconds':time.time()-start})
        raise
    finally:
        if log is not None:log.close()
        if distributed:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--scope',choices=('pilot','full'),required=True)
    p.add_argument('--run',required=True);p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--pyramid-weight',type=float,default=.05)
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=2)
    train(p.parse_args())


if __name__=='__main__':main()
