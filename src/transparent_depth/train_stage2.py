"""Portable fresh/resumed Stage2 training; no historical migration dependency.

Use torchrun and explicitly select visible GPUs in the invoking environment.
Training reads supervised GT background/masks for curriculum composition;
selection validation and deployment use only RGB-derived P and calibration.
The configuration chooses the legacy ClearGrasp, transCG256 or cleargrasp256 protocol.
Resumes require unchanged world size, source, preprocessing and configuration.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from v2_lidf.common import atomic_json, append_json, read_jsonl
from v2_lidf.curriculum import CurriculumSchedule, mix_background_depth
from v2_lidf.data import V2LIDFDataset
from v2_lidf.data_protocol import TRANSFER_PROTOCOLS, validate_partition
from v2_lidf.distributed import create_single_bucket_ddp
from v2_lidf.runtime import validate_distributed_config
from .stage2 import build_stage2, file_sha, source_hashes
from .training_utils import (seed_all, worker_seed, atomic_save, checkpoint_rng,
    restore_rng, transfer, parameter_groups, accumulation_window_size, validate,
    repair_selected_checkpoint, repair_epoch_history, repair_epoch_archive, emit_progress, assert_state_tree_equal)
from .training_cache import validate_training_caches, AuthenticatedTrainingDataset


def resolve_manifest_counts(cfg, input_manifest, world):
    """Bind loader/schedule counts to the authenticated manifest, not history."""
    import copy
    from collections import Counter
    from v2_lidf.data import load_input_manifest
    cfg = copy.deepcopy(cfg)
    if cfg.get('data_protocol') not in TRANSFER_PROTOCOLS:
        return cfg
    rows = load_input_manifest(input_manifest)
    counts = Counter(row['role'] for row in rows)
    test_selection = cfg.get('data_protocol') == 'cleargrasp256'
    if not counts['train'] or not counts['evaluation' if test_selection else 'val']:
        raise ValueError('Nonempty training and configured validation required')
    if test_selection and counts['val']:
        raise ValueError('cleargrasp256 has no selection validation split')
    scene_roles, rgb_roles = {}, {}
    for row in rows:
        if row['model_hw'] != cfg['input_hw'] or row.get('preprocessing') != cfg['preprocessing']:
            raise ValueError('Input preprocessing differs from training config')
        validate_partition(row, cfg['data_protocol'])
        digest = row['rgb_sha256']
        if digest in rgb_roles and rgb_roles[digest] != row['role']:
            raise ValueError('Duplicate RGB crosses experiment partitions')
        rgb_roles[digest] = row['role']
        if row['role'] in {'train', 'val'}:
            key = row['scene']
            if key in scene_roles and scene_roles[key] != row['role']:
                raise ValueError('Scene leakage between train and val')
            scene_roles[key] = row['role']
    cfg['counts'] = {r:counts[r] for r in ('train','val','evaluation')}
    per_rank = math.ceil(counts['train'] / world)
    cfg['curriculum']['steps_per_epoch'] = math.ceil(math.ceil(per_rank/cfg['batch_per_gpu'])/cfg['gradient_accumulation_steps'])
    cfg['sampler_protocol'] = {'kind':'DistributedSampler','drop_last':False,
        'unique_per_epoch':counts['train'],'presentations_per_epoch':per_rank*world,
        'padding_duplicates_per_epoch':per_rank*world-counts['train']}
    cfg['split_sha256'] = sorted({r['split_sha256'] for r in rows})
    return cfg


def validate_training_config(cfg, world):
    validate_distributed_config(cfg)
    positive = ('epochs', 'batch_per_gpu', 'gradient_accumulation_steps', 'expected_world_size')
    if any(type(cfg.get(k)) is not int or cfg[k] <= 0 for k in positive):
        raise ValueError('Training counts must be positive integers')
    if cfg['expected_world_size'] != world:
        raise ValueError('WORLD_SIZE differs from this fresh/resume protocol')
    stages = cfg['curriculum']['stage_epochs']
    transfer_protocol = cfg.get('data_protocol') in TRANSFER_PROTOCOLS
    if transfer_protocol:
        from v2_lidf.data_protocol import validate_protocol
        validate_protocol(cfg['preprocessing'])
        if (cfg['input_hw'] != [256,256] or cfg['model'].get('input_hw') != cfg['input_hw']
                or cfg['curriculum']['mix_resolution'] != cfg['input_hw']
                or cfg['model']['spvnas']['voxel_size'] != .05
                or cfg.get('phase_matched_validation') is not False):
            raise ValueError('Invalid 256 geometry/selection protocol')
        expected_schema = 'rgb_fusion_' + cfg['data_protocol'].removesuffix('256') + '_fullfield256_v1'
        if cfg['preprocessing']['schema'] != expected_schema:
            raise ValueError('Experiment/config preprocessing mismatch')
        if not cfg.get('diagnostic_smoke',False) and (cfg['epochs'] != 60 or stages != [20,20,20]):
            raise ValueError('Formal 256 protocol is 60 epochs, 20/20/20')
        if not cfg['stage1']['frozen'] or cfg['scheduler']['step_size'] >= cfg['epochs']:
            raise ValueError('Frozen Stage1 and effective within-run LR decay required')
    elif not cfg.get('diagnostic_smoke', False) and (cfg['epochs'] != 80 or stages != [20, 30, 30]
            or cfg['counts'] != {'train': 45453, 'val': 704, 'evaluation': 1226}):
        raise ValueError('Reduced/nonstandard acceptance protocols must explicitly set diagnostic_smoke=true')
    test_selection = cfg.get('data_protocol') == 'cleargrasp256'
    if test_selection:
        if (cfg.get('validation_mode') != 'real_val_test_selection' or cfg.get('selection_policy') != 'real_val_best'
                or cfg.get('test_set_selection') is not True or cfg['curriculum']['selection_start_epoch'] != 1
                or cfg.get('counts', {}).get('val', 0) != 0):
            raise ValueError('ClearGrasp requires explicit real-val test-set selection, all epochs, and no synthetic-val')
    elif cfg['curriculum']['selection_start_epoch'] != sum(stages[:2]) + 1:
        raise ValueError('Select only the final pure-P curriculum stage')
    if transfer_protocol and cfg.get('selection_metric') != 'transparent_mae_m':
        raise ValueError('256 selection requires transparent MAE')
    if sum(stages) != cfg['epochs']:
        raise ValueError('Curriculum stage lengths must sum to total epochs')
    if cfg['optimizer'] != 'Adam' or cfg['effective_global_batch'] != world * cfg['batch_per_gpu'] * cfg['gradient_accumulation_steps']:
        raise ValueError('Optimizer or effective batch protocol changed')
    model = cfg['model']
    if model.get('dense_allow_tf32') is not False or model['spvnas'].get('allow_tf32') is not False:
        raise ValueError('Both dense and sparse TF32 must remain disabled')
    if not model.get('rgb_pretrained') or not model['spvnas'].get('pretrained'):
        raise ValueError('Fresh training requires both authenticated pretrained backbones')
    return CurriculumSchedule(cfg['curriculum']['steps_per_epoch'], tuple(stages))


def training_contract(cfg, args, world, cache_info):
    root = Path(__file__).resolve().parents[1]
    sources = source_hashes()
    for relative in ('transparent_depth/train_stage2.py', 'transparent_depth/training_utils.py', 'transparent_depth/training_cache.py', 'transparent_depth/prepare_stage2_data.py',
                     'v2_lidf/data.py', 'v2_lidf/common.py', 'v2_lidf/curriculum.py', 'v2_lidf/distributed.py'):
        sources[relative] = file_sha(root / relative)
    if cfg.get('data_protocol') in TRANSFER_PROTOCOLS:
        for relative in ('v2_lidf/data_protocol.py','transparent_depth/evaluate256.py','transparent_depth/prepare_transfer_data.py'):
            sources[relative] = file_sha(root/relative)
    if cfg.get('data_protocol') == 'cleargrasp256':
        sources['transparent_depth/prepare_cleargrasp256.py'] = file_sha(root/'transparent_depth/prepare_cleargrasp256.py')
    return {'schema': 'transparent_depth.stage2.training.v1', 'config': cfg,
            'world_size': world, 'sources': sources,
            'input_manifest_sha256': file_sha(args.input_manifest),
            'label_manifest_sha256': file_sha(args.label_manifest),
            'stage1_complete_sha256': cache_info['stage1_complete_sha256'],
            'labels_complete_sha256': cache_info['labels_complete_sha256'],
            'ground_truth_access_during_training': True,
            'transparent_gt_depth_as_input': False,
            'pure_p_validation_gt_or_mask_used_as_input': False,
            'phase_matched_validation_gt_or_mask_used_as_input': False if cfg.get('phase_matched_validation') is False else 'alpha>0',
            'test_set_selection':cfg.get('test_set_selection',False),
            'selection_policy':cfg.get('selection_policy','internal_val'),
            'historical_trajectory_equivalence_claimed': False}


def broadcast_rank0(operation, rank, world):
    """Only rank zero performs filesystem decisions; late workers receive them."""
    result = [None]
    if rank == 0:
        try:
            result[0] = {'ok': True, 'value': operation()}
        except Exception as error:
            result[0] = {'ok': False, 'error': type(error).__name__ + ': ' + str(error)}
    if world > 1:
        dist.broadcast_object_list(result, src=0)
    if not result[0] or not result[0]['ok']:
        raise RuntimeError((result[0] or {}).get('error', 'Missing rank-zero decision'))
    return result[0]['value']


def establish_contract(output, contract, resume, rank, world):
    path = Path(output) / 'contract.json'
    def decide():
        if path.exists():
            if not resume or json.loads(path.read_text()) != contract:
                raise ValueError('Existing run requires --resume and identical inputs, source, config, world size')
        elif resume:
            raise FileNotFoundError('No training contract to resume')
        else:
            if any(p.name != '.training.lock' for p in Path(output).iterdir()):
                raise FileExistsError('Fresh training requires an empty output directory')
            atomic_json(path, contract)
        return True
    broadcast_rank0(decide, rank, world)
    if json.loads(path.read_text()) != contract:
        raise ValueError('Training contract changed after rank-zero registration')


def run(args):
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    cfg = json.loads(Path(args.config).read_text())
    cfg = resolve_manifest_counts(cfg, args.input_manifest, world)
    schedule = validate_training_config(cfg, world)
    smoke = bool(cfg.get('diagnostic_smoke', False))
    test_selection = cfg.get('selection_policy') == 'real_val_best'
    stop = getattr(args, 'stop_after_epoch', None)
    if stop is not None and (not smoke or not 0 < stop < cfg['epochs']):
        raise ValueError('Controlled stop is permitted only before completion of diagnostic smoke')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A process lifetime flock rejects another launcher without signalling it.
    lock = None
    if rank == 0:
        lock = (output / '.training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    cache_info = broadcast_rank0(lambda: validate_training_caches(cfg, args.input_manifest,
        args.label_manifest, args.stage1_root, args.label_root), rank, world)
    contract = training_contract(cfg, args, world, cache_info)
    contract_path = output / 'contract.json'
    last = output / 'last.pt'
    establish_contract(output, contract, args.resume, rank, world)
    train_set = AuthenticatedTrainingDataset(args.input_manifest, args.label_manifest, args.stage1_root,
        args.label_root, role='train', include_target=True, curriculum=True,
        label_sha256=cache_info['label_file_sha256'])
    selection_role = 'evaluation' if test_selection else 'val'
    val_set = V2LIDFDataset(args.input_manifest, None, args.stage1_root, None,
        role=selection_role, include_target=False)
    if len(train_set) != cfg['counts']['train'] or len(val_set) != cfg['counts'][selection_role]:
        raise ValueError('Training/validation roster differs from declared counts')
    label_rows = {row['cache_id']: row for row in read_jsonl(args.label_manifest) if row['role'] == selection_role}
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True,
                                 seed=cfg['seed'], drop_last=False)
    loader = DataLoader(train_set, batch_size=cfg['batch_per_gpu'], sampler=sampler,
        num_workers=cfg['workers_per_gpu'], worker_init_fn=worker_seed,
        pin_memory=True, persistent_workers=False)
    accumulation = cfg['gradient_accumulation_steps']
    if math.ceil(len(loader) / accumulation) != schedule.steps_per_epoch or sampler.total_size != cfg['sampler_protocol']['presentations_per_epoch']:
        raise ValueError('Actual loader coverage/update count differs from schedule')
    seed_all(cfg['seed'] + rank)
    model = build_stage2(cfg['model'], device, initialization=not args.resume,
                         assets_root=args.assets_root, audit_dir=output / 'initialization' / ('rank%d' % rank))
    if rank == 0:
        atomic_json(output/'runtime.json',model.runtime_audit)
    if world > 1:
        from v2_lidf.spvnas_fusion import convert_sparse_sync_batchnorm
        model = convert_sparse_sync_batchnorm(model)
    optimizer = torch.optim.Adam(parameter_groups(model, cfg))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **cfg['scheduler'])
    start, updates, best, saved = 0, 0, math.inf, None
    if args.resume:
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['contract'] != contract or len(saved['rng']) != world:
            raise ValueError('Committed checkpoint identity/topology differs')
        if cfg.get('data_protocol') in TRANSFER_PROTOCOLS:
            for name in ('torch','torchvision','torch_scatter','torchsparse_commit','torch_cuda'):
                if saved.get('runtime',{}).get(name) != model.runtime_audit[name]:
                    raise ValueError('Resume runtime ABI differs: '+name)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, updates = saved['epoch'] + 1, saved['updates']
        best = saved['best_score'] if saved['best_score'] is not None else math.inf
        if updates != start * schedule.steps_per_epoch or scheduler.last_epoch != start or saved['curriculum_next_alpha'] != schedule.alpha(updates):
            raise ValueError('Resume schedule differs from the committed optimizer history')
        if rank == 0:
            repair_selected_checkpoint(output, saved)
            repair_epoch_history(output, saved)
            repair_epoch_archive(output, saved)
    wrapped = model
    if world > 1:
        wrapped, audit = create_single_bucket_ddp(model, cfg['distributed'], [local_rank])
        if rank == 0:
            atomic_json(output / 'ddp.json', audit)
    if saved is not None:
        # Constructor/DDP broadcasts must not consume resumed training RNG.
        restore_rng(saved['rng'][rank])
        assert_state_tree_equal(saved['model'], model.state_dict(), 'model')
        assert_state_tree_equal(saved['optimizer'], optimizer.state_dict(), 'optimizer')
        assert_state_tree_equal(saved['scheduler'], scheduler.state_dict(), 'scheduler')
        assert_state_tree_equal(saved['rng'][rank], checkpoint_rng(), 'rng')
        atomic_json(output / ('resume_verification_rank%d.json' % rank), {
            'status': 'passed', 'rank': rank, 'updates': updates, 'start_epoch': start,
            'model_exact': True, 'optimizer_exact': True, 'scheduler_exact': True, 'rng_exact': True})
    for epoch in range(start, cfg['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        begun = time.monotonic()
        sums = torch.zeros(5, dtype=torch.float64, device=device)
        alpha_values = []
        initial_updates = updates
        for step, raw in enumerate(loader):
            batch = transfer(raw, device)
            alpha = schedule.alpha(updates)
            if step % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
                alpha_values.append(alpha)
            predicted = mix_background_depth(batch['depth_input'], batch['depth_gt'],
                batch['gt_valid'], batch['transparent_mask'], alpha)
            window = accumulation_window_size(step, len(loader), accumulation)
            commit = (step + 1) % accumulation == 0 or step + 1 == len(loader)
            with nullcontext() if world == 1 or commit else wrapped.no_sync():
                result = wrapped(batch['rgb'], predicted, batch['intrinsics'],
                    depth_gt=batch['depth_gt'], gt_valid=batch['gt_valid'], epoch=epoch)
                if smoke and not bool(torch.isfinite(result['depth']).all()):
                    raise FloatingPointError('Nonfinite smoke prediction')
                loss = result['losses']['total']
                healthy = torch.tensor(int(bool(torch.isfinite(loss))), device=device)
                if world > 1:
                    dist.all_reduce(healthy, op=dist.ReduceOp.MIN)
                if not bool(healthy):
                    raise FloatingPointError('Nonfinite training loss')
                (loss / window).backward()
            if commit:
                gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                healthy = torch.stack(gradients).all().int() if gradients else torch.zeros((), dtype=torch.int32, device=device)
                if world > 1:
                    dist.all_reduce(healthy, op=dist.ReduceOp.MIN)
                if not bool(healthy):
                    raise FloatingPointError('Nonfinite/missing training gradients')
                before = ({group['name']: [p.detach().cpu().clone() for p in group['params']]
                           for group in optimizer.param_groups} if smoke else None)
                optimizer.step()
                updates += 1
                if smoke:
                    changes = {group['name']: max(float((p.detach().cpu() - prior).abs().max())
                        for p, prior in zip(group['params'], before[group['name']]))
                        for group in optimizer.param_groups}
                    if not all(math.isfinite(v) and v > 0 for v in changes.values()):
                        raise FloatingPointError('Smoke requires a real finite parameter change in every optimizer group')
                    append_json(output / ('optimizer_changes_rank%d.jsonl' % rank),
                        {'updates': updates, 'alpha': alpha, 'loss_finite': True, 'gradients_finite': True,
                         'output_finite': True,
                         'parameter_group_max_abs_change': changes})
            count = batch['rgb'].shape[0]
            for i, key in enumerate(('total', 'position', 'probability', 'normal')):
                sums[i] += result['losses'][key].detach().double() * count
            sums[4] += count
            if rank == 0 and commit and (updates <= 3 or updates % 25 == 0):
                record = {'epoch': epoch + 1, 'microstep': step + 1, 'updates': updates,
                          'input_alpha': alpha, 'loss': float(loss.detach())}
                emit_progress(record, output / 'progress.json')
        if world > 1:
            dist.all_reduce(sums)
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
        if int(sums[4]) != sampler.total_size or updates - initial_updates != schedule.steps_per_epoch:
            raise RuntimeError('Incomplete epoch coverage')
        train_seconds = time.monotonic() - begun
        val = validate(model, val_set, label_rows, device, rank, world, alpha=0.0,
            selection_policy='real_val_best' if test_selection else 'internal_val')
        matched = (dict(val) if cfg.get('phase_matched_validation') is False or alpha_values[-1] == 0.0
            else validate(model, val_set, label_rows, device, rank, world, alpha=alpha_values[-1]))
        eligible = epoch + 1 >= cfg['curriculum']['selection_start_epoch']
        metric = cfg.get('selection_metric', 'abs_rel')
        selected = eligible and val[metric] < best
        if selected:
            best = val[metric]
        scheduler.step()
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, checkpoint_rng())
        else:
            states[0] = checkpoint_rng()
        if rank == 0:
            record = {'epoch': epoch + 1, 'updates': updates, 'smoke': smoke,
                'train_presentations': int(sums[4]), 'train_unique_expected': len(train_set),
                'train_seconds': train_seconds, 'validation': val,
                'validation_curriculum': matched, 'selected': selected,
                'selection_eligible': eligible, 'input_alpha_first': alpha_values[0],
                'input_alpha_last': alpha_values[-1],
                'losses': {key: float(sums[i] / sums[4]) for i, key in enumerate(('total', 'position', 'probability', 'normal'))}}
            saved = {'model': model.state_dict(), 'model_config': cfg['model'],
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'updates': updates, 'best_score': best if math.isfinite(best) else None,
                'validation': val, 'validation_curriculum': matched, 'rng': states,
                'contract': contract, 'epoch_record': record,
                'selection_metric': metric,
                'runtime': model.runtime_audit,
                'curriculum_next_alpha': schedule.alpha(updates)}
            atomic_save(last, saved)
            if selected:
                atomic_save(output / 'best.pt', saved)
            repair_epoch_history(output, saved)
            repair_epoch_archive(output, saved)
            emit_progress(record, output / 'progress.json')
        if world > 1:
            dist.barrier()
        if stop == epoch + 1:
            if rank == 0:
                atomic_json(output / 'smoke.interrupted.json', {'status': 'controlled_epoch_stop',
                    'smoke': True, 'epoch': epoch + 1, 'updates': updates, 'last_sha256': file_sha(last)})
            if world > 1:
                dist.barrier()
                dist.destroy_process_group()
            if lock:
                lock.close()
            return
    if rank == 0:
        records = read_jsonl(output / 'epochs.jsonl')
        receipts = read_jsonl(output / 'epoch_checkpoints.jsonl')
        wanted = list(range(1, cfg['epochs'] + 1))
        if [r['epoch'] for r in records] != wanted or [r['epoch'] for r in receipts] != wanted or updates != schedule.total_steps or not math.isfinite(best):
            raise RuntimeError('Cannot seal incomplete training')
        atomic_json(output / ('smoke.complete.json' if smoke else 'complete.json'), {'status': 'complete',
            'smoke': smoke, 'epochs': cfg['epochs'],
            'updates': updates, 'best_score': best, 'last_sha256': file_sha(last),
            'delivery_checkpoint': 'best.pt',
            'delivery_sha256': file_sha(output / 'best.pt'),
            'best_sha256': file_sha(output / 'best.pt'), 'contract_sha256': file_sha(contract_path),
            'finished_unix': time.time(), 'historical_trajectory_equivalence_claimed': False})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if lock:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'input-manifest', 'label-manifest', 'stage1-root', 'label-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--assets-root', help='Explicit verified official initialization assets; required for fresh training')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, help='Diagnostic smoke only: stop after a committed epoch')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
