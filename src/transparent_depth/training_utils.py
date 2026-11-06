"""Preserved training helpers; numerical bodies copied from the production run.
Epoch receipt paths alone use portable relative names."""
import os
import json
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from v2_lidf.common import append_json, file_sha, read_jsonl, atomic_json
from v2_lidf.data import nearest_exact_resize
from .training_cache import load_authenticated_native_labels as load_native_labels
from v2_lidf.curriculum import mix_background_depth

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_seed(worker_id):
    value = torch.initial_seed() % (2 ** 32)
    np.random.seed(value)
    random.seed(value)

def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.%d' % os.getpid())
    torch.save(value, str(tmp))
    os.replace(str(tmp), str(path))

def repair_selected_checkpoint(output, saved):
    """Recover a crash after committing last.pt but before committing best.pt."""
    best_path = Path(output) / 'best.pt'
    if saved['best_score'] is None:
        if best_path.exists():
            raise RuntimeError('Selected checkpoint exists before any eligible epoch')
        return
    metric = saved.get('selection_metric', 'abs_rel')
    last_was_selected = saved.get('epoch_record', {}).get(
        'selected', saved['validation'][metric] == saved['best_score'])
    if last_was_selected:
        # last.pt contains the selected model itself, so it is authoritative.
        atomic_save(best_path, saved)
    else:
        best = torch.load(str(best_path), map_location='cpu', weights_only=False)
        if (best['contract'] != saved['contract']
                or best['validation'][metric] != saved['best_score']
                or best['epoch'] > saved['epoch']):
            raise RuntimeError('Selected checkpoint does not match committed training history')

def repair_epoch_history(output, saved):
    """An epoch is committed by last.pt; recover a missing trailing log write."""
    path = Path(output) / 'epochs.jsonl'
    rows = read_jsonl(path) if path.exists() else []
    wanted = saved['epoch_record']
    epochs = [row['epoch'] for row in rows]
    if len(set(epochs)) != len(epochs) or epochs != sorted(epochs):
        raise RuntimeError('Epoch history contains duplicate or unordered entries')
    if epochs and epochs[-1] > wanted['epoch']:
        raise RuntimeError('Epoch history is ahead of the committed checkpoint')
    if epochs and epochs[-1] == wanted['epoch']:
        if rows[-1] != wanted:
            raise RuntimeError('Epoch history differs from committed checkpoint')
    else:
        append_json(path, wanted)

def repair_epoch_archive(output, saved):
    """Retain the last committed epoch and recover its receipt after a crash.

    Atomic replacement of last.pt never changes an older hard link. Thus each
    epoch keeps a complete resumable checkpoint without another serialization.
    """
    output = Path(output)
    last = output / 'last.pt'
    epoch = saved['epoch'] + 1
    path = output / ('epoch%03d.pt' % epoch)
    digest = file_sha(last)
    if not path.exists():
        temporary = path.with_name(path.name + '.tmp.%d' % os.getpid())
        if temporary.exists():
            temporary.unlink()
        os.link(str(last), str(temporary))
        os.replace(str(temporary), str(path))
    elif file_sha(path) != digest:
        raise RuntimeError('Archived epoch differs from the committed checkpoint')
    record = saved['epoch_record']
    receipt = {'epoch': epoch, 'path': path.name, 'sha256': digest,
               'updates': saved['updates'],
               'validation_abs_rel': (saved['validation'] or {}).get('abs_rel'),
               'validation_curriculum_abs_rel': (saved['validation_curriculum'] or {}).get('abs_rel'),
               'input_alpha': record['input_alpha_last'],
               'selection_eligible': record['selection_eligible']}
    manifest = output / 'epoch_checkpoints.jsonl'
    rows = read_jsonl(manifest) if manifest.exists() else []
    epochs = [row['epoch'] for row in rows]
    if epochs != sorted(set(epochs)) or (epochs and epochs[-1] > epoch):
        raise RuntimeError('Epoch checkpoint receipt order is inconsistent')
    if epochs and epochs[-1] == epoch:
        if rows[-1] != receipt:
            raise RuntimeError('Epoch checkpoint receipt differs from committed checkpoint')
    else:
        if epochs and epochs[-1] != epoch - 1:
            raise RuntimeError('Epoch checkpoint receipt history has a gap')
        append_json(manifest, receipt)
    return receipt

def accumulation_window_size(micro_step, micro_steps, accumulation_steps):
    """Normalize a partial final window by its actual microbatch count."""
    if not (isinstance(accumulation_steps, int) and accumulation_steps > 0
            and isinstance(micro_steps, int) and micro_steps > 0
            and isinstance(micro_step, int) and 0 <= micro_step < micro_steps):
        raise ValueError('Invalid gradient accumulation window')
    start = (micro_step // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, micro_steps - start)

def parameter_groups(model, cfg):
    """Match reference LIDF/SPVNAS/adapters learning rates without omissions."""
    branch = getattr(model, 'spvnas_branch', None)
    factors = cfg['learning_rate_factors']
    branch_ids = {id(p) for p in branch.parameters()} if branch is not None else set()
    groups = [{'name': 'lidf', 'params': [p for p in model.parameters()
                if p.requires_grad and id(p) not in branch_ids],
               'lr': cfg['learning_rate'] * factors['lidf']}]
    if branch is not None:
        groups += [{'name': 'spvnas', 'params': [p for p in branch.backbone.parameters()
                                                if p.requires_grad],
                    'lr': cfg['learning_rate'] * factors['spvnas']},
                   {'name': 'adapters', 'params': [p for part in (branch.input_adapter, branch.fusion)
                                                  for p in part.parameters() if p.requires_grad],
                    'lr': cfg['learning_rate'] * factors['adapters']}]
    groups = [group for group in groups if group['params']]
    ids = [id(p) for group in groups for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Optimizer parameter groups overlap or omit trainable parameters')
    return groups

def checkpoint_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())

def transfer(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}

def validation_depth_input(predicted, labels=None, alpha=0.0):
    """Explicit label-conditioned validation; alpha zero never accesses labels."""
    if alpha == 0.0:
        return mix_background_depth(predicted, alpha=alpha)
    shape = tuple(predicted.shape[-2:])
    converted = {key: torch.as_tensor(nearest_exact_resize(labels[key], shape),
                  device=predicted.device, dtype=dtype)[None]
                 for key, dtype in (('target', predicted.dtype), ('valid', torch.bool),
                                    ('transparent', torch.bool))}
    return mix_background_depth(predicted, converted['target'], converted['valid'],
                                converted['transparent'], alpha)

@torch.no_grad()
def validate(model, dataset, label_rows, device, rank, world, limit=0, alpha=0.0, selection_policy="internal_val"):
    """Native-support validation, with explicit optional GT-background input.

    Selection uses only alpha=0. RNG, every module's training flag, and all BN
    buffers are restored even if evaluation fails. Test selection requires the
    explicit real_val_best policy; its scores are not held-out estimates.
    """
    if selection_policy not in {'internal_val','real_val_best'}:
        raise ValueError('Unknown checkpoint selection policy')
    test_selection = selection_policy == 'real_val_best'
    if test_selection and alpha != 0:
        raise ValueError('Test-set selection must use pure P')
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    transfer_protocol = bool(dataset.rows and dataset.rows[0].get('preprocessing'))
    if transfer_protocol and alpha != 0:
        raise ValueError('256 validation always uses pure P; label-conditioned validation forbidden')
    grid_records = []
    count = min(len(dataset), limit) if limit else len(dataset)
    if count <= 0:
        raise ValueError('Validation requires nonempty image coverage')
    if not isinstance(alpha, (float, int)) or isinstance(alpha, bool) or not 0 <= alpha <= 1:
        raise ValueError('Validation alpha must be finite and in [0, 1]')
    # Validation sampling must be identical across epochs and resume.
    state = checkpoint_rng()
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    started = time.monotonic()
    try:
        model.eval()
        seed_all(81373 + rank)
        for index in range(rank, count, world):
            item = dataset[index]
            row = dataset.rows[index]
            expected_role = 'evaluation' if test_selection else 'val'
            if row.get('role', 'val') != expected_role:
                raise ValueError('Selection role differs from explicit policy')
            if test_selection and (not transfer_protocol or row.get('dataset')!='cleargrasp'
                    or row.get('official_split')!='real-val'
                    or row.get('preprocessing',{}).get('schema')!='rgb_fusion_cleargrasp_fullfield256_v1'):
                raise ValueError('Only cleargrasp256 real-val may enter explicit test selection')
            if transfer_protocol:
                from v2_lidf.data_protocol import validate_partition
                validate_partition(row)
            labels = load_native_labels(label_rows[row['cache_id']]) if alpha else None
            predicted = item['depth_input'].to(device)
            depth_input = validation_depth_input(predicted, labels, alpha)
            out = model(item['rgb'][None].to(device), depth_input[None],
                        item['intrinsics'][None].to(device), compute_loss=False)
            if transfer_protocol:
                from .evaluate256 import score_item
                if not bool(out['query_mask'].all()):
                    raise RuntimeError('Validation must query all pixels')
                record, _ = score_item(row, item['depth_input'][0].numpy(),
                    out['depth'][0, 0].cpu().numpy(), label_rows[row['cache_id']])
                grid_records.append(record)
                continue
            # Pure-P predictions are complete before labels are opened.
            if labels is None:
                labels = load_native_labels(label_rows[row['cache_id']])
            target = torch.as_tensor(labels['target'], device=device, dtype=torch.float32)
            valid = torch.as_tensor(labels['valid'], device=device, dtype=torch.bool)
            depth = F.interpolate(out['depth'], size=target.shape, mode='bilinear', align_corners=False)[0, 0]
            if not bool(torch.isfinite(depth).all()) or not bool((depth > 0).all()):
                raise FloatingPointError('Non-finite/non-positive validation prediction: ' + row['cache_id'])
            if not bool(valid.any()):
                raise ValueError('Empty native validation GT: ' + row['cache_id'])
            error = (depth.double()[valid] - target.double()[valid]).abs()
            totals[0] += (error / target.double()[valid]).mean()
            totals[1] += error.mean() * 1000
            totals[2] += error.square().mean().sqrt() * 1000
            totals[3] += out['refined_mask'].float().mean().double()
            totals[4] += 1
        if world > 1:
            dist.all_reduce(totals)
    finally:
        for buffer, original in buffers:
            buffer.copy_(original)
        for module, training in modes:
            module.training = training
        restore_rng(state)
    if transfer_protocol:
        from .evaluate256 import aggregate, metric_metadata
        if world > 1:
            all_records = [None] * world
            dist.all_gather_object(all_records, grid_records)
            grid_records = [r for records in all_records for r in records]
        if len(grid_records) != count:
            raise ValueError('Incomplete 256 validation coverage')
        metrics = aggregate(grid_records)
        score = metrics['Stage2']['transparent']['mae_m']
        if score is None:
            raise ValueError('No valid transparent supervision in internal validation')
        return {'images':count, 'transparent_mae_m':score,
            'abs_rel':metrics['Stage2']['all']['abs_rel'],
            'mae_mm':metrics['Stage2']['all']['mae_m']*1000,
            'rmse_mm':metrics['Stage2']['all']['rmse_m']*1000,
            'metrics256':metrics, **metric_metadata(), 'seconds':time.monotonic()-started,
            'input_alpha':0.0,'gt_or_mask_used_as_input':False,
            'test_set_selection':test_selection,'selection_policy':selection_policy,
            'protocol':dataset.rows[0]['preprocessing']['schema'] + ':selection_val_pure_p_image_macro'}
    n = int(totals[4].item())
    if n != count:
        raise ValueError('Validation coverage mismatch: %s != %s' % (n, count))
    return {'images': n, 'abs_rel': float(totals[0] / n), 'mae_mm': float(totals[1] / n),
            'rmse_mm': float(totals[2] / n), 'refined_fraction': float(totals[3] / n),
            'seconds': time.monotonic() - started, 'input_alpha': float(alpha),
            'gt_or_mask_used_as_input': bool(alpha),
            'protocol': 'phase_matched_label_conditioned' if alpha else 'pure_p'}


def _telemetry_failure(error, operation, payload):
    """Best-effort diagnostics on local storage when shared logging fails."""
    message = json.dumps({'event': 'telemetry_io_error', 'operation': operation,
                          'error': str(error), 'pid': os.getpid(),
                          'rank': os.environ.get('RANK', '0'), 'record_json': payload}) + '\n'
    try:
        os.write(2, message.encode('utf-8'))
    except OSError:
        pass
    # Explicitly local /tmp: runtime TMPDIR may itself be on shared storage.
    descriptor = None
    try:
        path = '/tmp/transparent_depth_telemetry_%d_%d.jsonl' % (os.getuid(), os.getpid())
        descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.write(descriptor, message.encode('utf-8'))
        if operation == 'stdout':
            # Also protect the interpreter's final stdout flush and later
            # library prints from the same disconnected shared log descriptor.
            os.dup2(descriptor, 1)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def emit_progress(record, path=None):
    """Progress is observational; checkpoint/contract writes remain mandatory."""
    payload = json.dumps(record)
    try:
        print(payload, flush=True)
    except OSError as error:
        _telemetry_failure(error, 'stdout', payload)
    if path is not None:
        try:
            atomic_json(path, record)
        except OSError as error:
            _telemetry_failure(error, 'progress_json', payload)


def assert_state_tree_equal(expected, actual, name='state'):
    """Exact resume evidence for tensors, optimizer groups and RNG state."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or expected.dtype != actual.dtype or not torch.equal(expected.cpu(), actual.detach().cpu()):
            raise RuntimeError('Resume tensor differs: ' + name)
    elif isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or expected.dtype != actual.dtype or not np.array_equal(expected, actual):
            raise RuntimeError('Resume NumPy state differs: ' + name)
    elif isinstance(expected, dict):
        if set(expected) != set(actual):
            raise RuntimeError('Resume mapping differs: ' + name)
        for key in expected:
            assert_state_tree_equal(expected[key], actual[key], name + '/' + str(key))
    elif isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise RuntimeError('Resume sequence differs: ' + name)
        for i, (left, right) in enumerate(zip(expected, actual)):
            assert_state_tree_equal(left, right, name + '/' + str(i))
    elif expected != actual:
        raise RuntimeError('Resume scalar differs: ' + name)
