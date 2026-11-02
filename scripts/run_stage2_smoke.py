#!/usr/bin/env python3
"""Real-data training acceptance, gated on a sealed portable RGB Stage1 batch.

This does not train a publishable model. It performs three complete tiny epochs,
three optimizer updates per epoch, and an exact committed-epoch resume. It never
fabricates Stage1 receipts, signals another process, or chooses GPUs implicitly.
"""
import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_ref(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def authenticate_stage1_acceptance(marker, predictions):
    """Authenticate the explicit user-approved policy and actual fresh RGB batch."""
    marker = Path(marker)
    record = json.loads(marker.read_text())
    if (record.get('schema') != 'portable_stage1_validation_complete_v2'
            or record.get('status') != 'passed_user_policy'
            or record.get('numeric_tolerance_user_authorized') is not True
            or record.get('numeric_tolerance_automatically_relaxed') is not False):
        raise ValueError('Stage1 user-policy acceptance is not complete')
    for key, path in (('fresh_predictions', Path(predictions)),
                      ('fresh_predictions_seal', Path(predictions).parent/'complete.json'),
                      ('source_inventory', ROOT/'provenance/stage1_runtime_sources.json'),
                      ('asset_manifest', ROOT/'provenance/stage1_assets.json')):
        reference = record[key] if key.startswith('fresh_') else record['bindings'][key]
        if reference != file_ref(path):
            raise ValueError('Stage1 acceptance source or fresh prediction binding changed: '+key)
    comparisons = record.get('policy_comparisons', [])
    if len(comparisons) != 4 or len({r['path'] for r in comparisons}) != 4:
        raise ValueError('Stage1 acceptance requires three cases and relocation')
    for reference in comparisons:
        if reference != file_ref(reference['path']):
            raise ValueError('Stage1 policy comparison changed')
        comparison = json.loads(Path(reference['path']).read_text())
        if (comparison.get('status') != 'passed_user_policy'
                or comparison.get('numeric_tolerance_user_authorized') is not True
                or comparison.get('numeric_tolerance_automatically_relaxed') is not False
                or not comparison.get('fields')
                or not all(value.get('accepted') is True for value in comparison['fields'].values())):
            raise ValueError('A required Stage1 policy comparison did not pass')
    return file_ref(marker)


def smoke_config(base, rows, world):
    from transparent_depth.prepare_stage2_data import read_roles
    cfg = copy.deepcopy(base)
    counts = Counter(role for row in rows for role in read_roles(row))
    if not 8 <= counts['train'] <= 16 or not 1 <= counts['val'] <= 4:
        raise ValueError('Smoke requires 8–16 genuine training samples and 1–4 validation samples')
    batch = 2
    microsteps = math.ceil(math.ceil(counts['train'] / world) / batch)
    candidates = [a for a in range(1, microsteps + 1) if math.ceil(microsteps / a) == 3]
    if not candidates:
        raise ValueError('This sample/world combination cannot provide exactly three optimizer updates per epoch')
    # Prefer a partial last accumulation window to exercise its actual divisor.
    accumulation = next((a for a in candidates if microsteps % a), candidates[0])
    presentations = math.ceil(counts['train'] / world) * world
    cfg.update(diagnostic_smoke=True, epochs=3, expected_world_size=world,
        batch_per_gpu=batch, gradient_accumulation_steps=accumulation,
        effective_global_batch=world*batch*accumulation, workers_per_gpu=0,
        counts={k: counts[k] for k in ('train','val','evaluation')},
        selection='Diagnostic only: pure-P validation selection in epoch 3; no quality/80-epoch claim',
        training_reproduction='Real portable Stage1 + fresh initialization smoke only; three tiny epochs and committed-epoch resume')
    cfg['curriculum'].update(stage_epochs=[1,1,1], steps_per_epoch=3,
        selection_start_epoch=3, transition='three middle-stage updates with alpha 1,.5,0')
    cfg['sampler_protocol'].update(unique_per_epoch=counts['train'], presentations_per_epoch=presentations,
        padding_duplicates_per_epoch=presentations-counts['train'])
    cfg['checkpoint_policy']='Diagnostic epoch001..003, last and pure-P best; smoke.complete.json only'
    return cfg


def wait_for_predictions(predictions, timeout, interval, acceptance=None):
    predictions = Path(predictions)
    begun = time.monotonic()
    last_notice = -math.inf
    while (not predictions.is_file() or not (predictions.parent / 'complete.json').is_file()
           or (acceptance and not Path(acceptance).is_file())):
        elapsed = time.monotonic() - begun
        if elapsed >= timeout:
            raise TimeoutError('Real portable Stage1 prediction seal did not arrive; no training was started')
        if elapsed - last_notice >= 60:
            print(json.dumps({'status':'waiting_for_real_portable_stage1', 'seconds':int(elapsed),
                              'predictions':str(predictions)}), flush=True)
            last_notice = elapsed
        time.sleep(min(interval, max(.01, timeout-elapsed)))


def run_command(command, environment, log):
    print(json.dumps({'command':command,'log':str(log)}), flush=True)
    with Path(log).open('ab') as stream:
        subprocess.run(command, env=environment, cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return {'command': command, 'exit_code': 0, 'log': file_ref(log)}


def verify(output, world):
    from transparent_depth.stage2 import file_sha
    run = output / 'training'
    complete = json.loads((run/'smoke.complete.json').read_text())
    interrupted = json.loads((run/'smoke.interrupted.json').read_text())
    epochs = read_rows(run/'epochs.jsonl')
    if (complete.get('smoke') is not True or complete['epochs'] != 3 or complete['updates'] != 9
            or (run/'complete.json').exists() or interrupted['updates'] != 3
            or [r['epoch'] for r in epochs] != [1,2,3]
            or [r['updates'] for r in epochs] != [3,6,9]
            or any(not r['smoke'] for r in epochs)
            or any(r['validation']['input_alpha'] != 0 or r['validation']['gt_or_mask_used_as_input'] for r in epochs)
            or any(r['train_presentations'] <= 0 or r['validation']['images'] <= 0 for r in epochs)
            or any(r['selected'] for r in epochs[:2]) or not epochs[2]['selection_eligible']):
        raise ValueError('Incomplete or incorrectly labelled diagnostic run')
    expected_alphas = [1.,1.,1.,1.,.5,0.,0.,0.,0.]
    ranks = []
    for rank in range(world):
        resume_path=run/('resume_verification_rank%d.json'%rank)
        resume=json.loads(resume_path.read_text())
        changes_path=run/('optimizer_changes_rank%d.jsonl'%rank)
        changes=read_rows(changes_path)
        if (resume.get('status') != 'passed' or resume['updates'] != 3
                or not all(resume[k] for k in ('model_exact','optimizer_exact','scheduler_exact','rng_exact'))
                or [r['updates'] for r in changes] != list(range(1,10))
                or [r['alpha'] for r in changes] != expected_alphas
                or any(not r['loss_finite'] or not r['gradients_finite'] for r in changes)):
            raise ValueError('Missing exact state restoration, schedule, or finite-gradient evidence')
        groups=set(changes[0]['parameter_group_max_abs_change'])
        maxima={g:max(r['parameter_group_max_abs_change'][g] for r in changes) for g in groups}
        if groups != {'lidf','spvnas','adapters'} or any(not math.isfinite(v) or v <= 0 for v in maxima.values()):
            raise ValueError('A trainable parameter group did not actually update')
        ranks.append({'rank':rank,'resume_sha256':file_sha(resume_path),
                      'optimizer_changes_sha256':file_sha(changes_path),'group_max_abs_change':maxima})
    return {'status':'passed','diagnostic_smoke':True,'formal_training_claimed':False,
        'epochs':3,'updates':9,'input_alphas':expected_alphas,'world_size':world,
        'stage1_source':'actual sealed portable RGB Stage1 predictions',
        'training_complete_sha256':file_sha(run/'smoke.complete.json'),
        'config_sha256':file_sha(output/'smoke.config.json'),'ranks':ranks,
        'transparent_input_is_always_P':'enforced by unchanged tested curriculum mixer',
        'finished_unix':time.time()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input-roster','label-roster','predictions','assets-root','output','gpus'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--config',default=str(ROOT/'configs/stage2_train.json'))
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--wait-timeout',type=float,default=21600)
    p.add_argument('--poll-seconds',type=float,default=30)
    p.add_argument('--stage1-acceptance',help='Require and authenticate the explicit Stage1 user-policy completion marker')
    args=p.parse_args()
    devices=args.gpus.split(',')
    if not devices or len(devices)!=len(set(devices)) or any(not d.isdigit() for d in devices):
        raise ValueError('--gpus must explicitly list unique physical GPU indices')
    if not 0 < args.poll_seconds <= 60 or args.wait_timeout <= 0:
        raise ValueError('Use a positive timeout and poll interval no longer than 60 seconds')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.smoke_runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'training').exists():
            raise FileExistsError('Smoke runner output already contains a training attempt; choose a new output')
        wait_for_predictions(args.predictions,args.wait_timeout,args.poll_seconds,args.stage1_acceptance)
        acceptance = (authenticate_stage1_acceptance(args.stage1_acceptance,args.predictions)
                      if args.stage1_acceptance else None)
        from transparent_depth.prepare_stage2_data import prepare, read_table
        from v2_lidf.common import atomic_json
        cfg=smoke_config(json.loads(Path(args.config).read_text()),read_table(args.input_roster),len(devices))
        # Authentication happens here before any GPU subprocess; never synthesize producer evidence.
        prepare(args.input_roster,args.label_roster,args.predictions,out/'prepared')
        atomic_json(out/'smoke.config.json',cfg)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
        env['PATH']=str(Path(args.python).resolve().parent)+os.pathsep+env.get('PATH','')
        env.setdefault('OMP_NUM_THREADS','2');env.setdefault('MAX_JOBS','2')
        env.setdefault('TORCH_EXTENSIONS_DIR',str(out/'extensions'))
        cmd=[args.python,'-m','torch.distributed.run','--standalone','--nproc-per-node='+str(len(devices)),
            '--log-dir',str(out/'torchrun'),'--redirects','3','--tee','3','-m','transparent_depth.train_stage2',
            '--config',str(out/'smoke.config.json'),
            '--input-manifest',str(out/'prepared/manifests/input_manifest.jsonl'),
            '--label-manifest',str(out/'prepared/manifests/label_manifest.jsonl'),
            '--stage1-root',str(out/'prepared/stage1'),'--label-root',str(out/'prepared/labels'),
            '--assets-root',str(Path(args.assets_root).resolve()),'--output',str(out/'training')]
        executions = [run_command(cmd+['--stop-after-epoch','1'],env,out/'fresh.log')]
        executions.append(run_command(cmd+['--resume'],env,out/'resume.log'))
        if acceptance and authenticate_stage1_acceptance(args.stage1_acceptance,args.predictions) != acceptance:
            raise ValueError('Stage1 acceptance changed during training smoke')
        receipt=verify(out,len(devices))
        receipt.update(runner_sha256=file_ref(__file__)['sha256'], executions=executions,
            stage1_acceptance=acceptance, source_predictions=file_ref(args.predictions),
            training_contract=file_ref(out/'training/contract.json'))
        atomic_json(out/'acceptance.complete.json',receipt)
        print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
