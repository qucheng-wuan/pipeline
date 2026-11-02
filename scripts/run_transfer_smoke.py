"""Single-GPU, real-cache acceptance. Never creates/fakes Stage1 predictions.

Use only the four-frame ClearGrasp or six-frame transCG smoke roster. Formal 60-epoch config is read
for boundary assertions and is never passed to the training launcher.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from transparent_depth.train_stage2 import resolve_manifest_counts, validate_training_config
from transparent_depth.training_cache import validate_training_caches
from v2_lidf.curriculum import CurriculumSchedule
from v2_lidf.data import atomic_json, read_jsonl


def boundary_checks(steps, formal):
    schedule=CurriculumSchedule(steps,tuple(formal['curriculum']['stage_epochs']))
    if formal['epochs']!=60 or schedule.stage_epochs!=(20,20,20):
        raise ValueError('Formal config must retain 60 epochs and 20/20/20')
    a,b,_=schedule.stage_steps
    points={'epoch1_first':0,'epoch20_last':a-1,'epoch21_first':a,
            'transition_mid':a+(b-1)//2,'epoch40_last':a+b-1,
            'epoch41_first':a+b,'epoch60_last':schedule.total_steps-1}
    values={k:schedule.alpha(v) for k,v in points.items()}
    assert values['epoch20_last']==values['epoch21_first']==1
    assert 0<values['transition_mid']<1
    assert values['epoch40_last']==values['epoch41_first']==values['epoch60_last']==0
    return {'steps_per_epoch':steps,'alpha':values,'update_indices':points}


def run(args):
    prepared=Path(args.prepared_root).resolve();output=Path(args.output).resolve()
    im=prepared/'manifests/input_manifest.jsonl';lm=prepared/'manifests/label_manifest.jsonl'
    cfg=resolve_manifest_counts(json.loads(Path(args.config).read_text()),im,1)
    test_selection = cfg['data_protocol']=='cleargrasp256'
    expected_counts = {'train':2,'val':0 if test_selection else 2,'evaluation':2}
    if not cfg.get('diagnostic_smoke') or cfg['epochs']!=4 or cfg['counts']!=expected_counts:
        raise ValueError('Smoke requires the independent four-epoch diagnostic config and exact configured sample counts')
    validate_training_config(cfg,1)
    validate_training_caches(cfg,im,lm,prepared/'stage1',prepared/'labels')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Smoke output must be new; retain interrupted evidence')
    output.mkdir(parents=True,exist_ok=True)
    formal=json.loads(Path(args.formal_config).read_text())
    if formal['data_protocol'] != cfg['data_protocol'] or formal['preprocessing'] != cfg['preprocessing']:
        raise ValueError('Smoke/formal experiment protocol mismatch')
    atomic_json(output/'formal_schedule_boundaries.json',boundary_checks(cfg['curriculum']['steps_per_epoch'],formal))
    train_output=output/'training'
    command=[sys.executable,'-m','transparent_depth.train_stage2','--config',str(Path(args.config).resolve()),
        '--input-manifest',str(im),'--label-manifest',str(lm),'--stage1-root',str(prepared/'stage1'),
        '--label-root',str(prepared/'labels'),'--assets-root',str(Path(args.assets_root).resolve()),'--output',str(train_output)]
    def launch(argv):
        print('Running: '+' '.join(argv),flush=True)
        subprocess.run(argv,check=True,cwd=ROOT)
    launch(command+['--stop-after-epoch','2'])
    launch(command+['--resume'])
    resume=json.loads((train_output/'resume_verification_rank0.json').read_text())
    if not all(resume[k] for k in ('model_exact','optimizer_exact','scheduler_exact','rng_exact')):
        raise ValueError('Resume did not exactly restore model/optimizer/scheduler/RNG')
    changes=read_jsonl(train_output/'optimizer_changes_rank0.jsonl')
    if len(changes)!=8 or not {0.,1.} <= {r['alpha'] for r in changes} or not any(0<r['alpha']<1 for r in changes):
        raise ValueError('Smoke did not cover all curriculum phases')
    for row in changes:
        if not all(row[k] for k in ('loss_finite','gradients_finite','output_finite')) or not all(
                math.isfinite(v) and v>0 for v in row['parameter_group_max_abs_change'].values()):
            raise ValueError('Missing finite, real parameter update')
    complete=json.loads((train_output/'smoke.complete.json').read_text())
    if complete['delivery_checkpoint']!='best.pt' or not (train_output/'best.pt').is_file():
        raise ValueError('Smoke must deliver a metric-selected best.pt')
    evaluations = [('cleargrasp','evaluation')] if test_selection else [('transcg','val'),('cleargrasp','evaluation')]
    for dataset,role in evaluations:
        launch([sys.executable,'-m','transparent_depth.evaluate256','--checkpoint',str(train_output/'best.pt'),
            '--input-manifest',str(im),'--label-manifest',str(lm),'--stage1-root',str(prepared/'stage1'),
            '--dataset',dataset,'--role',role,'--output',str(output/(dataset+'-'+role)),'--smoke','--visualizations','6'])
    report={'status':'passed','diagnostic_smoke':True,'formal_training_started':False,
        'optimizer_updates':len(changes),'alpha':[r['alpha'] for r in changes],
        'resume':resume,'accuracy_claim':False,'real_sealed_stage1_required':True}
    atomic_json(output/'acceptance.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-root',required=True);p.add_argument('--assets-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(ROOT/'configs/transcg_stage2_smoke.json'))
    p.add_argument('--formal-config',default=str(ROOT/'configs/transcg_stage2_60.json'))
    run(p.parse_args())
