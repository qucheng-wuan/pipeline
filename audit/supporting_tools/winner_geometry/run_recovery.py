"""Explicitly launched recovery of only the eight geometry workers and merge.

No model execution, package install, original failure overwrite, independent
sweep rehash, or source/threshold modification is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)

def verify_ref(item):
    assert sha(item['path']) == item['sha256'], 'Changed evidence: '+item['path']

def make_plan():
    cpu = read(BASE/'cpu_parity.complete.json')
    assert cpu['status'] == 'passed' and cpu['byte_exact'] and cpu['frozen_sources_unchanged']
    for key in ('shim', 'checker', 'failure_evidence', 'frozen_sources',
                'complete_label_batches_vs_portable_inputfile', 'native_file_reference'):
        verify_ref(cpu[key])
    assert all(sha(path) == digest for path,digest in read(cpu['frozen_sources']['path']).items())
    failure = read(BASE/'failure_snapshot.json')
    for item in failure['artifacts']:
        verify_ref(item['original']); verify_ref(item['snapshot'])
    verify_ref(failure['selection']); verify_ref(failure['independent_verification'])
    selected = read(failure['selection']['path'])
    verified = read(failure['independent_verification']['path'])
    assert selected['status'] == 'complete' and selected['completed_epochs'] == 80
    assert verified['status'] == 'passed' and verified['predictions_verified'] == 98080
    original = read(BASE/'failure_snapshot/execution.json')
    assert len(original['workers']) == 8 and all(w['returncode'] == 1 for w in original['workers'])
    jobs = []
    for w in original['workers']:
        command = w['command']
        assert command[command.index('--phase')+1] == 'geometry'
        assert int(command[command.index('--rank')+1]) == w['rank']
        jobs.append({'rank':w['rank'], 'physical_gpu':w['physical_gpu'], 'command':command,
                     'log':str(BASE/'execution'/('geometry_rank%02d.log'%w['rank']))})
    assert sorted(w['physical_gpu'] for w in jobs) == list(range(8))
    merge = list(jobs[0]['command']); merge[merge.index('--phase')+1] = 'merge'
    output = Path(merge[merge.index('--output')+1])
    return {'schema_version':'winner_geometry_exr_recovery_plan_v1',
            'runner':ref(Path(__file__)), 'cpu_parity':ref(BASE/'cpu_parity.complete.json'),
            'failure_evidence':ref(BASE/'failure_snapshot.json'), 'shim':cpu['shim'],
            'selection':failure['selection'], 'independent_verification':failure['independent_verification'],
            'resolved_selection':ref(output/'resolved_selection.json'),
            'original_execution':ref(output/'execution.json'),
            'selected_epoch':selected['selected_epoch'], 'output':str(output),
            'environment_overrides':{'CUDA_DEVICE_ORDER':'PCI_BUS_ID', 'OMP_NUM_THREADS':'2',
                'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'2', 'PYTHONDONTWRITEBYTECODE':'1',
                'PYTHONPATH':str(BASE/'shim'), 'TD_OPENEXR_FILE_COMPAT':'1',
                'TD_OPENEXR_COMPAT_AUDIT_DIR':str(BASE/'execution/shim_usage')},
            'workers':jobs, 'merge_command':merge,
            'independent_sweep_verification_rerun':False, 'original_commands_changed':False,
            'shared_environment_modified':False, 'model_executed':False}

def execute(plan):
    directory = BASE/'execution'; directory.mkdir(exist_ok=False)
    record = {'schema_version':'winner_geometry_exr_recovery_execution_v1',
              'status':'running', 'plan':ref(BASE/'launch_plan.json'), 'started_unix':time.time(),
              'supervisor_pid':os.getpid(), 'workers':[]}
    active = []
    common = dict(os.environ, **plan['environment_overrides'])
    record['inherited_runtime_environment'] = {key:common.get(key) for key in
                ('CUDA_HOME','LD_LIBRARY_PATH','PATH','TMPDIR')}
    for job in plan['workers']:
        stream = open(job['log'],'w')
        env = dict(common, CUDA_VISIBLE_DEVICES=str(job['physical_gpu']))
        process = subprocess.Popen(job['command'], env=env, stdout=stream, stderr=subprocess.STDOUT)
        record['workers'].append(dict(job, pid=process.pid, cuda_visible_devices=env['CUDA_VISIBLE_DEVICES']))
        active.append((process,stream))
    atomic(directory/'execution.json',record)
    for index,(process,stream) in enumerate(active):
        record['workers'][index]['returncode'] = process.wait(); stream.close()
        record['workers'][index]['log_ref'] = ref(record['workers'][index]['log'])
        atomic(directory/'execution.json',record)
    if any(w['returncode'] != 0 for w in record['workers']):
        record.update(status='failed_geometry_workers',finished_unix=time.time())
        atomic(directory/'execution.json',record)
        raise RuntimeError('Geometry recovery failed; evidence retained')
    for w in record['workers']:
        usage_path=directory/'shim_usage'/('usage_pid%d.json'%w['pid'])
        usage=read(usage_path)
        assert usage['shim_sha256'] == plan['shim']['sha256'] and usage['openexr_version']=='3.2.4'
        assert usage['files_opened'] > 0 and usage['cuda_visible_devices']==str(w['physical_gpu'])
        w['shim_usage']=ref(usage_path)
    log=directory/'merge.log'
    with log.open('w') as stream:
        result=subprocess.run(plan['merge_command'],env=dict(common,CUDA_VISIBLE_DEVICES=''),
                              stdout=stream,stderr=subprocess.STDOUT)
    record['merge']={'command':plan['merge_command'],'returncode':result.returncode,'log':ref(log)}
    record.update(status='merge_completed' if result.returncode==0 else 'failed_merge',finished_unix=time.time())
    atomic(directory/'execution.json',record)
    assert result.returncode==0, 'Merge failed; evidence retained'
    verify_ref(plan['resolved_selection']); verify_ref(plan['original_execution'])
    marker_path=Path(plan['output'])/'evaluation.complete.json'; marker=read(marker_path)
    assert marker['status']=='complete' and marker['images']==1226 and marker['test_images']==521
    assert marker['selected_epoch']==plan['selected_epoch'] and marker['model_executed'] is False
    assert marker['sweep_mae_absolute_difference_mm']<=marker['sweep_mae_tolerance_mm']==2e-5
    assert all(sha(path)==digest for path,digest in read(BASE/'frozen_sources_before.json').items())
    complete={'schema_version':'winner_geometry_exr_recovery_complete_v1','status':'passed_recovered_geometry',
              'reason':'Frozen label loader required OpenEXR.File; 3.2.4 only exposes InputFile',
              'plan':ref(BASE/'launch_plan.json'),'cpu_parity':plan['cpu_parity'],
              'failure_evidence':plan['failure_evidence'],'shim':plan['shim'],
              'actual_execution':ref(directory/'execution.json'),'evaluation':ref(marker_path),
              'resolved_selection':plan['resolved_selection'],'independent_verification':plan['independent_verification'],
              'selected_epoch':marker['selected_epoch'],'images':1226,'test_images':521,
              'frozen_sources_unchanged':True,'shared_environment_modified':False,
              'original_failure_artifacts_unchanged':True,'model_executed':False,
              'independent_sweep_verification_rerun':False,'finished_unix':time.time()}
    atomic(BASE/'complete.json',complete)
    print(json.dumps(complete),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Explicit root-authorized eight-GPU metrics recovery')
    args=parser.parse_args()
    plan=make_plan()
    saved=BASE/'launch_plan.json'
    if saved.exists():
        assert read(saved)==plan, 'Plan/source/evidence changed since review'
    else:
        atomic(saved,plan)
    if args.execute:
        execute(plan)
    else:
        print(json.dumps({'status':'plan_only_no_gpu_started','plan':ref(saved),'runner':plan['runner']}),flush=True)
