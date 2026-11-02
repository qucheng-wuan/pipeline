"""Coordinate the authorized eight-GPU historical checkpoint sweep.

Existing training processes are neither signalled nor stopped. Each child sees
one physical GPU; labels stay in its separate evaluator subprocess. A failed
worker blocks final selection but does not terminate other completed work.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/gluster_dcpu/hengxianli/fusion_v2_lidf_spvnas_20261030/continuation_gpu567_20261031/selection_full1226_transparent_mae_20261101')
sys.path.insert(0, str(PROJECT/'scripts'))
from sweep_checkpoints import atomic_json, read, ref


def main():
    if socket.gethostname().split('.')[0] != 'gpu07':
        raise ValueError('Registered historical sweep runs on gpu07')
    lock = (ROOT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    probe = read(ROOT/'probe_epoch063/complete.json')
    if (probe.get('e63_reference_parity',{}).get('status')!='passed'
            or probe['images']<8 or not probe['smoke']):
        raise ValueError('Complete real E63 parity probe must pass before the full sweep')
    sys.path.insert(0,'/home/hengxianli/FusionV2_LIDF_SPVNAS_GPU567_20261031/scripts')
    from run_full import environment,LIDF_PYTHON
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader'],text=True)
    cards=[tuple(p.strip() for p in line.split(',')) for line in inventory.splitlines()]
    ordered=sorted(cards,key=lambda x:tuple(int(p,16) for p in x[2].replace('.',':').split(':')))
    if [int(x[0]) for x in ordered]!=list(range(8)):
        raise ValueError('Physical GPU ordinal mapping differs from PCI_BUS_ID ordering')
    (ROOT/'logs').mkdir(exist_ok=True)
    workers=[]
    run={'schema_version':'full1226_sweep_eight_gpu_execution_v1','supervisor_pid':os.getpid(),
         'started_unix':time.time(),'contract':ref(ROOT/'contract.json'),'probe':ref(ROOT/'probe_epoch063/complete.json'),
         'scripts':{str(p):ref(p)['sha256'] for p in (Path(__file__),PROJECT/'scripts/sweep_checkpoints.py')},
         'gpu_inventory':cards,'existing_training_left_running':True,'workers':[]}
    for rank in range(8):
        env=environment(str(rank),lidf=True)
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1')
        command=[LIDF_PYTHON,'-u',str(PROJECT/'scripts/sweep_checkpoints.py'),'worker',
                 '--rank',str(rank),'--world','8','--wait-for-checkpoints']
        log=ROOT/'logs'/('worker%02d.log'%rank)
        stream=log.open('a')
        child=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
        workers.append((child,stream))
        run['workers'].append({'rank':rank,'physical_gpu':rank,'pid':child.pid,'command':command,
                               'log':str(log),'cuda_visible_devices':str(rank)})
    atomic_json(ROOT/'launch.json',run)
    pending=set(range(8))
    while pending:
        for rank in list(pending):
            code=workers[rank][0].poll()
            if code is not None:
                pending.remove(rank)
                workers[rank][1].close()
                run['workers'][rank].update(returncode=code,completion_observed_unix=time.time())
                atomic_json(ROOT/'execution_progress.json',run)
        if pending:
            time.sleep(10)
    run['finished_unix']=time.time()
    run['status']='workers_passed' if all(x['returncode']==0 for x in run['workers']) else 'worker_failure'
    atomic_json(ROOT/'execution.complete.json',run)
    if run['status']!='workers_passed':
        raise SystemExit('Worker failure: no all-80 selection generated; inspect individual logs')
    env=environment('',lidf=True)
    env.update(PYTHONDONTWRITEBYTECODE='1')
    command=[LIDF_PYTHON,'-u',str(PROJECT/'scripts/sweep_checkpoints.py'),'merge']
    with (ROOT/'logs/merge.log').open('a') as stream:
        result=subprocess.run(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT)
    run['merge']={'command':command,'returncode':result.returncode,'log':ref(ROOT/'logs/merge.log')}
    run['status']='complete_selection_requires_winner_geometry' if result.returncode==0 else 'merge_failure'
    if result.returncode==0:
        run['selection']=ref(ROOT/'selection.complete.json')
    atomic_json(ROOT/'execution.complete.json',run)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
