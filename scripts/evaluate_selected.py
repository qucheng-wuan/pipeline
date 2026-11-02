"""Full native geometry for the authenticated all-80, test-informed winner.

No model is imported or executed. The coordinator verifies the complete sweep
before spawning separate labelled evaluation processes. Paths are explicit;
portable installations can supply their label loader and metric source tree.
Original checkpoint contracts and the original sweep selection remain intact.
"""
import argparse
import copy
import csv
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import sweep_checkpoints as sweep

TIE_MM = 2e-5
METHODS = ('stage1_native', 'stage1_resized', 'stage1_lidf')
REGIONS = ('all', 'transparent', 'background')


def candidates_for_resolution(ranking, tolerance=TIE_MM):
    sweep.require(len(ranking)==80 and len({r['epoch'] for r in ranking})==80,
                  'Tie resolution requires all eighty candidates')
    ordered=sorted(ranking,key=lambda r:(r['transparent_mae_mm'],r['epoch']))
    for row in ordered:
        sweep.require(math.isfinite(row['transparent_mae_mm']), 'Nonfinite candidate score')
    if ordered[1]['transparent_mae_mm']-ordered[0]['transparent_mae_mm']>tolerance:
        return [ordered[0]['epoch']]
    return [r['epoch'] for r in ordered
            if r['transparent_mae_mm']-ordered[0]['transparent_mae_mm']<=tolerance]


def verify_score(actual, original, tolerance=TIE_MM):
    sweep.require(math.isfinite(actual) and math.isfinite(original)
                  and abs(actual-original)<=tolerance,
                  'Native winner MAE differs from sweep ranking beyond 2e-5 mm')


def verify_runtime_inputs(args):
    contract=sweep.read(args.selection_root/'contract.json')
    for path,key in ((args.input_manifest,'input_manifest'),
                     (args.label_manifest,'label_manifest'),
                     (args.stage1_root/'complete.json','stage1_seal')):
        sweep.require(sweep.sha(path)==contract[key]['sha256'], 'Final evaluation input identity differs: '+key)
    selection=sweep.read(args.selection_root/'selection.complete.json')
    verification=sweep.read(args.selection_root/'publication_verification.complete.json')
    sweep.require(verification['status']=='passed' and verification['epochs']==80
                  and verification['predictions_verified']==98080
                  and verification['selection']==sweep.ref(args.selection_root/'selection.complete.json'),
                  'Full sweep must pass independent publication verification first')
    sweep.require(selection['selection_policy']==sweep.POLICY,'Test-informed selection protocol differs')
    return contract, selection


def install_paths(args):
    sys.path.insert(0,str(args.production_src))
    sys.path.insert(0,str(args.metric_src))


def source_identity(args):
    loader_module=args.label_loader.split(':')[0]
    files=[Path(__file__).resolve(),Path(sweep.__file__).resolve(),
           args.metric_src/'native_fusion/geometry.py',args.metric_src/'native_fusion/evaluation.py',
           args.metric_src/'native_fusion/data.py',
           args.production_src/'v2_lidf/data.py']
    module_path=Path(*loader_module.split('.')).with_suffix('.py')
    loader_path=(args.metric_src/module_path if (args.metric_src/module_path).is_file()
                 else args.production_src/module_path)
    files.append(loader_path)
    return {str(path.resolve()):sweep.sha(path) for path in files}


def gpu_setup():
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):
        sweep.require(os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
                      and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1,
                      'Evaluation worker must see exactly one physical GPU')
        torch.cuda.set_device(0)
        return 'cuda:0'
    return 'cpu'


def load_batch(args, row, expected_assets, device):
    # Labels are opened only in resolve/geometry subprocesses, after the entire
    # sweep's complete prediction bytes have been independently authenticated.
    for item in expected_assets.values():
        sweep.require(sweep.sha(item['path'])==item['sha256'],'GT/calibration source differs from sealed baseline')
    module,name=args.label_loader.split(':')
    loader=getattr(importlib.import_module(module),name)
    result=loader(row['original_row'] if args.label_schema=='original' else row)
    batch=result[0] if isinstance(result,tuple) else result
    return {key:value.to(device) for key,value in batch.items()}


def labels_and_reference(args):
    from v2_lidf.data import load_label_manifest
    contract=sweep.read(args.selection_root/'contract.json')
    reference={r['cache_id']:r for r in sweep.lines(contract['reference_per_image']['path'])}
    labels={r['cache_id']:r for r in load_label_manifest(args.label_manifest,'evaluation')}
    roster=sweep.lines(args.selection_root/'roster.jsonl')
    sweep.require(set(labels)==set(reference)=={r['cache_id'] for r in roster},'Label and prediction roster differ')
    return labels,reference,roster


def native_prediction(args, epoch, row):
    import numpy as np
    import torch
    import torch.nn.functional as F
    path=args.selection_root/('epoch%03d'%epoch)/'predictions'/(row['cache_id']+'.npz')
    with np.load(str(path),allow_pickle=False) as payload:
        low=torch.from_numpy(payload['depth'].copy())[None,None]
        mask=torch.from_numpy(payload['refined_mask'].copy())[None,None].bool()
    native=F.interpolate(low,size=row['native_hw'],mode='bilinear',align_corners=False)
    return native,mask


def resolve_worker(args):
    install_paths(args)
    import torch
    from native_fusion.geometry import evaluate_depth
    _,selection=verify_runtime_inputs(args)
    device=gpu_setup()
    candidate_epochs=candidates_for_resolution(selection['ranking'])
    candidates={r['epoch']:r for r in selection['ranking']}
    result={'schema_version':'full1226_same_gpu_selection_resolution_v1',
        'original_selection':sweep.ref(args.selection_root/'selection.complete.json'),
        'verification':sweep.ref(args.selection_root/'publication_verification.complete.json'),
        'test_informed':True,'near_tie_tolerance_mm':TIE_MM,'candidate_epochs':candidate_epochs,
        'recomputed_on_one_gpu':len(candidate_epochs)>1,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES',''),
        'source_sha256':source_identity(args),'scores':{},'created_unix':time.time()}
    if len(candidate_epochs)>1:
        labels,reference,roster=labels_and_reference(args)
        values={epoch:[] for epoch in candidate_epochs}
        for index,row in enumerate(roster):
            batch=load_batch(args,labels[row['cache_id']],reference[row['cache_id']]['label_assets'],device)
            for epoch in candidate_epochs:
                native,_=native_prediction(args,epoch,row)
                metrics=evaluate_depth(native.to(device),batch['target'],batch['valid'] & batch['content'].bool(),
                                       batch['transparent'],intrinsics=None)
                score=metrics['transparent']['mae_mm']
                if score is not None:
                    values[epoch].append(score)
            if index%100==0:
                print(json.dumps({'phase':'same_gpu_tie_resolution','image':index,'candidates':candidate_epochs}),flush=True)
        for epoch,items in values.items():
            sweep.require(items,'Candidate has no valid transparent support')
            actual=math.fsum(items)/len(items)
            verify_score(actual,candidates[epoch]['transparent_mae_mm'])
            result['scores'][str(epoch)]={'transparent_mae_mm':actual,'images_with_transparent_support':len(items),
                'original_transparent_mae_mm':candidates[epoch]['transparent_mae_mm']}
        winner=min(candidate_epochs,key=lambda epoch:(result['scores'][str(epoch)]['transparent_mae_mm'],epoch))
    else:
        winner=candidate_epochs[0]
        result['scores'][str(winner)]={'transparent_mae_mm':candidates[winner]['transparent_mae_mm'],
                                     'original_transparent_mae_mm':candidates[winner]['transparent_mae_mm']}
    checkpoint=args.training_root/('epoch%03d.pt'%winner)
    sweep.require(sweep.sha(checkpoint)==candidates[winner]['checkpoint_sha256'],'Winning checkpoint changed')
    result.update(status='complete',selected_epoch=winner,selected_checkpoint=sweep.ref(checkpoint),
                  original_selected_epoch=selection['selected_epoch'],
                  final_selection_basis='full1226 transparent image-macro MAE; same-device candidate resolution if near tied')
    sweep.atomic_json(args.output/'resolved_selection.json',result)


def geometry_worker(args):
    install_paths(args)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from native_fusion.evaluation import evaluate_prediction
    verify_runtime_inputs(args)
    device=gpu_setup()
    resolution=sweep.read(args.output/'resolved_selection.json')
    sweep.require(resolution['source_sha256']==source_identity(args),'Resolution/evaluation source mismatch')
    epoch=resolution['selected_epoch']
    epoch_dir=args.selection_root/('epoch%03d'%epoch)
    prediction_seal=sweep.read(epoch_dir/'predictions.complete.json')
    predictions={r['cache_id']:r for r in prediction_seal['predictions']}
    stage1_receipts={r['cache_id']:r for r in sweep.lines(args.stage1_root/'inputs.sealed.jsonl')}
    labels,reference,roster=labels_and_reference(args)
    directory=args.output/'metrics'
    directory.mkdir(exist_ok=True)
    common={'resolved_selection':sweep.ref(args.output/'resolved_selection.json'),
            'prediction_seal':sweep.ref(epoch_dir/'predictions.complete.json'),
            'source_sha256':source_identity(args),'label_manifest':sweep.ref(args.label_manifest),
            'stage1_seal':sweep.ref(args.stage1_root/'complete.json')}
    started=time.monotonic()
    for index in range(args.rank,len(roster),args.world):
        row=roster[index];cid=row['cache_id']
        pred=epoch_dir/'predictions'/(cid+'.npz')
        native_path=args.stage1_root/'native'/(cid+'.npy')
        input_path=args.stage1_root/'inputs'/(cid+'.npz')
        source={**common,'prediction_sha256':sweep.sha(pred),
                'stage1_native_sha256':sweep.sha(native_path),'stage1_input_sha256':sweep.sha(input_path)}
        sweep.require(source['prediction_sha256']==predictions[cid]['prediction_sha256']
                      and source['stage1_native_sha256']==stage1_receipts[cid]['native_sha256']
                      and source['stage1_input_sha256']==stage1_receipts[cid]['input_sha256'],
                      'Prediction or frozen baseline changed after sealing')
        output=directory/(cid+'.json')
        if output.exists():
            previous=sweep.read(output)
            sweep.require(previous['source']==source,'Existing geometry record has different provenance')
            for item in previous['label_assets'].values():
                sweep.require(sweep.sha(item['path'])==item['sha256'],'Previously evaluated GT changed')
            continue
        lidf,refined=native_prediction(args,epoch,row)
        native=torch.from_numpy(np.load(str(native_path),allow_pickle=False).copy())[None,None]
        with np.load(str(input_path),allow_pickle=False) as payload:
            low=torch.from_numpy(payload['depth_input'].copy())[None,None]
        resized=F.interpolate(low,size=row['native_hw'],mode='bilinear',align_corners=False)
        assets=reference[cid]['label_assets']
        batch=load_batch(args,labels[cid],assets,device)
        metrics={name:evaluate_prediction(value.to(device),batch) for name,value in zip(METHODS,(native,resized,lidf))}
        for region in REGIONS:
            for key in ('pixels','normal_pixels','depth_induced_normal_pixels'):
                sweep.require(len({metrics[m][region][key] for m in METHODS})==1,'Methods use different GT support')
        record={**sweep.image_identity(row),'metrics':metrics,'label_assets':assets,'source':source,
                'normal_reference':'synthetic_released_camera_normals'
                    if bool(batch['normal_reference_available'].item()) else 'real_depth_induced_proxy',
                'refined_fraction_240x320':float(refined.float().mean())}
        sweep.atomic_json(output,record)
        if index%100<args.world:
            print(json.dumps({'phase':'winner_geometry','rank':args.rank,'index':index,
                              'epoch':epoch,'seconds':time.monotonic()-started}),flush=True)
    sweep.atomic_json(args.output/('geometry_rank%02d.complete.json'%args.rank),
                      {'status':'complete','rank':args.rank,'world':args.world,'epoch':epoch,
                       'images':len(range(args.rank,len(roster),args.world)),
                       'source':common,'finished_unix':time.time()})


def paired_difference(rows,region,metric,rng,draws=2000):
    import numpy as np
    values=[]
    for row in rows:
        a=row['metrics']['stage1_lidf'][region][metric]
        b=row['metrics']['stage1_native'][region][metric]
        sweep.require((a is None)==(b is None),'Paired support differs')
        if a is not None:
            values.append(a-b)
    if not values:
        return {'paired_images':0,'difference_lidf_minus_stage1':None,'image_bootstrap_95pct':None}
    values=np.asarray(values,dtype=float)
    samples=rng.integers(0,len(values),size=(draws,len(values)))
    return {'paired_images':len(values),'difference_lidf_minus_stage1':float(values.mean()),
            'image_bootstrap_95pct':np.quantile(values[samples].mean(1),[.025,.975]).tolist()}


def merge(args):
    install_paths(args)
    import numpy as np
    from native_fusion.evaluation import aggregate
    verify_runtime_inputs(args)
    roster=sweep.lines(args.selection_root/'roster.jsonl')
    resolution=sweep.read(args.output/'resolved_selection.json')
    rows=[sweep.read(args.output/'metrics'/(r['cache_id']+'.json')) for r in roster]
    sweep.validate_roster(rows,roster)
    expected_resolution=sweep.ref(args.output/'resolved_selection.json')
    for row in rows:
        sweep.require(row['source']['resolved_selection']==expected_resolution
                      and row['source']['source_sha256']==source_identity(args),'Mixed geometry record identity')
    parts={'full1226':rows,'test521':[r for r in rows if r['official_split'].endswith('test')]}
    sweep.require(len(parts['test521'])==521,'Expected521 test images')
    for group in sorted({r['benchmark_group'] for r in rows}):
        parts[group]=[r for r in rows if r['benchmark_group']==group]
    summary={name:aggregate(part) for name,part in parts.items()}
    actual=summary['full1226']['combined']['stage1_lidf']['transparent']['image_macro']['mae_mm']
    winner=resolution['selected_epoch']
    expected=resolution['scores'][str(winner)]['transparent_mae_mm']
    verify_score(actual,expected)
    original=resolution['scores'][str(winner)]['original_transparent_mae_mm']
    verify_score(actual,original)
    report=args.output/'reports';report.mkdir(exist_ok=True)
    sweep.atomic_json(report/'metrics.json',summary)
    sweep.atomic_lines(report/'per_image.jsonl',rows)
    differences={};rng=np.random.default_rng(23)
    for subset in ('full1226','test521'):
        differences[subset]={}
        for domain in ('combined','real','synthetic'):
            selected=[r for r in parts[subset] if domain=='combined' or r['domain']==domain]
            for region in REGIONS:
                for metric in ('mae_mm','abs_rel'):
                    differences[subset]['%s/%s/%s'%(domain,region,metric)]=paired_difference(selected,region,metric,rng)
    sweep.atomic_json(report/'paired_differences.json',differences)
    with (report/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['subset','domain','method','region','averaging','metric','value','images','pixels'])
        for subset,domains in summary.items():
            for domain,methods in domains.items():
                for method,regions in methods.items():
                    for region,item in regions.items():
                        for averaging in ('image_macro','pixel_pooled'):
                            for metric,value in item[averaging].items():
                                writer.writerow([subset,domain,method,region,averaging,metric,value,item['images'],item['pixels']])
    text=['完整1226逐轮重新选模后的LIDF-SPVNAS结果','',
          '选模使用全部80轮及完整1226（包括521测试图）的透明区域逐图宏平均MAE。',
          '这是明确的测试集参与选模结果，不能称独立盲测或与原704验证选模等同。',
          '原704验证AbsRel选模和原checkpoint合同均保留，没有改写。',
          '最终epoch=%d；透明MAE=%.9f mm；相对原sweep差=%+.9g mm。'%(winner,actual,actual-original),
          '全部预测先封存，随后独立GT评估进程读取标签；未重跑模型，未作GT尺度对齐。',
          '原生E08、E08缩放往返、LIDF三种方法共享GT有效像素支持；全图/透明/背景、真实/合成分别报告。',
          '合成法线参考独立cameraNormals，真实法线为GT深度诱导代理。',
          '按图bootstrap区间不计入场景相关、80次选模或历史测试曝光不确定性。','']
    for subset in ('full1226','test521'):
        for domain in ('combined','real','synthetic'):
            for region in REGIONS:
                for method in METHODS:
                    m=summary[subset][domain][method][region]['image_macro']
                    text.append('%s/%s/%s/%s: MAE %.6f mm, RMSE %.6f mm, AbsRel %.9f'%(subset,domain,region,method,m['mae_mm'],m['rmse_mm'],m['abs_rel']))
    (report/'RESULTS.zh-CN.txt').write_text('\n'.join(text)+'\n')
    names=('metrics.json','per_image.jsonl','paired_differences.json','comparison.csv','RESULTS.zh-CN.txt')
    complete={'schema_version':'full1226_test_informed_selected_geometry_v1','status':'complete',
        'images':1226,'test_images':521,'selected_epoch':winner,'selected_checkpoint':resolution['selected_checkpoint'],
        'selection_policy':sweep.POLICY,'resolved_selection':expected_resolution,
        'sweep_verification':sweep.ref(args.selection_root/'publication_verification.complete.json'),
        'source_sha256':source_identity(args),'reports_sha256':{n:sweep.sha(report/n) for n in names},
        'transparent_mae_mm':actual,'original_sweep_transparent_mae_mm':original,
        'sweep_mae_absolute_difference_mm':abs(actual-original),'sweep_mae_tolerance_mm':TIE_MM,
        'model_executed':False,'ground_truth_used_for_metric_and_selection':True,
        'ground_truth_used_as_prediction_input':False,'finished_unix':time.time()}
    sweep.atomic_json(args.output/'evaluation.complete.json',complete)
    print(json.dumps(complete),flush=True)


def child_command(args,phase,rank=0,world=1):
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--phase',phase,
             '--selection-root',str(args.selection_root),'--stage1-root',str(args.stage1_root),
             '--input-manifest',str(args.input_manifest),'--label-manifest',str(args.label_manifest),
             '--training-root',str(args.training_root),'--production-src',str(args.production_src),
             '--metric-src',str(args.metric_src),'--output',str(args.output),
             '--label-loader',args.label_loader,'--label-schema',args.label_schema,
             '--rank',str(rank),'--world',str(world)]
    return command


def coordinator(args):
    import sweep_verify
    while not (args.selection_root/'selection.complete.json').exists():
        state=args.selection_root/'execution.complete.json'
        if state.exists():
            sweep.require(sweep.read(state)['status'] not in ('worker_failure','merge_failure'),'Sweep failed; refusing evaluation')
        sweep.require(args.wait,'All80 sweep selection not complete')
        print('Waiting for authenticated all-80 selection.',flush=True)
        time.sleep(30)
    while True:
        state=sweep.read(args.selection_root/'execution.complete.json')
        if state['status']=='complete_selection_requires_winner_geometry':
            sweep.require(all(r['returncode']==0 for r in state['workers']),'Sweep worker failed')
            break
        sweep.require(args.wait,'Sweep workers/merge have not exited normally')
        time.sleep(10)
    frozen=sweep.read(args.selection_root/'contract.json')
    verify_args=argparse.Namespace(root=args.training_root.parent,output=args.selection_root,
        reference=Path(frozen['reference_evaluation']['path']).parent,
        train_project=Path(frozen['training_project']),
        v2_source=next(Path(p).parents[1] for p in frozen['source_sha256'] if p.endswith('/native_fusion/geometry.py')))
    sweep_verify.verify(verify_args)
    verify_runtime_inputs(args)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'logs').mkdir(exist_ok=True)
    gpus=args.gpus.split(',')
    sweep.require(len(set(gpus))==len(gpus) and all(g in {str(i) for i in range(8)} for g in gpus),'GPU list must be unique0–7')
    def environment(gpu):
        return dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    execution={'schema_version':'selected_geometry_execution_v1','started_unix':time.time(),
               'source_sha256':source_identity(args),'workers':[],'supervisor_pid':os.getpid()}
    command=child_command(args,'resolve')
    with (args.output/'logs/resolution.log').open('a') as stream:
        result=subprocess.run(command,env=environment(gpus[0]),stdout=stream,stderr=subprocess.STDOUT)
    execution['resolution']={'command':command,'returncode':result.returncode,'log':sweep.ref(args.output/'logs/resolution.log')}
    sweep.require(result.returncode==0,'Same-device selection resolution failed')
    processes=[]
    for rank,gpu in enumerate(gpus):
        command=child_command(args,'geometry',rank,len(gpus))
        log=args.output/'logs'/('geometry_rank%02d.log'%rank)
        stream=log.open('a')
        process=subprocess.Popen(command,env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT)
        processes.append((process,stream))
        execution['workers'].append({'rank':rank,'physical_gpu':int(gpu),'pid':process.pid,'command':command,'log':str(log)})
    sweep.atomic_json(args.output/'execution.json',execution)
    for rank,(process,stream) in enumerate(processes):
        code=process.wait();stream.close()
        execution['workers'][rank]['returncode']=code
    sweep.atomic_json(args.output/'execution.json',execution)
    sweep.require(all(x['returncode']==0 for x in execution['workers']),'Geometry worker failed')
    merge(args)
    execution.update(status='complete',finished_unix=time.time(),evaluation=sweep.ref(args.output/'evaluation.complete.json'))
    sweep.atomic_json(args.output/'execution.json',execution)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-root',type=Path,required=True)
    parser.add_argument('--stage1-root',type=Path,required=True)
    parser.add_argument('--input-manifest',type=Path,required=True)
    parser.add_argument('--training-root',type=Path,required=True,help='Directory containing epoch001.pt through epoch080.pt')
    parser.add_argument('--production-src',type=Path,required=True)
    parser.add_argument('--metric-src',type=Path,help='Directory containing native_fusion; defaults to production-src')
    parser.add_argument('--label-manifest',type=Path,help='Defaults to authenticated sweep contract reference')
    parser.add_argument('--output',type=Path,help='Defaults to selection-root/selected_geometry')
    parser.add_argument('--label-loader',default='native_fusion.benchmark:load_evaluation_labels')
    parser.add_argument('--label-schema',choices=('original','separated'),default='original')
    parser.add_argument('--phase',choices=('coordinator','resolve','geometry','merge'),default='coordinator')
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world',type=int,default=1)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    args.metric_src=args.metric_src or args.production_src
    args.output=args.output or args.selection_root/'selected_geometry'
    if args.label_manifest is None:
        args.label_manifest=Path(sweep.read(args.selection_root/'contract.json')['label_manifest']['path'])
    sweep.require(0<=args.rank<args.world<=8,'Invalid rank/world')
    functions={'coordinator':coordinator,'resolve':resolve_worker,'geometry':geometry_worker,'merge':merge}
    functions[args.phase](args)


if __name__=='__main__':
    main()
