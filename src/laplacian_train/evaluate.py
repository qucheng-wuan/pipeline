"""Portable native-resolution Stage1 evaluation: predict/seal before label reads.

Uses the unchanged metric and normal-support mathematics. An explicitly named
checkpoint is evaluated; this tool never reselects E08 or uses test labels to
change the public frozen Stage1 asset manifest.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .common import numerics, read, sha, write
from .model import LearnableLaplacianFusion, apply_gates_full
from .prepare import FullInputBuilder, forbid_label_reads
from native_fusion.benchmark import load_evaluation_labels, read_prediction, save_prediction
from native_fusion.evaluation import evaluate_prediction, aggregate


def predict(args):
    numerics()
    builder=FullInputBuilder(args.split,args.device)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=LearnableLaplacianFusion(state.get('model_config',state.get('config',{}).get('model_settings'))).to(args.device).eval()
    model.load_state_dict(state['model'],strict=True)
    checkpoint_sha=sha(args.checkpoint)
    records=[]
    with torch.inference_mode(),forbid_label_reads():
        for row in builder.rows:
            full=builder(row)
            value=apply_gates_full(full,model(full['context'])['gate_delta'])['depth']
            asset=save_prediction(args.output/'predictions'/(row['cache_id']+'.npy'),value,row['native_hw'])
            records.append({'sample_id':row['sample_id'],'cache_id':row['cache_id'],'rgb_sha256':row['rgb_sha256'],
                            'domain':row['domain'],'native_hw':row['native_hw'],'asset':asset})
    path=args.output/'predictions.json'
    write(path,records)
    write(args.output/'predictions.sealed.json',{'status':'complete','split':args.split,'images':len(records),
        'predictions_sha256':sha(path),'checkpoint_sha256':checkpoint_sha,'ground_truth_access':False})


def evaluate(args):
    from .prepare import MANIFEST
    from native_fusion.data import load_manifest
    numerics()
    rows=load_manifest(MANIFEST,args.split)
    marker=read(args.output/'predictions.sealed.json')
    path=args.output/'predictions.json'
    if marker['status']!='complete' or marker['ground_truth_access'] is not False or marker['predictions_sha256']!=sha(path):
        raise ValueError('all predictions must be authenticated before label access')
    records=read(path)
    if len(records)!=len(rows) or len({r['sample_id'] for r in records})!=len(rows):
        raise ValueError('prediction coverage differs')
    indexed={r['sample_id']:r for r in records}
    if set(indexed)!={r['sample_id'] for r in rows}:raise ValueError('prediction roster differs')
    # Verify every immutable array before opening the first label.
    for row in rows:
        r=indexed[row['sample_id']]
        if any(r[k]!=row[k] for k in ('cache_id','rgb_sha256','native_hw','domain')):raise ValueError('identity changed')
        read_prediction(Path(r['asset']['path']),r['asset']['sha256'],row['native_hw'])
    output=[]
    for row in rows:
        r=indexed[row['sample_id']]
        prediction=torch.from_numpy(read_prediction(Path(r['asset']['path']),r['asset']['sha256'],row['native_hw']))[None,None]
        batch,assets=load_evaluation_labels(row)
        output.append({'sample_id':row['sample_id'],'domain':row['domain'],
            'normal_reference':'synthetic_released_camera_normals' if bool(batch['normal_reference_available']) else 'real_depth_induced_proxy',
            'label_assets':assets,'metrics':{'stage1':evaluate_prediction(prediction,batch)}})
    write(args.output/'per_image.json',output)
    write(args.output/'summary.json',aggregate(output))
    write(args.output/'evaluation.complete.json',{'status':'complete','images':len(output),
        'prediction_seal_sha256':sha(args.output/'predictions.sealed.json'),
        'summary_sha256':sha(args.output/'summary.json'),'selection_changed':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('predict','evaluate'))
    p.add_argument('--split',choices=('val','test'),required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.action=='predict':
        if args.checkpoint is None:p.error('predict requires --checkpoint')
        predict(args)
    else:evaluate(args)


if __name__=='__main__':main()
