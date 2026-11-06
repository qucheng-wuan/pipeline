"""Pure-P 256-grid evaluation; labels are opened only after model prediction."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from v2_lidf.data import V2LIDFDataset, load_label_manifest, atomic_json, atomic_npz
from v2_lidf.data_protocol import resize_supervision, TRANSFER_PROTOCOLS, validate_partition
from .stage2 import build_stage2, file_sha, source_hashes
from .training_cache import load_authenticated_native_labels, validate_prediction_caches


DELTA_THRESHOLDS = (1.05, 1.10, 1.25)
METRIC_KEYS = ('mae_m', 'rmse_m', 'abs_rel', *(f'delta_{t:.2f}' for t in DELTA_THRESHOLDS))
EVALUATION_SCHEMA = 'fullfield256_evaluation_v2'


def metric_metadata():
    return {'metric_protocol': 'depth256_rftrans_thresholds_v2',
            'delta_thresholds': list(DELTA_THRESHOLDS),
            'delta_comparison': 'strict_less_than', 'delta_storage': 'fraction_0_to_1'}


def region_metrics(prediction, target, support):
    prediction, target = np.asarray(prediction), np.asarray(target)
    support = np.asarray(support, dtype=bool)
    if prediction.shape != target.shape or support.shape != target.shape:
        raise ValueError('Metric grid mismatch')
    if not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise FloatingPointError('Nonfinite/nonpositive model prediction')
    count = int(support.sum())
    if not count:
        return {'valid_pixels': 0, 'empty': True}
    p, g = prediction[support].astype(np.float64), target[support].astype(np.float64)
    if not np.isfinite(g).all() or np.any(g <= 0):
        raise ValueError('Invalid GT inside evaluation support')
    error = np.abs(p-g)
    ratio = np.maximum(p/g, g/p)
    return dict(valid_pixels=count, empty=False, mae_m=float(error.mean()),
        rmse_m=float(np.sqrt(np.mean(error**2))), abs_rel=float(np.mean(error/g)),
        **{f'delta_{t:.2f}': float(np.mean(ratio < t)) for t in DELTA_THRESHOLDS})


def score_predictions(baseline, prediction, labels):
    masks = {'all': labels['valid'], 'transparent': labels['valid'] & labels['transparent'],
             'background': labels['valid'] & ~labels['transparent']}
    return {name:{region:region_metrics(depth, labels['target'], valid)
                  for region, valid in masks.items()}
            for name, depth in [('P256', baseline), ('Stage2', prediction)]}


def aggregate(records):
    result = {}
    for model in ('P256','Stage2'):
        result[model] = {}
        for region in ('all','transparent','background'):
            rows = [r['metrics'][model][region] for r in records]
            good = [r for r in rows if not r['empty']]
            value = {'images': len(rows), 'nonempty_images': len(good),
                     'empty_images': len(rows)-len(good), 'valid_pixels': sum(r['valid_pixels'] for r in rows)}
            for key in METRIC_KEYS:
                value[key] = float(np.mean([r[key] for r in good])) if good else None
            result[model][region] = value
    return result


def score_item(row, baseline, prediction, label_row):
    if row.get('preprocessing') != label_row.get('preprocessing'):
        raise ValueError('Prediction/supervision preprocessing mismatch')
    labels = resize_supervision(load_authenticated_native_labels(label_row), row['preprocessing'])
    return {'sample_id':row['sample_id'], 'cache_id':row['cache_id'],
            'dataset':row['dataset'], 'scene':row['scene'],
            'camera':row['camera'], 'metrics':score_predictions(baseline,prediction,labels)}, labels


def visualize(path, rgb, baseline, prediction, labels, depth_range):
    """Shared fixed metric colour scale, no GT scale fitting or image min/max."""
    import cv2
    from PIL import Image, ImageDraw
    lo, hi = depth_range
    def color(x):
        gray = np.round(np.clip((x-lo)/(hi-lo),0,1)*255).astype(np.uint8)
        return cv2.cvtColor(cv2.applyColorMap(gray,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    target = color(labels['target']); target[~labels['valid']]=0
    mask = np.zeros((*labels['valid'].shape,3),np.uint8)
    mask[labels['valid']]=[70,70,70]; mask[labels['valid'] & labels['transparent']]=[255,255,255]
    tiles = [rgb, color(baseline), color(prediction), target, mask]
    canvas = Image.new('RGB',(256*5,282),'white'); draw=ImageDraw.Draw(canvas)
    for i,(tile,name) in enumerate(zip(tiles,['RGB256','P256 metres','Stage2 metres','GT256 (invalid black)','valid / transparent'])):
        canvas.paste(Image.fromarray(tile),(i*256,26));draw.text((i*256+5,6),name,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def evaluate(args):
    device = torch.device(args.device)
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg = saved['contract']['config']
    diagnostic = bool(args.smoke or cfg.get('diagnostic_smoke',False))
    if cfg.get('data_protocol') not in TRANSFER_PROTOCOLS:
        raise ValueError('256 evaluation requires a supported 256 checkpoint')
    if cfg['data_protocol']=='cleargrasp256' and (cfg.get('selection_policy')!='real_val_best'
            or cfg.get('test_set_selection') is not True):
        raise ValueError('Checkpoint predates the explicit real-val best-checkpoint protocol')
    for name, digest in source_hashes().items():
        if saved['contract']['sources'].get(name) != digest:
            raise ValueError('Checkpoint model source changed: '+name)
    validate_prediction_caches(cfg,args.input_manifest,args.label_manifest,args.stage1_root)
    model = build_stage2(saved['model_config'],device,initialization=False)
    model.load_state_dict(saved['model'],strict=True); model.eval()
    dataset = V2LIDFDataset(args.input_manifest,None,args.stage1_root,None,
                          role=args.role,include_target=False)
    labels = {r['cache_id']:r for r in load_label_manifest(args.label_manifest,args.role)}
    rows = []
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Evaluation output must be new')
    output.mkdir(parents=True,exist_ok=True)
    indices = [i for i,r in enumerate(dataset.rows) if r['dataset']==args.dataset]
    if args.limit:
        indices=indices[:args.limit]
    for position,index in enumerate(indices):
        row = dataset.rows[index]
        validate_partition(row, cfg['data_protocol'])
        if row['preprocessing'] != cfg['preprocessing']:
            raise ValueError('Checkpoint preprocessing differs from evaluation')
        item=dataset[index]
        torch.manual_seed(int(row['rgb_sha256'][:8],16));torch.cuda.manual_seed_all(int(row['rgb_sha256'][:8],16))
        with torch.inference_mode():
            out=model(item['rgb'][None].to(device),item['depth_input'][None].to(device),
                item['intrinsics'][None].to(device),compute_loss=False)
        if not bool(out['query_mask'].all()):
            raise ValueError('Evaluation must query every pixel')
        prediction=out['depth'][0,0].cpu().numpy();baseline=item['depth_input'][0].numpy()
        result, supervision = score_item(row,baseline,prediction,labels[row['cache_id']]);rows.append(result)
        atomic_npz(output/'supervision'/(row['cache_id']+'.npz'), **supervision)
        atomic_npz(output/'predictions'/(row['cache_id']+'.npz'), P256=baseline, D256=prediction)
        if position < args.visualizations:
            with np.load(Path(args.stage1_root)/'inputs'/(row['cache_id']+'.npz')) as cache:
                rgb=cache['rgb'].copy()
            visualize(output/'visualizations'/(row['cache_id']+'.png'),rgb,baseline,prediction,
                      supervision,cfg['preprocessing']['depth_range_m'])
    if not rows:
        raise ValueError('No requested evaluation samples')
    groups=defaultdict(list)
    for row in rows:
        groups[row['dataset']+'/'+row['scene']+'/'+row['camera']].append(row)
    report={'schema':EVALUATION_SCHEMA, **metric_metadata(), 'diagnostic_smoke':diagnostic,
        'accuracy_claim':False if diagnostic else ('test-set-selected result; not independent held-out accuracy'
            if cfg.get('test_set_selection') else 'dataset-specific; see Stage1 exposure disclosure'),
        'test_set_selection':cfg.get('test_set_selection',False),
        'selection_policy':cfg.get('selection_policy','internal_val'),
        'checkpoint_epoch':saved['epoch']+1,
        'checkpoint_sha256':file_sha(args.checkpoint),'preprocessing':cfg['preprocessing'],
        'input_manifest_sha256':file_sha(args.input_manifest),'label_manifest_sha256':file_sha(args.label_manifest),
        'stage1_seal_sha256':file_sha(Path(args.stage1_root)/'complete.json'),
        'aggregation':'unweighted mean of per-image metrics over nonempty regions; RMSE averaged per image',
        'units':{'MAE':'m','RMSE':'m','AbsRel':'dimensionless',
                 'delta':'fraction; max(P/G,G/P)<t; t=1.05,1.10,1.25; multiply by 100 only for percent display'},
        'empty_regions':'omitted from macro means, counted explicitly; no valid prediction filtering',
        'gt_scale_alignment':False,'native_upsampling':False,'input_gt_or_mask_access':False,
        'stage1_exposure':'Frozen V2/Laplacian were historically selected using ClearGrasp validation (including real-val); not an untouched held-out test of the complete pipeline.',
        'overall':aggregate(rows),'groups':{k:aggregate(v) for k,v in groups.items()},'images':rows}
    atomic_json(output/'metrics.json',report)
    return {'images':len(rows),'report':str(output/'metrics.json'),'diagnostic_smoke':diagnostic}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','input-manifest','label-manifest','stage1-root','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--dataset',choices=['cleargrasp','transcg'],default='cleargrasp')
    p.add_argument('--role',choices=['val','evaluation'],default='evaluation')
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--visualizations',type=int,default=4);p.add_argument('--smoke',action='store_true')
    print(json.dumps(evaluate(p.parse_args()),indent=2))


if __name__=='__main__':main()
