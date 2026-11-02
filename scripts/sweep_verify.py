"""Independent final publication gate for the frozen, all-80 sweep.

Run only after selection.complete.json exists. Rehash every original checkpoint
and prediction; bind the reported winner to the original committed checkpoint.
Does not run models or modify any training/evaluation source artifacts.
"""
import argparse
from pathlib import Path
import time

import sweep_checkpoints as sweep


def verify(args):
    sweep.verify_contract(args)
    selected = sweep.read(args.output/'selection.complete.json')
    sweep.require(selected['status']=='complete' and selected['completed_epochs']==80
                  and selected['selection_policy']==sweep.POLICY,
                  'Cannot publish partial or differently selected sweep')
    roster = sweep.lines(args.output/'roster.jsonl')
    entries, evidence = [], {}
    for epoch in range(1,81):
        directory=args.output/('epoch%03d'%epoch)
        marker=sweep.validate_epoch(args,directory,roster,verify_predictions=True)
        committed=sweep.checkpoint_for_epoch(args,epoch)
        sweep.require(committed is not None,'Missing committed epoch')
        checkpoint,receipt,record=committed
        seal=sweep.read(directory/'predictions.complete.json')
        sweep.require(marker['epoch']==epoch and marker['checkpoint_sha256']==receipt['sha256']
                      and seal['checkpoint_receipt']==receipt and seal['epoch_record']==record,
                      'Evaluated checkpoint/metadata differs from original archived epoch')
        predictions={r['cache_id']:r for r in seal['predictions']}
        for row in sweep.lines(directory/'per_image.jsonl'):
            sweep.require(row['prediction_sha256']==predictions[row['cache_id']]['prediction_sha256'],
                          'Per-image metrics refer to different prediction bytes')
        entries.append(marker)
        evidence[str(checkpoint.resolve())]=receipt['sha256']
        evidence[str(directory/'complete.json')]=sweep.sha(directory/'complete.json')
    ranked=sweep.ranking(entries)
    expected=[{k:r[k] for k in ('epoch','checkpoint_sha256','images','smoke','transparent_mae_mm','metrics','per_image')}
              for r in ranked]
    sweep.require(selected['ranking']==expected and selected['selected_epoch']==ranked[0]['epoch'],
                  'Winner/ranking differs from complete per-epoch metrics')
    actual=sweep.ref(args.root/'training'/('epoch%03d.pt'%ranked[0]['epoch']))
    sweep.require(selected['selected_checkpoint']==actual and actual['sha256']==ranked[0]['checkpoint_sha256'],
                  'Selected file differs from the evaluated winning checkpoint')
    result={'schema_version':'full1226_sweep_publication_verification_v1','status':'passed',
            'selection':sweep.ref(args.output/'selection.complete.json'),'winner':actual,
            'selected_epoch':ranked[0]['epoch'],'epochs':80,'predictions_verified':80*1226,
            'uses_test_labels_for_checkpoint_selection':True,
            'independent_test_performance_claimed':False,
            'geometry_verification_scope':'separate winner geometry report still required',
            'evidence_sha256':evidence,'verifier':sweep.ref(Path(__file__)),
            'finished_unix':time.time()}
    sweep.atomic_json(args.output/'publication_verification.complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=sweep.ROOT)
    parser.add_argument('--output',type=Path,default=sweep.OUTPUT)
    parser.add_argument('--reference',type=Path,default=sweep.REFERENCE)
    parser.add_argument('--train-project',type=Path,default=sweep.TRAIN_PROJECT)
    parser.add_argument('--v2-source',type=Path,default=sweep.V2_SOURCE)
    args=parser.parse_args()
    result=verify(args)
    print('All 80 original checkpoints and 98080 predictions authenticated; winner E%03d.'%result['selected_epoch'])


if __name__=='__main__':
    main()
