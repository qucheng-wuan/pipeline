"""Publication selection, numerical-admission and paired-report tests."""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))
import evaluate_selected as selected
import sweep_checkpoints as sweep


def ranking():
    return [dict(epoch=epoch,transparent_mae_mm=50.+epoch) for epoch in range(1,81)]


def test_near_tie_resolves_all_close_candidates_not_just_top_two():
    rows=ranking()
    rows[0]['transparent_mae_mm']=50.
    rows[1]['transparent_mae_mm']=50.+1e-5
    rows[2]['transparent_mae_mm']=50.+1.9e-5
    assert selected.candidates_for_resolution(rows)==[1,2,3]
    rows[1]['transparent_mae_mm']=51.
    rows[2]['transparent_mae_mm']=52.
    assert selected.candidates_for_resolution(rows)==[1]
    with pytest.raises(ValueError,match='eighty'):
        selected.candidates_for_resolution(rows[:-1])


def test_geometry_ranking_difference_has_explicit_mm_limit():
    selected.verify_score(50.+8.2e-6,50.)
    for bad in (50.+2.1e-5,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='ranking'):
            selected.verify_score(bad,50.)


def test_paired_bootstrap_matches_registered_original_including_empty_support():
    from transparent_depth.statistics import paired_difference
    rows=[]
    for baseline,current in ((10.,8.),(3.,4.),(None,None),(50.,20.)):
        rows.append({'metrics':{'stage1_native':{'transparent':{'mae_mm':baseline}},
                                'stage1_lidf':{'transparent':{'mae_mm':current}}}})
    a=selected.paired_difference(rows,'transparent','mae_mm',np.random.default_rng(23))
    b=paired_difference(rows,'transparent','mae_mm',np.random.default_rng(23))
    assert a==b and a['paired_images']==3
    rows[-1]['metrics']['stage1_lidf']['transparent']['mae_mm']=None
    with pytest.raises(ValueError,match='support'):
        selected.paired_difference(rows,'transparent','mae_mm',np.random.default_rng(23))


def test_native_prediction_only_reads_sealed_arrays_and_preserves_resize_math(tmp_path):
    import torch
    import torch.nn.functional as F
    path=tmp_path/'epoch004/predictions'
    path.mkdir(parents=True)
    low=np.arange(12,dtype=np.float32).reshape(3,4)+.5
    mask=np.ones_like(low,dtype=bool)
    np.savez_compressed(path/'sample.npz',depth=low,refined_mask=mask)
    out,refined=selected.native_prediction(argparse.Namespace(selection_root=tmp_path),4,
                                          {'cache_id':'sample','native_hw':[7,9]})
    assert torch.equal(out,F.interpolate(torch.from_numpy(low)[None,None],size=[7,9],mode='bilinear',align_corners=False))
    assert refined.shape==(1,1,3,4) and bool(refined.all())


def test_parameterized_child_command_keeps_user_paths(tmp_path):
    args=argparse.Namespace(selection_root=tmp_path/'selection',stage1_root=tmp_path/'stage1',
        input_manifest=tmp_path/'inputs.jsonl',label_manifest=tmp_path/'labels.jsonl',
        training_root=tmp_path/'training',production_src=tmp_path/'portable/src',metric_src=tmp_path/'metrics/src',
        output=tmp_path/'out',label_loader='transparent_depth.evaluation:load_evaluation_labels',label_schema='separated')
    cmd=selected.child_command(args,'geometry',3,8)
    for name in ('selection-root','stage1-root','input-manifest','training-root','production-src','metric-src'):
        assert cmd[cmd.index('--'+name)+1]==str(getattr(args,name.replace('-','_')))
    assert cmd[cmd.index('--rank')+1]=='3'


def test_incomplete_sweep_cannot_start_labelled_evaluation(tmp_path,monkeypatch):
    import sweep_verify
    touched=[]
    monkeypatch.setattr(sweep_verify,'verify',lambda args:touched.append('verification'))
    with pytest.raises(ValueError,match='not complete'):
        selected.coordinator(argparse.Namespace(selection_root=tmp_path,wait=False))
    assert touched==[]
