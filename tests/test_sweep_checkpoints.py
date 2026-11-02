"""Meaningful rejection, support, selection and original-metric tests; CPU only."""
import copy
import importlib.util
import math
from pathlib import Path
import sys

import pytest

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sweep', P/'scripts/sweep_checkpoints.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def identity(cid='one', domain='real', split='real-test'):
    return dict(cache_id=cid,sample_id=cid,rgb_sha256='a'*64,domain=domain,
                official_split=split,benchmark_group=domain+'_novel',native_hw=[12,16])


def test_exact_roster_rejects_duplicate_missing_and_changed_rgb():
    expected=[identity(),identity('two')]
    sweep.validate_roster(copy.deepcopy(expected),expected)
    for bad in ([expected[0]], [expected[0],expected[0]],
                [dict(expected[0],rgb_sha256='b'*64),expected[1]]):
        with pytest.raises(ValueError):
            sweep.validate_roster(bad,expected)


def test_all_eighty_required_and_early_non_val_best_allowed_with_honest_test_label():
    entries=[dict(epoch=i,images=1226,smoke=False,transparent_mae_mm=1. if i in (4,63) else 2.) for i in range(1,81)]
    assert sweep.ranking(entries)[0]['epoch']==4
    with pytest.raises(ValueError,match='All 80'):
        sweep.ranking(entries[:-1])
    with pytest.raises(ValueError,match='Duplicate'):
        sweep.ranking(entries+[entries[0]])
    for replacement in (dict(entries[0],smoke=True),dict(entries[0],images=8),dict(entries[0],transparent_mae_mm=float('nan'))):
        with pytest.raises(ValueError):
            sweep.ranking([replacement]+entries[1:])
    assert sweep.POLICY['uses_test_labels_for_checkpoint_selection'] is True
    assert sweep.POLICY['independent_test_performance_claimed'] is False


def test_empty_transparent_support_excluded_not_zero_and_macro_not_pixel_weighted():
    def row(cid,n,value,domain):
        fields={'pixels':n,**{k:value for k in sweep.METRICS}}
        return {**identity(cid,domain,domain+'-test'),'metrics':{r:dict(fields) for r in sweep.REGIONS}}
    rows=[row('a',1,10.,'real'),row('b',100,30.,'synthetic'),row('c',0,None,'real')]
    summary=sweep.aggregate(rows)
    all_=summary['full1226']['combined']['transparent']
    assert all_['image_macro']['mae_mm']==20.
    assert all_['metric_image_counts']['mae_mm']==2 and all_['images']==3
    assert summary['test521']['real']['all']['image_macro']['mae_mm']==10.


def test_depth_only_calls_original_geometry_with_identical_depth_values_and_support():
    import torch
    from native_fusion.geometry import evaluate_depth
    gt=torch.linspace(.2,1.2,192,dtype=torch.float32).reshape(1,1,12,16)
    pred=gt*1.1+.01
    valid=torch.ones_like(gt,dtype=torch.bool);valid[...,0,0]=False
    transparent=torch.zeros_like(valid);transparent[...,3:8,4:11]=True
    K=torch.tensor([[[50.,0.,7.5],[0.,50.,5.5],[0.,0.,1.]]])
    no_normal=evaluate_depth(pred,gt,valid,transparent,intrinsics=None)
    full=evaluate_depth(pred,gt,valid,transparent,intrinsics=K)
    assert sweep.depth_only(no_normal)==sweep.depth_only(full)
    assert no_normal['transparent']['pixels']==35
    expected=(pred.double()-gt.double()).abs()[valid].sum().item()/191*1000
    assert math.isclose(no_normal['all']['mae_mm'],expected,rel_tol=1e-15)
    bad=pred.clone();bad[...,4,4]=float('nan')
    with pytest.raises(ValueError,match='cannot shrink'):
        evaluate_depth(bad,gt,valid,transparent,intrinsics=None)


def test_source_identity_binds_sweep_script_and_numerical_dependencies_without_historical_paths(tmp_path):
    import argparse
    project, metrics = tmp_path/'training', tmp_path/'metrics'
    (project/'src/v2_lidf').mkdir(parents=True)
    (metrics/'native_fusion').mkdir(parents=True)
    for name in ('model.py','spvnas_fusion.py','predict.py','data.py'):
        (project/'src/v2_lidf'/name).write_text('# independent fixture: '+name)
    for name in ('geometry.py','data.py','evaluation.py'):
        (metrics/'native_fusion'/name).write_text('# metric fixture: '+name)
    args=argparse.Namespace(train_project=project,v2_source=metrics)
    sources=sweep.execution_sources(args)
    assert str(P/'scripts/sweep_checkpoints.py') in sources
    for relative in ('model.py','spvnas_fusion.py','predict.py','data.py'):
        assert str(project/'src/v2_lidf'/relative) in sources
    geometry = metrics/'native_fusion/geometry.py'
    assert str(geometry) in sources
    geometry.write_text('# changed numerical dependency')
    assert sweep.execution_sources(args)[str(geometry)] != sources[str(geometry)]
