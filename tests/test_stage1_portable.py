"""CPU checks for the portable Stage1 contract and retained numerical model."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import sys

import torch

from transparent_depth.stage1 import PROJECT, Stage1Assets, forbid_label_reads, prepare_flex_gemm_runtime
from laplacian_train.model import (LearnableLaplacianFusion, apply_gates_full,
    fixed_depth_and_bands, mean_coefficients, build_gate_context)


class PortableStage1Tests(unittest.TestCase):
    def test_trace_passes_configured_cache_to_predictor_without_project_fallback(self):
        import importlib.util
        from types import SimpleNamespace
        from transparent_depth import stage1
        spec=importlib.util.spec_from_file_location('portable_trace_cache_test',PROJECT/'scripts/trace_stage1_live.py')
        trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
        class AfterBootstrap(Exception):pass
        old_temp=tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ),patch.object(sys,'path',list(sys.path)):
                root=Path(folder);project=root/'relocated_project';project.mkdir()
                asset_root=root/'assets';asset_root.mkdir()
                seed=asset_root/'seed.json';seed.write_text('{"kernel":"frozen"}\n')
                manifest=root/'assets.json';manifest.write_text(json.dumps({'schema':'portable_stage1_assets_v1','assets':[
                    {'id':'flex_gemm_autotune_cache','path':'seed.json','bytes':seed.stat().st_size,
                     'sha256':hashlib.sha256(seed.read_bytes()).hexdigest()}]}))
                rgb=root/'input.png';rgb.write_bytes(b'rgb')
                args=SimpleNamespace(project=project,assets=manifest,asset_root=asset_root,device='cpu',
                    rgb=rgb,output=root/'trace',cache_root=root/'shared_runtime',deny_prefix=[])
                observed={}
                def predictor(*args_,**kwargs):
                    observed['cache_argument']=kwargs.get('runtime_cache')
                    observed['bootstrap']=prepare_flex_gemm_runtime(stage1.Stage1Assets(manifest,asset_root),kwargs.get('runtime_cache'))
                    raise AfterBootstrap()
                with patch.object(stage1,'PROJECT',project),patch.object(stage1,'Stage1Predictor',side_effect=predictor):
                    with self.assertRaises(AfterBootstrap):trace.run_portable(args,None)
                self.assertEqual(observed['cache_argument'],args.cache_root.resolve())
                self.assertTrue(Path(observed['bootstrap']['runtime_cache_path']).is_relative_to(args.cache_root))
                self.assertFalse((project/'outputs').exists())
                self.assertEqual(json.loads((args.output/'file_access_guard.json').read_text())['denied_paths'],[])
        finally:tempfile.tempdir=old_temp
        self.assertFalse(torch.cuda.is_initialized())

    def test_flex_seed_is_private_verified_and_never_reads_home_cache(self):
        from transparent_depth.input_guard import RGBInputGuard
        old_temp=tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ):
                root=Path(folder);assets_root=root/'assets';assets_root.mkdir()
                seed=assets_root/'seed.json';raw=b'{"NVIDIA GeForce RTX 3090":{"kernel":{}}}\n';seed.write_bytes(raw)
                digest=hashlib.sha256(raw).hexdigest();manifest=root/'assets.json'
                manifest.write_text(json.dumps({'schema':'portable_stage1_assets_v1','assets':[
                    {'id':'flex_gemm_autotune_cache','path':'seed.json','bytes':len(raw),'sha256':digest}]}))
                assets=Stage1Assets(manifest,assets_root);cache=root/'runtime';cache.mkdir()
                home_cache=Path.home()/'.flex_gemm/autotune_cache.json'
                os.environ['FLEX_GEMM_AUTOTUNE_CACHE_PATH']=str(home_cache)
                guard=RGBInputGuard(project=PROJECT,asset_root=assets_root,rgb_paths=[],
                    output_root=root/'output',cache_root=cache,deny_prefixes=[home_cache.parent])
                with guard:receipt=prepare_flex_gemm_runtime(assets,cache)
                target=Path(receipt['runtime_cache_path'])
                self.assertNotEqual(target,seed);self.assertTrue(target.is_relative_to(cache))
                self.assertEqual(target.read_bytes(),raw);self.assertEqual(receipt['seed_sha256'],digest)
                self.assertEqual(os.environ['FLEX_GEMM_AUTOTUNE_CACHE_PATH'],str(target))
                self.assertFalse(any(str(home_cache.parent) in p for p in guard.reads))
                self.assertEqual(guard.denials,[])
                target.write_text('{"runtime_autotune_updates":true}')
                self.assertEqual(seed.read_bytes(),raw)
                second=prepare_flex_gemm_runtime(assets,cache)
                self.assertNotEqual(second['runtime_cache_path'],str(target))
                self.assertEqual(Path(second['runtime_cache_path']).read_bytes(),raw)
        finally:tempfile.tempdir=old_temp
        self.assertFalse(torch.cuda.is_initialized())

    def test_flex_bootstrap_rejects_preimported_runtime(self):
        with patch.dict(sys.modules,{'flex_gemm':object()}):
            with self.assertRaisesRegex(RuntimeError,'before the authenticated'):
                prepare_flex_gemm_runtime(None)

    def test_training_prepare_seal_rejects_old_producer(self):
        from native_fusion import prepare
        row={'sample_id':'id','cache_id':'id','rgb_sha256':'a'*64,'native_hw':[2,3],
             'split':'train','domain':'synthetic','target_depth_path':'target.exr'}
        identity={key:row[key] for key in ('sample_id','cache_id','rgb_sha256','native_hw','split','domain')}
        class Assets:
            manifest_sha256='b'*64
            def verify_sources(self):return 'c'*64
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'prepare_receipts').mkdir()
            record={'input_identity':identity,'row':row,'ground_truth_read':False,
                'assets_sha256':'b'*64,'sources_sha256':'d'*64,'files':[],'residual':{}}
            (root/'prepare_receipts/id.json').write_text(json.dumps(record))
            with patch.object(prepare,'normalize_rows',return_value=[row]),patch('transparent_depth.stage1.Stage1Assets',Assets):
                with self.assertRaisesRegex(ValueError,'producer changed'):prepare.seal(root/'input.jsonl',root)
                record['sources_sha256']='c'*64;record['row']={**row,'target_depth_path':'different.exr'}
                (root/'prepare_receipts/id.json').write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,'identity/input contract'):prepare.seal(root/'input.jsonl',root)

    def test_model_loss_and_metrics_are_original_bytes(self):
        origins=json.loads((PROJECT/'provenance/stage1_source_origins.json').read_text())
        paths={'src/native_fusion/'+n for n in ('model.py','geometry.py','losses.py','evaluation.py')}
        paths|={'src/laplacian_train/'+n for n in ('model.py','losses.py','objective.py')}
        paths.add('src/laplacian_fusion/pyramid.py')
        indexed={r['path']:r['original_sha256'] for r in origins}
        for relative in paths:
            self.assertEqual(hashlib.sha256((PROJECT/relative).read_bytes()).hexdigest(),indexed[relative])

    def test_relative_asset_registry_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);file=root/'model.pt';file.write_bytes(b'original weight bytes')
            manifest=root/'assets.json'
            entry={'id':'model','path':'model.pt','bytes':file.stat().st_size,
                   'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
            manifest.write_text(json.dumps({'schema':'portable_stage1_assets_v1','assets':[entry]}))
            assets=Stage1Assets(manifest,root)
            self.assertEqual(assets.verify('model'),file)
            file.write_bytes(b'changed weight bytes!')
            with self.assertRaises(ValueError):assets.verify('model')
            entry['path']='../escape.pt'
            manifest.write_text(json.dumps({'schema':'portable_stage1_assets_v1','assets':[entry]}))
            with self.assertRaises(ValueError):Stage1Assets(manifest,root).path('model')

    def test_label_guard_blocks_before_open(self):
        with forbid_label_reads():
            for path in ('/not-present/image.exr','/not-present/a-segmentation-mask.png','/not-present/a-mask.png'):
                with self.assertRaises(AssertionError):Path(path).open('rb')

    def test_no_historical_root_literals_in_portable_entrypoints(self):
        roots=['src/native_fusion','src/laplacian_train','src/laplacian_fusion']
        files=[p for root in roots for p in (PROJECT/root).glob('*.py')]
        files += [PROJECT/'src/transparent_depth/stage1.py',PROJECT/'src/transparent_depth/stage1_experts.py']
        for path in files:
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Constant) and isinstance(node.value,str):
                    self.assertFalse(node.value.startswith(('/home/hengxianli/','/ssd/polyu/','/mnt/gluster_dcpu/')),str(path))

    def test_historical_teacher_map_contains_only_rgb_identity(self):
        data=json.loads((PROJECT/'provenance/stage1_seed_identities.json').read_text())
        self.assertIs(data['ground_truth_values_included'],False)
        self.assertEqual(len(data['identities']),46679)
        self.assertTrue(all(len(k)==64 and isinstance(v,str) and not v.startswith('/') for k,v in data['identities'].items()))

    def test_zero_adapter_and_full_image_mean_on_odd_geometry(self):
        torch.set_num_threads(2)
        g=torch.Generator().manual_seed(19)
        v2=.5+torch.rand(1,1,65,97,generator=g)
        moge=v2*torch.exp(.04*torch.randn(1,1,65,97,generator=g))
        fixed,bands=fixed_depth_and_bands(v2,moge)
        context=build_gate_context(torch.rand(1,3,65,97,generator=g),torch.rand(1,1,65,97,generator=g),
            torch.randn(1,3,65,97,generator=g),torch.ones(1,1,65,97),v2,moge,bands)
        model=LearnableLaplacianFusion()
        gate=model(context)['gate_delta']
        self.assertTrue(torch.equal(gate,torch.zeros_like(gate)))
        inputs={'fixed_depth':fixed,'bands':bands,'mean_coeff':mean_coefficients(bands)}
        output=apply_gates_full(inputs,gate)
        self.assertTrue(torch.equal(output['depth'],fixed))
        varied=.3*torch.randn(1,6,24,32,generator=g).tanh()
        output=apply_gates_full(inputs,varied)
        self.assertLess(abs(float(output['log_correction'].mean())),1e-8)
        self.assertTrue(torch.isfinite(output['depth']).all())
        self.assertTrue((output['depth']>0).all())


if __name__=='__main__':unittest.main()
