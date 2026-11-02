"""CPU-only behavioral tests for persistent export and file access isolation."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transparent_depth import stage1_batch as batch
from transparent_depth.input_guard import RGBInputGuard


class BatchTests(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder);rgb=root/'rgb';rgb.mkdir();output=root/'export';output.mkdir()
        entries=[]
        for i in range(3):
            path=rgb/f'{i}.png';Image.fromarray(np.full((7,9,3),i,dtype=np.uint8)).save(path)
            entries.append({'sample_id':f'image/{i}','rgb_path':str(path),'rgb_sha256':batch.sha256(path),'native_hw':[7,9]})
        manifest=root/'rgb.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in entries))
        assets=root/'assets';assets.mkdir();asset_manifest=root/'assets.json';asset_manifest.write_text('{}')
        args=SimpleNamespace(manifest=manifest,output=output,assets=asset_manifest,asset_root=assets,
            expected_count=3,rank=0,world=2,device='cpu',cache_root=None,deny_prefix=[])
        return args,entries

    def test_persistent_workers_resume_and_tamper_rejection(self):
        instances=[]
        class Assets:
            def __init__(self,manifest,root):self.manifest=manifest;self.root=root;self.manifest_sha256='a'*64
            def verify_sources(self):return 'b'*64
        class Predictor:
            device=torch.device('cpu')
            def __init__(self,*args,**kwargs):
                self.runtime_cache=Path(kwargs['runtime_cache'])
                instances.append(self)
            def predict(self,path):
                with Image.open(path) as image:shape=(image.height,image.width)
                return np.full(shape,.75,dtype=np.float32),{'rgb_sha256':batch.sha256(path)}
        with tempfile.TemporaryDirectory() as folder:
            args,rows=self.fixture(folder)
            # Restore the process environment/temp directory changed by the production cache setup.
            import os
            previous_temp=tempfile.tempdir
            try:
                with patch.dict(os.environ),patch.object(batch,'Stage1Assets',Assets),patch.object(batch,'Stage1Predictor',Predictor):
                    first=batch.predict(args);self.assertEqual(first['new_images'],2)
                    self.assertEqual(len(instances),1)
                    self.assertEqual(instances[-1].runtime_cache,(args.output/'runtime_cache/rank_000').resolve())
                    self.assertEqual(os.environ['TD_STAGE1_RUNTIME_CACHE'],str(instances[-1].runtime_cache))
                    self.assertFalse(instances[-1].runtime_cache.is_relative_to(batch.PROJECT/'outputs'))
                    args.cache_root=Path(folder)/'explicit_runtime_cache'
                    args.rank=1;second=batch.predict(args);self.assertEqual(second['new_images'],1)
                    self.assertEqual(len(instances),2)
                    self.assertEqual(instances[-1].runtime_cache,args.cache_root.resolve())
                    self.assertEqual(os.environ['TD_STAGE1_RUNTIME_CACHE'],str(args.cache_root.resolve()))
                    self.assertFalse(instances[-1].runtime_cache.is_relative_to(batch.PROJECT/'outputs'))
                    args.rank=0;resumed=batch.predict(args);self.assertEqual(resumed['reused_images'],2)
                    self.assertEqual(resumed['predictor_instances'],0);self.assertEqual(len(instances),2)
                    marker=batch.seal(args);self.assertEqual(marker['images'],3)
                    self.assertEqual(batch.seal(args),marker)
                    prediction=next((args.output/'predictions').glob('*.npy'))
                    prediction.write_bytes(b'corrupt')
                    with self.assertRaises(ValueError):batch.seal(args)
            finally:tempfile.tempdir=previous_temp
        self.assertFalse(torch.cuda.is_initialized())

    def test_manifest_rejects_labels_duplicates_and_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            args,rows=self.fixture(folder)
            for edited in ([{**r,'target':'not-allowed'} for r in rows],rows+[rows[0]],rows[:2]):
                args.manifest.write_text(''.join(json.dumps(r)+'\n' for r in edited))
                with self.assertRaises(ValueError):batch.read_rgb_manifest(args.manifest,3)
            rows[0]['rgb_path']='label.exr';args.manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):batch.read_rgb_manifest(args.manifest,3)

    def test_guard_blocks_old_path_and_labels_including_named_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);project=root/'project';project.mkdir();assets=root/'assets';assets.mkdir()
            rgb=root/'input.png';rgb.write_bytes(b'rgb');label=project/'gt.exr';label.write_bytes(b'label')
            old=root/'old';old.mkdir();file=old/'depth.pt';file.write_bytes(b'old prediction')
            guard=RGBInputGuard(project=project,asset_root=assets,rgb_paths=[rgb,label],
                output_root=root/'out',cache_root=root/'cache',deny_prefixes=[old])
            with guard:
                self.assertEqual(rgb.read_bytes(),b'rgb')
                with self.assertRaises(PermissionError):label.read_bytes()
                with self.assertRaises(PermissionError):file.read_bytes()
            self.assertEqual(len(guard.denials),2)


if __name__=='__main__':unittest.main()
