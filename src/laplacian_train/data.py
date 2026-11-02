"""Authenticated native patches with full-image Laplacian geometry.

The cached bands and global mean coefficients are made from complete images.
Cropping only removes supervised pixels, never recomputes a pyramid or its mean.
Labels live in separate files and are returned in a separate dictionary.
"""
from __future__ import annotations
import os

import hashlib
import io
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(os.environ.get('TD_STAGE1_LAP_ROOT', 'artifacts/stage1_training/laplacian')).resolve()
SCHEMA = 'fusion_v2_trainable_laplacian_native_patch_v1'
PATCH_SIZE = 256
HALO = 32
LEVELS = 6
GATE_HW = (24, 32)
SEED = 20261023
INPUT_KEYS = ('fixed_depth', 'bands', 'context', 'mean_coeff', 'native_hw', 'crop_top_left')
LABEL_KEYS = ('target', 'valid', 'transparent', 'target_normal', 'normal_valid', 'K', 'camera_valid', 'core')
IDENTITY_KEYS = ('cache_id', 'sample_id', 'rgb_sha256', 'split', 'domain', 'native_hw')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_asset(asset):
    raw = Path(asset['path']).read_bytes()
    if len(raw) != asset['bytes'] or hashlib.sha256(raw).hexdigest() != asset['sha256']:
        raise ValueError('changed Laplacian cache asset: ' + asset['path'])
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


def validate_inputs(payload):
    if set(payload) != set(INPUT_KEYS):
        raise ValueError('only the explicit RGB-derived input whitelist is allowed')
    shapes = {'fixed_depth': (1, PATCH_SIZE, PATCH_SIZE), 'bands': (LEVELS, PATCH_SIZE, PATCH_SIZE),
              'context': (16, *GATE_HW), 'mean_coeff': (LEVELS, *GATE_HW),
              'native_hw': (2,), 'crop_top_left': (2,)}
    for key, shape in shapes.items():
        value = payload[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError('invalid input: ' + key)
        if key in ('native_hw', 'crop_top_left'):
            if value.dtype != torch.int64:
                raise ValueError('native pixel metadata must be int64')
        elif value.dtype != torch.float32:
            raise ValueError('cached floating inputs must retain FP32: ' + key)
    if not (payload['fixed_depth'] > 0).all():
        raise ValueError('fixed depth must be positive optical-Z metres')
    hw, origin = payload['native_hw'], payload['crop_top_left']
    if (origin < 0).any() or (origin + PATCH_SIZE > hw).any():
        raise ValueError('native crop lies outside the source image')


def validate_labels(payload):
    if set(payload) != set(LABEL_KEYS):
        raise ValueError('supervision field mismatch')
    booleans = ('valid', 'transparent', 'normal_valid', 'camera_valid', 'core')
    for key in LABEL_KEYS:
        shape = (() if key == 'camera_valid' else (3, 3) if key == 'K'
                 else (3 if key == 'target_normal' else 1, PATCH_SIZE, PATCH_SIZE))
        value = payload[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError('invalid supervision: ' + key)
        if key in booleans and value.dtype != torch.bool:
            raise ValueError('supervision support must be boolean: ' + key)
    if (payload['target'][payload['valid']] <= 0).any():
        raise ValueError('valid target depth must be positive')
    expected = torch.zeros_like(payload['core'])
    expected[..., HALO:-HALO, HALO:-HALO] = True
    if not torch.equal(expected, payload['core']):
        raise ValueError('supervised core must be central 192x192')
    k = payload['K']
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not torch.equal(k[2], k.new_tensor([0, 0, 1])):
        raise ValueError('invalid camera matrix')


def verify_record(record, identity, contract_sha256, root, *, labels=True):
    if (any(record[k] != identity[k] for k in IDENTITY_KEYS)
            or record.get('contract_sha256') != contract_sha256
            or record.get('input_ground_truth_access') is not False
            or record.get('test_labels_read') is not False or record['split'] != 'train'):
        raise ValueError('native patch identity or producer mismatch')
    for key, folder, validator in (('input_asset', 'inputs', validate_inputs),
                                    ('label_asset', 'labels', validate_labels)):
        if key == 'label_asset' and not labels:
            continue
        asset = record[key]
        expected = Path(root) / 'cache' / folder / 'train' / f"{record['cache_id']}.pt"
        if Path(asset['path']).resolve() != expected.resolve():
            raise ValueError('cache asset path differs from its identity')
        validator(load_asset(asset))


class NativePatchDataset(Dataset):
    """One fixed native patch for every image in the sealed pilot/full roster."""
    def __init__(self, root=ROOT, scope='pilot', *, include_target=True, verify_sources=True):
        if scope not in ('pilot', 'full'):
            raise ValueError('scope must be pilot or full; test labels are never supported')
        self.root, self.scope, self.include_target = Path(root), scope, include_target
        self.plan = read_json(self.root / f'cache/plan_{scope}.json')
        self.contract_sha256 = digest(self.plan['contract'])
        if verify_sources:
            for path, expected in self.plan['contract']['source_sha256'].items():
                if sha256(Path(__file__).resolve().parents[2] / path) != expected:
                    raise ValueError('cache producer source changed: ' + path)
        path = self.root / f'cache/manifest_{scope}.jsonl'
        marker = read_json(self.root / f'cache/complete_{scope}.json')
        self.rows = read_jsonl(path)
        if (marker.get('status') != 'complete' or marker['manifest_sha256'] != sha256(path)
                or marker['contract_sha256'] != self.contract_sha256
                or marker['plan_sha256'] != digest(self.plan)
                or marker['images'] != len(self.rows) or len(self.rows) != len(self.plan['images'])
                or marker.get('test_labels_read') is not False):
            raise ValueError('native Laplacian training cache is not complete and authenticated')
        if len({r['cache_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('duplicate native image in training cache')
        for row, expected in zip(self.rows, self.plan['images'], strict=True):
            if any(row[k] != expected[k] for k in IDENTITY_KEYS):
                raise ValueError('cache roster differs from planned image order')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if (row['split'] != 'train' or row['contract_sha256'] != self.contract_sha256
                or row['input_ground_truth_access'] is not False or row['test_labels_read'] is not False):
            raise ValueError('training cache provenance mismatch')
        inputs = load_asset(row['input_asset'])
        validate_inputs(inputs)
        if inputs['native_hw'].tolist() != row['native_hw'] or inputs['crop_top_left'].tolist() != row['crop_top_left']:
            raise ValueError('cached input pixel coordinates differ from receipt')
        result = {'inputs': inputs, **{k: row[k] for k in ('cache_id', 'sample_id', 'domain', 'cluster')},
                  'normal_reference': row['normal_reference']}
        if self.include_target:
            labels = load_asset(row['label_asset'])
            validate_labels(labels)
            result['labels'] = labels
        return result


PatchDataset = NativePatchDataset
