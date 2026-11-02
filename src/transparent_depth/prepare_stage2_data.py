"""Prepare portable RGB/P/K caches and separate curriculum supervision.

Input CSV/JSONL accepts only sample_id, rgb_path, intrinsics (native integer-
centre 3x3 JSON), role/roles, and optional identity/calibration/domain fields.
Labels live in a different CSV/JSONL and must explicitly declare rectified
optical-Z metres. A sealed portable Stage1 batch supplies native RGB predictions.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from v2_lidf.data import (INPUT_SCHEMA, LABEL_SCHEMA, load_input_manifest,
    validate_input_row, prepare_label_cache, atomic_json, atomic_npz, resolve_data_path)
from .stage2 import file_sha, read_intrinsics, resize_native_inputs

INPUT_FIELDS = {'sample_id', 'rgb_path', 'rgb_sha256', 'intrinsics', 'intrinsics_json',
    'pixel_convention', 'role', 'roles', 'domain', 'official_split', 'benchmark_group'}
TRANSFER_INPUT_FIELDS = {'dataset', 'scene', 'camera', 'split_sha256', 'intrinsics_source', 'intrinsics_source_sha256'}
LABEL_FIELDS = {'sample_id', 'target_depth_path', 'transparent_mask_path', 'normal_path',
                'depth_semantics', 'normal_camera_valid'}
TRANSFER_LABEL_FIELDS = {'depth_divisor', 'camera'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(newline='') as stream:
            return [{k: v for k, v in row.items() if v not in ('', None)} for row in csv.DictReader(stream)]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve(path, relative_to):
    value = Path(resolve_data_path(path))
    return value.resolve() if value.is_absolute() else (Path(relative_to).parent / value).resolve()


def read_roles(row):
    roles = row.get('roles', [row.get('role')])
    if isinstance(roles, str):
        roles = json.loads(roles) if roles.startswith('[') else [roles]
    if not roles or len(set(roles)) != len(roles) or set(roles) - {'train', 'val', 'evaluation'}:
        raise ValueError('Roles must be unique train/val/evaluation values')
    if 'train' in roles and len(roles) != 1:
        raise ValueError('One sample identity may not be both training and validation/evaluation')
    return roles


def load_predictions(path):
    path = Path(path).resolve()
    seal_path = path.parent / 'complete.json'
    seal = json.loads(seal_path.read_text())
    contract = json.loads((path.parent / 'contract.json').read_text())
    if (seal.get('schema') != 'portable_stage1_predictions_seal_v1'
            or seal.get('status') != 'complete' or seal.get('ground_truth_access') is not False
            or seal.get('all_arrays_sha256_verified') is not True
            or seal.get('manifest_sha256') != file_sha(path)
            or seal.get('contract_sha256') != fingerprint(contract)
            or contract.get('ground_truth_inputs') is not False):
        raise ValueError('Stage1 predictions must have a complete RGB-only source seal')
    rows = read_table(path)
    indexed = {row['sample_id']: row for row in rows}
    if len(indexed) != len(rows) or len(rows) != seal['images']:
        raise ValueError('Incomplete/duplicate Stage1 prediction roster')
    producer_weights = None
    for row in rows:
        metadata = row.get('metadata', {})
        weights = metadata.get('weight_sha256')
        if (not isinstance(weights, dict) or not weights.get('v2') or not weights.get('laplacian_e08')
                or metadata.get('ground_truth_access') is not False
                or metadata.get('assets_sha256') != contract['assets_sha256']
                or metadata.get('source_manifest_sha256') != contract['sources_sha256']):
            raise ValueError('Stage1 receipt omits authenticated producer asset/source identity')
        if producer_weights is not None and weights != producer_weights:
            raise ValueError('Stage1 roster mixes different producer weights')
        producer_weights = weights
        ref = row['prediction']
        relative = Path(ref['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Stage1 prediction path must be manifest-relative')
        native = path.parent / relative
        if (row.get('ground_truth_access') is not False or row.get('status') != 'complete'
                or row.get('contract_sha256') != seal['contract_sha256']
                or ref.get('dtype') != 'float32' or ref.get('unit') != 'm'
                or ref.get('coordinate') != 'optical_z'
                or file_sha(native) != ref['sha256'] or native.stat().st_size != ref['bytes']):
            raise ValueError('Stage1 prediction receipt/bytes differ: ' + row['sample_id'])
    return indexed, {**seal, 'producer_weights_sha256': producer_weights,
                     'producer_contract': contract}


def prepare(input_roster, label_roster, predictions, output, preprocessing=None):
    input_roster, predictions, output = Path(input_roster), Path(predictions), Path(output)
    raw_inputs = read_table(input_roster)
    if not raw_inputs:
        raise ValueError('Input roster is empty')
    if preprocessing is not None:
        from v2_lidf.data_protocol import validate_protocol
        preprocessing = validate_protocol(preprocessing)
    model_hw = preprocessing['model_hw'] if preprocessing else [240, 320]
    # Parse only the RGB/K roster and RGB-only predictions before any labels.
    predicted, producer_seal = load_predictions(predictions)
    contract = {'schema': 'transparent_depth.stage2.prepare.v1',
        'rgb_k_roster_sha256': file_sha(input_roster),
        'predictions_manifest_sha256': file_sha(predictions),
        'producer_seal_sha256': file_sha(predictions.parent / 'complete.json'),
        'source_sha256': file_sha(__file__), 'model_hw': model_hw,
        'input_gt_or_mask_access': False}
    if preprocessing:
        from v2_lidf import data_protocol, data
        contract.update(preprocessing=preprocessing,
            preprocessing_sources={m.__name__: file_sha(m.__file__) for m in (data_protocol, data)},
            resize_source_sha256=file_sha(Path(__file__).with_name('stage2.py')))
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / 'preparation.contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Preparation output belongs to a different source/roster; use a new directory')
    atomic_json(contract_path, contract)
    inputs, receipts, per_sample, seen_roles, roles_by_sample = [], {}, {}, set(), {}
    stage1 = output / 'stage1'
    for raw in raw_inputs:
        if set(raw) - (INPUT_FIELDS | TRANSFER_INPUT_FIELDS if preprocessing else INPUT_FIELDS):
            raise ValueError('RGB/K roster contains unapproved fields, including possible labels')
        sample = raw['sample_id']
        if not isinstance(sample, str) or not sample:
            raise ValueError('sample_id must be a nonempty string')
        roles = read_roles(raw)
        if sample not in predicted:
            raise ValueError('No sealed RGB Stage1 prediction: ' + sample)
        rgb_path = resolve(raw['rgb_path'], input_roster)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        rgb_sha = file_sha(rgb_path)
        source = predicted[sample]
        if source['rgb_sha256'] != rgb_sha or raw.get('rgb_sha256', rgb_sha) != rgb_sha or source['native_hw'] != list(rgb.shape[:2]):
            raise ValueError('RGB identity/shape differs from Stage1')
        if raw.get('pixel_convention') != 'integer_pixel_centres':
            raise ValueError('Declare pixel_convention=integer_pixel_centres explicitly')
        if ('intrinsics' in raw) == ('intrinsics_json' in raw):
            raise ValueError('Supply exactly one native intrinsics array or intrinsics_json path')
        if 'intrinsics_json' in raw:
            k = read_intrinsics(resolve(raw['intrinsics_json'], input_roster))
        else:
            value = raw['intrinsics']
            k = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=np.float64)
        native_path = predictions.parent / source['prediction']['path']
        native = np.load(native_path, allow_pickle=False)
        rgb_low, depth_low, k_low = resize_native_inputs(rgb, native, k, model_hw,
            preprocessing['rgb_interpolation'] if preprocessing else 'INTER_LINEAR')
        cache_id = 'sample_' + hashlib.sha256(sample.encode()).hexdigest()[:24]
        base = {'schema_version': INPUT_SCHEMA, 'id': cache_id, 'item_id': cache_id,
            'cache_id': cache_id, 'sample_id': sample, 'rgb_path': str(rgb_path), 'rgb_sha256': rgb_sha,
            'native_hw': list(rgb.shape[:2]), 'model_hw': model_hw,
            'intrinsics_native': k.tolist(), 'intrinsics': k_low.tolist(),
            'intrinsics_pixel_convention': 'integer_pixel_centres',
            'intrinsics_source_kind': 'explicit_user_calibration',
            'unit': 'm', 'coordinate': 'optical_z', 'domain': raw['domain'],
            'official_split': raw['official_split'], 'benchmark_group': raw.get('benchmark_group', raw['domain'])}
        if preprocessing:
            base.update(preprocessing=preprocessing, **{k: raw[k] for k in
                ('dataset', 'scene', 'camera', 'split_sha256', 'intrinsics_source', 'intrinsics_source_sha256')})
            base['intrinsics_source_kind'] = ('official_transcg_info' if raw['dataset']=='transcg'
                                               else ('released_synthetic_per_frame_fov_integer_centres' if raw['domain']=='synthetic'
                                                     else 'released_real_yaml_integer_centres'))
            if raw['rgb_path'].startswith('${'):
                base['rgb_path'] = raw['rgb_path']
        if base['domain'] not in {'real', 'synthetic'}:
            raise ValueError('Domain must be real or synthetic')
        if sample in per_sample and per_sample[sample] != base:
            raise ValueError('Shared sample identity has inconsistent RGB/K/metadata')
        per_sample[sample] = base
        for role in roles:
            key = (sample, role)
            if key in seen_roles:
                raise ValueError('Duplicate sample/role identity')
            seen_roles.add(key)
            row = dict(base, role=role)
            validate_input_row(row)
            inputs.append(row)
        all_roles = roles_by_sample.setdefault(sample, set())
        all_roles.update(roles)
        if 'train' in all_roles and len(all_roles) > 1:
            raise ValueError('Shared train/validation identity is forbidden')
        input_path = stage1 / 'inputs' / (cache_id + '.npz')
        identity = {'sample_id': sample, 'rgb_sha256': rgb_sha,
                    'native_prediction_sha256': source['prediction']['sha256'],
                    'intrinsics_native': k.tolist(), 'preparation_contract_sha256': fingerprint(contract),
                    'producer_contract_sha256': producer_seal['contract_sha256']}
        receipt_path = stage1 / 'receipts' / (cache_id + '.json')
        reusable = False
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            reusable = receipt.get('source_identity') == identity and file_sha(input_path) == receipt['input_sha256']
            if not reusable:
                raise ValueError('Existing input cache changed: ' + cache_id)
        if not reusable:
            atomic_npz(input_path, rgb=rgb_low, depth_input=depth_low, intrinsics=k_low,
                metadata_json=np.array(canonical({'item_id': cache_id, 'rgb_sha256': rgb_sha,
                    'source_identity': identity, 'ground_truth_access': False,
                    'intrinsics_pixel_convention': 'integer_pixel_centres'})))
            receipt = {'cache_id': cache_id, 'rgb_sha256': rgb_sha,
                       'input_sha256': file_sha(input_path), 'source_identity': identity}
            atomic_json(receipt_path, receipt)
        if 'evaluation' in roles:
            native_output = stage1 / 'native' / (cache_id + '.npy')
            native_output.parent.mkdir(parents=True, exist_ok=True)
            if not native_output.exists():
                shutil.copyfile(native_path, native_output)
            if file_sha(native_output) != source['prediction']['sha256']:
                raise ValueError('Native evaluation baseline changed')
            receipt['native_sha256'] = source['prediction']['sha256']
        receipts[cache_id] = {**receipts.get(cache_id, {}), **receipt}
    if set(per_sample) != set(predicted):
        raise ValueError('RGB/K roster must cover exactly the sealed Stage1 sample IDs')
    manifests = output / 'manifests'
    manifests.mkdir(exist_ok=True)
    input_manifest = manifests / 'input_manifest.jsonl'
    input_manifest.write_text(''.join(canonical(row)+'\n' for row in inputs))
    load_input_manifest(input_manifest)
    sealed = stage1 / 'inputs.sealed.jsonl'
    sealed.write_text(''.join(canonical(receipts[key])+'\n' for key in sorted(receipts)))
    input_marker = {'status': 'complete', 'input_manifest_sha256': file_sha(input_manifest),
        'sealed_inputs_manifest_sha256': file_sha(sealed), 'all_arrays_sha256_verified': True,
        'ground_truth_access': False, 'samples': len(receipts),
        'role_counts': dict(Counter(row['role'] for row in inputs)),
        'source_predictions_sha256': file_sha(predictions), 'preparation_contract_sha256': fingerprint(contract),
        'producer_weights_sha256': producer_seal['producer_weights_sha256'],
        'producer_contract': producer_seal['producer_contract'],
        'producer_contract_sha256': producer_seal['contract_sha256'],
        'preparation_contract': contract}
    atomic_json(stage1 / 'complete.json', input_marker)
    # Supervision is deliberately opened only after RGB/P/K caches are sealed.
    labels, label_receipts = [], {}
    if label_roster:
        label_contract = {'label_roster_sha256': file_sha(label_roster),
                          'input_manifest_sha256': file_sha(input_manifest), 'source_sha256': file_sha(__file__)}
        label_contract_path = output / 'labels' / 'preparation.contract.json'
        if label_contract_path.exists() and json.loads(label_contract_path.read_text()) != label_contract:
            raise ValueError('Supervision preparation source changed; use a new output directory')
        atomic_json(label_contract_path, label_contract)
        originals = read_table(label_roster)
        indexed = {row['sample_id']: row for row in originals}
        if len(indexed) != len(originals) or set(indexed) != set(per_sample):
            raise ValueError('Separate supervision roster must cover the declared input samples exactly')
        for row in inputs:
            raw = indexed[row['sample_id']]
            allowed_semantics = {'already_rectified_optical_z_m'}
            if preprocessing:
                allowed_semantics.add('transcg_uint16_optical_z')
            if set(raw) - (LABEL_FIELDS | TRANSFER_LABEL_FIELDS if preprocessing else LABEL_FIELDS) or raw.get('depth_semantics') not in allowed_semantics:
                raise ValueError('Labels require an explicit supported optical-Z unit contract')
            camera_valid = raw['normal_camera_valid']
            if isinstance(camera_valid, str):
                camera_valid = json.loads(camera_valid.lower())
            if type(camera_valid) is not bool:
                raise ValueError('normal_camera_valid must be explicit boolean')
            label = {key: row[key] for key in ('item_id','cache_id','sample_id','role','domain','official_split',
                'benchmark_group','rgb_sha256','native_hw','model_hw','intrinsics_native')}
            label.update(schema_version=LABEL_SCHEMA, depth_semantics=raw['depth_semantics'],
                target_depth_path=str(resolve(raw['target_depth_path'], label_roster)),
                transparent_mask_path=str(resolve(raw['transparent_mask_path'], label_roster)),
                normal_path=str(resolve(raw['normal_path'], label_roster)) if raw.get('normal_path') else None,
                normal_camera_valid=camera_valid)
            if preprocessing:
                required_semantics = {'transcg':'transcg_uint16_optical_z',
                                      'cleargrasp':'already_rectified_optical_z_m'}
                if raw['depth_semantics'] != required_semantics.get(row['dataset']):
                    raise ValueError('Dataset and depth encoding disagree')
                label.update(preprocessing=preprocessing, **{k: row[k] for k in
                    ('dataset', 'scene', 'camera', 'split_sha256')})
                for field in ('target_depth_path', 'transparent_mask_path'):
                    if raw[field].startswith('${'):
                        label[field] = raw[field]
                if label['depth_semantics'] == 'transcg_uint16_optical_z':
                    if raw.get('camera') != row['camera']:
                        raise ValueError('Supervision camera differs from RGB/K')
                    label['depth_divisor'] = raw['depth_divisor']
            if row['role'] in {'train', 'val'}:
                prepare_label_cache(label, output / 'labels', overwrite=False)
                cache_path = output / 'labels' / (row['cache_id'] + '.npz')
                with np.load(cache_path, allow_pickle=False) as cache:
                    meta = json.loads(str(cache['metadata_json'].item()))
                label['source_sha256'] = meta['source_sha256']
                label_receipts[row['cache_id']] = {'cache_id': row['cache_id'], 'role': row['role'],
                    'rgb_sha256': row['rgb_sha256'], 'label_sha256': file_sha(cache_path)}
            else:
                label['source_sha256'] = {'depth': file_sha(label['target_depth_path']),
                    'transparent_mask': file_sha(label['transparent_mask_path'])}
            if label['normal_path']:
                label['source_sha256']['normal'] = file_sha(label['normal_path'])
            labels.append(label)
        (manifests / 'label_manifest.jsonl').write_text(''.join(canonical(row)+'\n' for row in labels))
    elif any(row['role'] in {'train', 'val'} for row in inputs):
        raise ValueError('Training/validation preparation requires the separate supervision roster')
    if label_roster:
        label_seal = output / 'labels' / 'labels.sealed.jsonl'
        label_seal.parent.mkdir(parents=True, exist_ok=True)
        label_seal.write_text(''.join(canonical(label_receipts[key])+'\n' for key in sorted(label_receipts)))
        atomic_json(output / 'labels' / 'complete.json', {'status': 'complete',
            'schema': 'transparent_depth.supervision_cache.v1',
            'label_manifest_sha256': file_sha(manifests / 'label_manifest.jsonl'),
            'sealed_labels_manifest_sha256': file_sha(label_seal),
            'all_arrays_sha256_verified': True, 'images': len(label_receipts),
            'training_ground_truth_access': True, 'transparent_gt_depth_used_as_model_input': False})
    result = {'status': 'complete', 'samples': len(per_sample), 'roles': input_marker['role_counts'],
        'input_manifest_sha256': file_sha(input_manifest),
        'label_manifest_sha256': file_sha(manifests/'label_manifest.jsonl') if label_roster else None,
        'training_ground_truth_access': bool(label_roster), 'rgb_p_k_cache_ground_truth_access': False,
        'transparent_gt_depth_used_as_model_input': False}
    atomic_json(output / 'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-roster', '--inputs', dest='inputs', required=True)
    parser.add_argument('--label-roster', '--labels', dest='labels')
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', help='Training or smoke JSON containing the versioned preprocessing protocol')
    args = parser.parse_args()
    import cv2
    import torch
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    protocol = json.loads(Path(args.config).read_text())['preprocessing'] if args.config else None
    print(json.dumps(prepare(args.inputs, args.labels, args.predictions, args.output, protocol), indent=2))


if __name__ == '__main__':
    main()
