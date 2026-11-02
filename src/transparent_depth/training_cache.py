"""Portable producer and supervision authentication for fresh/resumed training."""
import hashlib
import json
from pathlib import Path

from v2_lidf.data import V2LIDFDataset, load_input_manifest, load_label_manifest, load_native_labels
from .stage2 import file_sha
from v2_lidf.data_protocol import TRANSFER_PROTOCOLS, validate_partition


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_prediction_caches(cfg, input_manifest, label_manifest, stage1_root):
    """Authenticate RGB/P/K producer for training OR independent evaluation."""
    stage1_root = Path(stage1_root)
    inputs = load_input_manifest(input_manifest)
    labels = load_label_manifest(label_manifest)
    producer = json.loads((stage1_root / 'complete.json').read_text())
    source = producer.get('producer_contract', {})
    preparation = producer.get('preparation_contract', {})
    weights = producer.get('producer_weights_sha256', {})
    expected = cfg['stage1']
    if cfg.get('data_protocol') in TRANSFER_PROTOCOLS:
        from v2_lidf import data_protocol, data
        project = Path(__file__).resolve().parents[2]
        if (source.get('sources_sha256') != file_sha(project/'provenance/stage1_runtime_sources.json')
                or source.get('assets_sha256') != file_sha(project/'provenance/stage1_assets.json')):
            raise ValueError('Stage1 producer source/asset registry differs from this release')
        if (preparation.get('preprocessing') != cfg['preprocessing']
                or preparation.get('model_hw') != cfg['input_hw']
                or preparation.get('preprocessing_sources') != {m.__name__:file_sha(m.__file__) for m in (data_protocol,data)}
                or preparation.get('resize_source_sha256') != file_sha(Path(__file__).with_name('stage2.py'))):
            raise ValueError('Stale 256 preprocessing cache; regenerate from sealed native P')
        for row in inputs:
            validate_partition(row, cfg['data_protocol'])
        by_input = {(r['role'],r['cache_id']):r for r in inputs}
        for row in labels:
            other = by_input.get((row['role'],row['cache_id']),{})
            if any(row.get(k) != other.get(k) for k in ('model_hw','preprocessing','split_sha256','intrinsics_native','camera','dataset','scene')):
                raise ValueError('Input/label geometry or split identity differs')
    if (producer.get('status') != 'complete' or producer.get('ground_truth_access') is not False
            or producer.get('input_manifest_sha256') != file_sha(input_manifest)
            or producer.get('sealed_inputs_manifest_sha256') != file_sha(stage1_root / 'inputs.sealed.jsonl')
            or producer.get('all_arrays_sha256_verified') is not True
            or producer.get('producer_contract_sha256') != fingerprint(source)
            or producer.get('preparation_contract_sha256') != fingerprint(preparation)
            or preparation.get('source_sha256') != file_sha(Path(__file__).with_name('prepare_stage2_data.py'))
            or source.get('ground_truth_inputs') is not False
            or weights.get('v2') != expected['checkpoint_sha256']
            or weights.get('laplacian_e08') != expected['adapter_checkpoint_sha256']
            or weights != expected['expected_asset_sha256']):
        raise ValueError('Stage1 cache producer/weights do not match the frozen RGB+E08 protocol')
    receipts = read_rows(stage1_root / 'inputs.sealed.jsonl')
    indexed = {r['cache_id']: r for r in receipts}
    if len(indexed) != len(receipts) or set(indexed) != {r['cache_id'] for r in inputs}:
        raise ValueError('Stage1 sealed roster differs from the inputs')
    for receipt in receipts:
        if (receipt.get('source_identity', {}).get('producer_contract_sha256') != producer['producer_contract_sha256']
                or receipt.get('source_identity', {}).get('preparation_contract_sha256') != producer['preparation_contract_sha256']
                or file_sha(stage1_root / 'inputs' / (receipt['cache_id'] + '.npz')) != receipt['input_sha256']):
            raise ValueError('Stage1 cache changed after producer sealing')
    if {(r['role'],r['cache_id']) for r in inputs} != {(r['role'],r['cache_id']) for r in labels}:
        raise ValueError('Input/supervision manifest coverage differs')
    return inputs, labels


def validate_training_caches(cfg, input_manifest, label_manifest, stage1_root, label_root):
    stage1_root, label_root = Path(stage1_root), Path(label_root)
    inputs, labels = validate_prediction_caches(cfg,input_manifest,label_manifest,stage1_root)
    seal = json.loads((label_root / 'complete.json').read_text())
    manifest = label_root / 'labels.sealed.jsonl'
    if (seal.get('schema') != 'transparent_depth.supervision_cache.v1' or seal.get('status') != 'complete'
            or seal.get('label_manifest_sha256') != file_sha(label_manifest)
            or seal.get('sealed_labels_manifest_sha256') != file_sha(manifest)
            or seal.get('all_arrays_sha256_verified') is not True
            or seal.get('training_ground_truth_access') is not True
            or seal.get('transparent_gt_depth_used_as_model_input') is not False):
        raise ValueError('An authenticated separate curriculum label cache is required')
    entries = read_rows(manifest)
    label_index = {r['cache_id']: r for r in entries}
    expected_rows = {r['cache_id']: r for r in labels if r['role'] in {'train','val'}}
    if len(label_index) != len(entries) or set(label_index) != set(expected_rows):
        raise ValueError('Supervision cache coverage differs from train/validation roster')
    for key, receipt in label_index.items():
        row = expected_rows[key]
        if (receipt['role'] != row['role'] or receipt['rgb_sha256'] != row['rgb_sha256']
                or file_sha(label_root / (key+'.npz')) != receipt['label_sha256']
                or not row.get('source_sha256', {}).get('depth')):
            raise ValueError('Supervision cache changed: ' + key)
    return {'stage1_complete_sha256': file_sha(stage1_root/'complete.json'),
            'labels_complete_sha256': file_sha(label_root/'complete.json'),
            'label_file_sha256': {k: r['label_sha256'] for k,r in label_index.items()}}


class AuthenticatedTrainingDataset(V2LIDFDataset):
    def __init__(self, *args, label_sha256, **kwargs):
        self.label_sha256 = label_sha256
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        key = self.rows[index]['cache_id']
        if file_sha(self.label_root / (key+'.npz')) != self.label_sha256[key]:
            raise ValueError('Supervision changed after training started: '+key)
        return super().__getitem__(index)


def load_authenticated_native_labels(row):
    # Called by validation only after its pure-P forward has completed.
    sources = row['source_sha256']
    for key, field in (('depth','target_depth_path'), ('transparent_mask','transparent_mask_path')):
        if file_sha(row[field]) != sources[key]:
            raise ValueError('Native validation supervision changed: '+row['cache_id'])
    return load_native_labels(row)
