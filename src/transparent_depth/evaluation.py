"""Label-only native metrics, isolated from RGB inference and bundle loading.

Depth/normal/edge mathematics lives once in native_fusion.evaluation/geometry,
which is copied verbatim from the production evaluator. This adapter restores
its boundary-coordinate camera convention from Stage2 integer-centre K.
"""
from pathlib import Path
import json
import numpy as np
import torch

from v2_lidf.data import load_native_labels, load_label_manifest, read_exr_channels
from .stage2 import file_sha


def load_evaluation_labels(row, device='cpu'):
    labels = load_native_labels(row, include_normals=False)
    camera = row.get('original_row', {}).get('camera', {})
    calibration_valid = row.get('normal_camera_valid', camera.get('normal_camera_valid'))
    if calibration_valid is None:
        raise ValueError('Specify normal_camera_valid; default-FOV geometry must not be silently accepted')
    k = labels['intrinsics'].copy()
    k[0, 2] += .5
    k[1, 2] += .5
    if camera and (camera.get('pixel_center_offset') != .5 or
                   not np.allclose(k, camera['intrinsics'], atol=1e-9, rtol=1e-9)):
        raise ValueError('Stage2 and evaluation camera coordinates differ')
    target = torch.from_numpy(labels['target'])[None, None]
    valid = torch.from_numpy(labels['valid'])[None, None]
    batch = {'target': target, 'valid': valid, 'content': torch.ones_like(valid),
             'transparent': torch.from_numpy(labels['transparent'])[None, None],
             'intrinsics': torch.from_numpy(k)[None],
             'normal_camera_valid': torch.tensor([bool(calibration_valid)]),
             'normal_reference_available': torch.tensor([bool(row.get('normal_path'))])}
    if row.get('normal_path'):
        # Preserve the production torch float32 normal normalization exactly.
        # NumPy normalization can differ by a float32 ULP near metric thresholds.
        raw = torch.stack([torch.from_numpy(value.copy()) for value in
                           read_exr_channels(row['normal_path'], ('R', 'G', 'B'))])
        finite = torch.isfinite(raw).all(0, keepdim=True)
        clean = torch.where(torch.isfinite(raw), raw, torch.zeros_like(raw))
        norm = torch.linalg.vector_norm(clean, dim=0, keepdim=True)
        valid_normal = finite & ~(clean == -1).all(0, keepdim=True) & (norm >= .5) & (norm <= 1.5)
        cv = clean * clean.new_tensor([1., -1., -1.])[:, None, None]
        normal = torch.where(valid_normal, torch.nn.functional.normalize(cv, dim=0, eps=1e-6), torch.zeros_like(cv))
        batch.update(target_normal=normal[None], normal_valid=valid_normal[None])
    return {name: value.to(device) for name, value in batch.items()}


def evaluate_native_prediction(prediction, label_row, device='cpu'):
    from native_fusion.evaluation import evaluate_prediction
    value = np.asarray(prediction)
    if value.dtype != np.float32 or tuple(value.shape) != tuple(label_row['native_hw']):
        raise ValueError('Prediction must be native float32 HW')
    if not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError('Prediction must contain finite positive optical-Z metres')
    tensor = torch.from_numpy(value.copy())[None, None].to(device)
    # This API is explicitly a labelled metric operation, never inference.
    return evaluate_prediction(tensor, load_evaluation_labels(label_row, device))


def evaluate_sealed_predictions(prediction_manifest, label_manifest, output, expected_images=1226, device='cpu'):
    """Accept an explicit sealed prediction roster before any GT file is read.

    Roster: {status:'complete', labels_used_as_input:false, predictions:[
      {cache_id, methods:{name:{path:<relative .npy>,sha256}}}]}. No checkpoint
    selection claim is inferred; caller must publish its selection provenance.
    """
    from native_fusion.evaluation import aggregate
    manifest = Path(prediction_manifest)
    seal = json.loads(manifest.read_text())
    rows = seal.get('predictions', [])
    if seal.get('status') != 'complete' or seal.get('labels_used_as_input') is not False:
        raise ValueError('A completed label-free prediction seal is required')
    if len(rows) != expected_images or len({r['cache_id'] for r in rows}) != expected_images:
        raise ValueError('Incomplete or duplicate prediction roster')
    # Authenticate the complete set before opening even the label manifest.
    for row in rows:
        for ref in row['methods'].values():
            path = (manifest.parent / ref['path']).resolve()
            if file_sha(path) != ref['sha256']:
                raise ValueError('Sealed prediction changed: ' + row['cache_id'])
    labels = {row['cache_id']: row for row in load_label_manifest(label_manifest, 'evaluation')}
    if set(labels) != {row['cache_id'] for row in rows}:
        raise ValueError('Prediction/label rosters differ')
    records = []
    for row in rows:
        label = labels[row['cache_id']]
        methods = {name: np.load(manifest.parent / ref['path'], allow_pickle=False)
                   for name, ref in row['methods'].items()}
        metrics = {name: evaluate_native_prediction(value, label, device) for name, value in methods.items()}
        records.append({key: label[key] for key in ('cache_id', 'sample_id', 'domain', 'official_split', 'benchmark_group')}
                       | {'metrics': metrics, 'normal_reference': 'synthetic_released_camera_normals'
                          if label.get('normal_path') else 'real_depth_induced_proxy'})
    tests = [row for row in records if row['official_split'].endswith('test')]
    if expected_images == 1226 and len(tests) != 521:
        raise ValueError('The full ClearGrasp report must retain the original 521 test images')
    summary = {'all': aggregate(records), 'test': aggregate(tests) if tests else None,
               'images': len(records), 'test_images': len(tests),
               'prediction_manifest_sha256': file_sha(manifest),
               'label_manifest_sha256': file_sha(label_manifest),
               'selection': seal.get('selection'),
               'note': 'Full1226 includes official validation and historical exposure; a sweep winner is test-informed.'}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'per_image.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (output / 'complete.json').write_text(json.dumps({'status': 'complete', 'images': len(records),
        'metrics_sha256': file_sha(output / 'metrics.json'),
        'per_image_sha256': file_sha(output / 'per_image.jsonl')}, indent=2) + '\n')
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-images', type=int, default=1226)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    evaluate_sealed_predictions(args.predictions, args.labels, args.output, args.expected_images, args.device)


if __name__ == '__main__':
    main()
