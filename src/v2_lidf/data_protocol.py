"""Versioned full-field 256 protocol shared by preparation and evaluation.

Depth is optical Z in metres. Native validity for area integration is exactly
finite and positive; the fixed metric range is applied to the integrated GT.
No inpainting, depth normalization, cropping, or image-scale depth correction.
"""
import copy

import cv2
import numpy as np

PROTOCOL = {
    'schema': 'rgb_fusion_transcg_fullfield256_v1',
    'model_hw': [256, 256],
    'field_of_view': 'full_resize_no_crop_no_pad',
    'rgb_interpolation': 'INTER_AREA',
    'prediction_interpolation': 'INTER_LINEAR',
    'gt_interpolation': 'valid_weighted_INTER_AREA',
    'mask_interpolation': 'nearest_exact_pixel_centres',
    'coverage_threshold': 0.5,
    'coverage_epsilon': 1e-8,
    'depth_range_m': [0.1, 2.0],
    'depth_range_application': 'inclusive_after_valid_positive_area_integration',
    'unit': 'm',
    'coordinate': 'optical_z',
    'pixel_convention': 'integer_pixel_centres',
    'intrinsics_resize': 'half_pixel_once',
    'augmentation': 'none',
}

CLEARGRASP_PROTOCOL = dict(PROTOCOL, schema='rgb_fusion_cleargrasp_fullfield256_v1')
TRANSFER_PROTOCOLS = ('transcg256', 'cleargrasp256')


def validate_partition(row, experiment=None):
    """Selection policy is part of the experiment, never inferred from labels."""
    schema = row.get('preprocessing', {}).get('schema')
    inferred = {PROTOCOL['schema']: 'transcg256',
                CLEARGRASP_PROTOCOL['schema']: 'cleargrasp256'}.get(schema)
    if inferred is None or (experiment is not None and inferred != experiment):
        raise ValueError('Experiment/preprocessing identity mismatch')
    role = row['role']
    if inferred == 'cleargrasp256':
        expected = {'train': ('synthetic', 'train'), 'evaluation': ('real', 'real-val')}
        if (row.get('dataset') != 'cleargrasp' or role not in expected
                or (row.get('domain'), row.get('official_split')) != expected[role]):
            raise ValueError('ClearGrasp partition leakage or unsupported dataset')
    elif role in {'train', 'val'}:
        if row.get('dataset') != 'transcg' or row.get('official_split') != 'train':
            raise ValueError('Training/selection may only use official transCG train scenes')
    return inferred


def validate_protocol(value):
    if not isinstance(value, dict) or set(value) != set(PROTOCOL):
        raise ValueError('Complete explicit 256 preprocessing protocol required')
    if value['schema'] not in (PROTOCOL['schema'], CLEARGRASP_PROTOCOL['schema']):
        raise ValueError('Unsupported experiment preprocessing schema')
    for key in PROTOCOL:
        if key not in {'depth_range_m', 'schema'} and value[key] != PROTOCOL[key]:
            raise ValueError('Unsupported preprocessing setting: ' + key)
    limits = value['depth_range_m']
    if (len(limits) != 2 or not np.isfinite(limits).all()
            or not 0 < limits[0] < limits[1]):
        raise ValueError('Explicit finite positive metric depth interval required')
    return copy.deepcopy(value)


def weighted_area_depth(depth, protocol):
    protocol = validate_protocol(protocol)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError('GT must be a 2D optical-Z image')
    size = tuple(reversed(protocol['model_hw']))
    native_valid = np.isfinite(depth) & (depth > 0)
    coverage = cv2.resize(native_valid.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    numerator = cv2.resize(np.where(native_valid, depth, 0).astype(np.float32),
                           size, interpolation=cv2.INTER_AREA)
    target = (numerator / np.maximum(coverage, protocol['coverage_epsilon'])).astype(np.float32)
    low, high = protocol['depth_range_m']
    valid = ((coverage >= protocol['coverage_threshold']) & np.isfinite(target)
             & (target >= low) & (target <= high))
    # Values with insufficient support remain in the artifact for auditing;
    # only valid pixels can supervise any loss or curriculum input.
    return target, valid, coverage.astype(np.float32)


def resize_supervision(native, protocol):
    from .data import nearest_exact_resize
    target, valid, coverage = weighted_area_depth(native['target'], protocol)
    return {'target': target, 'valid': valid, 'coverage': coverage,
            'transparent': nearest_exact_resize(native['transparent'], protocol['model_hw']).astype(bool)}
