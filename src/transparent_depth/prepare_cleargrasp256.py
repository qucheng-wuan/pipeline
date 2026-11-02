"""Read-only ClearGrasp audit and strictly separated, portable 256 rosters.

Audit decodes every RGB/mask, checks every EXR structure and frame calibration, and
hashes both encoded and decoded RGB for split leakage. No default/fallback K.
Use --decode-all for exhaustive GT pixel decoding. Smoke always decodes its GT.
An audit with errors cannot produce a full roster. Smoke selects four verified
frames explicitly and records all outstanding full-dataset errors.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import math
import os
from pathlib import Path
import re
import hashlib

import numpy as np
from PIL import Image

from v2_lidf.data import atomic_json, atomic_npz, load_native_labels, resize_intrinsics
from v2_lidf.data_protocol import CLEARGRASP_PROTOCOL, validate_protocol, resize_supervision
from .prepare_transfer_data import cleargrasp_records, portable_path, visualize_supervision
from .prepare_stage2_data import canonical, fingerprint
from .stage2 import file_sha


def synthetic_calibration(path):
    meta = json.loads(Path(path).read_text())
    h, w = meta['image']['height_px'], meta['image']['width_px']
    if any(type(v) is not int or v <= 0 for v in (h, w)):
        raise ValueError('Per-frame JSON must declare positive integer image dimensions')
    fov = meta['camera']['field_of_view']
    x, y = float(fov['x_axis_rads']), float(fov['y_axis_rads'])
    if not all(math.isfinite(v) and 0 < v < math.pi for v in (x, y)):
        raise ValueError('Per-frame FOV must be in radians and within (0, pi)')
    # Renderer centres are boundary coordinates W/2,H/2. Convert once to
    # integer pixel centres; subsequent resize_intrinsics uses half-pixel.
    k = [[w/(2*math.tan(x/2)), 0., (w-1)/2],
         [0., h/(2*math.tan(y/2)), (h-1)/2], [0., 0., 1.]]
    return [h, w], k


def synthetic_records(root, role):
    root = Path(root).resolve()
    if role not in ('train', 'val'):
        raise ValueError('Synthetic partition must be train or val')
    split = 'train' if role == 'train' else 'synthetic-val'
    specs = {'rgb': ('rgb-imgs', '-rgb.jpg'),
             'gt': ('depth-imgs-rectified', '-depth-rectified.exr'),
             'mask': ('segmentation-masks', '-segmentation-mask.png'),
             'calibration': ('json-files', '-masks.json')}
    tables, faults = {}, []
    for kind, (folder, suffix) in specs.items():
        table = {}
        for path in sorted(root.rglob('*'+suffix)):
            if path.parent.name != folder:
                faults.append({'path': str(path.relative_to(root)), 'error': 'Unexpected modality directory'})
                continue
            frame = path.name[:-len(suffix)]
            if not re.fullmatch(r'\d+', frame):
                faults.append({'path': str(path.relative_to(root)), 'error': 'Non-numeric frame'})
                continue
            key = (path.parent.parent.relative_to(root).as_posix(), int(frame))
            if key in table:
                faults.append({'key': list(key), 'error': 'Ambiguous duplicate '+kind})
            table[key] = path
        tables[kind] = table
    keys = set().union(*(set(t) for t in tables.values()))
    if not keys:
        raise ValueError('Empty synthetic dataset: '+str(root))
    records = []
    for scene, frame in sorted(keys):
        key = scene, frame
        missing = [name for name, table in tables.items() if key not in table]
        sample = 'cleargrasp/%s/%s/%09d' % (split, scene, frame)
        if missing:
            faults.append({'sample_id': sample, 'error': 'Missing paired modalities', 'missing': missing})
            continue
        rgb = tables['rgb'][key]
        normal = rgb.parent.parent/'camera-normals'/(rgb.name.removesuffix('-rgb.jpg')+'-cameraNormals.exr')
        records.append(dict(sample_id=sample, **{n:t[key] for n,t in tables.items()},
            normal=normal if normal.is_file() else None, root=root,
            root_var='CLEARGRASP_TRAIN_ROOT' if role=='train' else 'CLEARGRASP_SYNTHETIC_VAL_ROOT',
            dataset='cleargrasp', scene=scene, camera='synthetic_renderer', role=role,
            domain='synthetic', official_split=split, depth_semantics='already_rectified_optical_z_m'))
    return records, faults, {k:len(v) for k,v in tables.items()}


def discover(roots):
    records, faults, counts = [], [], {}
    for role in ('train',):
        rows, errors, sizes = synthetic_records(roots[role], role)
        records.extend(rows); faults.extend(errors); counts[role] = sizes
    real = cleargrasp_records(Path(roots['evaluation']).resolve())
    for row in real:
        row.update(root_var='CLEARGRASP_REAL_VAL_ROOT', domain='real', normal=None)
    records.extend(real)
    counts['evaluation'] = {kind:len(real) for kind in ('rgb','gt','mask','calibration')}
    return records, faults, counts


def check_record(task):
    record, protocol = task[:2]
    decode_gt = task[2] if len(task) > 2 else True
    row = {'sample_id':record['sample_id'], 'role':record['role'], 'scene':record['scene'],
           'camera':record['camera'], 'domain':record['domain'], 'official_split':record['official_split']}
    errors = []
    try:
        import cv2
        cv2.setNumThreads(1)
        # Complete image integrity/duplicate checks even when calibration is
        # absent. Observed RGB dimensions never become an invented camera K.
        with Image.open(record['rgb']) as image:
            image.load()
            if image.mode != 'RGB':
                raise ValueError('RGB image must be RGB mode')
            hw = [image.height, image.width]
            decoded = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        row.update(native_hw=hw, decoded_rgb_sha256=decoded,
            rgb_relative=record['rgb'].relative_to(record['root']).as_posix(),
            source_sha256={kind:file_sha(record[kind]) for kind in ('rgb','gt','mask','calibration')})
        try:
            calibrated_hw, k = (synthetic_calibration(record['calibration']) if record['domain']=='synthetic'
                                 else (record['native_hw'], record['k']))
            if (np.shape(k) != (3, 3) or not np.isfinite(k).all() or k[0][0] <= 0 or k[1][1] <= 0
                    or not np.array_equal(k[2], [0, 0, 1])):
                raise ValueError('Invalid camera matrix')
            if calibrated_hw != hw:
                raise ValueError('RGB dimensions differ from per-frame calibration')
            row.update(intrinsics_native=k, intrinsics256=resize_intrinsics(k, hw, [256,256]).tolist())
        except Exception as error:
            errors.append('calibration: '+type(error).__name__+': '+str(error))
        with Image.open(record['mask']) as mask_image:
            mask_image.load()
            mask = np.asarray(mask_image).copy()
            if mask_image.mode != 'L' or list(mask.shape) != hw or not np.isin(mask,[0,255]).all():
                raise ValueError('Mask dimensions or 0/255 encoding mismatch')
        import OpenEXR
        exr = OpenEXR.InputFile(str(record['gt']))
        try:
            header = exr.header(); window = header['dataWindow']
            if [window.max.y-window.min.y+1,window.max.x-window.min.x+1] != hw:
                raise ValueError('EXR dimensions differ from RGB')
            if 'R' not in header['channels'] or not exr.isComplete():
                raise ValueError('Missing depth channel or incomplete EXR chunks')
        finally:
            exr.close()
        row.update(mask_encoding=[0,255], depth_unit='m', depth_coordinate='optical_z', depth_conversion=False,
            gt_pixel_decode=decode_gt, exr_complete=True,
            optional_normal_present=bool(record.get('normal')),
            normals_used='existing_loss_GT256_K256_dx_cross_dy_valid_current_right_down')
        if decode_gt:
            from v2_lidf.data import read_exr_channels
            depth = read_exr_channels(str(record['gt']))[0]
            valid = np.isfinite(depth) & (depth>0)
            native = {'target':depth,'valid':valid,'transparent':mask==255}
            low = resize_supervision(native, protocol)
            row.update(native_positive_valid=int(valid.sum()), valid256=int(low['valid'].sum()),
                transparent_valid256=int((low['valid'] & low['transparent']).sum()),
                zero_coverage256=int((low['coverage']==0).sum()),
                gt_native_positive_range_m=([float(depth[valid].min()),float(depth[valid].max())]
                                           if valid.any() else None))
            if not low['valid'].any():
                raise ValueError('Empty GT after fixed coverage/range')
    except Exception as error:
        errors.append(type(error).__name__+': '+str(error))
    row['status'] = 'error' if errors else 'ok'
    if errors:
        row['error'] = '; '.join(errors)
    return row


def duplicates(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if 'source_sha256' in row and 'decoded_rgb_sha256' in row:
            digest = row['source_sha256']['rgb'] if field=='encoded' else row['decoded_rgb_sha256']
            groups[digest].append(row)
    return [{'sha256':key, 'sample_ids':[r['sample_id'] for r in group],
             'roles':sorted({r['role'] for r in group}),
             'cross_split':len({r['role'] for r in group})>1}
            for key,group in groups.items() if len(group)>1]


def historical_comparison(rows, reference):
    if reference is None:
        return {'status':'not_checked', 'exposure':'Historical frozen Stage1 used ClearGrasp train and validation including real-val.'}
    with gzip.open(reference, 'rt') as source:
        old = [json.loads(line) for line in source if line.strip()]
    prefixes = {'train':'cleargrasp-dataset-train/',
                'evaluation':'cleargrasp-dataset-test-val/real-val/'}
    report = {'reference_sha256':file_sha(reference), 'partitions':{}}
    for role, prefix in prefixes.items():
        prior = {r['rgb_path'][len(prefix):]:r for r in old if r['rgb_path'].startswith(prefix)}
        current = {r['rgb_relative']:r for r in rows if r['role']==role and 'rgb_relative' in r}
        # A frame may exist in the old reference yet have been evaluation-only.
        selected = {r['rgb_sha256'] for r in prior.values() if 'val' in r['roles']}
        report['partitions'][role] = {'current_rgb_decoded':len(current), 'historical_inventory':len(prior),
            'historical_roles':dict(Counter(','.join(r['roles']) for r in prior.values())),
            'additional_paths':sorted(set(current)-set(prior)), 'missing_or_failed_paths':sorted(set(prior)-set(current)),
            'changed_rgb_paths':sorted(p for p in set(prior)&set(current)
                                      if prior[p]['rgb_sha256']!=current[p]['source_sha256']['rgb']),
            'historical_stage1_selection_rgb_sha_matches':sum(r['source_sha256']['rgb'] in selected for r in current.values())}
    report['exposure'] = 'Frozen experts/V2/Laplacian have ClearGrasp training/validation history; synthetic-val and real-val are not unseen by the complete pipeline. Stage2 splits remain isolated.'
    return report


def audit(roots, output, protocol, workers=2, reference=None, decode_all=False):
    protocol = validate_protocol(protocol)
    if protocol['schema'] != CLEARGRASP_PROTOCOL['schema']:
        raise ValueError('Use the independent cleargrasp256 configuration')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Audit output must be new; preserve earlier evidence')
    output.mkdir(parents=True, exist_ok=True)
    records, pairing_errors, counts = discover(roots)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool, (output/'frames.jsonl').open('w') as stream:
        for index, result in enumerate(pool.map(check_record, ((r,protocol,decode_all) for r in records), chunksize=8), 1):
            rows.append(result); stream.write(canonical(result)+'\n')
            if index % 500 == 0:
                stream.flush(); print(json.dumps({'checked':index, 'total':len(records),
                    'errors':sum(r['status']!='ok' for r in rows)}), flush=True)
    dups = {kind:duplicates(rows,kind) for kind in ('encoded','decoded')}
    errors = [r for r in rows if r['status']!='ok']
    summary = {'schema':'cleargrasp256_audit_v1', 'preprocessing':protocol,
        'status':'passed' if not pairing_errors and not errors and not any(
            d['cross_split'] for group in dups.values() for d in group) else 'blocked',
        'source_sha256':file_sha(__file__), 'frames_sha256':file_sha(output/'frames.jsonl'),
        'discovered_counts':counts, 'calibrated_and_checked_counts':dict(Counter(r['role'] for r in rows if r['status']=='ok')),
        'decoded_counts':dict(Counter(r['role'] for r in rows if 'decoded_rgb_sha256' in r)),
        'native_sizes':{role:dict(Counter('x'.join(map(str,r['native_hw'])) for r in rows
            if r['role']==role and 'native_hw' in r)) for role in roots},
        'pairing_errors':pairing_errors, 'decode_or_calibration_errors':errors,
        'duplicate_rgb':dups, 'historical_comparison':historical_comparison(rows,reference),
        'optional_camera_normals':'Not used/read as labels; original loss derives normals from GT256/K256, valid current/right/down stencil. Optional file existence recorded.',
        'gt_pixel_decode_scope':'all' if decode_all else 'EXR dimensions/channel/completeness/SHA256 for all; pixel decoding in smoke and mandatory Stage2 label-cache construction',
        'stage1_prediction_executed':False, 'raw_data_mutated':False}
    atomic_json(output/'audit.json', summary)
    print(json.dumps({k:summary[k] for k in ('status','decoded_counts','native_sizes')},indent=2))
    return summary


def roster(roots, audit_root, output, protocol, smoke=False):
    protocol = validate_protocol(protocol)
    audit_root, output = Path(audit_root), Path(output)
    summary = json.loads((audit_root/'audit.json').read_text())
    if (summary['preprocessing'] != protocol or summary['source_sha256'] != file_sha(__file__)
            or summary['frames_sha256'] != file_sha(audit_root/'frames.jsonl')):
        raise ValueError('Stale/mismatched audit; rerun audit in a new directory')
    if summary['status'] != 'passed' and not smoke:
        raise ValueError('Full roster blocked by audit errors; no automatic sample exclusion')
    records, pairing_errors, counts = discover(roots)
    if counts != summary['discovered_counts'] or pairing_errors != summary['pairing_errors']:
        raise ValueError('Dataset inventory changed after audit')
    rows = [json.loads(line) for line in (audit_root/'frames.jsonl').read_text().splitlines()]
    checked = {r['sample_id']:r for r in rows}
    selected = records
    if smoke:
        selected = []
        cross = {s for group in summary['duplicate_rgb'].values() for d in group if d['cross_split'] for s in d['sample_ids']}
        # Deterministic selection from metadata only; cover two native sizes in
        # synthetic splits, two cameras/scenes where available in real-val.
        for role in ('train','evaluation'):
            good = [r for r in records if r['role']==role and checked[r['sample_id']]['status']=='ok'
                    and r['sample_id'] not in cross]
            first = good[0]
            selected.append(first)
            a = checked[first['sample_id']]
            second = next((r for r in good[1:] if checked[r['sample_id']]['native_hw'] != a['native_hw']), None)
            second = second or next((r for r in good[1:] if r['scene']!=first['scene']), None) or good[1]
            selected.append(second)
    identity = [{'sample_id':r['sample_id'], 'role':r['role'], 'rgb_sha256':checked[r['sample_id']].get('source_sha256',{}).get('rgb')}
                for r in records]
    split = {'policy':'official train / no synthetic-val / real-val explicit test-set checkpoint selection',
             'smoke':smoke, 'samples':identity, 'selected':[r['sample_id'] for r in selected],
             'audit_sha256':file_sha(audit_root/'audit.json')}
    digest = fingerprint(split)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Roster output must be new')
    output.mkdir(parents=True,exist_ok=True)
    rgb_rows, inputs, labels = [], [], []
    for record in selected:
        row = checked[record['sample_id']]
        if row['status']!='ok':
            raise ValueError('Unaudited/corrupt sample in roster')
        if any(file_sha(record[k]) != row['source_sha256'][k] for k in ('rgb','gt','mask','calibration')):
            raise ValueError('Source file changed after audit: '+record['sample_id'])
        rgb = dict(sample_id=record['sample_id'], rgb_path=portable_path(record['rgb'],record),
                   rgb_sha256=row['source_sha256']['rgb'], native_hw=row['native_hw'])
        inp = {k:v for k,v in rgb.items() if k!='native_hw'}
        inp.update(intrinsics=row['intrinsics_native'], pixel_convention='integer_pixel_centres',
            **{k:record[k] for k in ('domain','role','official_split','dataset','scene','camera')},
            split_sha256=digest, benchmark_group='cleargrasp/'+record['scene']+'/'+record['camera'],
            intrinsics_source=portable_path(record['calibration'],record),
            intrinsics_source_sha256=row['source_sha256']['calibration'])
        label = dict(sample_id=record['sample_id'], target_depth_path=portable_path(record['gt'],record),
            transparent_mask_path=portable_path(record['mask'],record),
            depth_semantics='already_rectified_optical_z_m', normal_camera_valid=True, camera=record['camera'])
        rgb_rows.append(rgb); inputs.append(inp); labels.append(label)
        if smoke:
            native = load_native_labels(dict(label, target_depth_path=str(record['gt']),
                transparent_mask_path=str(record['mask']), native_hw=row['native_hw'], intrinsics_native=row['intrinsics_native']))
            low = resize_supervision(native,protocol)
            if not low['valid'].any():
                raise ValueError('Selected smoke frame has no valid GT')
            checked[record['sample_id']] = check_record((record,protocol,True))
            if checked[record['sample_id']]['status'] != 'ok':
                raise ValueError('Smoke full GT decoding failed')
            key = fingerprint(record['sample_id'])[:24]
            atomic_npz(output/'supervision_audit'/(key+'.npz'), **low)
            visualize_supervision(output/'visualizations'/(key+'.png'),record['rgb'],low,protocol)
    for name, values in [('rgb',rgb_rows),('inputs',inputs),('labels',labels)]:
        (output/(name+'.jsonl')).write_text(''.join(canonical(row)+'\n' for row in values))
    atomic_json(output/'split.json',split)
    contract = {'schema':'cleargrasp256_rosters_v1', 'source_sha256':file_sha(__file__),
        'preprocessing':protocol, 'split_sha256':digest, 'smoke':smoke,
        'counts':dict(Counter(r['role'] for r in selected)), 'full_audit_status':summary['status'],
        'samples_audit':[checked[r['sample_id']] for r in selected] if smoke else None,
        'roster_sha256':{name:file_sha(output/(name+'.jsonl')) for name in ('rgb','inputs','labels')}}
    atomic_json(output/'contract.json',contract)
    return {'counts':contract['counts'], 'split_sha256':digest, 'smoke':smoke}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['audit','roster'])
    for name, env in [('train','CLEARGRASP_TRAIN_ROOT'),('real-val','CLEARGRASP_REAL_VAL_ROOT')]:
        parser.add_argument('--'+name+'-root',default=os.environ.get(env),required=not os.environ.get(env))
    parser.add_argument('--config',default='configs/cleargrasp256_stage2_60.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--audit-root'); parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--decode-all',action='store_true',help='Additionally decompress every full-resolution GT pixel; no model inference')
    parser.add_argument('--historical-reference',default='data/cleargrasp_reference/rgb_k.jsonl.gz')
    args = parser.parse_args()
    protocol = json.loads(Path(args.config).read_text())['preprocessing']
    roots = {'train':args.train_root, 'evaluation':args.real_val_root}
    if args.operation=='audit':
        if not 1 <= args.workers <= 8:
            parser.error('Audit workers must be 1..8')
        report = audit(roots,args.output,protocol,args.workers,args.historical_reference,args.decode_all)
        if report['status']!='passed':
            raise SystemExit(2)
    else:
        if not args.audit_root:
            parser.error('roster requires --audit-root')
        print(json.dumps(roster(roots,args.audit_root,args.output,protocol,args.smoke),indent=2))


if __name__=='__main__':
    main()
