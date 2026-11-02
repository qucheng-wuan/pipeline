"""Read-only transCG / ClearGrasp discovery and separated portable rosters."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import yaml

from .prepare_stage2_data import canonical, fingerprint
from .stage2 import file_sha
from v2_lidf.data import atomic_json, atomic_npz, load_native_labels, resize_intrinsics
from v2_lidf.data_protocol import validate_protocol, resize_supervision


def scene_split(official, seed=23):
    train, test = sorted(official['train']), sorted(official['test'])
    if set(train) & set(test) or len(set(train + test)) != official['total_scenes']:
        raise ValueError('Invalid official scene split')
    shuffled = train.copy()
    random.Random(seed).shuffle(shuffled)
    val = sorted(shuffled[:max(1, round(len(train) * .1))])
    return {'seed': seed, 'algorithm': 'Python Random(seed).shuffle(sorted official train); round(10%)',
            'train': sorted(set(train) - set(val)), 'val': val, 'official_test': test}


def transcg_records(root, info, split):
    official = json.loads((info / 'metadata.json').read_text())
    scene_paths = {}
    # Accept both the official flat layout and the existing archive-part layout.
    # An identity appearing in both is an error, never silently deduplicated.
    paths = list(root.glob('scene*/metadata.json')) + list(root.glob('transcg-*/scene*/metadata.json'))
    for path in paths:
        sid = int(path.parent.name.removeprefix('scene'))
        if sid in scene_paths:
            raise ValueError('Duplicate scene identity: %d' % sid)
        scene_paths[sid] = path
    if set(scene_paths) != set(official['train'] + official['test']):
        raise ValueError('Scene coverage differs from official metadata')
    records, metadata_hashes = [], {}
    for sid, path in sorted(scene_paths.items()):
        meta = json.loads(path.read_text())
        expected_split = 'train' if sid in official['train'] else 'test'
        if meta['split'] != expected_split:
            raise ValueError('Scene split disagrees with official root metadata')
        metadata_hashes[str(sid)] = file_sha(path)
        role = 'train' if sid in split['train'] else ('val' if sid in split['val'] else 'evaluation')
        for c, camera in ((1, 'D435'), (2, 'L515')):
            views = meta[camera + '_valid_perspective_list']
            if len(views) != meta[camera + '_valid_perspective_num'] or len(set(views)) != len(views):
                raise ValueError('Invalid official perspective list')
            calibration = info / 'camera_intrinsics' / ('%d-camIntrinsics-%s.npy' % (c, camera))
            k = np.load(calibration, allow_pickle=False)
            if k.shape != (3, 3) or not np.isfinite(k).all() or not np.allclose(k[2], [0, 0, 1]):
                raise ValueError('Invalid official camera intrinsics')
            for perspective in views:
                if type(perspective) is not int or not 0 <= perspective < official['perspective_num']:
                    raise ValueError('Invalid perspective ID')
                view = path.parent / str(perspective)
                rgb, gt, mask = [view / name for name in
                    ('rgb%d.png' % c, 'depth%d-gt.png' % c, 'depth%d-gt-mask.png' % c)]
                if not all(p.is_file() for p in (rgb, gt, mask)):
                    raise FileNotFoundError('Missing official valid view: ' + str(view))
                records.append(dict(sample_id='transcg/scene%d/perspective%d/camera%d' % (sid, perspective, c),
                    rgb=rgb, gt=gt, mask=mask, root=root, root_var='TRANSCG_ROOT', dataset='transcg',
                    scene='scene%d' % sid, camera=camera, role=role, official_split=expected_split,
                    k=k.tolist(), native_hw=[720, 1280], calibration=calibration,
                    depth_semantics='transcg_uint16_optical_z', depth_divisor=1000 if c == 1 else 4000))
    counts = Counter((r['official_split'], r['camera']) for r in records)
    for part in ('train', 'test'):
        for camera in ('D435', 'L515'):
            if counts[part, camera] != official['%s_%s_samples' % (part, camera)]:
                raise ValueError('Official per-camera sample count mismatch')
    return records, metadata_hashes


def cleargrasp_records(root):
    suffixes = {'rgb': '-transparent-rgb-img.jpg', 'gt': '-opaque-depth-img.exr', 'mask': '-mask.png'}
    tables = {}
    for kind, suffix in suffixes.items():
        table = {}
        for path in root.rglob('*' + suffix):
            frame = path.name[:-len(suffix)]
            if not re.fullmatch(r'\d+', frame):
                raise ValueError('Non-numeric ClearGrasp frame ID: ' + str(path))
            key = (str(path.parent.relative_to(root)), int(frame))
            if key in table:
                raise ValueError('Ambiguous duplicate ClearGrasp frame')
            table[key] = path
        tables[kind] = table
    if not tables['rgb'] or any(set(t) != set(tables['rgb']) for t in tables.values()):
        raise ValueError('ClearGrasp RGB/opaque GT/mask frame sets must match exactly')
    result = []
    for key, rgb in sorted(tables['rgb'].items()):
        parent = rgb.parent
        while not (parent / 'camera_intrinsics.yaml').is_file():
            if parent == root or root not in parent.parents:
                raise FileNotFoundError('No corresponding camera_intrinsics.yaml for ' + str(rgb))
            parent = parent.parent
        calibration = parent / 'camera_intrinsics.yaml'
        camera = yaml.safe_load(calibration.read_text())
        k = [[float(camera['fx']), 0., float(camera['cx'])],
             [0., float(camera['fy']), float(camera['cy'])], [0., 0., 1.]]
        result.append(dict(sample_id='cleargrasp/real-val/%s/%09d' % key,
            rgb=rgb, gt=tables['gt'][key], mask=tables['mask'][key], root=root,
            root_var='CLEARGRASP_ROOT', dataset='cleargrasp', scene=key[0] or 'real-val',
            camera=camera['camera_name'], role='evaluation', official_split='real-val',
            k=k, native_hw=[int(camera['yres']), int(camera['xres'])], calibration=calibration,
            depth_semantics='already_rectified_optical_z_m'))
    return result


def portable_path(path, record):
    return '${%s}/%s' % (record['root_var'], path.relative_to(record['root']).as_posix())


def visualize_supervision(path, rgb_path, labels, protocol):
    """Data-only audit panels, with no predicted-depth substitute."""
    import cv2
    from PIL import ImageDraw
    with Image.open(rgb_path) as image:
        rgb=cv2.resize(np.asarray(image),(256,256),interpolation=cv2.INTER_AREA)
    lo,hi=protocol['depth_range_m']
    values=np.uint8(np.clip((labels['target']-lo)/(hi-lo),0,1)*255)
    target=cv2.cvtColor(cv2.applyColorMap(values,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    target[~labels['valid']]=0
    mask=np.repeat((labels['transparent'].astype(np.uint8)*255)[...,None],3,axis=2)
    coverage=np.repeat(np.uint8(np.clip(labels['coverage'],0,1)*255)[...,None],3,axis=2)
    canvas=Image.new('RGB',(1024,282),'white');draw=ImageDraw.Draw(canvas)
    for i,(tile,title) in enumerate(zip([rgb,target,mask,coverage],['RGB256','GT256 metres / invalid black','transparent mask','GT coverage'])):
        canvas.paste(Image.fromarray(tile),(i*256,26));draw.text((i*256+4,6),title,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def prepare_rosters(transcg_root, cleargrasp_root, info_root, output, protocol,
                    smoke=False, include_official_test=False):
    protocol = validate_protocol(protocol)
    root, cg, info, output = map(lambda p: Path(p).resolve(),
        (transcg_root, cleargrasp_root, info_root, output))
    info_source = json.loads((info/'source.json').read_text())
    for relative,digest in info_source['files'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or file_sha(info/relative) != digest:
            raise ValueError('Official transCG metadata/calibration checksum mismatch')
    official = json.loads((info / 'metadata.json').read_text())
    split = scene_split(official)
    records, meta = transcg_records(root, info, split)
    cg_records = cleargrasp_records(cg)
    split.update(official_metadata_sha256=file_sha(info / 'metadata.json'), scene_metadata_sha256=meta)
    split_digest = fingerprint(split)
    all_counts = Counter(r['role'] for r in records)
    selected = [r for r in records if include_official_test or r['role'] != 'evaluation']
    if smoke:
        selected = [next(r for r in records if r['role'] == role and r['camera'] == camera)
                    for role in ('train', 'val') for camera in ('D435', 'L515')]
        selected += cg_records[:2]
    else:
        selected += cg_records
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(schema='transfer_rosters_v1', split_sha256=split_digest, preprocessing=protocol,
        smoke=smoke, include_official_test=include_official_test, source_sha256=file_sha(__file__),
        info_provenance_sha256=file_sha(info/'source.json'),
        calibration_sha256={p.name: file_sha(p) for p in (info/'camera_intrinsics').glob('*.npy')})
    old = output / 'contract.json'
    if old.exists() and json.loads(old.read_text()) != contract:
        raise ValueError('Roster contract changed; use a fresh output directory')
    atomic_json(old, contract)
    atomic_json(output/'split_seed23.json', split)
    rgb_rows, input_rows, label_rows, audit = [], [], [], []
    for record in selected:
        with Image.open(record['rgb']) as image:
            hw = [image.height, image.width]
            if image.mode != 'RGB' or hw != record['native_hw']:
                raise ValueError('Native RGB/calibration dimensions disagree')
        rgb_sha = file_sha(record['rgb'])
        rgb = dict(sample_id=record['sample_id'], rgb_path=portable_path(record['rgb'], record),
                   rgb_sha256=rgb_sha, native_hw=hw)
        inputs = {k: v for k, v in rgb.items() if k != 'native_hw'}
        inputs.update(intrinsics=record['k'], pixel_convention='integer_pixel_centres', domain='real',
            role=record['role'], official_split=record['official_split'], dataset=record['dataset'],
            scene=record['scene'], camera=record['camera'], split_sha256=split_digest,
            benchmark_group=record['dataset']+'/'+record['scene']+'/'+record['camera'],
            intrinsics_source=record['calibration'].name, intrinsics_source_sha256=file_sha(record['calibration']))
        labels = dict(sample_id=record['sample_id'], target_depth_path=portable_path(record['gt'], record),
            transparent_mask_path=portable_path(record['mask'], record),
            depth_semantics=record['depth_semantics'], normal_camera_valid=True, camera=record['camera'])
        if 'depth_divisor' in record:
            labels['depth_divisor'] = record['depth_divisor']
        # Label audit has no path to model input rosters. Full mode validates
        # all supervision; smoke validates only these six selected real views.
        native = load_native_labels(dict(labels, target_depth_path=str(record['gt']),
            transparent_mask_path=str(record['mask']), native_hw=hw, intrinsics_native=record['k']))
        low = resize_supervision(native, protocol)
        if not low['valid'].any():
            raise ValueError('Empty supervision after configured range/coverage: '+record['sample_id'])
        if smoke:
            key = fingerprint(record['sample_id'])[:24]
            atomic_npz(output/'supervision_audit'/(key+'.npz'),**low)
            visualize_supervision(output/'visualizations'/(key+'.png'),record['rgb'],low,protocol)
        audit.append(dict(sample_id=record['sample_id'], native_hw=hw, camera=record['camera'],
            role=record['role'], mask_encoding=[0, 1] if record['dataset']=='transcg' else [0, 255],
            depth_divisor=record.get('depth_divisor', 1), valid256=int(low['valid'].sum()),
            transparent_valid256=int((low['valid'] & low['transparent']).sum()),
            zero_coverage256=int((low['coverage']==0).sum()),
            gt_valid_m_range=[float(low['target'][low['valid']].min()), float(low['target'][low['valid']].max())],
            intrinsics_native=record['k'], intrinsics256=resize_intrinsics(record['k'], hw, [256,256]).tolist()))
        rgb_rows.append(rgb); input_rows.append(inputs); label_rows.append(labels)
    for name, rows in [('rgb', rgb_rows), ('inputs', input_rows), ('labels', label_rows)]:
        path = output/(name+'.jsonl')
        content = ''.join(canonical(r)+'\n' for r in rows)
        if path.exists() and path.read_text() != content:
            raise ValueError('Roster bytes changed; use a new output directory')
        path.write_text(content)
    summary = dict(samples=len(selected), counts=dict(Counter(r['role'] for r in selected)),
        full_transcg_counts=dict(all_counts), cleargrasp_real_val=len(cg_records),
        scene_counts={k:len(split[k]) for k in ('train','val','official_test')},
        split_sha256=split_digest, preprocessing=protocol, samples_audit=audit,
        roster_sha256={n:file_sha(output/(n+'.jsonl')) for n in ('rgb','inputs','labels')},
        root_environment={'TRANSCG_ROOT':str(root), 'CLEARGRASP_ROOT':str(cg)},
        no_sensor_depth_read=True, stage1_prediction_executed=False)
    atomic_json(output/'audit.json', summary)
    return {k:v for k,v in summary.items() if k not in {'samples_audit','root_environment'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcg-root', default=os.environ.get('TRANSCG_ROOT'), required=not os.environ.get('TRANSCG_ROOT'))
    parser.add_argument('--cleargrasp-root', default=os.environ.get('CLEARGRASP_ROOT'), required=not os.environ.get('CLEARGRASP_ROOT'))
    parser.add_argument('--info-root', default='data/transcg_info')
    parser.add_argument('--config', default='configs/transcg_stage2_60.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--include-official-test', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    print(json.dumps(prepare_rosters(args.transcg_root,args.cleargrasp_root,args.info_root,args.output,
        cfg['preprocessing'],args.smoke,args.include_official_test),indent=2))


if __name__ == '__main__':
    main()
