"""Explicit two-environment RGB -> E08 -> LIDF-SPVNAS inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from . import __version__
from .assets import destination


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def find_project_root(explicit=None):
    if explicit:
        root = Path(explicit).expanduser().resolve()
    elif os.environ.get('TRANSPARENT_DEPTH_HOME'):
        root = Path(os.environ['TRANSPARENT_DEPTH_HOME']).expanduser().resolve()
    else:
        candidates = [Path.cwd(), *Path(__file__).resolve().parents]
        root = next((p for p in candidates if (p / 'pyproject.toml').is_file()
                     and (p / 'configs').is_dir()), None)
        if root is None:
            raise ValueError('Run inside the repository or pass --project-root')
    if not (root / 'configs').is_dir() or not (root / 'src/transparent_depth').is_dir():
        raise ValueError('Project root does not contain this repository')
    return root


def project_path(root, name):
    """Release defaults must stay relocatable and contained in the checkout."""
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Release configuration paths must be relative to the project')
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Release configuration path escapes the project')
    return result


def inference_commands(args, root, config, intermediate):
    stage1 = config['stage1']
    stage2 = config['stage2']
    asset_root = Path(args.asset_root).resolve() if args.asset_root else project_path(root, config['asset_root'])
    stage1_python = args.stage1_python or os.environ.get('TRANSPARENT_DEPTH_STAGE1_PYTHON')
    stage2_python = args.stage2_python or os.environ.get('TRANSPARENT_DEPTH_STAGE2_PYTHON')
    if not stage1_python or not stage2_python:
        raise ValueError('Provide --stage1-python and --stage2-python, or their TRANSPARENT_DEPTH_*_PYTHON environment variables')
    commands = [
        [stage1_python, '-m', 'transparent_depth.stage1', '--rgb', str(args.rgb),
         '--output', str(intermediate), '--assets', str(project_path(root, stage1['assets_manifest'])),
         '--asset-root', str(asset_root), '--device', args.device],
        [stage2_python, '-m', 'transparent_depth.stage2', '--rgb', str(args.rgb),
         '--depth-input', str(intermediate), '--intrinsics', str(args.intrinsics),
         '--bundle', str(destination(asset_root, stage2['bundle'])), '--output', str(args.output),
         '--device', args.device],
    ]
    if getattr(args, 'runtime_cache', None):
        commands[0] += ['--runtime-cache', str(args.runtime_cache.expanduser().resolve())]
    return commands


def validate_release_assets(root, config, asset_root):
    """Bind the selected experiment to the actual assets before starting CUDA."""
    stage1 = config['stage1']
    manifest = project_path(root, stage1['assets_manifest'])
    if sha256(manifest) != stage1['assets_manifest_sha256']:
        raise ValueError('Stage1 asset registry differs from the verified release')
    stage2 = config['stage2']
    bundle = destination(asset_root, stage2['bundle'])
    metadata_path = bundle / 'model.json'
    if sha256(metadata_path) != stage2['bundle_manifest_sha256']:
        raise ValueError('Stage2 bundle identity differs from the selected release')
    metadata = json.loads(metadata_path.read_text())
    selection = config['selection']
    origin = metadata['origin']
    if (stage2['checkpoint_epoch'] != selection['checkpoint_epoch']
            or origin['epoch_one_based'] != selection['checkpoint_epoch']
            or origin['checkpoint_sha256'] != selection['checkpoint_sha256']
            or origin['checkpoint_sha256'] != stage2['source_checkpoint_sha256']
            or origin['selection'] != selection):
        raise ValueError('Loaded model is not the declared test-selected checkpoint')
    weights = metadata['weights']
    if (weights['file'] != 'model.pt' or weights['sha256'] != stage2['weights_sha256']
            or sha256(bundle / 'model.pt') != stage2['weights_sha256']):
        raise ValueError('Stage2 weights differ from the selected release')
    return {'stage1_assets_manifest_sha256': stage1['assets_manifest_sha256'],
            'stage2_bundle_manifest_sha256': stage2['bundle_manifest_sha256'],
            'stage2_weights_sha256': stage2['weights_sha256']}


def infer(args):
    root = find_project_root(args.project_root)
    config_path = Path(args.config).resolve() if args.config else project_path(root, 'configs/release.json')
    config = json.loads(config_path.read_text())
    if config.get('status') != 'verified_release':
        raise ValueError('Release selection and portability verification are not complete')
    asset_root = Path(args.asset_root).resolve() if args.asset_root else project_path(root, config['asset_root'])
    asset_identities = validate_release_assets(root, config, asset_root)
    for name in ('rgb', 'intrinsics'):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.output = args.output.expanduser().resolve()
    if args.output.suffix != '.npy':
        raise ValueError('--output must end in .npy (float32 metric depth)')
    metadata_path = args.output.with_suffix('.json')
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError('Output or metadata already exists; use a new output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(root / 'src')
    environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # The caller controls physical GPU visibility. The two children inherit
    # the exact same restriction and use the selected logical CUDA device.
    with tempfile.TemporaryDirectory(prefix='transparent-depth-') as directory:
        intermediate = Path(directory) / 'stage1.npy'
        commands = inference_commands(args, root, config, intermediate)
        for command in commands:
            subprocess.run(command, cwd=str(root), env=environment, check=True)
        if not args.output.is_file():
            raise RuntimeError('Stage two did not produce the requested output')
    report = {
        'schema': 'transparent_depth_prediction_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'rgb_sha256': sha256(args.rgb), 'intrinsics_sha256': sha256(args.intrinsics),
        'release_config_sha256': sha256(config_path), 'depth_sha256': sha256(args.output),
        'selection': config['selection'], 'units': 'metres',
        'runtime_inputs': ['RGB', 'known_camera_intrinsics'],
        'sensor_depth_used': False, 'gt_depth_used_as_input': False,
        'gt_transparent_mask_used_as_input': False,
        'asset_identities': asset_identities,
    }
    temporary = metadata_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(metadata_path)
    print(json.dumps({'depth': str(args.output), 'metadata': str(metadata_path)}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    inference = commands.add_parser('infer', help='Run both frozen RGB depth stages')
    inference.add_argument('--rgb', type=Path, required=True)
    inference.add_argument('--intrinsics', type=Path, required=True, help='JSON with a calibrated 3x3 native-image K matrix')
    inference.add_argument('--output', type=Path, required=True)
    inference.add_argument('--project-root', type=Path)
    inference.add_argument('--config', type=Path)
    inference.add_argument('--asset-root', type=Path)
    inference.add_argument('--runtime-cache', type=Path, help='Writable Stage1 JIT/cache directory')
    inference.add_argument('--stage1-python')
    inference.add_argument('--stage2-python')
    inference.add_argument('--device', default='cuda:0')
    inference.set_defaults(function=infer)
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
