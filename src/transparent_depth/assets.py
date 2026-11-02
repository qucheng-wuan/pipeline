"""Fetch pinned assets without storing model bytes or credentials in Git."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def destination(root, relative):
    root = Path(root).resolve()
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts or not name.parts:
        raise ValueError('Asset path must be relative and contained in the asset root')
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Asset destination escapes the asset root')
    return path


def check_file(path, entry):
    path = Path(path)
    return (path.is_file() and path.stat().st_size == entry['bytes']
            and file_sha(path) == entry['sha256'])


def validate_manifest(manifest):
    if manifest.get('schema') != 'transparent_depth_release_assets_v1':
        raise ValueError('Unknown asset manifest schema')
    assets = manifest.get('assets')
    if not isinstance(assets, list) or not assets:
        raise ValueError('Asset manifest is empty')
    seen = set()
    for asset in assets:
        path = asset['path']
        destination(Path.cwd() / 'asset-path-validation', path)
        if path in seen:
            raise ValueError('Duplicate asset destination')
        seen.add(path)
        digest = asset['sha256']
        if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or type(asset['bytes']) is not int or asset['bytes'] < 0):
            raise ValueError('Malformed asset identity')
        source = asset['download']
        if source['kind'] not in {'github_release', 'huggingface', 'https'}:
            raise ValueError('Unsupported asset source')
        if source['kind'] == 'github_release':
            for part in source['parts']:
                name = part['name']
                if Path(name).name != name or any(c in name for c in '*?[]\n\r'):
                    raise ValueError('Unsafe release asset name')
        if source['kind'] == 'https' and not source['url'].startswith('https://'):
            raise ValueError('Only HTTPS asset downloads are accepted')
    return assets


def fetch_asset(asset, root, working, *, force=False):
    target = destination(root, asset['path'])
    if check_file(target, asset):
        return {'path': asset['path'], 'status': 'already_verified'}
    if target.exists() and not force:
        raise FileExistsError('Asset exists with a different identity: %s (use --force to replace it)' % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.download.%d' % os.getpid())
    source = asset['download']
    try:
        if source['kind'] == 'github_release':
            with temporary.open('wb') as output:
                for part in source['parts']:
                    part_path = working / part['name']
                    if not check_file(part_path, part):
                        if shutil.which('gh'):
                            subprocess.run(['gh', 'release', 'download', source['tag'],
                                '--repo', source['repository'], '--pattern', part['name'],
                                '--dir', str(working), '--clobber'], check=True)
                        else:
                            from urllib.parse import quote
                            url = 'https://github.com/%s/releases/download/%s/%s' % (
                                source['repository'], quote(source['tag'],safe=''),quote(part['name'],safe=''))
                            with urllib.request.urlopen(url,timeout=120) as response, part_path.open('wb') as stream:
                                shutil.copyfileobj(response,stream,length=4*1024*1024)
                    if not check_file(part_path, part):
                        raise ValueError('GitHub release part identity mismatch: ' + part['name'])
                    with part_path.open('rb') as stream:
                        shutil.copyfileobj(stream, output, length=4 * 1024 * 1024)
        elif source['kind'] == 'huggingface':
            from huggingface_hub import hf_hub_download
            upstream = Path(hf_hub_download(repo_id=source['repository'],
                filename=source['filename'], revision=source['revision']))
            shutil.copyfile(upstream, temporary)
        else:
            with urllib.request.urlopen(source['url'], timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
        if not check_file(temporary, asset):
            raise ValueError('Downloaded asset identity mismatch: ' + asset['path'])
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': asset['path'], 'status': 'downloaded_and_verified'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, default=Path('artifacts'))
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--component',choices=('all','stage1','stage2'),default='all')
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    assets = validate_manifest(manifest)
    if args.component != 'all':
        assets = [a for a in assets if a['path'].startswith(args.component+'/')]
        if not assets:
            raise ValueError('No assets for requested component')
    args.asset_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.downloads-', dir=str(args.asset_root)) as directory:
        for asset in assets:
            if args.verify_only:
                if not check_file(destination(args.asset_root, asset['path']), asset):
                    parser.exit(1, 'Missing or changed asset: ' + asset['path'] + '\n')
                result = {'path': asset['path'], 'status': 'verified'}
            else:
                result = fetch_asset(asset, args.asset_root, Path(directory), force=args.force)
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
