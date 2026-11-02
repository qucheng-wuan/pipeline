#!/usr/bin/env python3
"""Download verified official initialization assets for fresh Stage2 training.

These are unnecessary when deploying a complete trained Stage2 bundle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

BASE = 'https://hanlab18.mit.edu/files/SPVNAS/spvnas_specialized/SemanticKITTI_val_SPVNAS@20GMACs/'
ASSETS = (
    ('resnet34-333f7ec4.pth', 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
     '333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef'),
    ('SemanticKITTI_val_SPVNAS@20GMACs/net.config', BASE + 'net.config',
     '3bedf00c36f4ba926504fcc05cc00734b01d74f0803a4563583e315ec4f3c72d'),
    ('SemanticKITTI_val_SPVNAS@20GMACs/init', BASE + 'init',
     'd8f8e53ab5774f610c33a9d7d023beb82fdc3f846276162b205906112a27af85'),
)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(root):
    root = Path(root)
    records = []
    for relative, url, digest in ASSETS:
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.download')
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as output:
                while True:
                    block = source.read(4 << 20)
                    if not block:
                        break
                    output.write(block)
            if sha(temporary) != digest:
                raise ValueError('Official downloaded artifact SHA256 mismatch: ' + relative)
            os.replace(temporary, path)
        if sha(path) != digest:
            raise ValueError('Existing initialization artifact changed: ' + relative)
        records.append({'id': relative, 'path': relative, 'sha256': digest,
                        'bytes': path.stat().st_size, 'kind': 'training_initialization', 'download_url': url})
    (root / 'initialization.assets.json').write_text(json.dumps({'assets': records}, indent=2) + '\n')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(fetch(parser.parse_args().output), indent=2))
