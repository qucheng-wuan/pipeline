"""CPU-only actual EXR and complete label-batch equivalence audit."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

BASE = Path('/mnt/gluster_dcpu/hengxianli/transparent_depth_release_20261101/audit/winner_geometry_recovery')
SELECTION = Path('/mnt/gluster_dcpu/hengxianli/fusion_v2_lidf_spvnas_20261030/continuation_gpu567_20261031/selection_full1226_transparent_mae_20261101')
PROJECT = Path('/home/hengxianli/RGBOnly_Fusion_Laplacian_LIDF_SPVNAS')
OLD_METRIC = Path('/home/hengxianli/FAR_MoGe_TransNormal_V2_20261021/src')
OLD_MODEL = Path('/home/hengxianli/FusionV2_LIDF_SPVNAS_GPU567_20261031/src')

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()

def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def prepare():
    root = BASE / 'failure_snapshot'
    root.mkdir(exist_ok=False)
    originals = [SELECTION/'selected_geometry/execution.json',
                 SELECTION/'selected_geometry/resolved_selection.json',
                 SELECTION/'evaluate_selected.wait.log']
    originals += sorted((SELECTION/'selected_geometry/logs').glob('geometry_rank*.log'))
    execution = read(originals[0])
    assert len(execution['workers']) == 8 and all(r['returncode'] == 1 for r in execution['workers'])
    copied = []
    for source in originals:
        destination = root / source.name
        shutil.copyfile(source, destination)
        assert sha(source) == sha(destination)
        copied.append({'original': ref(source), 'snapshot': ref(destination)})
    contract = read(SELECTION/'contract.json')
    rows = {}
    with Path(contract['label_manifest']['path']).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['role'] == 'evaluation' and row['domain'] not in rows:
                rows[row['domain']] = row
            if set(rows) == {'real', 'synthetic'}:
                break
    assert set(rows) == {'real', 'synthetic'} and rows['synthetic'].get('normal_path')
    write(BASE/'samples.json', rows)
    source_paths = list(contract['source_sha256'])
    source_paths += [str(PROJECT/'scripts/evaluate_selected.py'),
                     str(PROJECT/'scripts/sweep_verify.py'),
                     str(PROJECT/'src/transparent_depth/evaluation.py')]
    write(BASE/'frozen_sources_before.json', {p: sha(p) for p in sorted(set(source_paths))})
    write(BASE/'failure_snapshot.json', {'status': 'preserved_actual_failure',
          'failure': 'OpenEXR 3.2.4 has no File API used by frozen native label readers',
          'failed_workers': 8, 'artifacts': copied,
          'selection': ref(SELECTION/'selection.complete.json'),
          'independent_verification': ref(SELECTION/'publication_verification.complete.json'),
          'shared_environment_modified': False, 'frozen_sources_modified': False,
          'created_unix': time.time()})
    print('Original eight failures copied and authenticated; real/synthetic samples selected.', flush=True)

def array_info(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}

def exact(a, b, name):
    assert array_info(a) == array_info(b), 'Not byte-exact: ' + name
    return array_info(a)

def raw_arrays(OpenEXR, row):
    import numpy as np
    result = {}
    with OpenEXR.File(str(row['target_depth_path']), separate_channels=True) as source:
        result['depth'] = np.array(source.channels()['R'].pixels, dtype=np.float32)
    if row.get('normal_path'):
        with OpenEXR.File(str(row['normal_path']), separate_channels=True) as source:
            result['normal'] = np.stack([np.array(source.channels()[k].pixels, dtype=np.float32)
                                         for k in ('R', 'G', 'B')])
    return result

def old():
    import numpy as np
    import torch
    import OpenEXR
    assert os.environ.get('TD_OPENEXR_FILE_COMPAT') == '1'
    assert hasattr(OpenEXR, '_transparent_depth_file_compat'), 'Adapter did not activate'
    sys.path[:0] = [str(OLD_METRIC), str(OLD_MODEL), str(PROJECT/'src')]
    import native_fusion.benchmark as original
    import native_fusion.data as native_data
    import v2_lidf.data as stage2_data
    import transparent_depth.evaluation as portable
    records = {}
    for domain, row in read(BASE/'samples.json').items():
        batch, assets = original.load_evaluation_labels(row['original_row'])
        other = portable.load_evaluation_labels(row)
        assert batch.keys() == other.keys()
        infos = {}
        for key in batch:
            a, b = batch[key].cpu().numpy(), other[key].cpu().numpy()
            infos[key] = exact(a, b, domain + '/' + key)
            assert torch.equal(batch[key], other[key]), 'Non-equal tensor ' + key
        raw = raw_arrays(OpenEXR, row)
        raw_path = BASE/(domain+'_shim_raw.npz')
        assert not raw_path.exists()
        np.savez(raw_path, **raw)
        records[domain] = {'cache_id': row['cache_id'], 'batch_fields': infos,
                           'label_assets': assets, 'raw_arrays': ref(raw_path)}
    sample = next(iter(read(BASE/'samples.json').values()))['target_depth_path']
    for kwargs in ({'separate_channels': False}, {'separate_channels': True, 'header_only': True}, {}):
        try:
            OpenEXR.File(sample, **kwargs)
        except TypeError:
            pass
        else:
            raise AssertionError('Unsupported API accepted')
    with OpenEXR.File(sample, separate_channels=True) as source:
        try:
            source.header()
        except AttributeError:
            pass
        else:
            raise AssertionError('Unimplemented API unexpectedly accepted')
    try:
        source.channels()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Use after close accepted')
    modules = [original, native_data, stage2_data, portable, OpenEXR]
    write(BASE/'portable_parity.json', {'status': 'passed', 'byte_exact': True,
          'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
          'python': sys.executable, 'torch_version': torch.__version__,
          'openexr_version': importlib.metadata.version('OpenEXR'),
          'modules': [ref(module.__file__) for module in modules],
          'shim': ref(BASE/'shim/sitecustomize.py'), 'samples': records,
          'unsupported_api_rejected': True, 'test_scope': 'two actual native label batches; no model execution'})
    print('Real/synthetic original shim labels versus portable InputFile labels: all fields byte-exact.', flush=True)

def native():
    import numpy as np
    import OpenEXR
    assert not os.environ.get('TD_OPENEXR_FILE_COMPAT')
    assert not hasattr(OpenEXR, '_transparent_depth_file_compat')
    assert hasattr(OpenEXR, 'File')
    records = {}
    for domain, row in read(BASE/'samples.json').items():
        raw = raw_arrays(OpenEXR, row)
        path = BASE/(domain+'_native_file_raw.npz')
        assert not path.exists()
        np.savez(path, **raw)
        records[domain] = {'file': ref(path), 'arrays': {k: array_info(v) for k,v in raw.items()}}
    write(BASE/'native_file_reference.json', {'status': 'completed_reference_read',
          'python': sys.executable, 'openexr_version': importlib.metadata.version('OpenEXR'),
          'openexr_module': ref(OpenEXR.__file__), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
          'native_file_api_unmodified': True, 'samples': records})
    print('Unmodified fresh Stage1 native OpenEXR.File arrays captured on CPU.', flush=True)

def compare():
    import numpy as np
    records = {}
    for domain in ('real', 'synthetic'):
        with np.load(BASE/(domain+'_shim_raw.npz'), allow_pickle=False) as a, np.load(BASE/(domain+'_native_file_raw.npz'), allow_pickle=False) as b:
            assert a.files == b.files
            records[domain] = {key: exact(a[key], b[key], domain+'/'+key) for key in a.files}
    before = read(BASE/'frozen_sources_before.json')
    assert all(sha(path) == digest for path,digest in before.items()), 'Frozen source changed'
    portable = read(BASE/'portable_parity.json')
    assert portable['status'] == 'passed' and portable['byte_exact']
    write(BASE/'cpu_parity.complete.json', {'schema_version': 'winner_geometry_exr_compat_cpu_v1',
          'status': 'passed', 'byte_exact': True, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
          'raw_depth_and_camera_normal_vs_native_file': records,
          'complete_label_batches_vs_portable_inputfile': ref(BASE/'portable_parity.json'),
          'native_file_reference': ref(BASE/'native_file_reference.json'),
          'shim': ref(BASE/'shim/sitecustomize.py'), 'checker': ref(Path(__file__)),
          'failure_evidence': ref(BASE/'failure_snapshot.json'),
          'frozen_sources': ref(BASE/'frozen_sources_before.json'), 'frozen_sources_unchanged': True,
          'shared_environment_modified': False, 'model_executed': False,
          'finished_unix': time.time()})
    print('PASS: shim versus native File raw bytes, and complete original versus portable batches; frozen sources unchanged.', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'old', 'native', 'compare'))
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU audit requires explicitly hidden GPUs'
    globals()[args.phase]()
