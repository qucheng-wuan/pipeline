#!/usr/bin/env python3
"""CPU review of one recorded CLI run's exact local CUDA-extension classification.

This preserves every original observation and failure. It does not execute a
model, change numerical limits, or accept an arbitrary external module.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

CONTROLLER_SHA = 'b17a3c7ae43d36a282726fa739fd3dadeb97581b6d92ff1c3269c51b50659dd4'
SO_SHA = 'ea7ef559059918b826cd14c08cffee80662cf8f5d01b8d0a528962e56195af80'
CPP_SHA = 'c6bddfc3481d7760e8b916ec0e24be249bdaf764a739d79f83f0e1085e3daeeb'
CUDA_SHA = 'ef16dc71b245395e2b7ce6669310c5c4a00a3efdfc5da4092db2cb0b542e0403'
MODULE = 'v2_lidf_ray_aabb'
UUID = 'GPU-9c9f5c94-e2c6-4f01-6419-8bd0dfc4ba37'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def save_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'controller', 'launch-receipt', 'project'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only verification requires CUDA_VISIBLE_DEVICES empty')
    attempt, project = args.attempt.resolve(), args.project.resolve()
    output = attempt / 'posthoc_verification'
    output.mkdir(exist_ok=True)
    require(not (output / 'complete.json').exists(), 'Preserve completed review evidence')
    require(sha(args.controller) == CONTROLLER_SHA, 'Unexpected controller implementation')
    launch = read(args.launch_receipt)
    require(launch['controller'] == ref(args.controller) and launch['output'] == str(attempt), 'Launch identity differs')
    command = launch['command']
    def argument(name):
        return command[command.index('--' + name) + 1]
    require(Path(argument('output')).resolve() == attempt and Path(argument('project')).resolve() == project,
            'Controller arguments identify another attempt')
    options = {name.replace('-', '_'): Path(argument(name)) for name in
               ('asset-root', 'stage1-python', 'stage2-python', 'cuda-home', 'stage1-acceptance',
                'single-training-acceptance', 'ddp-training-acceptance', 'cached-validation',
                'cached-quality-acceptance')}
    a = SimpleNamespace(project=project, output=attempt, gpu_uuid=UUID, **options)
    require(argument('gpu-uuid') == UUID and argument('gpu-index') == '4', 'GPU authorization differs')
    spec = importlib.util.spec_from_file_location('recorded_cli_controller', args.controller)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    execution, failure = read(attempt / 'execution.json'), read(attempt / 'failure.json')
    require(execution['returncode'] == 0 and execution['execution_completed_normally'] is True,
            'The actual full CLI must have exited successfully')
    require(failure['controller'] == ref(args.controller)
            and failure['error'] == "ValueError('Relocated module/old-source isolation failed')",
            'Original failure was not solely observer classification')
    require(execution['environment']['CUDA_VISIBLE_DEVICES'] == UUID, 'Actual execution GPU differs')
    source_copy, sample = read(attempt / 'checkout.copy.json'), read(attempt / 'input.json')
    checkout = Path(source_copy['destination']).resolve()
    require(checkout == attempt / 'checkout', 'Relocated checkout differs')
    # Derive the exact declared extension directory from the unchanged controller.
    environment = controller.environment(a, checkout)
    extension_root = Path(environment['TORCH_EXTENSIONS_DIR']).resolve()
    require(extension_root == attempt / 'runtime_cache/stage2_extensions', 'Unexpected declared JIT cache')
    build = extension_root / MODULE
    binary = build / (MODULE + '.so')
    require(not binary.is_symlink() and sha(binary) == SO_SHA, 'Runtime binary differs from the observed exact extension')
    observations = sorted((attempt / 'module_observations').glob('modules_*.json'))
    require(len(observations) == 3, 'Require exactly CLI, Stage1 and Stage2 observations')
    normalized_path = None
    for path in observations:
        observation = read(path)
        require(not observation['denied_reads'] and observation['cuda_visible_devices'] == UUID,
                'Old-source access was denied or the process used another GPU')
        if MODULE in observation['modules']:
            require(normalized_path is None and observation['status'] == 'failed'
                    and observation['foreign_project_modules'] == [str(binary)]
                    and observation['modules'][MODULE] == ref(binary),
                    'Only the exact declared runtime extension may be reclassified')
            normalized_path = path.resolve()
        else:
            require(observation['status'] == 'passed' and not observation['foreign_project_modules'],
                    'An unrelated observation failed')
        for name, value in observation['modules'].items():
            require(sha(value['path']) == value['sha256'], 'Observed module bytes changed: ' + name)
            if name != MODULE:
                require(Path(value['path']).resolve().is_relative_to(checkout), 'Unexpected external numerical module')
    require(normalized_path is not None, 'Expected the recorded extension classification failure')
    source_dir = checkout / 'src/v2_lidf/vendor/lidf/extensions/ray_aabb'
    cpp, cuda = source_dir / 'ray_aabb_cuda.cpp', source_dir / 'ray_aabb_cuda_kernel.cu'
    require(sha(cpp) == CPP_SHA and sha(cuda) == CUDA_SHA, 'Compiled numerical source changed')
    ninja = build / 'build.ninja'
    text = ninja.read_text()
    expected_builds = [f'build ray_aabb_cuda.o: compile {cpp}',
                       f'build ray_aabb_cuda_kernel.cuda.o: cuda_compile {cuda}',
                       'build v2_lidf_ray_aabb.so: link ray_aabb_cuda.o ray_aabb_cuda_kernel.cuda.o']
    require([line for line in text.splitlines() if line.startswith('build ')] == expected_builds,
            'Compilation or linking used unexpected input files')
    fresh = a.stage2_python.parent.parent
    require('cxx = g++-11\n' in text and f'nvcc = {a.cuda_home}/bin/nvcc\n' in text
            and '-ccbin gcc-11' in text and '-gencode=arch=compute_86,code=sm_86' in text
            and str(fresh / 'lib/python3.10/site-packages/torch/include') in text
            and str(fresh / 'lib/python3.10/site-packages/torch/lib') in text
            and str(project) not in text, 'Declared fresh compiler/runtime source chain differs')
    toolchain = {'nvcc': subprocess.check_output([str(a.cuda_home / 'bin/nvcc'), '--version'], text=True),
                 'gcc': subprocess.check_output(['gcc-11', '-dumpfullversion'], text=True).strip(),
                 'gxx': subprocess.check_output(['g++-11', '-dumpfullversion'], text=True).strip()}
    require('release 12.1' in toolchain['nvcc'] and toolchain['gcc'].startswith('11.')
            and toolchain['gxx'].startswith('11.'), 'Compiler versions differ')
    products = [build / name for name in ('ray_aabb_cuda.o', 'ray_aabb_cuda_kernel.cuda.o',
                                         'v2_lidf_ray_aabb.so', '.ninja_log')]
    require(all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 for path in products),
            'Actual object/link products are missing')
    targets = {line.split('\t')[3] for line in (build / '.ninja_log').read_text().splitlines()
               if line and not line.startswith('#')}
    require({'ray_aabb_cuda.o', 'ray_aabb_cuda_kernel.cuda.o', 'v2_lidf_ray_aabb.so'} <= targets,
            'Ninja log does not record the object and final linking products')
    raw_paths = observations + [attempt / name for name in
        ('failure.json', 'execution.json', 'depth.npy', 'depth.json', 'depth.npy.json', 'gate.json',
         'checkout.copy.json', 'input.json', 'gpu_admission.json')]
    raw_before = {str(path): sha(path) for path in raw_paths}
    config, stage1, evidence, cached = controller.gates(a)
    save_new(output / 'gate.json', {'status': 'passed', 'evidence_sha256': evidence,
             'cached_validation': cached, 'controller': ref(args.controller), 'verifier': ref(__file__)})
    original_read = controller.read
    def reviewed_read(path):
        value = original_read(path)
        if Path(path).resolve() == normalized_path:
            value = copy.deepcopy(value)
            value['status'] = 'passed'
            value['foreign_project_modules'] = []
        return value
    controller.read = reviewed_read
    try:
        result = controller.verify_result(a, config, source_copy, sample)
    finally:
        controller.read = original_read
    require(raw_before == {str(path): sha(path) for path in raw_paths}, 'An original artifact changed during CPU review')
    complete = {'schema': 'portable_final_cli_actual_integration_v1',
        'status': 'passed_candidate_cli_test_after_runtime_classification_review',
        'raw_observer_classification_failed': True, 'raw_controller_failure': ref(attempt / 'failure.json'),
        'published': False, 'original_project_promoted': False, 'real_gpu_execution': True,
        'posthoc_cpu_only': True, 'model_executed_during_posthoc': False,
        'model_or_numeric_tolerance_changed': False, 'full_live_1226_regression_claimed': False,
        'verification_use': 'candidate_cli_test_only_not_published', 'cached_validation': cached,
        'classification_review': {'authorized_by': '/root', 'scope': 'Only the exact observed local JIT extension is declared runtime',
            'module_name': MODULE, 'binary': ref(binary), 'declared_torch_extensions_dir': str(extension_root),
            'build_ninja': ref(ninja), 'numerical_sources': [ref(cpp), ref(cuda)],
            'build_products': [ref(path) for path in products], 'toolchain': toolchain,
            'raw_observation': ref(normalized_path), 'in_memory_changes_only': ['status', 'foreign_project_modules'],
            'raw_files_modified': False, 'other_foreign_modules': [], 'denied_reads': []},
        'controller': ref(args.controller), 'verifier': ref(__file__), 'launch_receipt': ref(args.launch_receipt),
        'gate': ref(output / 'gate.json'), 'source_copy': ref(attempt / 'checkout.copy.json'),
        'input': ref(attempt / 'input.json'), 'execution': ref(attempt / 'execution.json'),
        'gpu_admission': ref(attempt / 'gpu_admission.json'), 'result': result,
        'actual_cli_seconds': execution['finished_unix'] - execution['started_unix'],
        'raw_artifact_sha256': raw_before, 'completed_unix': time.time()}
    save_new(output / 'complete.json', complete)
    target = project / 'audit/cli_integration.json'
    require(not target.exists(), 'Do not overwrite an existing published CLI receipt')
    shutil.copyfile(output / 'complete.json', target)
    tool = project / 'audit/verify_cli_runtime_classification.py'
    require(not tool.exists(), 'Do not overwrite an existing audit tool')
    shutil.copyfile(__file__, tool)
    require(sha(target) == sha(output / 'complete.json') and sha(tool) == sha(__file__), 'Audit copy changed bytes')
    print(json.dumps({'status': complete['status'], 'complete': ref(output / 'complete.json'),
                      'project_receipt': ref(target), 'project_audit_tool': ref(tool),
                      'actual_cli_seconds': complete['actual_cli_seconds'], 'result': result}, indent=2))


if __name__ == '__main__':
    main()
