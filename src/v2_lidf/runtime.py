"""Portable runtime constraints; no historical experiment paths are required."""
import os
import hashlib
import json
from pathlib import Path

DISTRIBUTED_PROTOCOL = {
    'strategy': 'single_gradient_bucket_after_backward', 'torch_version': '2.1.2',
    'bucket_cap_mb': 512, 'first_bucket_cap_mb': 512, 'find_unused_parameters': False,
    'broadcast_buffers': True, 'gradient_as_bucket_view': False, 'static_graph': False,
}
TORCHSPARSE_COMMIT = '385f5ce8718fcae93540511b7f5832f4e71fd835'


def validate_distributed_config(config):
    if config.get('distributed') != DISTRIBUTED_PROTOCOL:
        raise ValueError('Training requires the pinned PyTorch 2.1.2 single gradient bucket protocol')


def validate_runtime(require_sparse=True):
    """Inspect versions and the installed sparse source, without launching CUDA."""
    import torch
    import torchvision
    import torch_scatter
    expected = [('torch', torch.__version__, '2.1.2+cu121'),
                ('torchvision', torchvision.__version__, '0.16.2+cu121'),
                ('torch_scatter', torch_scatter.__version__, '2.1.2+pt21cu121')]
    profile = os.environ.get('TD_RUNTIME_PROFILE', 'reference')
    if profile == 'blackwell-cu130':
        expected = [('torch', torch.__version__, '2.11.0+cu130'),
                    ('torchvision', torchvision.__version__, '0.26.0+cu130'),
                    ('torch_scatter', torch_scatter.__version__, '2.1.2+pt211cu130')]
    elif profile != 'reference':
        raise RuntimeError('Unknown TD_RUNTIME_PROFILE: ' + profile)
    for name, actual, wanted in expected:
        if str(actual) != wanted:
            raise RuntimeError('%s=%s; this release requires %s' % (name, actual, wanted))
    result = {name: str(actual) for name, actual, _ in expected}
    if require_sparse:
        import importlib.util
        import importlib.metadata
        specification = importlib.util.find_spec('torchsparse')
        if specification is None or not specification.origin:
            raise RuntimeError('Pinned TorchSparse installation is missing')
        # Importing TorchSparse calls cuda.get_device_capability in this commit.
        # Source/ABI inspection must remain usable before selecting any GPU.
        package = Path(specification.origin).resolve().parent
        expected_sources = json.loads((Path(__file__).with_name('assets') / 'torchsparse_python_sources.json').read_text())
        for relative, digest in expected_sources['files'].items():
            path = package / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError('TorchSparse source differs from pinned commit: ' + relative)
        result['torchsparse_commit'] = expected_sources['commit']
        result['torchsparse_version'] = importlib.metadata.version('torchsparse')
    return result


def validate_device_runtime(device=None):
    """Check target architecture before loading/rebuilding custom CUDA code.

    CUDA extensions are built from local sources for the selected device. A
    driver alone is not a build toolchain, and copying a .so is not migration.
    """
    import shutil
    import subprocess
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    result = validate_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable')
    target = torch.device(device or 'cuda:0')
    major, minor = torch.cuda.get_device_capability(target)
    arch = '%d.%d' % (major, minor)
    compiled = torch.cuda.get_arch_list()
    if 'sm_%d%d' % (major, minor) not in compiled:
        raise RuntimeError('PyTorch build does not advertise this GPU architecture; install a compatible profile')
    profile = os.environ.get('TD_RUNTIME_PROFILE', 'reference')
    if major >= 10 and profile == 'reference':
        raise RuntimeError('Reference CUDA12.1 profile cannot target this GPU; use CUDA13 profile')
    toolkit = Path(CUDA_HOME or '') / 'bin/nvcc'
    if not CUDA_HOME or not toolkit.is_file():
        raise RuntimeError('CUDA toolkit/nvcc required to build LIDF extensions from source')
    version = subprocess.check_output([str(toolkit),'--version'],text=True)
    wanted = '13.0' if profile == 'blackwell-cu130' else '12.1'
    if 'release '+wanted not in version:
        raise RuntimeError('CUDA toolkit must match selected profile '+wanted)
    if not shutil.which('ninja'):
        raise RuntimeError('ninja is required for local CUDA compilation')
    declared = os.environ.get('TORCH_CUDA_ARCH_LIST')
    if declared and arch not in declared.replace('+PTX','').replace(';',' ').split():
        raise RuntimeError('TORCH_CUDA_ARCH_LIST does not include the selected GPU: '+arch)
    result.update(device=torch.cuda.get_device_name(target),compute_capability=arch,
                  torch_cuda=torch.version.cuda,toolkit=CUDA_HOME,
                  extensions_cache=os.environ.get('TORCH_EXTENSIONS_DIR','torch default per-environment cache'))
    return result
