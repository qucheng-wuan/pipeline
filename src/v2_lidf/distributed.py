"""Pinned DDP reduction ordering for models with conditional SyncBN branches.

PyTorch 2.1.2 rebuilds gradient buckets after its first iteration. Its small
first bucket has a separate cap from bucket_cap_mb; both must cover this whole
FP32 model so DDP reduction cannot interleave with a peer's SyncBN backward.
The model separately orders sparse SyncBN backward before RGB SyncBN backward.
"""
import inspect
import threading

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .runtime import validate_distributed_config
from .common import file_sha

_CONSTRUCTION_LOCK = threading.Lock()


def create_single_bucket_ddp(model, protocol, device_ids=None):
    """Return DDP plus its construction audit; never change a persistent global.

The private first-bucket setting is deliberately version-pinned and restored
even if DDP construction fails. One dtype/device and two sufficiently large
caps also keep subsequent rebuilt buckets together, regardless of hook order.
"""
    validate_distributed_config({'distributed': protocol})
    if str(torch.__version__).split('+', 1)[0] != protocol['torch_version']:
        raise RuntimeError('Single-bucket DDP is pinned to PyTorch 2.1.2')
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError('Single-bucket DDP requires trainable parameters')
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1 or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError('Single-bucket DDP requires all parameters float32 on one device')
    if getattr(model, '_ddp_params_and_buffers_to_ignore', None):
        raise ValueError('Single-bucket DDP may not omit model parameters or buffers')
    if any(isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)) and module.sparse
           for module in model.modules()):
        raise ValueError('Single-bucket DDP requires dense parameter gradients')
    parameter_count = sum(parameter.numel() for parameter in trainable)
    gradient_bytes = sum(parameter.numel() * parameter.element_size() for parameter in trainable)
    regular_bytes = protocol['bucket_cap_mb'] * 1024 ** 2
    first_bytes = protocol['first_bucket_cap_mb'] * 1024 ** 2
    if gradient_bytes > min(regular_bytes, first_bytes):
        raise ValueError('Both DDP bucket caps must cover every trainable gradient')
    if not hasattr(dist, '_DEFAULT_FIRST_BUCKET_BYTES'):
        raise RuntimeError('Pinned PyTorch first-bucket setting is unavailable')
    with _CONSTRUCTION_LOCK:
        original_first_bytes = dist._DEFAULT_FIRST_BUCKET_BYTES
        try:
            dist._DEFAULT_FIRST_BUCKET_BYTES = first_bytes
            wrapped = DistributedDataParallel(model, device_ids=device_ids,
                bucket_cap_mb=protocol['bucket_cap_mb'],
                find_unused_parameters=protocol['find_unused_parameters'],
                broadcast_buffers=protocol['broadcast_buffers'],
                gradient_as_bucket_view=protocol['gradient_as_bucket_view'],
                static_graph=protocol['static_graph'])
        finally:
            dist._DEFAULT_FIRST_BUCKET_BYTES = original_first_bytes
        global_restored = dist._DEFAULT_FIRST_BUCKET_BYTES == original_first_bytes
    logging = wrapped._get_ddp_logging_data()
    buckets = [int(value.strip()) for value in str(logging['bucket_sizes']).split(',') if value.strip()]
    if buckets != [gradient_bytes]:
        raise RuntimeError('Actual DDP initial buckets do not match the complete gradient set')
    source = inspect.getsourcefile(DistributedDataParallel)
    audit = {'schema_version': 'v2_lidf_spvnas_single_bucket_ddp_v1',
             'protocol': dict(protocol), 'torch_version': str(torch.__version__),
             'parameter_dtype': 'float32', 'parameter_device': str(next(iter(devices))),
             'parameter_count': parameter_count, 'gradient_bytes': gradient_bytes,
             'bucket_cap_bytes': regular_bytes, 'first_bucket_cap_bytes': first_bytes,
             'initial_bucket_sizes': buckets,
             'previous_first_bucket_bytes': original_first_bytes,
             'first_bucket_global_restored': global_restored,
             'torch_ddp_source': source, 'torch_ddp_source_sha256': file_sha(source)}
    return wrapped, audit
