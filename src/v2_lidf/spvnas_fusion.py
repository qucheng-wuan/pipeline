"""Optional Stage-1 branch for the pinned official SPVNAS implementation.

This revision uses TorchSparse batch-FIRST coordinates. No semantic classifier,
GT xyz, or ray-grid modifications are used. Importing this file needs only torch.
"""
import hashlib
from contextvars import ContextVar
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
VENDOR = Path(__file__).resolve().parent / 'vendor' / 'spvnas'
MODEL_IDS = {f'spvnas_fusion_{s}g': f'SemanticKITTI_val_SPVNAS@{s}GMACs' for s in (20, 65)}
MAPPING_CHECKS = ContextVar('spvnas_mapping_checks', default=True)


def fusion_config(options):
    """Validate the explicit integration settings; never substitute a backbone."""
    defaults = dict(variant='spvnas_fusion_20g', pretrained=True,
        repo_dir=str(VENDOR), voxel_size=.05, coordinate_scale=1.,
        spatial_shape=[64, 64, 64], runtime_checks=True, allow_tf32=False)
    if not isinstance(options, dict):
        raise TypeError('spvnas must be a configuration dictionary')
    unknown = set(options) - set(defaults) - {'assets_dir', 'audit_dir'}
    if unknown:
        raise ValueError('Unknown SPVNAS settings: ' + str(sorted(unknown)))
    result = dict(defaults, **options)
    if result['variant'] != 'spvnas_fusion_20g':
        raise ValueError('This experiment fixes the official 20GMACs architecture')
    for name in ('pretrained', 'runtime_checks', 'allow_tf32'):
        if not isinstance(result[name], bool):
            raise ValueError(name + ' must be Boolean')
    for name in ('assets_dir', 'audit_dir', 'repo_dir'):
        if name not in result or not Path(result[name]).is_absolute():
            raise ValueError('SPVNAS ' + name + ' must be an absolute path')
    if Path(result['repo_dir']).resolve() != VENDOR.resolve():
        raise ValueError('SPVNAS must use the experiment-owned pinned source')
    if result['voxel_size'] != .05 or result['coordinate_scale'] != 1.:
        raise ValueError('The fixed metric SPVNAS grid is 0.05 m with coordinate_scale=1')
    if list(result['spatial_shape']) != [64, 64, 64]:
        raise ValueError('The fixed sparse spatial extent is [64,64,64]')
    return SimpleNamespace(**result)


def mean_pool(features, inverse, count):
    out = features.new_zeros((count, features.shape[1]))
    out.index_add_(0, inverse, features)
    counts = torch.bincount(inverse, minlength=count).to(features).clamp_min(1)
    return out / counts[:, None]


def quantize(xyz, batch, voxel_size, coordinate_scale=1.0, check=True):
    """Exact deduplication, including batch ID, with a checked inverse map.

    Camera metres -> (forward, left, up) = (z, -x, -y). The configurable
    scale applies both to xyz features and quantization; default preserves metres.
    The 5cm RGB-D grid matches the pretrained pres/vres (0.05) and is
    independent of LIDF's original ray grid.
    """
    if voxel_size <= 0 or coordinate_scale <= 0:
        raise ValueError('voxel_size and coordinate_scale must be positive')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or batch.shape != (len(xyz),):
        raise ValueError('Expected Nx3 point cloud with matching N batch indices')
    if batch.dtype not in (torch.int32, torch.int64):
        raise ValueError('Batch indices must be integers')
    if check and (not torch.isfinite(xyz).all() or (batch < 0).any()):
        raise ValueError('Invalid input points/batch indices')
    metric = torch.stack((xyz[:, 2], -xyz[:, 0], -xyz[:, 1]), 1) * coordinate_scale
    grid = torch.floor(metric / voxel_size).long()
    # Translation per sample, never a batch-wide minimum (batch order invariant).
    for bid in torch.unique(batch):
        selected = batch == bid
        grid[selected] -= grid[selected].amin(0)
    coords = torch.cat((batch.long()[:, None], grid), 1)
    unique, inverse = torch.unique(coords, dim=0, return_inverse=True)
    if check:
        assert torch.equal(unique[inverse], coords), 'Quantization inverse mismatch'
        assert torch.equal(unique[inverse, 0], batch), 'Cross-batch quantization'
        if unique.numel() and unique.max() > torch.iinfo(torch.int32).max:
            raise ValueError('Sparse coordinates overflow int32')
    return metric, unique.int(), inverse


def mean_pool_matched(features, indices, count):
    """A missing sparse-grid match is not voxel zero; exclude it from pooling."""
    matched = indices >= 0
    return mean_pool(features[matched], indices[matched].long(), count)


def point_to_voxel_checked(x, z):
    """Official point_to_voxel with invalid hash matches excluded, not clamped.

    The pinned upstream utils.py clamps -1 to 0 and thereby mixes samples at
    sparse downsampling boundaries. Keep the searched layers/weights unchanged;
    preserve valid assignments and zero-fill voxels with no matching points.
    """
    from torchsparse import SparseTensor
    from core.models.utils import sphashquery
    if len(z.C) == 0 or len(x.C) == 0:
        features = (z.F[:0] if len(x.C) == 0 else
                    z.F.new_zeros((len(x.C), z.F.shape[1])) + z.F.sum() * 0.)
        result = SparseTensor(features, x.C, x.s, spatial_range=x.spatial_range)
        result._caches = x._caches
        return result
    indices = z._caches.idx_query.get(x.s)
    if indices is None:
        query = torch.cat((z.C[:, :1].int(), torch.floor(z.C[:, 1:] / x.s[0]).int()), 1)
        indices = sphashquery(query, x.C).reshape(-1)
        z._caches.idx_query[x.s] = indices
    matched = indices >= 0
    if MAPPING_CHECKS.get():
        assert torch.equal(z.C[matched, 0].int(), x.C[indices[matched].long(), 0]), 'SPVNAS cross-batch voxel assignment'
    result = SparseTensor(mean_pool_matched(z.F, indices, len(x.C)), x.C, x.s,
                          spatial_range=x.spatial_range)
    result._caches = x._caches
    return result


class FeatureIdentity(nn.Module):
    # SPVNAS.forward calls this DynamicLinear interface even after determinize().
    def set_in_channel(self, in_channel=None, constraint=None):
        pass

    def forward(self, x):
        return x


def load_specialized(config, variant, audit_path):
    """Official specialized construction, with local artifacts and strict audit.

    Require official model_zoo.spvnas_specialized, local official artifacts and
    strict tensor verification. Missing dependencies/weights abort construction.
    """
    model_id = MODEL_IDS[variant]
    directory = Path(config.assets_dir) / model_id
    config_path, weights_path = directory / 'net.config', directory / 'init'
    if not config_path.is_file() or (config.pretrained and not weights_path.is_file()):
        raise FileNotFoundError(f'Official artifacts missing: {directory}; run scripts/spvnas/fetch_weights.py')
    expected = json.loads((VENDOR / 'OFFICIAL_ASSETS.json').read_text())[model_id]
    for path in (config_path, weights_path) if config.pretrained else (config_path,):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[path.name]['sha256']:
            raise ValueError('Official SPVNAS asset hash mismatch: ' + str(path))
    sys.path.insert(0, str(Path(config.repo_dir).resolve()))
    # Deliberately lazy: the original variant never imports TorchSparse.
    import torchsparse
    # TorchSparse has its own precision switches, independent of torch.backends.
    # Keep FP32 storage; TF32 acceleration is an explicit experiment setting.
    torchsparse.backends.allow_tf32 = getattr(config, 'allow_tf32', False)
    torchsparse.backends.allow_fp16 = False
    official = importlib.import_module('core.models.semantic_kitti.spvnas')
    if Path(official.__file__).resolve() != VENDOR / 'core/models/semantic_kitti/spvnas.py':
        raise RuntimeError('An unrelated core.models module shadows the pinned SPVNAS source')
    official.point_to_voxel = point_to_voxel_checked
    net_config = json.loads(config_path.read_text())
    try:
        zoo = importlib.import_module('model_zoo')
    except ModuleNotFoundError as exc:
        raise RuntimeError('Official model_zoo.spvnas_specialized is unavailable; repair dependencies before training') from exc
    def local_artifact(url, **kwargs):
        filename = url.rsplit('/', 1)[-1]
        if url.rsplit('/', 2)[-2] != model_id or filename not in ('net.config', 'init'):
            raise ValueError('Unexpected official artifact request: ' + url)
        return str(directory / filename)
    original_download, original_dist = zoo.download_url, zoo.dist
    try:
        zoo.download_url = local_artifact
        zoo.dist = SimpleNamespace(local_rank=lambda: torch.cuda.current_device())
        # Official determinization runs dummy inference and consumes torch RNG.
        # Keep model construction from perturbing training/data sampler streams.
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            model = zoo.spvnas_specialized(model_id, pretrained=config.pretrained)
    finally:
        zoo.download_url, zoo.dist = original_download, original_dist
    audit = dict(model_id=model_id, net_config=net_config,
                 source='https://hanlab18.mit.edu/files/SPVNAS/spvnas_specialized/' + model_id,
                 config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 pretrained=config.pretrained,
                 loader='official model_zoo.spvnas_specialized',
                 missing_keys=[], unexpected_keys=[], mismatched_keys=[])
    if config.pretrained:
        state = torch.load(weights_path, map_location='cpu')['model']
        current = model.state_dict()
        audit.update(weights_path=str(weights_path), weights_bytes=weights_path.stat().st_size,
                     weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                     missing_keys=sorted(set(current) - set(state)),
                     unexpected_keys=sorted(set(state) - set(current)),
                     mismatched_keys=[k for k in set(state) & set(current) if state[k].shape != current[k].shape])
        assert not (audit['missing_keys'] or audit['unexpected_keys'] or audit['mismatched_keys'])
        assert all(torch.equal(current[k].detach().cpu(), state[k]) for k in state)
        audit['all_loaded_tensors_match_official_checkpoint'] = True
    audit['point_to_voxel_fix'] = 'Exclude unmatched -1 indices instead of clamping to voxel 0; preserve batch isolation'
    audit['empty_support_policy'] = ('Zero-row convolution/point mapping traverses all normalization nodes; '
        'if at most one sample globally has support, sparse-branch BN uses existing running statistics')
    audit['removed_after_loading'] = 'classifier (SemanticKITTI logits)'
    audit['new_parameters'] = ['input_adapter (6 -> 4)', 'fusion (feature_channels -> pnet_out)']
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    Path(audit_path).write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    model.classifier = FeatureIdentity()
    install_empty_safe_operators(model, official)
    return model, net_config['output_channels'][-1]


class SPVNASFusion(nn.Module):
    def __init__(self, config, variant, output_channels, audit_path):
        super().__init__()
        self.config = config
        self.backbone, channels = load_specialized(config, variant, audit_path)
        self.input_adapter = nn.Linear(6, 4)
        # Preserve metric xyz and initialize reflectance proxy to RGB luminance.
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            self.input_adapter.bias.zero_()
            self.input_adapter.weight[:3, :3].copy_(torch.eye(3))
            self.input_adapter.weight[3, 3:].copy_(torch.tensor([.299, .587, .114]))
        self.fusion = nn.Linear(channels, output_channels)
        # Small nonzero residual enables backbone/input gradients on the first step.
        nn.init.normal_(self.fusion.weight, std=1e-4)
        nn.init.zeros_(self.fusion.bias)
        self.last_diagnostics = {}

    def forward(self, xyz, rgb, batch, point_ids, voxel_ids, voxel_batch, roi_min, roi_max, batch_size,
                backward_dependency=None):
        from torchsparse import SparseTensor
        check = getattr(self.config, 'runtime_checks', False)
        if check and not torch.isfinite(xyz).all():
            raise ValueError('Invalid input points')
        # Crop to the (expanded) LIDF ray-grid box before quantization. Points
        # outside it never fall into an occupied voxel, so dropping them loses
        # no pooled information and bounds the sparse coordinate span.
        inside = ((xyz >= roi_min) & (xyz <= roi_max)).all(1)
        remap = torch.full((len(xyz),), -1, dtype=torch.long, device=xyz.device)
        inside_ids = inside.nonzero(as_tuple=True)[0]
        remap[inside_ids] = torch.arange(inside_ids.numel(), device=xyz.device)
        xyz, rgb, batch = xyz[inside], rgb[inside], batch[inside]
        metric, coords, inverse = quantize(xyz, batch, self.config.voxel_size, self.config.coordinate_scale, check=check)
        # LIDF RGB was ImageNet standardized by the dataset.
        color = rgb * rgb.new_tensor([.229, .224, .225]) + rgb.new_tensor([.485, .456, .406])
        features = self.input_adapter(torch.cat((metric, color), 1))
        if backward_dependency is not None:
            # Keep the edge even for zero rows, after support selection. This
            # orders independent SyncBN branches without changing any feature
            # value or introducing a nonzero gradient into the RGB encoder.
            features = features + backward_dependency
        # TorchSparse otherwise infers a global spatial extent from this batch,
        # changing boundary downsampling when a larger sample is added/removed.
        # A fixed sparse extent (not a dense allocation) keeps examples independent.
        extent = tuple(self.config.spatial_shape)
        if len(extent) != 3 or any(n <= 0 or n % 16 for n in extent):
            raise ValueError('spatial_shape must contain three positive multiples of 16')
        if check and (coords[:, 1:] >= coords.new_tensor(extent)).any():
            raise ValueError('Point cloud exceeds configured SPVNAS spatial_shape')
        sparse = SparseTensor(feats=mean_pool(features, inverse, len(coords)), coords=coords,
                              spatial_range=(batch_size, *extent))
        token = MAPPING_CHECKS.set(check)
        # A completely empty rank still calls SyncBN with zero observations.
        # If globally zero/one examples have support, a coarse sparse level can
        # have <=1 observation and its sample variance is undefined. Use the
        # already stored BN statistics for this exceptional whole-microbatch
        # case, uniformly on all ranks. Convolutions remain in train mode.
        active = xyz.new_tensor(torch.unique(batch).numel())
        if self.training and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(active)
        norm_states = []
        if self.training and int(active) <= 1:
            norm_states = [(module, module.training) for module in self.backbone.modules()
                           if isinstance(module, nn.modules.batchnorm._BatchNorm)]
            for module, _ in norm_states:
                module.eval()
        try:
            unique_features = self.backbone(sparse)
        finally:
            for module, training in norm_states:
                module.train(training)
            MAPPING_CHECKS.reset(token)
        assert unique_features.shape[0] == len(coords)
        # Official forward returns z3.F on z.C == input coordinates, preserving order.
        point_features = unique_features[inverse]
        kept_ids = remap[point_ids]
        if check:
            assert (kept_ids >= 0).all(), 'LIDF occupied-voxel points outside SPVNAS ROI crop'
            assert torch.equal(batch[kept_ids], voxel_batch[voxel_ids]), 'LIDF voxel batch mismatch'
        pooled = mean_pool(point_features[kept_ids], voxel_ids, len(voxel_batch))
        self.last_diagnostics = {
            'spvnas_input_point_count': xyz.new_tensor(len(xyz)),
            'spvnas_sparse_voxel_count': xyz.new_tensor(len(coords)),
            'spvnas_empty_rank': xyz.new_tensor(float(len(coords) == 0)),
            'spvnas_bn_running_fallback': xyz.new_tensor(float(bool(norm_states))),
        }
        return self.fusion(pooled)


def _empty_sparse_convolution(self, value):
    """Zero-row convolution with an autograd edge; no synthetic support point.

    Sparse CUDA hash/kmap kernels do not define a zero-coordinate workload.
    Nonempty calls use the original pinned operator unchanged. The empty path
    retains the ordinary module/normalization traversal and parameter names.
    """
    if len(value.F):
        return self._spvnas_nonempty_forward(value)
    from torchsparse import SparseTensor
    zero = value.F.sum() * 0. + self.kernel.reshape(-1)[0] * 0.
    if self.bias is not None:
        zero = zero + self.bias.reshape(-1)[0] * 0.
    stride = tuple(a // b if self.transposed else a * b
                   for a, b in zip(value.s, self.stride))
    spatial_range = value.spatial_range
    if spatial_range is not None:
        spatial = []
        for extent, step, padding, kernel in zip(spatial_range[1:], self.stride,
                                                 self.padding, self.kernel_size):
            if self.transposed:
                spatial.append((extent - 1) * step - 2 * padding + self.dilation * (kernel - 1) + 1)
            else:
                spatial.append((extent + 2 * padding - self.dilation * (kernel - 1) - 1) // step + 1)
        spatial_range = (spatial_range[0], *spatial)
    result = SparseTensor(value.F.new_empty((0, self.out_channels)) + zero,
                          value.C, stride, spatial_range=spatial_range)
    result._caches = value._caches
    return result


def voxel_to_point_empty_safe(x, z, nearest=False):
    from core.models.utils import PointTensor, voxel_to_point
    if len(z.C) == 0 or len(x.C) == 0:
        # Preserve the requested point ordering even if a coarse sparse level
        # contains no neighbors. Zero rows still retain every preceding BN edge.
        features = x.F.new_zeros((len(z.C), x.F.shape[1])) + x.F.sum() * 0.
        result = PointTensor(features, z.C)
        result._caches = z._caches
        return result
    return voxel_to_point(x, z, nearest=nearest)


def _singleton_batchnorm(self, value):
    features = value if torch.is_tensor(value) else value.F
    if self.training and features.numel() // features.shape[1] == 1:
        normalized = F.batch_norm(features, self.running_mean, self.running_var,
            self.weight, self.bias, False, 0., self.eps)
        if torch.is_tensor(value):
            return normalized
        from torchsparse.nn.utils import fapply
        return fapply(value, lambda _: normalized)
    return self._spvnas_regular_forward(value)


def install_empty_safe_operators(model, official):
    """Keep real-input layers/weights intact while defining zero/singleton cases.

    Binding instance forwards leaves state_dict keys and Parameter objects
    unchanged. This is installed after strict official weight verification.
    Empty tensors traverse the same BN nodes as nonempty peers, so SyncBN can
    contribute zero observations instead of deadlocking or adding fake points.
    """
    from torchsparse.nn import Conv3d
    official.voxel_to_point = voxel_to_point_empty_safe
    for module in model.modules():
        if isinstance(module, Conv3d) and not hasattr(module, '_spvnas_nonempty_forward'):
            module._spvnas_nonempty_forward = module.forward
            module.forward = MethodType(_empty_sparse_convolution, module)
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not hasattr(module, '_spvnas_regular_forward'):
                module._spvnas_regular_forward = module.forward
                module.forward = MethodType(_singleton_batchnorm, module)


class SafeSyncBatchNorm(nn.SyncBatchNorm):
    """Native SyncBN, with running-stat fallback for a local single observation."""
    def forward(self, value):
        sync = (torch.distributed.is_available() and torch.distributed.is_initialized()
                and torch.distributed.get_world_size(self.process_group) > 1)
        if self.training and not sync and value.numel() // value.shape[1] == 1:
            return F.batch_norm(value, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0., self.eps)
        return super().forward(value)


class SparseSyncBatchNorm(SafeSyncBatchNorm):
    """Sync features while preserving TorchSparse coordinates and tensor caches."""
    def forward(self, value):
        from torchsparse.nn.utils import fapply
        return fapply(value, super().forward)


def convert_sparse_sync_batchnorm(module):
    """PyTorch's blanket converter erases sparse BatchNorm.forward; preserve it."""
    from torchsparse.nn import BatchNorm
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            cls = SparseSyncBatchNorm if isinstance(child, (BatchNorm, SparseSyncBatchNorm)) else SafeSyncBatchNorm
            replacement = cls(child.num_features, child.eps, child.momentum,
                                               child.affine, child.track_running_stats)
            if child.affine:
                replacement.weight, replacement.bias = child.weight, child.bias
            replacement.running_mean = child.running_mean
            replacement.running_var = child.running_var
            replacement.num_batches_tracked = child.num_batches_tracked
            replacement.training = child.training
            setattr(module, name, replacement)
        else:
            convert_sparse_sync_batchnorm(child)
    return module
