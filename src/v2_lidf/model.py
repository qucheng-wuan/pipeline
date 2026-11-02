"""RGB-only pseudo-depth adaptation of NVlabs' official LIDF stage one.

The learned networks and AABB kernels are vendored without topology changes.
Only the supplied input depth creates occupied voxels: curriculum training may
mix GT background into that input, while deployment uses frozen RGB predictions.
Query selection is independent of GT segmentation; inference does not require GT.
See vendor/lidf/NOTICE.md.
"""
from functools import lru_cache
import os
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torch_scatter import scatter_max, scatter_min, scatter_log_softmax

from .geometry import occupied_voxels, pixel_rays, surface_normals, unproject_depth
from .vendor.lidf.implicit_net import IEF, IMNet, get_embedder
from .vendor.lidf.pointnet import PointNet2Stage
from .vendor.lidf.resnet_dilated import Resnet34_8s


@lru_cache(maxsize=2)
def _extension(name):
    """Resolve CUDA sources relative to this file, independent of the cwd."""
    from torch.utils.cpp_extension import load
    directory = Path(__file__).resolve().parent / "vendor" / "lidf" / "extensions" / name
    return load("v2_lidf_" + name,
                [str(directory / (name + "_cuda.cpp")),
                 str(directory / (name + "_cuda_kernel.cu"))], verbose=False)


def stable_group_argmax(scores, groups, group_count):
    """Highest raw score per ray; exact ties choose the lowest pair index.

    Softmax preserves ordering mathematically, but float32 can round distinct
    scores to equal probabilities and make CUDA scatter argmax nondeterministic.
    Select before softmax and resolve true ties with integer scatter_min.
    Empty groups retain the sentinel ``scores.numel()``.
    """
    scores = scores.detach()
    maximum, _ = scatter_max(scores, groups, dim=0, dim_size=group_count)
    pair_count = scores.numel()
    pair_index = torch.arange(pair_count, dtype=torch.long, device=scores.device)
    eligible = torch.where(scores == maximum[groups], pair_index,
                           torch.full_like(pair_index, pair_count))
    selected = torch.full((group_count,), pair_count, dtype=torch.long, device=scores.device)
    selected, _ = scatter_min(eligible, groups, dim=0, out=selected, dim_size=group_count)
    return selected


class LIDFStage1(nn.Module):
    def __init__(self, grid_resolution=8, valid_sample_num=10000,
                 train_query_num=20000, query_chunk_size=4096,
                 maxpool_label_epochs=6, pos_weight=100., prob_weight=.5,
                 normal_weight=10., spvnas=None, rgb_pretrained=False,
                 dense_allow_tf32=False):
        super().__init__()
        if dense_allow_tf32 is not False:
            raise ValueError('This experiment requires dense_allow_tf32=False')
        # LIDF makes a discrete candidate choice from nearby logits. TF32
        # convolution can vary with RGB batch size enough to reverse that
        # choice. Use the same FP32 precision for dense and sparse networks.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.precision_audit = {
            'forward_dtype': 'float32', 'forward_autocast_enabled': False,
            'torch_cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'torch_cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        }
        if grid_resolution <= 0 or query_chunk_size <= 0:
            raise ValueError("grid_resolution and query_chunk_size must be positive")
        if valid_sample_num == 0 or valid_sample_num < -1:
            raise ValueError("valid_sample_num must be positive or -1 for all")
        if train_query_num == 0 or train_query_num < -1:
            raise ValueError("train_query_num must be positive or -1 for all")
        self.grid_resolution = grid_resolution
        self.valid_sample_num = valid_sample_num
        self.train_query_num = train_query_num
        self.query_chunk_size = query_chunk_size
        self.maxpool_label_epochs = maxpool_label_epochs
        self.pos_weight, self.prob_weight, self.normal_weight = pos_weight, prob_weight, normal_weight
        # Match train_lidf.yaml exactly: 32 RGB channels, 128 voxel channels,
        # ROI 8x8 -> 2x2, 8/4 position/direction frequencies, two IEF updates.
        if not isinstance(rgb_pretrained, bool):
            raise ValueError('rgb_pretrained must be Boolean')
        self.resnet_model = Resnet34_8s(inp_ch=3, out_ch=32, pretrained=rgb_pretrained)
        self.rgb_initialization = self.resnet_model.initialization_audit
        self.pnet_model = PointNet2Stage(input_channels=6, output_channels=128, gf_dim=32)
        self.embed_fn, embed_ch = get_embedder(8)
        self.embeddirs_fn, dir_ch = get_embedder(4)
        decoder_channels = 128 + 32 * 4 + 2 * embed_ch + dir_ch
        self.offset_dec = IEF(torch.device("cpu"), inp_dim=decoder_channels, out_dim=1,
                              gf_dim=64, n_iter=2, use_sigmoid=False)
        self.prob_dec = IMNet(inp_dim=decoder_channels, out_dim=1, gf_dim=64, use_sigmoid=False)
        self.spvnas_branch = None
        if spvnas is not None:
            from .spvnas_fusion import SPVNASFusion, fusion_config
            config = fusion_config(spvnas)
            audit = Path(config.audit_dir) / ('weight_audit_rank%s_pid%d.json' %
                                            (os.environ.get('RANK', '0'), os.getpid()))
            self.spvnas_branch = SPVNASFusion(config, config.variant, 128, audit)
            import torchsparse
            record = json.loads(audit.read_text())
            record['rgb_initialization'] = self.rgb_initialization
            record['precision'] = dict(self.precision_audit,
                torchsparse_allow_tf32=bool(torchsparse.backends.allow_tf32),
                torchsparse_allow_fp16=bool(torchsparse.backends.allow_fp16))
            audit.write_text(json.dumps(record, indent=2))

    def _batch_voxel_features(self, xyz, rgb_points, good_input, pixels,
                              backward_dependency=None):
        """One sparse backbone call per microbatch, including empty ranks.

        PointNet retains its original per-image input and voxel ordering. Only
        the extra SPVNAS context is batched, with explicit global point/voxel
        offsets. Query/support RNG calls retain their original interleaving.
        This method accepts input geometry only, never target depth or masks.
        """
        batch = xyz.shape[0]
        prepared, clouds, colors, bids, pids, vids, vbids = [], [], [], [], [], [], []
        point_offset = voxel_offset = 0
        for image_idx in range(batch):
            queries = self._queries(pixels, xyz.device)
            support_ids = self._support_indices(good_input[image_idx].reshape(-1))
            points, color = xyz[image_idx, support_ids], rgb_points[image_idx, support_ids]
            voxels = occupied_voxels(points, self.grid_resolution)
            nvoxel = len(voxels['bounds'])
            features = (self.pnet_model(torch.cat((voxels['relative_points'],
                        color[voxels['point_indices']]), dim=-1), voxels['inverse'])
                        if nvoxel else xyz.new_empty((0, 128)))
            prepared.append(dict(queries=queries, support_ids=support_ids, voxels=voxels,
                                 features=features, voxel_offset=voxel_offset))
            clouds.append(points)
            colors.append(color)
            bids.append(torch.full((len(points),), image_idx, device=xyz.device, dtype=torch.long))
            pids.append(voxels['point_indices'] + point_offset)
            vids.append(voxels['inverse'] + voxel_offset)
            vbids.append(torch.full((nvoxel,), image_idx, device=xyz.device, dtype=torch.long))
            point_offset += len(points)
            voxel_offset += nvoxel
        side = 2. / self.grid_resolution
        branch_options = ({} if backward_dependency is None else
                          {'backward_dependency': backward_dependency})
        residual = self.spvnas_branch(torch.cat(clouds), torch.cat(colors), torch.cat(bids),
            torch.cat(pids), torch.cat(vids), torch.cat(vbids),
            xyz.new_tensor([-1., -1., 0.]) - side * .5,
            xyz.new_tensor([1., 1., 2.]) + side * .5, batch, **branch_options)
        if residual.shape != (voxel_offset, 128):
            raise ValueError('SPVNAS feature shape does not match LIDF occupied voxels')
        for item in prepared:
            start = item['voxel_offset']
            item['features'] = item['features'] + residual[start:start + len(item['features'])]
        return prepared, residual

    def _support_indices(self, valid):
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        if self.valid_sample_num == -1 or indices.numel() <= self.valid_sample_num:
            return indices
        if self.training:
            return indices[torch.randperm(indices.numel(), device=indices.device)[:self.valid_sample_num]]
        # Deterministic and spatially spread evaluation support, including endpoints.
        select = torch.linspace(0, indices.numel() - 1, self.valid_sample_num,
                                device=indices.device).round().long()
        return indices[select]

    def _queries(self, pixels, device):
        indices = torch.arange(pixels, device=device)
        if self.training and 0 < self.train_query_num < pixels:
            # A uniform circular raster interval gives every pixel inclusion
            # probability train_query_num / pixels while retaining local query
            # neighbours. Only query indices wrap: normal stencils still use
            # actual image neighbours and exclude the bottom/right borders.
            start = torch.randint(pixels, (), device=device)
            indices = (start + indices[:self.train_query_num]) % pixels
        return indices

    def forward(self, rgb: torch.Tensor, depth_input: torch.Tensor,
                intrinsics: torch.Tensor, depth_gt: Optional[torch.Tensor] = None,
                gt_valid: Optional[torch.Tensor] = None, epoch: int = 0,
                compute_loss: Optional[bool] = None) -> Dict[str, object]:
        """Predict [B,1,H,W] metric Z depth; no external mask is accepted.

        RGB is ImageNet-normalized. K is calibrated to these exact H,W pixels.
        eval() queries every pixel and never uses GT to select a prediction.
        train() samples queries and uses GT voxel labels only for the first six
        epochs, as in the official curriculum. For missing positive labels the
        learned termination score is used instead of an arbitrary voxel.
        """
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must be [B,3,H,W]")
        batch, _, height, width = rgb.shape
        if depth_input.shape != (batch, 1, height, width):
            raise ValueError("depth_input must be [B,1,H,W], aligned with RGB")
        if not rgb.is_cuda or not depth_input.is_cuda or not intrinsics.is_cuda:
            raise ValueError("LIDF stage one requires CUDA tensors")
        if rgb.device != depth_input.device or rgb.device != intrinsics.device:
            raise ValueError("all inputs must be on the same CUDA device")
        if not torch.isfinite(rgb).all():
            raise ValueError("RGB contains nonfinite values")
        compute_loss = depth_gt is not None if compute_loss is None else compute_loss
        if compute_loss and depth_gt is None:
            raise ValueError("depth_gt is required when compute_loss=True")
        if depth_gt is not None and depth_gt.shape != depth_input.shape:
            raise ValueError("depth_gt must match depth_input")
        if gt_valid is not None and gt_valid.shape != depth_input.shape:
            raise ValueError("gt_valid must match depth_input")
        # The custom kernels are float32. Disable AMP here even if the caller
        # uses mixed precision elsewhere in the frozen first-stage pipeline.
        with torch.autocast(device_type="cuda", enabled=False):
            return self._forward_float(rgb.float(), depth_input.float(), intrinsics.float(),
                                       depth_gt.float() if compute_loss else None,
                                       gt_valid, int(epoch), bool(compute_loss))

    def _forward_float(self, rgb, depth_input, intrinsics, depth_gt, gt_valid, epoch, compute_loss):
        batch, _, height, width = rgb.shape
        pixels = height * width
        good_input = torch.isfinite(depth_input) & (depth_input > 0)
        # Invalid first-stage inputs have no useful fallback; explicitly output
        # zero there unless LIDF supplies a valid prediction, and count them.
        base_depth = torch.where(good_input, depth_input, torch.zeros_like(depth_input))
        xyz = unproject_depth(base_depth, intrinsics).permute(0, 2, 3, 1).reshape(batch, pixels, 3)
        rays = pixel_rays(intrinsics, height, width).reshape(batch, pixels, 3)
        rgb_points = rgb.permute(0, 2, 3, 1).reshape(batch, pixels, 3)
        rgb_features = self.resnet_model(rgb)
        # A graph-connected zero makes completely unsupported minibatches safe
        # to backpropagate while leaving their fallback output unchanged.
        # Touch every trainable tensor, including the probability head when no
        # ray has a positive label. DDP must see a zero gradient instead of an
        # unused parameter on empty-support ranks/minibatches.
        zero = sum((parameter.reshape(-1)[0] * 0. for parameter in self.parameters()
                    if parameter.requires_grad), rgb.new_zeros(()))
        # An empty-support rank must still execute every SyncBatchNorm backward
        # collective in ResNet. Touching parameters alone skips those operators
        # and can mismatch another rank's SyncBN and DDP reductions.
        rgb_zero = rgb_features.sum() * 0.
        zero = zero + rgb_zero
        prepared = None
        if self.spvnas_branch is not None:
            # A value-zero dependency makes the RGB SyncBN backward wait for
            # the complete sparse branch on every rank. Otherwise empty sparse
            # shortcuts can interleave these independent collective sequences.
            prepared, sparse_residual = self._batch_voxel_features(
                xyz, rgb_points, good_input, pixels,
                backward_dependency=rgb_zero if self.training and torch.is_grad_enabled() else None)
            # Empty ranks and heads with no supervised rays must still run the
            # complete sparse SyncBatchNorm backward sequence on zero rows.
            zero = zero + sparse_residual.sum() * 0.
        pos_sum, prob_sum = zero, zero
        pos_count = rgb.new_zeros(())
        prob_count = rgb.new_zeros(())
        counts = {key: rgb.new_zeros(()) for key in (
            "query_count", "intersect_count", "refined_count", "no_intersection_count",
            "invalid_prediction_count", "empty_support_image_count", "support_count",
            "in_volume_support_count", "occupied_voxel_count", "gt_valid_query_count",
            "gt_covered_query_count")}
        depths, refined_masks, query_masks = [], [], []
        clean_gt, supervision_valid, gt_xyz = None, None, None
        if compute_loss:
            supervision_valid = torch.isfinite(depth_gt) & (depth_gt > 0)
            if gt_valid is not None:
                supervision_valid = supervision_valid & gt_valid.bool()
            clean_gt = torch.where(supervision_valid, depth_gt, torch.zeros_like(depth_gt))
            # Exactly the same K as input points and rays: no hidden GT camera.
            gt_xyz = unproject_depth(clean_gt, intrinsics).permute(0, 2, 3, 1).reshape(batch, pixels, 3)
        for image_idx in range(batch):
            out = base_depth[image_idx, 0].reshape(-1).clone()
            refined = torch.zeros(pixels, dtype=torch.bool, device=rgb.device)
            queries = (self._queries(pixels, rgb.device) if prepared is None
                       else prepared[image_idx]['queries'])
            query_mask = torch.zeros_like(refined)
            query_mask[queries] = True
            counts["query_count"] += queries.numel()
            support_ids = (self._support_indices(good_input[image_idx].reshape(-1))
                           if prepared is None else prepared[image_idx]['support_ids'])
            counts["support_count"] += support_ids.numel()
            voxels = (occupied_voxels(xyz[image_idx, support_ids], self.grid_resolution)
                      if prepared is None else prepared[image_idx]['voxels'])
            bounds = voxels["bounds"]
            counts["occupied_voxel_count"] += bounds.shape[0]
            counts["in_volume_support_count"] += voxels["point_indices"].numel()
            if bounds.numel() == 0:
                counts["empty_support_image_count"] += 1
                counts["no_intersection_count"] += queries.numel()
                if compute_loss:
                    counts["gt_valid_query_count"] += supervision_valid[image_idx].reshape(-1)[queries].sum()
            else:
                if prepared is None:
                    point_rgb = rgb_points[image_idx, support_ids[voxels["point_indices"]]]
                    voxel_features = self.pnet_model(torch.cat((voxels["relative_points"], point_rgb), dim=-1),
                                                     voxels["inverse"])
                else:
                    voxel_features = prepared[image_idx]['features']
                voxel_bid = torch.zeros(bounds.shape[0], dtype=torch.int32, device=rgb.device)
                for query_ids in queries.split(self.query_chunk_size):
                    nquery = query_ids.numel()
                    query_rays = rays[image_idx, query_ids].contiguous()
                    ray_bid = torch.zeros(nquery, dtype=torch.int32, device=rgb.device)
                    hit, distances = _extension("ray_aabb").forward(query_rays, bounds, ray_bid, voxel_bid)
                    pair = torch.nonzero(hit, as_tuple=False)
                    has_hit = hit.bool().any(dim=0)
                    counts["intersect_count"] += has_hit.sum()
                    counts["no_intersection_count"] += (~has_hit).sum()
                    query_gt_valid = None
                    if compute_loss:
                        query_gt_valid = supervision_valid[image_idx].reshape(-1)[query_ids]
                        counts["gt_valid_query_count"] += query_gt_valid.sum()
                    if pair.numel() == 0:
                        continue
                    vi, ri = pair[:, 0], pair[:, 1]
                    directions = query_rays[ri]
                    enter = directions * distances[vi, ri, 0, None]
                    leave = directions * distances[vi, ri, 1, None]
                    pixel_ids = query_ids[ri]
                    uv = torch.stack((pixel_ids % width, pixel_ids // width), dim=-1)
                    ul, br = uv - 4, uv + 4
                    ul = torch.stack((ul[:, 0].clamp(0, width - 1), ul[:, 1].clamp(0, height - 1)), -1)
                    br = torch.stack((br[:, 0].clamp(0, width - 1), br[:, 1].clamp(0, height - 1)), -1)
                    rois = torch.cat((rgb.new_full((pair.shape[0], 1), image_idx), ul, br), dim=-1).float()
                    roi_features = roi_align(rgb_features, rois, output_size=2,
                                             spatial_scale=1., aligned=True).flatten(1)
                    embedding = torch.cat((voxel_features[vi], roi_features,
                                           self.embed_fn(enter), self.embed_fn(leave),
                                           self.embeddirs_fn(directions)), dim=-1)
                    offsets = self.offset_dec(embedding)
                    scores = self.prob_dec(embedding)[:, 0]
                    pair_positions = enter + offsets * (3. ** .5) * voxels["side"] * directions
                    selected = stable_group_argmax(scores, ri, nquery)
                    if compute_loss:
                        point_membership = _extension("pcl_aabb").forward(
                            gt_xyz[image_idx, query_ids].contiguous(), bounds, ray_bid, voxel_bid)
                        labels = point_membership[vi, ri].bool() & query_gt_valid[ri]
                        positive_per_ray = torch.bincount(ri[labels], minlength=nquery) > 0
                        counts["gt_covered_query_count"] += positive_per_ray.sum()
                        if labels.any():
                            log_probability = scatter_log_softmax(scores, ri, dim=0)
                            prob_sum = prob_sum - log_probability[labels].sum()
                            prob_count = prob_count + labels.sum()
                        if self.training and epoch < self.maxpool_label_epochs:
                            teacher_selected = stable_group_argmax(labels.float(), ri, nquery)
                            selected = torch.where(positive_per_ray, teacher_selected, selected)
                    # Empty groups' argmax is a sentinel equal to pair count.
                    candidate_z = pair_positions[selected.clamp_max(pair.shape[0] - 1), 2]
                    valid_prediction = has_hit & torch.isfinite(candidate_z) & (candidate_z > 0)
                    counts["invalid_prediction_count"] += (has_hit & ~valid_prediction).sum()
                    counts["refined_count"] += valid_prediction.sum()
                    out[query_ids] = torch.where(valid_prediction, candidate_z, out[query_ids])
                    refined[query_ids] = valid_prediction
            if compute_loss:
                valid = supervision_valid[image_idx].reshape(-1)[queries]
                # Evaluate even fallback queries: never discard difficult samples
                # from denominators. Their frozen fallback has zero derivative.
                pred_xyz = rays[image_idx, queries] * (out[queries] / rays[image_idx, queries, 2])[:, None]
                pos_sum = pos_sum + (pred_xyz[valid] - gt_xyz[image_idx, queries][valid]).abs().sum()
                pos_count = pos_count + 3 * valid.sum()
            depths.append(out.reshape(1, height, width))
            refined_masks.append(refined.reshape(1, height, width))
            query_masks.append(query_mask.reshape(1, height, width))
        depth = torch.stack(depths)
        refined_mask = torch.stack(refined_masks)
        query_mask = torch.stack(query_masks)
        position_loss = pos_sum / pos_count.clamp_min(1)
        probability_loss = prob_sum / prob_count.clamp_min(1)
        normal_loss = zero
        if compute_loss:
            # Official training convention fills unsampled neighbours with GT.
            # This tensor exists only inside the loss and never feeds inference.
            normal_prediction_depth = torch.where(query_mask, depth, clean_gt)
            gt_normal = surface_normals(unproject_depth(clean_gt, intrinsics))
            pred_normal = surface_normals(unproject_depth(normal_prediction_depth, intrinsics))
            normal_valid = supervision_valid.clone()
            normal_valid[:, :, :, :-1] &= supervision_valid[:, :, :, 1:]
            normal_valid[:, :, :-1, :] &= supervision_valid[:, :, 1:, :]
            normal_valid[:, :, -1, :] = False
            normal_valid[:, :, :, -1] = False
            normal_valid = normal_valid & query_mask & (torch.linalg.vector_norm(gt_normal, dim=1, keepdim=True) > .5)
            normal_distance = (1. - F.cosine_similarity(pred_normal, gt_normal, dim=1)) * .5
            normal_loss = (normal_distance * normal_valid[:, 0]).sum() / normal_valid.sum().clamp_min(1)
        total = self.pos_weight * position_loss + self.prob_weight * probability_loss + self.normal_weight * normal_loss
        diagnostics = {key: value.detach() for key, value in counts.items()}
        if self.spvnas_branch is not None:
            diagnostics.update({key: value.detach()
                for key, value in self.spvnas_branch.last_diagnostics.items()})
        diagnostics.update({
            "total_pixel_count": rgb.new_tensor(batch * pixels),
            "input_invalid_count": (~good_input).sum().detach(),
            "intersection_coverage": (counts["intersect_count"] / counts["query_count"].clamp_min(1)).detach(),
            "refined_query_fraction": (counts["refined_count"] / counts["query_count"].clamp_min(1)).detach(),
            "replaced_pixel_fraction": (counts["refined_count"] / (batch * pixels)).detach(),
            "fallback_query_fraction": (1. - counts["refined_count"] / counts["query_count"].clamp_min(1)).detach(),
            "gt_voxel_coverage": (counts["gt_covered_query_count"] / counts["gt_valid_query_count"].clamp_min(1)).detach(),
            "probability_positive_count": prob_count.detach(),
        })
        return {"depth": depth, "losses": {"total": total, "position": position_loss,
                "probability": probability_loss, "normal": normal_loss},
                "diagnostics": diagnostics, "refined_mask": refined_mask, "query_mask": query_mask}
