This source subset is copied unchanged from `third_party/spvnas` in
qucheng-wuan/LIDF-SPVNAS commit a0b2771385bc802e2fbd52da2c4407ae86ac7e70,
which pins mit-han-lab/spvnas commit 5c53224ca685844eb27caedf22817a89233ac0ce.
Its MIT license is retained in LICENSE. The experiment's integration code lives
outside this source subset, in `v2_lidf/spvnas_fusion.py` and `model.py`.

The integration starts from the reference repository's SPVNASFusion, retaining:

- Official model_zoo.spvnas_specialized, manual_select and determinize, followed
  by strict verification of every pretrained tensor before classifier removal.
- 20GMACs SemanticKITTI searched topology, 48 classification-preceding features;
  this is a fixed searched architecture, not a fresh architecture search.
- Camera metres transformed to (z,-x,-y), a 0.05 m fixed sparse grid, per-sample
  coordinate translation, batch-first integer coordinates, exact inverse maps,
  fixed sparse spatial_range (B,64,64,64), and both mean-pooling operations.
- Excluding unmatched -1 point-to-voxel hash assignments, never clamping them
  to voxel zero. This reference integration fix deliberately differs from the
  vendored upstream utility and preserves batch isolation.
- Inverse ImageNet RGB normalization before the 6-to-4 input adapter; xyz
  identity and RGB luminance initialization; 48-to-128 fusion with Normal(0,1e-4)
  weight and zero bias. The residual augments retained PointNet voxel features.

Adaptations for the frozen V2 E08 pipeline:

- Only supplied input depth and calibrated K construct support geometry. There
  is no GT/mask argument in the SPVNAS branch. Training curriculum composition
  occurs upstream; evaluation input is frozen RGB-derived depth only.
- Support/query sampling, fixed LIDF AABB volume, full-image inference, fallback,
  losses, and deterministic raw-score ray selection retain the preceding local
  implementation. Consequently this is not a bitwise reproduction of the
  reference project's complete RGB-D dataset/trainer/evaluation pipeline.
- All local images are assembled into one SPVNAS forward per microbatch. Global
  point and voxel offsets are explicit; PointNet is still evaluated with the
  same per-image points and voxel ordering. No cross-image context is introduced.
- Zero-row sparse convolutions and point/voxel maps bypass undefined empty CUDA
  hash operations, retain differentiable zero edges, and traverse the ordinary
  backbone's normalization nodes. Nonempty convolutions call the pinned native
  implementation unchanged; no artificial points are inserted.
- At most one globally supported example can produce a singleton at a coarse
  sparse level. For that exceptional microbatch only, branch BN uses existing
  running statistics, with convolution training mode retained. Ordinary batches
  use sparse-aware SyncBatchNorm; zero-row ranks contribute zero observations.
  Local unconverted singleton BN also uses running statistics. These boundary
  semantics require the explicit empty/singleton/mixed-rank GPU preflight.
- Official asset hashes are mandatory; construction RNG is restored around
  specialized-backbone determinization. Per-rank load audits include the actual
  legacy ImageNet ResNet34 source/hash and the separate random 32-channel head.
- Dense cuDNN/matmul and sparse operators explicitly disable TF32. On actual
  ClearGrasp inputs, the default cuDNN TF32 path changed RGB features between
  B=1 and B=2 and could reverse near-tied LIDF candidate logits. FP32 removed
  those reversals in the independent preflight; the ray-selection rule remains
  unchanged. CUDA scatter sums may still differ in the last FP32 bit.
- During training with gradients, the sparse input features include the
  value-zero dependency `rgb_features.sum()*0`, including when no points survive
  support selection. This orders sparse SyncBN backward before RGB SyncBN
  backward on every rank; otherwise empty-rank shortcuts can interleave the two
  collective sequences. Evaluation does not add this dependency. Neither
  feature values nor mathematical gradients are changed, and no GT is involved.
- Distributed training also uses one full gradient bucket so that DDP gradient
  reduction cannot interleave with SyncBN backward on an empty rank. In pinned
  PyTorch 2.1.2, both the initial/rebuilt first-bucket limit and ordinary bucket
  limit must cover all gradient bytes; changing only `bucket_cap_mb` is
  insufficient. The shared distributed helper validates this runtime protocol.

TorchSparse must be commit 385f5ce8718fcae93540511b7f5832f4e71fd835. The old tag
v2.1.0 is not the same interface despite the matching version string. Reference
environment: Python 3.10, torch 2.1.2+cu121, torchvision 0.16.2+cu121,
torch-scatter 2.1.2+pt21cu121, torchpack 0.3.1, CUDA toolkit 12.1; sparsehash
commit f93c0c69e959c1c77611d0ba8d107aa971338811. New environments must pass actual
forward/backward, batch-isolation and distributed tests before training.
