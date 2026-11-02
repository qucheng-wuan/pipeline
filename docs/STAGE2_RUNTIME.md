# Stage2 runtime, assets, training and numerical provenance

The deployed chain takes RGB and known camera intrinsics. `--depth-input` is the
native optical-Z output of the frozen RGB Stage1, in metres. It is not a sensor
measurement. There is no inference mask/GT argument. Stage2 reads RGB, predicted
P and K at 240×320, queries the whole image, then bilinearly restores native
resolution. Integer pixel centres put the upper-left centre at (0,0). The resize
uses `c'=(c+0.5)*scale-0.5`; this differs from simply scaling the principal point.

## Installation

Use an isolated Python 3.10 environment and CUDA **toolkit** 12.1 with GCC/G++11.
The tested GPU family is RTX3090 / SM8.6. A different GPU requires setting
`TORCH_CUDA_ARCH_LIST` and repeating numerical parity checks. A CUDA driver alone
cannot build the kernels. No compiled `.so` is distributed as a portable binary.

```bash
conda create -n transparent-depth-stage2 python=3.10
conda activate transparent-depth-stage2
export CUDA_HOME=/your/cuda-12.1
export CC=gcc-11 CXX=g++-11
bash scripts/install_stage2.sh
```

The installer pins PyTorch2.1.2+cu121, torchvision0.16.2+cu121,
torch_scatter2.1.2+pt21cu121 and TorchSparse commit
`385f5ce8718fcae93540511b7f5832f4e71fd835` (its version string is 2.1.0;
installing a generic 2.1.0 tag is insufficient). Sparsehash is pinned to
`f93c0c69e959c1c77611d0ba8d107aa971338811`. The runtime checks exact Python source
hashes of TorchSparse. CUDA binaries are built locally and thus have no cross-
machine byte-identity guarantee. Keep CUDA_HOME/CC/CXX set for the first LIDF
kernel JIT build; `TORCH_EXTENSIONS_DIR` may point to a writable local cache.
No GPU compile/inference is triggered merely by importing the portable CLI.

A Python installation must include development headers (`Python.h`); a minimal
system interpreter without its matching development package is insufficient.
The installer checks Python3.10, headers, GCC/G++11 and nvcc12.1 before building.
`TD_STAGE2_DEPENDENCY_DIR` selects an absolute writable directory for the pinned
Sparsehash/TorchSparse source and build artifacts; it defaults to the ignored
`.deps/stage2` directory. Set `TMPDIR`, `PIP_CACHE_DIR` and
`TORCH_EXTENSIONS_DIR` to storage with adequate free space. A fresh installation
can need10–15GiB including wheels, unpacked dependencies and build intermediates.
Existing official wheels may be installed into the new environment first after
SHA verification; no old `site-packages` directory needs to be linked or copied.
TorchSparse is compiled from its pinned commit with CUDA architecture8.6 by
default. The final version/source inspection explicitly hides CUDA devices,
and `pip check` must pass. This installation check does not establish CUDA
numerical parity; the separately recorded GPU acceptance does.

## Deployment

```bash
python -m transparent_depth.stage2 \
  --rgb example.png --depth-input stage1_native.npy \
  --intrinsics camera.json --bundle /data/transparent-depth-assets/stage2 \
  --output final_native.npy --device cuda:0
```

`camera.json` may contain a bare 3×3 integer-centre matrix or
`{"pixel_convention":"integer_pixel_centres","intrinsics":[...]}`.
Predicted P must be positive finite native float32 HW. The CLI hashes RGB,
P, the selected model bundle and the output in a sidecar metadata file.
It applies precisely the production OpenCV linear resize, ImageNet RGB
normalization, and native PyTorch bilinear resize (`align_corners=False`).
The same RGB SHA seeds model sampling as the production prediction exporter.

`export_stage2_bundle(checkpoint_path, output_dir, selection_metadata)` converts
an explicitly trusted training checkpoint on CPU to model.pt + model.json.
All model tensors are retained and load strictly. Metadata records original
checkpoint SHA, epoch, update count, selection rule and relative package-source
hashes. It is relocation-safe and deliberately omits old contract/migration
admission paths. The full trained state needs neither ResNet initialization
weights nor the SPVNAS initialization weights. The small official `net.config`
is included because it defines the specialized architecture. Export only after
finalizing package sources; modifying a bound source invalidates the bundle.

## Fresh training and resume

`configs/stage2_train.json` describes a **fresh** 3-GPU 80-epoch run: 20 epochs
alpha1, 30 epochs linearly decreasing from1 to0, then30 epochs alpha0. Each epoch
covers all45,453 training images. Batch2/GPU, accumulation5 gives an effective
batch30 and1,516 optimizer updates/epoch (partial final accumulation uses its
actual microbatch count). Adam rates are1e-4 for LIDF and SPVNAS,1e-3 for adapters;
StepLR decays after epoch60, so the reduced rates apply from epoch61. Both dense
and sparse TF32 are disabled, forward uses FP32 without autocast, and no gradient
clipping is added.

The training input is `P + alpha*(G-P)` only at valid nontransparent GT pixels.
Transparent pixels and invalid GT always use P. Training uses GT background
and GT segmentation; it must not be described as label-free. Every selection
validation uses pure P. The diagnostic phase-matched validation is explicitly
label-conditioned when alpha>0 and never selects a checkpoint. Only epochs51–80
are eligible for best.pt, using704-image native full-image mean AbsRel.

To reuse the audited ClearGrasp split without manually writing the rosters, run
this command from the repository root:

```bash
python scripts/prepare_cleargrasp_rosters.py \
  --data-root /your/data --output outputs/cleargrasp
```

It creates `rgb_only.jsonl`, `input_roster.jsonl` and a separate `label_roster.jsonl`
for the commands below. The compressed reference preserves original native K,
sample order and calibration flags. See the [dataset preparation guide](../data/cleargrasp_reference/README.md)
for the expected directory layout, rectified-Z requirement and optional RGB SHA
verification. This metadata command does not open GT arrays.

For custom data, create three separate CSV/JSONL rosters. The RGB-only Stage1 roster has
`sample_id`, `rgb_path`, `rgb_sha256`, and optional `native_hw`. The Stage2 input
roster contains `sample_id`, `rgb_path`, native `intrinsics` (3×3),
`pixel_convention: "integer_pixel_centres"`, `role` (`train`, `val`, or
`evaluation`; alternatively `roles`), `domain` (`real` or `synthetic`), and
`official_split`. The separate label roster contains `sample_id`,
`target_depth_path`, `transparent_mask_path`, optional `normal_path`,
`normal_camera_valid`, and `depth_semantics: "already_rectified_optical_z_m"`.
Depth EXR values must already be optical Z in metres; this tool performs no
radial-to-Z or sensor-unit conversion. Relative image/label paths resolve against
their corresponding roster. A validation/evaluation sample can share an ID and
prediction; training IDs must remain separate.

Generate actual Stage1 predictions in the Stage1 environment. Stage1 accepts
only the RGB roster and cannot read K, roles or label paths. These commands use
the asset directory from the root README; official Stage2 initialization assets
are downloaded into its separate `stage2_initialization` subdirectory:

```bash
# Stage1 environment: use the expected number of unique RGB sample IDs.
CUDA_VISIBLE_DEVICES=0 python -m transparent_depth.stage1_batch predict \
  --manifest outputs/cleargrasp/rgb_only.jsonl --output outputs/caches/cleargrasp_p \
  --asset-root /data/transparent-depth-assets --rank 0 --world 1 --device cuda:0 \
  --expected-count 46679
python -m transparent_depth.stage1_batch seal \
  --manifest outputs/cleargrasp/rgb_only.jsonl --output outputs/caches/cleargrasp_p \
  --asset-root /data/transparent-depth-assets --rank 0 --world 1 --device cuda:0 \
  --expected-count 46679
# Stage2 environment: authenticate the producer, then create separate caches.
python -m transparent_depth.prepare_stage2_data \
  --input-roster outputs/cleargrasp/input_roster.jsonl \
  --label-roster outputs/cleargrasp/label_roster.jsonl \
  --predictions outputs/caches/cleargrasp_p/predictions.jsonl --output outputs/caches/stage2
python scripts/fetch_stage2_initialization.py \
  --output /data/transparent-depth-assets/stage2_initialization
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --standalone --nproc-per-node=3 \
  -m transparent_depth.train_stage2 --config configs/stage2_train.json \
  --input-manifest outputs/caches/stage2/manifests/input_manifest.jsonl \
  --label-manifest outputs/caches/stage2/manifests/label_manifest.jsonl \
  --stage1-root outputs/caches/stage2/stage1 --label-root outputs/caches/stage2/labels \
  --assets-root /data/transparent-depth-assets/stage2_initialization --output runs/stage2
# Repeat the same training command with --resume for the committed epoch.
```

The count46679 follows the historical released roster:45453 training samples plus
1226 evaluation identities, with704 validation identities included in evaluation.
Use the actual unique count of your roster and set the declared counts in a
separate diagnostic configuration when performing a smaller acceptance run.
The preparer requires exact sample-ID coverage from the sealed predictions.

For a **fresh8-GPU** run use `configs/stage2_train_8gpu.json`,
`--nproc-per-node=8`, and an explicitly authorized8-device list. That profile uses
batch2/GPU and accumulation2: effective batch32,1421 updates/epoch,45456 sample
presentations including3 DistributedSampler padding repeats. The3-GPU profile
uses batch2/accumulation5, effective batch30 and1516 updates/epoch. Both cover all
45453 unique samples each epoch. Neither claims to replay the historical
8-to-3-GPU optimizer trajectory.

The shown GPU lists are examples. The operator must select authorized devices.
No entrypoint kills another process. An output-directory flock rejects concurrent
training launchers, and only rank0 makes the fresh/resume decision before
broadcasting it to other ranks. Source, configuration, manifests, producer asset
SHAs (including E08), Stage1 seal, separate supervision seal, world size and update
count are bound in the resume contract. Input and label cache files are SHA
checked before training; label files are SHA checked again on access, while input
metadata is checked on access. Native validation labels are checked before
scoring. All checkpoint/contract writes are mandatory; progress telemetry
is best effort. Checkpoints retain optimizer, scheduler and per-rank RNG states,
which are checked exactly on restoration. Epoch archives and best/last are
repaired after a partially committed report/best write.

`scripts/run_stage2_smoke.py` waits for a **real** portable Stage1 seal and runs
three tiny epochs of three actual optimizer updates each. It interrupts after
epoch1 and restores the complete state before continuing; alpha values are
`1,1,1,1,.5,0,0,0,0`. With10 training and2 validation samples, one GPU uses
batch2/accumulation2 and exercises a partial final accumulation window. All
trainable parameter groups must actually change, and losses/gradients must be
finite. This produces `smoke.complete.json` and `acceptance.complete.json`, never
a formal training completion marker or a model-quality claim:

```bash
# Waits without using a GPU until predictions.jsonl and complete.json exist.
CUDA_HOME=/your/cuda-12.1 CC=gcc-11 CXX=g++-11 \
python scripts/run_stage2_smoke.py \
  --input-roster data/smoke_rgb_k.jsonl --label-roster data/smoke_labels.jsonl \
  --predictions caches/smoke_native_p/predictions.jsonl \
  --assets-root /data/transparent-depth-assets/stage2_initialization --output runs/training_smoke \
  --gpus 4 --wait-timeout 21600
```

The released historical run changed from8 to3 GPUs after epoch29. A fresh3-GPU
run is **not** claimed to reproduce that exact stochastic trajectory. The model,
losses, input composition and validation mathematics remain the same. The
portable training entrypoint is new orchestration; only tested GPU parity
results should be presented as evidence, not inferred from source copying.

## Evaluation and selection disclosure

`transparent_depth.evaluation` wraps the preserved native_fusion evaluator.
Synthetic released camera normals remain independent normal references;
real data uses a depth-induced proxy. Default-FOV cases retain their invalid
camera-normal flag. The wrapper converts integer-centre K back to the metric
engine's boundary-coordinate convention (+0.5 principal point). It never applies
per-image GT scale or shift alignment. `evaluate_sealed_predictions` authenticates
all prediction SHA values before opening labels and reports full1226 and the
original521 test subset separately.

If the final epoch is chosen by all80×1226 transparent MAE, that choice is
**test-informed**; the same1226 score is not an untouched test estimate. Full1226
already includes the704 validation images used during training. Keep history/
exposure disclosures with any released score. The winner and its final metrics
must come from the completed sweep; an interim epoch is not a final result.

## Parity checklist

CPU tests cover unchanged numerical-source digests, half-pixel resizing,
complete-state export/reload and directory relocation, fail-closed source/weight
checks, curriculum GT isolation, exact alpha0, transition endpoints, partial
accumulation and interrupted best-checkpoint recovery. The package includes
native geometry/maths tests and should compare real evaluation samples with the
sealed production metrics.

Before declaring GPU parity, use the same checkpoint and cached RGB/P/K in the
same pinned runtime and compare old/new low-resolution predictions, refined
masks and native predictions. Then test actual live RGB→Stage1→Stage2 separately.
Compare finite/positive outputs, full query coverage and unchanged P fallback.
Batch order/single-vs-batch, empty/singleton sparse support, custom kernels on a
nonzero logical CUDA device and a two-step distributed backward are meaningful
checks. Report measured maximum/mean differences and exact mask agreement;
source identity alone does not establish CUDA numerical equality.

CPU native-metric parity was checked against the archived E63 outputs on one
real and one synthetic image. Every depth/normal/edge support count matched
exactly; the maximum relative difference across scalar metrics and sums was
4.37e-9. This verifies the metric adapter on those two images only. It does not
constitute Stage2 CUDA model parity or a new quality result. The reproducible
check is scripts/validate_stage2_metrics.py.

For release E58, the complete same-P comparison covers all 1,226 images.
Transparent MAE is 55.9103744437 mm versus the historical 55.9103739407 mm;
all 18 subset/domain/region MAE changes are below 0.000003381 mm. The original
1e-6 m per-pixel check is still recorded as failed: five low-resolution pixels
exceed it, with a maximum difference of 56.46 mm. Each image's mean absolute
prediction difference remains below 0.001 mm and every refined mask matches.
Plain old-code self-replays reproduce the same near-tied hard-argmax jumps;
the numerical sources and full model state were preserved.

The additional quality review follows the requested approximately matching
training behavior. It is separate from strict pixel parity and does not change
model operators, weights or the original failure. See the final report and
`audit/stage2_quality_acceptance.json` for the operational limits, complete
partition comparisons, repeatability traces and explicitly reviewed scope.
