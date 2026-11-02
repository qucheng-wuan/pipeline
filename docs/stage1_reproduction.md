# Stage1: RGB → frozen V2 → learned Laplacian E08

The public Stage1 is `full_pyramid_s20261023_e08`, SHA256
`db5f9b61e767dea8bd9ad93c45779327f640dfb8c51bcfeccfee11c333f74256`.
Its frozen V2 checkpoint is
`32faa8cae83a3acd536507df21e986f28a79c9ea70165c1998fed31babdb4e7d`.
Original checkpoint bytes, historical training settings and selection records
are retained; portability does not rewrite their embedded provenance.

## Inference

Use Python 3.11, PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128.
`bash scripts/setup_stage1.sh /chosen/environment` installs pinned dependencies.
Stage2 has a separate Python/PyTorch environment. Do not import its sparse
runtime into this process. No historical checkout or installed historical
package overlay is added to `sys.path` by the portable Stage1.

Place the 20 files in `provenance/stage1_assets.json` under an asset root,
retaining their listed relative paths and SHA256 values. They total
8,757,990,002 bytes. The publication/download tooling supplies these separately
from ordinary Git source files. `stage1_local_bindings.json` is a local export
aid only and must be ignored by Git; inference never reads it.

```bash
PYTHONPATH=src /chosen/environment/bin/python -m transparent_depth.stage1 \
  --rgb /data/example.png --output /results/stage1.npy \
  --assets provenance/stage1_assets.json --asset-root /model-assets --device cuda:0
```

Output is native-resolution FP32 optical-Z depth in metres plus a JSON receipt.
Every expert runs from RGB. Input GT depth, masks, normals and camera calibration
are absent. Stage2 separately uses the known camera K. The historical producer
used sealed RGB-derived expert caches for bulk experiments; the portable public
entrypoint deliberately exercises the full live path and does not require those
caches or their manifests.

The frozen dependency chain is MoGe-3 step0299; a shared DINOv3 H+/16 feature
backbone for the FAR DPT decoder and R5; calibrated TOM-Net; TransNormal with its
own DINO cross-attention encoder/projector; R5 with frozen O2/O4 anchors; native
V2; and the learned Laplacian adapter. DINO feature taps are 7/15/23/31 with
shape 4×1280×24×32. The standalone DINO loader consumes all 615 HF tensors and
reverses their original packed-QKV map. No FAR attention LoRA is enabled.

## Numerical protocol retained

Experts use the original 384×512 antialiased letterbox and explicit FP16 cache
round-trips. Their per-component precision is intentional: MoGe `high` matmul
with cuDNN TF32; R5 `highest` with cuDNN TF32; FAR `highest` without cuDNN TF32;
DINO extraction FP16; TransNormal BF16. V2 uses native tiles of 384 with halo16
and BF16 neural layers. Do not apply Stage2's blanket FP32 setting to Stage1.

Laplacian fusion is calculated on the entire native image before Stage2's
320×240 resize. The six log-depth bands are lifted through each dyadic parent
size, including odd dimensions. A 16-channel 24×32 context produces six gate
maps. The exact full-image interpolation-adjoint mean correction is retained;
an area-pool or crop-average replacement changes the model. The public live
path retains the historical second MoGe call rather than silently optimizing
away a potential numerical difference.

TransNormal RNG uses an RGB-only map of 46,679 exact file SHA256 values to
historical sample IDs. This small provenance asset contains no label pixels or
label paths. New RGB bytes use `rgb-sha256/<digest>`. Renaming unchanged files
therefore preserves the seed. Historical manifest access is unnecessary.

## Reproduce V2 and adapter training

All numerical model, geometry, objective and aggregation functions are included.
The V2 trainer retains AdamW, the original 8-epoch warmup/cosine schedule,
deterministic crop formula, validation eligibility and image-macro metrics.
The adapter trainer retains full-image band preparation, the fixed TOM-weighted
patch draw, independent GT-normal supervision, the three loss groups and
AdamW. The actual E08 run used 8 GPUs, batch8 per rank, 8 epochs, seed20261023,
learning rate0.0003, weight decay0.0001 and pyramid weight0.05. Its V2 and all
experts remain frozen. The historical settings JSONs under `provenance` are
evidence; new runs write their own source/asset bindings.

Supply a JSONL dataset roster with `sample_id`, safe unique `cache_id`, `split`
(`train`/`val`/`test`), `domain`, `rgb_path`, `target_depth_path`,
`transparent_mask_path` and optional prevalidated `camera`. Relative paths use
`TD_DATA_ROOT`. The original V2 roster is 45,453 training, 704 validation and
521 historical test images. The final two-stage report separately includes all
1,226 requested evaluation images; it overlaps validation and is retrospective.
K in V2/Laplacian training uses boundary pixel centres `(u+0.5,v+0.5)`.
Synthetic independent normals are read from the released `camera-normals`
directory; unreliable default-FOV cases disable K-dependent normal losses.

```bash
export PYTHONPATH="$PWD/src"
export TD_DATA_ROOT=/cleargrasp
export TD_ASSET_ROOT=/model-assets
export TD_STAGE1_NATIVE_ROOT=/new-run/native
export TD_STAGE1_LAP_ROOT=/new-run/laplacian

# Run workers on deliberately allocated GPUs, with distinct rank/world values.
python -m native_fusion.prepare worker --manifest /data/roster.jsonl \
  --output "$TD_STAGE1_NATIVE_ROOT" --rank 0 --world 1 --device cuda:0
python -m native_fusion.prepare seal --manifest /data/roster.jsonl \
  --output "$TD_STAGE1_NATIVE_ROOT"

# Optional: retrain the original V2 head from its original initialization.
torchrun --standalone --nproc_per_node=8 -m native_fusion.train \
  --manifest "$TD_STAGE1_NATIVE_ROOT/manifests/dataset.jsonl" \
  --output /new-run/v2 --epochs 8 --batch-size 2 --seed 20261021

# Reproduce the E08 adapter experiment with the published V2 kept frozen.
python scripts/init_stage1_training.py --root "$TD_STAGE1_LAP_ROOT"
python -m laplacian_train.prepare plan --root "$TD_STAGE1_LAP_ROOT" --scope full
python -m laplacian_train.prepare worker --root "$TD_STAGE1_LAP_ROOT" --scope full \
  --rank 0 --world 1 --device cuda:0
python -m laplacian_train.prepare seal --root "$TD_STAGE1_LAP_ROOT" --scope full
torchrun --standalone --nproc_per_node=8 -m laplacian_train.train \
  --root "$TD_STAGE1_LAP_ROOT" --scope full --run full_pyramid_reproduction \
  --epochs 8 --batch-size 8 --learning-rate 0.0003 --pyramid-weight 0.05 --seed 20261023
```

Full-image validation/test use `python -m laplacian_train.evaluate predict`
with `--split`, `--checkpoint` and `--output`, followed by the same command
with `evaluate`. Predictions are sealed and all arrays authenticated before
label reading. Frozen model accuracy reproduction and a newly retrained CUDA
trajectory are different claims: the latter is not promised to match bitwise.

## Required portability/parity acceptance

The old E08 one-image smoke passed for `cup-with-waves-train/000000187`:
cached-vs-live max depth difference 1.65104866e-5 m, mean1.51809887e-9 m.
These old measurements are not a certificate for this new implementation.
Repeated CUDA interpolation-adjoint reductions already differed by up to
2.3841858e-7 m in the old final output; strict bitwise equality between all
independent executions would misstate that evidence.

The new acceptance must run the legacy and portable live paths on the same
RGB, device, weights, environment and teacher seed; compare component tensors,
V2 depth, bands, context, coefficients, gate maps and final native depth. Also
test one real image and an odd-size novel RGB. Prohibit legacy cache/source/GT
reads in the portable process; compare actual repeated-run noise before
interpreting differences. Then verify the Stage1→Stage2 resize and half-pixel K
transform, Stage2 fallback and final native restoration. Record real results,
source/asset SHA, precision and peak memory. Source inventories are explicitly
not execution receipts. Any source changes require a new acceptance receipt.

## Third-party terms

MoGe code is MIT except included DINOv2 code under Apache-2.0. Its source is
pinned to `74fbce054ebed49800de42d0ad0e83495065719a`; the historical checkout's
training-only dataloader patch is retained as supplied, not represented as a
clean upstream tree. DINOv3 uses the separate DINOv3 License and commit
`6876159a11b4df116f30f667f8c9888617df0751`. TransNormal source and weights are
CC BY-NC4.0, source commit `11b573907151e665426f02e5ccedee294bf0e058`, model
revision `796f97ad3d82ea2e455bed88afe3c33aca8e115f`. TOM-Net is MIT, upstream
commit `f4c60113653ea6f73e0a20a944691888bfb3065e`; its PyTorch port and trained
checkpoint are part of this user's model. Preserve the supplied third-party
license notices. The private repository does not relicense all third-party
weights under a single project license.

## Fresh full-roster RGB export

`transparent_depth.stage1_batch` keeps one complete frozen predictor alive per
worker. The input JSONL deliberately accepts only `sample_id`, `rgb_path`,
`rgb_sha256` and optional `native_hw`; it rejects GT, mask, depth-cache and K
fields. K belongs to the separately prepared Stage2 manifest. Relative RGB paths
resolve against the input manifest. Give each worker a distinct rank, an explicit
GPU allocation and the same world size. Existing valid predictions are verified
and skipped, so interrupted workers can continue without rerunning completed
images. All source/weight bytes must still match the original export contract.

```bash
python -m transparent_depth.stage1_batch predict --manifest /data/rgb1226.jsonl \
  --output /results/stage1_live --asset-root /model-assets \
  --rank 0 --world 8 --device cuda:0 --expected-count 1226
# Repeat for the other ranks using their deliberately assigned visible GPU.
CUDA_VISIBLE_DEVICES='' python -m transparent_depth.stage1_batch seal \
  --manifest /data/rgb1226.jsonl --output /results/stage1_live \
  --asset-root /model-assets --expected-count 1226
```

The seal verifies every native FP32 `.npy`, every SHA, exact roster coverage and
successful worker isolation receipts. `predictions.jsonl` names each native
optical-Z prediction under `prediction.path`, relative to that manifest, with
its hash, geometry and original RGB hash. Stage2's fresh-data preparation consumes
this manifest. The open guard permits only this checkout, the explicit model
asset directory, named RGB files, its own output/cache and system/runtime files.
It also blocks recognized label filenames even when explicitly named as RGB.
Its receipt honestly records Python audit/OpenCV/OpenEXR scope; it is an
engineering tripwire rather than an OS sandbox. `--deny-prefix` can explicitly
block every historical project/cache root during acceptance.

## Isolated traces and relocation

Run `scripts/trace_stage1_live.py --mode legacy` and `--mode portable` in separate
processes with `--rgb`, its exact `--rgb-sha256`, a new `--output` directory and the
assigned `--device`. Portable mode also receives `--asset-root`. Legacy mode
requires explicit `--legacy-project` directories and the selected
`--checkpoint`/`--checkpoint-sha256`; it always forces live expert computation and
blocks tensor-cache loaders. No legacy path is embedded in the portable entry.
The legacy process may read its original provenance metadata; the portable
process must pass its strict read guard without those directories.

The trace records both actual MoGe calls, DINO features, TOM logits/probability,
TransNormal outputs, R5/FAR outputs, every V2 input, native V2, fixed depth/bands,
context, exact mean coefficients, learned gates and final depth. BF16 values are
stored losslessly as FP32 with the source dtype recorded. Hooks only copy outputs;
CUDA synchronization means trace timings are not normal inference timings.
`complete.json` says `trace_complete`, never numerical parity. Compare separately:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/compare_stage1_traces.py \
  --reference /checks/legacy --candidate /checks/portable \
  --output /checks/comparison.json --atol 0 --rtol 0
```

Default comparison requires exact values and preserves failed comparisons.
A different tolerance must be explicitly declared and justified by repeated-run
noise and per-layer attribution; the comparator never increases it automatically.

`scripts/prepare_stage1_relocation.py` copies source/provenance into a new directory,
checks every sealed source hash, and runs a CPU-only import check with CUDA hidden.
It writes an explicit pending GPU trace command that forbids the original project
root. Its CPU receipt does not claim relocated GPU inference succeeded. The
actual relocated inference and all-1,226 end-to-end evaluation remain separate
acceptance steps; report only their real completed receipts.

The twentieth asset is the original 6,963-byte FlexGEMM RTX 3090 kernel-selection
configuration (14 kernel entries), SHA256
`e40097daa6f432754cc1a27762ae5910a114df5034d9d218a61109f4021a39cd`.
It is runtime configuration, not a depth or teacher-output cache. Before importing
FlexGEMM, each predictor verifies this seed and copies it to a new private writable
cache. It preserves the existing adaptive-autotuning policy and permits later
runtime updates only in that private copy; the frozen asset remains unchanged.
No `~/.flex_gemm` cache is consulted. An already-imported FlexGEMM runtime is
rejected because its previously loaded cache cannot be authenticated retroactively.
Keep one persistent predictor per worker process.

The CLI accepts `--runtime-cache /writable/runtime`; the Python API accepts
`Stage1Predictor(..., runtime_cache=...)`. Without either, the standalone API uses
`outputs/stage1_runtime/pid-PID` under the checkout. Batch/trace entrypoints already
supply their isolated runtime-cache root. Legacy acceptance also seeds a private
copy from the identical frozen asset before its subprocess imports FlexGEMM.
