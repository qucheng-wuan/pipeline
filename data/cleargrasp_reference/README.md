# ClearGrasp reference split

These three compressed JSONL files contain metadata only. They contain no RGB
images, GT depth/mask/normal arrays, or predicted depths. Obtain the dataset
separately, subject to its original terms.

The reference preserves the audited sample identities and their first-occurrence
order: **45,453 training + 704 validation + 1,226 evaluation**. Validation is part
of evaluation, so there are **46,679 unique samples**. Repeated validation and
evaluation identities share one record with both roles. No training identity
crosses roles, and the recorded train/evaluation RGB SHA sets do not overlap.
This roster preserves the original split; the published 1,226-image evaluation
is not an unseen test because it includes the 704 selection-validation images.

`rgb_k.jsonl.gz` stores native-resolution integer-centre K, RGB paths/SHA values,
roles and identity metadata. `labels.jsonl.gz` stores separate supervision paths,
the explicit rectified optical-Z-metre convention, and the original normal-camera
validity flag. `calibration_provenance.jsonl.gz` preserves calibration origin and
source SHA separately; it is never included in the model's input roster.
13,656 unique samples use the recorded official default FOV for missing legacy
metadata. The others use released per-frame synthetic calibration (32,737) or
real-camera YAML calibration (286). The defaults are not per-image measurements.

Arrange existing data under a parent directory, preserving these two names:

```text
/your/data/
  cleargrasp-dataset-train/
    cup-with-waves-train/rgb-imgs/...
    cup-with-waves-train/depth-imgs-rectified/...
    ...
  cleargrasp-dataset-test-val/
    real-val/...
    synthetic-test/...
    ...
```

The referenced `depth-imgs-rectified` files must already contain **optical-Z
metres**, with the exact released/previously prepared file layout. This tool does
not convert raw radial distance, rescale depth units, or create GT. Referenced
masks and normals must also exist. Calibration numeric values are already in the
reference, so historical project code and calibration-source files are not needed
to materialize the rosters.

From the repository root, using Python 3.10 or newer:

```bash
python scripts/prepare_cleargrasp_rosters.py \
  --data-root /your/data --output outputs/cleargrasp
```

This creates `rgb_only.jsonl` (only sample ID, RGB path/SHA, native size),
`input_roster.jsonl` (only permitted RGB/K/role fields), `label_roster.jsonl`
(separate GT paths), `calibration_provenance.jsonl`, and `complete.json`.
The default verifies compressed-reference SHA values, sample order/counts,
roles, path confinement, required file existence and RGB SHA field format.
It **does not read RGB bytes or any GT arrays**. Add `--verify-rgb-sha` to read
every RGB and verify its recorded SHA; this can be expensive for the full data.
The Stage1 predictor independently hashes each RGB when actually predicting.
The output marker distinguishes these verification scopes explicitly.

Generate genuine Stage1 predictions with the Stage1 interpreter, then create
Stage2 caches with the Stage2 interpreter. Do not substitute GT or sensor depth
for this prediction manifest:

```bash
# Select authorized physical devices explicitly in the environment.
/envs/stage1/bin/python -m transparent_depth.stage1_batch predict \
  --manifest outputs/cleargrasp/rgb_only.jsonl --output outputs/caches/cleargrasp_p \
  --asset-root /data/transparent-depth-assets --rank 0 --world 1 --device cuda:0 --expected-count 46679
/envs/stage1/bin/python -m transparent_depth.stage1_batch seal \
  --manifest outputs/cleargrasp/rgb_only.jsonl --output outputs/caches/cleargrasp_p \
  --asset-root /data/transparent-depth-assets --expected-count 46679
/envs/stage2/bin/python -m transparent_depth.prepare_stage2_data \
  --input-roster outputs/cleargrasp/input_roster.jsonl \
  --label-roster outputs/cleargrasp/label_roster.jsonl \
  --predictions outputs/caches/cleargrasp_p/predictions.jsonl --output outputs/caches/stage2
```

Continue with the fresh 3-GPU or 8-GPU training command in
[the Stage2 runtime guide](../../docs/STAGE2_RUNTIME.md). Both profiles consume
the same complete 45,453-image training split. Training uses GT background and
GT masks for the curriculum; pure-P selection validation and deployment do not
use GT/masks as model inputs.

`manifest.json` records compressed-file SHA values, original manifest SHA values,
role/calibration counts, and ordered sample-ID SHA. The optional
`--export-from-audited-manifests INPUT LABEL` argument is a maintainer export
operation; normal setup needs only this repository and the dataset directory.
