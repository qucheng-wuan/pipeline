# Verification evidence

The final selection, experiment tables and release verification scope are in
[`reports/FINAL_REPORT.zh-CN.md`](../reports/FINAL_REPORT.zh-CN.md). These smaller
receipts preserve the supporting checks; they do not substitute for that report.

| Evidence | Actual scope |
|---|---|
| `stage2_fresh_runtime.json` | Independent Python environment, pinned dependencies and a fresh TorchSparse source build; CPU isolation checks |
| `stage2_gpu_core_parity_e63.json` | Original/portable Stage2 CUDA comparison using E63, including sparse boundary and empty-support cases; this is not final-checkpoint selection |
| `stage2_native_metric_parity.json` | Metric-adapter comparison on one real and one synthetic image |
| `training_initialization_download.json` | Actual fresh official ResNet34/SPVNAS downloads and source-declared SHA256 checks |
| `cleargrasp_rosters.json` | Complete reference metadata export/materialization and comparison against the audited split; GT arrays were not opened |
| `stage1_portability.json` | Three live legacy/portable cases, relocated-checkout CUDA inference and twelve newly generated RGB-only predictions, under the explicitly authorized policy |
| `stage2_training_{single,ddp,acceptance}.json` | Actual fresh-data single-/two-GPU training, three curriculum phases, nine optimizer updates per run and exact checkpoint-state restoration; short diagnostics, not new 80-epoch quality measurements |
| `stage1_reproducibility/` | Preserved zero-tolerance failure, original V2 self-repeatability diagnostics and the user-authorized numerical policy |

The original V2 uses CUDA `scatter_add` reductions. Its own repeated inference
can differ slightly before the downstream BF16 computations. The failed
zero-tolerance comparison remains a failure. The separately authorized Stage1
policy requires exact expert outputs and all V2 inputs; each of four metric
depth outputs must satisfy maximum absolute difference **0.05 mm** and mean
absolute difference **0.001 mm**. Nonmetric downstream intermediates are reported
as diagnostics, not claimed exact. This policy does not relax Stage2 parity.

The published diagnostic JSON files contain hashes and summaries. Referenced
large arrays and raw execution logs remain in the original local audit archive
and are not included in Git. Original absolute paths identify historical
evidence; production inference and fresh training do not read these directories.
No fresh 80-epoch training score or full-roster live-RGB score is inferred from
small-sample or short-training checks.
