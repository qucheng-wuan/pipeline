#!/usr/bin/env bash
# Run each stage's CPU checks in its actual pinned environment.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'Usage: bash scripts/test_cpu.sh /envs/stage1/bin/python /envs/stage2/bin/python' >&2
  exit 2
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stage1_python="$1"
stage2_python="$2"
export CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_dir/src"
cd "$project_dir"
"$stage1_python" -m pytest -q \
  tests/test_stage1_portable.py tests/test_stage1_batch.py tests/test_stage1_trace.py
"$stage2_python" -m pytest -q \
  tests/test_stage2_portable.py tests/test_stage2_preparation.py tests/test_stage2_full.py \
  tests/test_portable_cli_assets.py tests/test_sweep_checkpoints.py tests/test_evaluate_selected.py \
  tests/test_build_release_report.py tests/test_cached_quality_acceptance.py \
  tests/test_cleargrasp_rosters.py tests/test_transfer256.py tests/test_cleargrasp256.py \
  tests/test_evaluate256_metrics.py tests/test_rescore256.py
