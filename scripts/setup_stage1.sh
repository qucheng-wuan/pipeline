#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stage1_env="${1:-$project_dir/.venv-stage1}"
python3.11 -m venv "$stage1_env"
"$stage1_env/bin/python" -m pip install --upgrade pip setuptools wheel
"$stage1_env/bin/python" -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
"$stage1_env/bin/python" -m pip install -r "$project_dir/requirements-stage1.txt"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$project_dir/src" "$stage1_env/bin/python" -c 'import torch, torchvision; assert torch.__version__ == "2.8.0+cu128"; assert torchvision.__version__ == "0.23.0+cu128"; print("Stage1 environment installed; GPU parity remains a separate check.")'
