#!/usr/bin/env bash
# Run in an isolated Python3.10 environment; build for the actual target GPU.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_profile="${TD_RUNTIME_PROFILE:-reference}"
dependency_dir="${TD_STAGE2_DEPENDENCY_DIR:-$project_dir/.deps/stage2-$runtime_profile}"
[[ "$dependency_dir" = /* ]] || { echo "TD_STAGE2_DEPENDENCY_DIR must be absolute" >&2; exit 2; }
: "${CUDA_HOME:?Set CUDA_HOME to the selected CUDA toolkit (not merely the driver)}"
export PATH="$CUDA_HOME/bin:$PATH"
case "$runtime_profile" in
  reference)
    export CC="${CC:-gcc-11}" CXX="${CXX:-g++-11}"
    cuda_release=12.1; cuda_index=cu121; requirements_file=stage2.txt
    scatter_version=2.1.2+pt21cu121; scatter_wheels=https://data.pyg.org/whl/torch-2.1.0+cu121.html ;;
  blackwell-cu130)
    export CC="${CC:-gcc}" CXX="${CXX:-g++}"
    cuda_release=13.0; cuda_index=cu130; requirements_file=stage2-cu130.txt
    scatter_version=2.1.2+pt211cu130; scatter_wheels=https://data.pyg.org/whl/torch-2.11.0+cu130.html ;;
  *) echo "Unknown TD_RUNTIME_PROFILE: $runtime_profile" >&2; exit 2 ;;
esac
export TD_RUNTIME_PROFILE="$runtime_profile"
export FORCE_CUDA=1
export PYTHONNOUSERSITE=1
export MAX_JOBS="${MAX_JOBS:-2}"
: "${TORCH_CUDA_ARCH_LIST:?Set the target GPU compute capability explicitly, e.g. 8.6, 8.9, 9.0 or 12.0}"
python - <<'CHECK'
import pathlib, sys, sysconfig
assert sys.version_info[:2] == (3, 10), 'Stage2 requires Python 3.10'
assert pathlib.Path(sysconfig.get_path('include'), 'Python.h').is_file(), 'Python development headers are required to compile CUDA extensions'
CHECK
for compiler in "$CC" "$CXX"; do
  command -v "$compiler" >/dev/null
  if [[ "$runtime_profile" = reference ]]; then
    [[ "$("$compiler" -dumpversion)" = 11* ]] || { echo 'Use GCC/G++11 for the reference CUDA build' >&2; exit 2; }
  fi
done
"$CUDA_HOME/bin/nvcc" --version | grep -q "release $cuda_release"
python -m pip install 'setuptools==69.5.1' wheel
python -m pip install --extra-index-url "https://download.pytorch.org/whl/$cuda_index" -r "$project_dir/requirements/$requirements_file"
python -m pip install --no-deps "torch-scatter==$scatter_version" --find-links "$scatter_wheels"
mkdir -p "$dependency_dir"
if [ ! -d "$dependency_dir/sparsehash/.git" ]; then
  git clone https://github.com/sparsehash/sparsehash.git "$dependency_dir/sparsehash"
fi
git -C "$dependency_dir/sparsehash" diff --quiet
git -C "$dependency_dir/sparsehash" diff --cached --quiet
git -C "$dependency_dir/sparsehash" checkout --detach f93c0c69e959c1c77611d0ba8d107aa971338811
(
  cd "$dependency_dir/sparsehash"
  ./configure --prefix="$dependency_dir/sparsehash_install"
  make -j "$MAX_JOBS"
  make install
)
export CPLUS_INCLUDE_PATH="$dependency_dir/sparsehash_install/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
if [ ! -d "$dependency_dir/torchsparse/.git" ]; then
  git clone https://github.com/mit-han-lab/torchsparse.git "$dependency_dir/torchsparse"
fi
git -C "$dependency_dir/torchsparse" diff --quiet
git -C "$dependency_dir/torchsparse" diff --cached --quiet
git -C "$dependency_dir/torchsparse" checkout --detach 385f5ce8718fcae93540511b7f5832f4e71fd835
python -m pip install --force-reinstall --no-deps --no-build-isolation "$dependency_dir/torchsparse"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$project_dir/src" python -c 'import torch; from v2_lidf.runtime import validate_runtime; print(validate_runtime()); assert not torch.cuda.is_initialized()'
# The package remains editable so CUDA .cpp/.cu and topology assets are local.
python -m pip install --no-deps -e "$project_dir"

python -m pip check
