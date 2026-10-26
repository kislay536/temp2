#!/bin/bash
# Unified venv builder for all 12 rtgs/splatonic/slamio environments.
#
# Every target below is pinned to the same stack: python3.10, torch==2.3.1
# built against cu121, and TORCH_CUDA_ARCH_LIST=8.6 (A40 / Ampere, sm_86).
# This was NOT the case historically -- previous_works' six repos actually
# ran on torch 2.7.1+cu128 (their checked-in environment.yml files are stale
# and were never the real setup), while slamio-main/slamio-final pinned
# torch 2.3.1+cu121 via conda. See envs/README.md for the full story and for
# why CUDA extensions are still compiled with a wrapped CUDA 12.8 nvcc even
# though the Python-side torch package is cu121: cu12.1's own nvcc cannot
# parse this host's GCC 14 system headers at all (confirmed experimentally --
# not just the sinpi/cospi mismatch previous_works patched for cu12.8, but
# unfixable `_Float32`/`type_traits` intrinsic mismatches). Mixing a newer
# nvcc to build the extension against an older-cu-runtime torch is safe and
# standard; torch.version.cuda still reports 12.1 at runtime.
#
# Usage:
#   bash scripts/setup_env.sh                       # build all 12
#   bash scripts/setup_env.sh rtgs-splatam slamio-opt-monogs   # just these
#   bash scripts/setup_env.sh list                   # print target names
#
# Idempotent: each venv/extension is only (re)built if missing, so re-running
# after a partial failure is safe and cheap.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
PY310="${PY310:-/home/karya/.local/bin/python3.10}"
SYSTEM_CUDA_ROOT="${SYSTEM_CUDA_ROOT:-/usr/lib/cuda-12.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"   # A40 (Ampere, sm_86)
# CUDA's per-arch "targets/<arch>-linux" directory name -- x86_64-linux on
# this cluster, but aarch64-linux (Jetson) or sbsa-linux (ARM server CUDA
# toolkits) elsewhere. Override on non-x86_64 hosts.
CUDA_TARGET_ARCH="${CUDA_TARGET_ARCH:-x86_64-linux}"
# ENVS_DIR is override-able so a second GPU generation's envs (e.g. H100,
# sm_90) can be built as a fully separate tree -- never share a venv/compiled
# extension between two GPU architectures. See scripts/setup_env_h100.sbatch.
ENVS_DIR="${ENVS_DIR:-$ROOT/envs}"
WRAPPED_CUDA="$ENVS_DIR/.cuda-wrapped"

ALL_TARGETS=(
  rtgs-splatam rtgs-monogs rtgs-gaussianslam
  splatonic-splatam splatonic-monogs splatonic-gaussianslam
  slamio-baseline-splatam slamio-baseline-monogs slamio-baseline-gaussianslam
  slamio-opt-splatam slamio-opt-monogs slamio-opt-gaussianslam
)

if [[ "${1:-}" == "list" ]]; then
  printf '%s\n' "${ALL_TARGETS[@]}"
  exit 0
fi

TARGETS=("${@:-${ALL_TARGETS[@]}}")

green()  { echo -e "\033[1;32m$*\033[0m"; }
red()    { echo -e "\033[1;31m$*\033[0m"; }

wrap_cuda() {
  # Shared across all 12 targets; guarded with flock since setup_env.sbatch
  # runs multiple targets concurrently as SLURM array tasks and two of them
  # racing to build this directory at once would corrupt it.
  mkdir -p "$ENVS_DIR"
  (
    flock -x 200
    if [ -x "$WRAPPED_CUDA/bin/nvcc" ]; then return; fi
    _wrap_cuda_build
  ) 200>"$ENVS_DIR/.cuda-wrapped.lock"
}

_wrap_cuda_build() {
  mkdir -p "$WRAPPED_CUDA/bin"
  for f in "$SYSTEM_CUDA_ROOT"/bin/*; do
    local fname; fname=$(basename "$f")
    if [ "$fname" = "nvcc" ]; then
      cp "$SYSTEM_CUDA_ROOT/bin/nvcc" "$WRAPPED_CUDA/bin/nvcc.real"
      cat > "$WRAPPED_CUDA/bin/nvcc" << 'INNER'
#!/bin/bash
exec "$(dirname "$0")/nvcc.real" -Xcompiler=-fpermissive -include cstdint -include cfloat "$@"
INNER
      chmod +x "$WRAPPED_CUDA/bin/nvcc"
    else
      ln -sf "$f" "$WRAPPED_CUDA/bin/$fname"
    fi
  done
  mkdir -p "$WRAPPED_CUDA/targets/$CUDA_TARGET_ARCH"
  cp -r "$SYSTEM_CUDA_ROOT/targets/$CUDA_TARGET_ARCH/include" "$WRAPPED_CUDA/targets/$CUDA_TARGET_ARCH/"
  ln -sf "$SYSTEM_CUDA_ROOT/targets/$CUDA_TARGET_ARCH/lib" "$WRAPPED_CUDA/targets/$CUDA_TARGET_ARCH/lib"
  ln -sf "$WRAPPED_CUDA/targets/$CUDA_TARGET_ARCH/include" "$WRAPPED_CUDA/include"
  for d in "$SYSTEM_CUDA_ROOT"/*; do
    local base; base=$(basename "$d")
    [ "$base" = "bin" ] || [ "$base" = "include" ] || [ "$base" = "targets" ] || ln -sf "$d" "$WRAPPED_CUDA/$base"
  done
  local math_h="$WRAPPED_CUDA/targets/$CUDA_TARGET_ARCH/include/crt/math_functions.h"
  sed -i \
    -e 's/sinpi(double x);/sinpi(double x) noexcept (true);/g' \
    -e 's/sinpif(float x);/sinpif(float x) noexcept (true);/g' \
    -e 's/cospi(double x);/cospi(double x) noexcept (true);/g' \
    -e 's/cospif(float x);/cospif(float x) noexcept (true);/g' \
    "$math_h"
  echo "=== CUDA toolchain wrapped at $(date) ==="
}

install_common() {
  local py="$1"
  "$py" -m pip install --upgrade pip -q
  if ! "$py" -c "import torch" 2>/dev/null; then
    # TORCH_PIP_SPEC/TORCH_INDEX_URL are override-able: the pinned
    # 2.3.1+cu121 wheels only exist for x86_64 on download.pytorch.org --
    # aarch64 platforms (e.g. Jetson Thor) need NVIDIA's own Jetson wheel
    # index and typically a different torch version tied to the JetPack
    # release; a workstation on a newer CUDA-only index may also need a
    # different pin. See envs/README.md's "other hardware" section.
    local spec="${TORCH_PIP_SPEC:-torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1}"
    local index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
    "$py" -m pip install $spec --index-url "$index_url" -q
  fi
  # Pinned everywhere for consistency: open3d/faiss/matplotlib etc. all pull
  # numpy>=2 transitively, which breaks np.unicode_/np.float_-style calls in
  # these codebases. Re-pin after every batch of deps, not just once.
  "$py" -m pip install -q "numpy==1.26.4"
}

setup_target() {
  local name="$1"
  local venv="$ENVS_DIR/$name"
  echo "############################################################"
  echo "=== $name: setup started on $(hostname) at $(date) ==="

  [ -x "$venv/bin/python" ] || "$PY310" -m venv "$venv"
  local py="$venv/bin/python"
  install_common "$py"

  wrap_cuda
  export CUDA_HOME="$WRAPPED_CUDA"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$SYSTEM_CUDA_ROOT/targets/$CUDA_TARGET_ARCH/lib:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST

  local slamio_root=""
  case "$name" in
    slamio-baseline-*) slamio_root="$ROOT/slamio-main" ;;
    slamio-opt-*)      slamio_root="$ROOT/slamio-final" ;;
  esac

  case "$name" in
    rtgs-splatam)
      "$py" -m pip install -q "tqdm==4.65.0" Pillow opencv-python imageio matplotlib \
        kornia natsort pyyaml lpips torchmetrics cyclonedds pytorch-msssim "plyfile==0.8.1" ninja wandb plotly
      "$py" -m pip install -q "numpy==1.26.4"
      cd "$ROOT/previous_works/rtgs-splatam"
      "$py" -c "import diff_gaussian_rasterization" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q \
          "git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110"
      "$py" -c "import diff_gaussian_rasterization; print('$name: CUDA extensions OK')"
      ;;

    rtgs-monogs|splatonic-monogs)
      "$py" -m pip install -q opencv-python munch trimesh "evo==1.11.0" open3d \
        torchmetrics imgviz PyOpenGL glfw PyGLM wandb lpips rich tqdm pyyaml plyfile ninja pynvml
      "$py" -m pip install -q "numpy==1.26.4"
      cd "$ROOT/previous_works/$name"
      "$py" -c "import simple_knn" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q ./submodules/simple-knn
      "$py" -c "import diff_gaussian_rasterization" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q ./submodules/diff-gaussian-rasterization
      if [[ "$name" == "splatonic-monogs" ]]; then
        "$py" -c "import track_rasterization" 2>/dev/null || \
          "$py" -m pip install --no-build-isolation -q ./submodules/track-rasterization
        "$py" -c "import simple_knn, diff_gaussian_rasterization, track_rasterization; print('$name: CUDA extensions OK')"
      else
        "$py" -c "import simple_knn, diff_gaussian_rasterization; print('$name: CUDA extensions OK')"
      fi
      ;;

    rtgs-gaussianslam|splatonic-gaussianslam)
      "$py" -m pip install -q "open3d==0.18.0" wandb trimesh pytorch_msssim torchmetrics \
        tqdm imageio opencv-python plyfile faiss-gpu-cu12
      "$py" -m pip install -q "numpy==1.26.4"
      cd "$ROOT/previous_works/$name"
      "$py" -c "import simple_knn" 2>/dev/null || "$py" -m pip install --no-build-isolation -q \
        "git+https://github.com/VladimirYugay/simple-knn.git@c7e51a06a4cd84c25e769fee29ab391fe5d5ff8d"
      "$py" -c "import gaussian_rasterizer" 2>/dev/null || "$py" -m pip install --no-build-isolation -q \
        "git+https://github.com/VladimirYugay/gaussian_rasterizer.git@9c40173fcc8d9b16778a1a8040295bc2f9ebf129"
      if [[ "$name" == "splatonic-gaussianslam" ]]; then
        "$py" -c "import map_rasterizer" 2>/dev/null || "$py" -m pip install --no-build-isolation -q ./map-rasterization
        "$py" -c "import track_rasterizer" 2>/dev/null || "$py" -m pip install --no-build-isolation -q ./track-rasterization
      fi
      "$py" -c "import evaluate_3d_reconstruction" 2>/dev/null || "$py" -m pip install -q \
        "git+https://github.com/eriksandstroem/evaluate_3d_reconstruction_lib.git@9b3cc08be5440db9c375cc21e3bd65bb4a337db7"
      if [[ "$name" == "splatonic-gaussianslam" ]]; then
        "$py" -c "import simple_knn, gaussian_rasterizer, map_rasterizer, track_rasterizer, faiss; print('$name: CUDA extensions + faiss OK')"
      else
        "$py" -c "import simple_knn, gaussian_rasterizer, faiss; print('$name: CUDA extensions + faiss OK')"
      fi
      ;;

    splatonic-splatam)
      "$py" -m pip install -q tqdm Pillow opencv-python imageio matplotlib kornia \
        natsort pyyaml lpips open3d torchmetrics pytorch-msssim plyfile ninja wandb plotly
      "$py" -m pip install -q "numpy==1.26.4"
      cd "$ROOT/previous_works/splatonic-splatam"
      "$py" -c "import diff_gaussian_rasterization" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q ./diff-gaussian-rasterization-w-depth
      "$py" -c "import track_rasterization" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q ./track-rasterization
      "$py" -c "import map_rasterization" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q ./map-rasterization
      "$py" -c "import diff_gaussian_rasterization, track_rasterization, map_rasterization; print('$name: CUDA extensions OK')"
      ;;

    slamio-baseline-splatam|slamio-opt-splatam)
      "$py" -m pip install -q setuptools "opencv-python-headless==4.8.1.78" open3d kornia natsort \
        pyyaml wandb lpips torchmetrics pytorch-msssim faiss-gpu-cu12 tqdm Pillow imageio matplotlib
      "$py" -m pip install -q "numpy==1.26.4"
      cd "$slamio_root/SplaTAM"
      "$py" -c "import diff_gaussian_rasterization" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q "$slamio_root/submodules/diff-gaussian-rasterization"
      "$py" -c "import diff_gaussian_rasterization; print('$name: CUDA extensions OK')"
      ;;

    slamio-baseline-monogs|slamio-opt-monogs)
      "$py" -m pip install -q setuptools "opencv-python-headless==4.8.1.78" open3d plyfile scipy \
        wandb torchmetrics imgviz PyOpenGL glfw PyGLM rich fvcore munch trimesh "evo==1.11.0" lpips ruff \
        tqdm matplotlib
      "$py" -m pip install -q "numpy==1.26.4"
      cd "$slamio_root/MonoGS"
      "$py" -c "import diff_gaussian_rasterization" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q "$slamio_root/submodules/diff-gaussian-rasterization"
      "$py" -c "import simple_knn" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q "$slamio_root/MonoGS/submodules/simple-knn"
      "$py" -c "import diff_gaussian_rasterization, simple_knn; print('$name: CUDA extensions OK')"
      ;;

    slamio-baseline-gaussianslam|slamio-opt-gaussianslam)
      "$py" -m pip install -q setuptools "opencv-python-headless==4.8.1.78" open3d wandb trimesh \
        pytorch_msssim torchmetrics tqdm imageio plyfile faiss-gpu-cu12 scipy matplotlib pyyaml
      "$py" -m pip install -q "numpy==1.26.4"
      cd "$slamio_root/Gaussian-SLAM"
      "$py" -c "import diff_gaussian_rasterization" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q "$slamio_root/submodules/diff-gaussian-rasterization"
      "$py" -c "import simple_knn" 2>/dev/null || \
        "$py" -m pip install --no-build-isolation -q "$slamio_root/MonoGS/submodules/simple-knn"
      "$py" -c "import evaluate_3d_reconstruction" 2>/dev/null || "$py" -m pip install -q \
        "git+https://github.com/eriksandstroem/evaluate_3d_reconstruction_lib.git"
      "$py" -c "import diff_gaussian_rasterization, simple_knn; print('$name: CUDA extensions OK')"
      ;;

    *)
      red "Unknown target: $name"
      return 1
      ;;
  esac
  echo "=== $name: setup done at $(date) ==="
}

FAILED=()
for t in "${TARGETS[@]}"; do
  setup_target "$t" || FAILED+=("$t")
done

echo ""
green "=== Summary ==="
for t in "${TARGETS[@]}"; do
  if printf '%s\n' "${FAILED[@]:-}" | grep -qx "$t"; then
    red "  FAIL  $t"
  else
    green "  OK    $t"
  fi
done
[[ ${#FAILED[@]} -eq 0 ]]
