#!/bin/bash
#SBATCH --job-name=slamio-envs-h100
#SBATCH --partition=h100nvl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-11%6
#SBATCH --output=envs-h100/logs/%x_%A_%a.log
#
# H100 twin of setup_env.sbatch. Builds the same 12 targets into a fully
# separate envs-h100/ tree (never shares a venv or compiled CUDA extension
# with the A40 envs/ tree), with the same python/torch/cuda version pins --
# only TORCH_CUDA_ARCH_LIST changes (9.0, Hopper, vs. A40's 8.6, Ampere).
# h100nvl chosen over h100sxm: 3 healthy nodes (24 GPUs) vs. h100sxm's 1
# healthy node (8 GPUs, 2 of its 3 nodes drained) -- see envs-h100/README.md.
#
# Usage:
#   mkdir -p envs-h100/logs
#   sbatch scripts/setup_env_h100.sbatch                 # all 12, 6 at a time
#   sbatch --array=0-2 scripts/setup_env_h100.sbatch      # just the first 3
#
# Progress: squeue -u $USER ; tail -f envs-h100/logs/slamio-envs-h100_<jobid>_<idx>.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

export ENVS_DIR="$PWD/envs-h100"
export TORCH_CUDA_ARCH_LIST="9.0"

mapfile -t TARGETS < <(bash scripts/setup_env.sh list)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"

echo "=== array task $SLURM_ARRAY_TASK_ID -> $TARGET on $(hostname), ENVS_DIR=$ENVS_DIR, ARCH=$TORCH_CUDA_ARCH_LIST ==="
bash scripts/setup_env.sh "$TARGET"
