#!/bin/bash
# Single-condition launcher: sbatch kd_lab/experiments/run_one.sbatch <config.yaml>
# Any GPU (v100 or a100); --time caps a run under the 4-GPU-hour checkpoint.
# 32GB+ GPU: the full-vocab divergences materialize [B,T,151936] tensors, which OOM a 16GB card.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=kd-run
#SBATCH --output=logs/run_%j.log
#SBATCH --error=logs/run_%j.log
set -eo pipefail
module purge
module load cuda/12.3.0
module load cuDNN/9.10.2
module load anaconda3/2024.06
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROJ_DIR="${PROJ_DIR:-/home/patodia.pa/kd-lab}"
cd "$PROJ_DIR"
mkdir -p logs
source .venv_hpc/bin/activate
export HF_HOME="${HF_HOME:-/home/patodia.pa/.cache/huggingface}"
CFG="${1:?usage: sbatch run_one.sbatch <config.yaml>}"
echo "running config: $CFG on $(hostname)"
python -m kd_lab.experiments.run --config "$CFG"
echo RUN_DONE
