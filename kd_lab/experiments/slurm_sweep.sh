#!/bin/bash
#================================================================
# On-policy distillation sweep: one A100 array task per generated config.
# Matches the Explorer (Northeastern) conventions used in vit-from-scratch.
#
# Usage:
#   python -m kd_lab.experiments.sweep --base configs/pointer_chase_base.yaml --out configs/generated
#   # set --array=0-(N-1) below to the printed config count, then:
#   sbatch kd_lab/experiments/slurm_sweep.sh configs/generated
#
# --time is 04:00:00 to keep a single run under the 4-GPU-hour checkpoint (BUILDPLAN section 0).
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --job-name=kd-opd
#SBATCH --output=logs/opd_%A_%a.out
#SBATCH --error=logs/opd_%A_%a.err
#SBATCH --array=0-35%4

set -eo pipefail

module purge
module load cuda/12.3.0
module load cuDNN/9.10.2
module load anaconda3/2024.06

# Explorer proxy for outbound (HF model downloads). Set AFTER module purge, which clears it.
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ_DIR="${PROJ_DIR:-/home/patodia.pa/kd-lab}"
cd "$PROJ_DIR"
mkdir -p logs
source .venv_hpc/bin/activate
# Shared HF cache so array tasks reuse downloaded weights instead of each re-downloading.
export HF_HOME="${HF_HOME:-/home/patodia.pa/.cache/huggingface}"

CONFIG_DIR="${1:-configs/generated}"
mapfile -t CONFIGS < <(ls "${CONFIG_DIR}"/*.yaml | sort)
CFG="${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "array task ${SLURM_ARRAY_TASK_ID:-0}: ${CFG}"

# Results go under the shared repo dir (results/), never node-local /tmp.
python -m kd_lab.experiments.run --config "${CFG}"
