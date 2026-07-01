#!/usr/bin/env bash
# SLURM array launcher: one A100 task per generated per-condition config.
#
# Usage:
#   python -m kd_lab.experiments.sweep --base configs/pointer_chase_base.yaml --out configs/generated
#   # set --array below to 0-(N-1) where N is the printed config count, then:
#   sbatch kd_lab/experiments/slurm_sweep.sh configs/generated
#
# Fill in the TODO placeholders for your cluster. Keep a single run under the 4-GPU-hour
# checkpoint; if a run trends longer, stop and reassess (BUILDPLAN section 0).

#SBATCH --job-name=kd-opd
#SBATCH --partition=TODO_PARTITION      # e.g. gpu, a100
#SBATCH --account=TODO_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-35%4                   # 0-(N-1); %4 caps concurrent tasks. Update N per sweep.
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

CONFIG_DIR="${1:-configs/generated}"
mapfile -t CONFIGS < <(ls "${CONFIG_DIR}"/*.yaml | sort)
CFG="${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "array task ${SLURM_ARRAY_TASK_ID:-0}: ${CFG}"

# Put outputs on shared home, never node-local /tmp (a later job must read results/).
python -m kd_lab.experiments.run --config "${CFG}"
