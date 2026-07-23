#!/bin/bash

# Wrapper for evaluation/eval_checkpoints.py (same convention as
# check_val_best.sh / start_job.sh): load the Python module, source the repo-root
# venv, then evaluate the gathered PIP checkpoints on the AMAI TSPTW test sets.
#
# The orchestrator spawns the per-family workers (eval_pomo.py / eval_am.py) with
# the venv's own interpreter, so run it from the repo root. Either run on a login
# node with a GPU, or submit as a GPU job (the #SBATCH defaults below apply then):
#
#   bash evaluation/eval_checkpoints.sh --device cuda                 # full grid
#   bash evaluation/eval_checkpoints.sh --modes best --sizes n20 n50
#   sbatch evaluation/eval_checkpoints.sh --device cuda               # as a job
#
# NOTE for sbatch: slurm does not create the log dir, so out/eval_checkpoints/
# must exist at submission (the mkdir below creates it for future runs).
#
#SBATCH -J eval_checkpoints
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8GB
#SBATCH -A hpc-prf-tqff
#SBATCH --output=out/eval_checkpoints/%j.out
#SBATCH --error=out/eval_checkpoints/%j.err

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run from the repo root (the dir this script's parent lives in), regardless of
# whether launched via bash or sbatch.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    repo_root="$SLURM_SUBMIT_DIR"
else
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$repo_root" || { echo "Error: could not cd to $repo_root"; exit 1; }

module load lang/Python/3.12.3-GCCcore-13.2.0

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $(pwd)/.venv"
    exit 1
fi

# Log to a timestamped file too, so a dropped ssh connection loses nothing.
# -u: unbuffered, so progress lines appear immediately despite the tee pipe.
mkdir -p out/eval_checkpoints
log_file="out/eval_checkpoints/$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $log_file"
python -u evaluation/eval_checkpoints.py "$@" 2>&1 | tee "$log_file"
exit "${PIPESTATUS[0]}"
