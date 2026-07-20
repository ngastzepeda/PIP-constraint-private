#!/bin/bash

#SBATCH --output=out/gather_checkpoints/%j.out
#SBATCH --error=out/gather_checkpoints/%j.err
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -J "gather_ckpts"
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -p normal
#SBATCH -A hpc-prf-tqff

# Wrapper for gather_checkpoints.py (same convention as start_job.sh):
# load the Python module, source the venv, then run the gather script.
#
#   bash gather_checkpoints.sh                # copy into checkpoints/ (no git)
#   bash gather_checkpoints.sh --git-add      # ...and git add the folder
#   bash gather_checkpoints.sh --include_all  # also gather unfinished runs
#
# By default only runs that reached their target epoch are gathered.
# Submit with sbatch (not bash) to run on the compute nodes, e.g. because the
# login node's filesystem reads are too slow/unstable:
#
#   sbatch gather_checkpoints.sh                     # gather all FINISHED runs into checkpoints/
#   sbatch gather_checkpoints.sh --match skillvrp    # only skillvrp runs
#   sbatch gather_checkpoints.sh --include-all       # also gather ongoing/interrupted runs

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# progress lines must reach the terminal/log immediately, even when piped
export PYTHONUNBUFFERED=1

# Use the directory from which the job was submitted
# SLURM_SUBMIT_DIR is only set when run via sbatch; fall back to pwd so
# `bash gather_checkpoints.sh` also works for quick local runs
current_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$current_folder" || { echo "Error: Could not change to $current_folder"; exit 1; }

module load lang/Python/3.12.3-GCCcore-13.2.0

# Source the virtual environment (needed for torch to read epoch/score from ckpts)
if [ -f "$current_folder/.venv/bin/activate" ]; then
    source "$current_folder/.venv/bin/activate"
else
    echo "Error: Virtual environment not found in $current_folder/.venv"
    exit 1
fi

python "$current_folder/gather_checkpoints.py" "$@"
