#!/bin/bash

#SBATCH --output=out/experiments/%j.out
#SBATCH --error=out/experiments/%j.err
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "nayeli"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nayeli.gast.zepeda@univie.ac.at
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A hpc-prf-tqff

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
# tensorboard_logger ships pre-generated protobuf code that is incompatible with
# protobuf>=4 unless the pure-python implementation is used
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Use the directory from which the job was submitted (repo root)
current_folder="$SLURM_SUBMIT_DIR"
cd "$current_folder" || { echo "Error: Could not change to $current_folder"; exit 1; }

module load lang/Python/3.10.4-GCCcore-11.3.0

# Source the virtual environment
if [ -f "$current_folder/.venv/bin/activate" ]; then
    source "$current_folder/.venv/bin/activate"
else
    echo "Error: Virtual environment not found in $current_folder/.venv"
    exit 1
fi

# First two arguments select the implementation directory (POMO+PIP or AM+PIP) and the
# entry script (train.py or run.py); the rest are forwarded. Scripts must run from
# inside their directory (data paths are relative to it).
program_dir="$1"; shift
entry_script="$1"; shift
cd "$current_folder/$program_dir" || { echo "Error: Could not change to $program_dir"; exit 1; }

# Start the cluster job
python "$entry_script" "$@"
