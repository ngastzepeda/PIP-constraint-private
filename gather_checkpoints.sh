#!/bin/bash

# Wrapper for gather_checkpoints.py (same convention as submit_jobs.sh):
# load the Python module, source the venv, then run the gather script.
#
#   bash gather_checkpoints.sh                # copy into checkpoints/ (no git)
#   bash gather_checkpoints.sh --git-add      # ...and git add the folder

module load lang/Python/3.12.3-GCCcore-13.2.0

# Source the virtual environment (needed for torch to read the pip decoder
# metadata and for the wandb lookup that disambiguates resumed AM val bests)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $(pwd)/.venv"
    exit 1
fi

python "gather_checkpoints.py" "$@"
