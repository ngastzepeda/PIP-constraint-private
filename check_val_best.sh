#!/bin/bash

# Wrapper for check_val_best.py (same convention as submit_jobs.sh):
# load the Python module, source the venv, then run the checker.
# Needs network access to wandb -> run on the login node, not in a job.
#
#   bash check_val_best.sh                              # full grid
#   bash check_val_best.sh --sizes 100 --variants pomo pip pipd

module load lang/Python/3.12.3-GCCcore-13.2.0

# Source the virtual environment (needed for wandb, and torch via the
# helpers imported from gather_checkpoints.py)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $(pwd)/.venv"
    exit 1
fi

# -u: unbuffered output, so progress lines appear immediately even when
# piped (e.g. ... | tee check.log) or over a laggy ssh connection
python -u "check_val_best.py" "$@"
