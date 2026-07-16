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

# Everything is also logged to a timestamped file, so a dropped ssh
# connection loses nothing. -u: unbuffered output, so progress lines appear
# immediately (both on the terminal and in the log) despite the tee pipe.
mkdir -p out/check_val_best
log_file="out/check_val_best/$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $log_file"
python -u "check_val_best.py" "$@" 2>&1 | tee "$log_file"
# exit with the checker's status (1 = something AT RISK/UNKNOWN), not tee's
exit "${PIPESTATUS[0]}"
