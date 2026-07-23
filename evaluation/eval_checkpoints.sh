#!/bin/bash

# Workstation wrapper for evaluation/eval_checkpoints.py: activates the
# repo-root venv, runs the eval, and logs stdout+stderr to a timestamped file
# under logs/. Extra args are forwarded (e.g. --modes best --sizes n20 n50);
# a later --device overrides the --device cuda default below.
#
#   bash evaluation/eval_checkpoints.sh
#   bash evaluation/eval_checkpoints.sh --modes best --sizes n20 n50
#   bash evaluation/eval_checkpoints.sh --device mps

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || { echo "Error: could not cd to $repo_root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $(pwd)/.venv"
    exit 1
fi

mkdir -p logs
log_file="logs/$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $log_file"
python -u evaluation/eval_checkpoints.py --device cuda "$@" 2>&1 | tee "$log_file"
exit "${PIPESTATUS[0]}"
