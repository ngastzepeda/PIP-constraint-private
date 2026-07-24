#!/bin/bash

# Workstation wrapper for evaluation/benchmarks/eval_benchmarks.py: activates the
# repo-root venv, evaluates the gathered checkpoints on the TSPTW benchmark
# libraries (out-of-distribution), and logs stdout+stderr to a timestamped file
# under logs/. Extra args are forwarded; a later flag of the same name overrides
# the default set below.
#
#   bash evaluation/benchmarks/eval_benchmarks.sh                     # n100, best, all variants
#   bash evaluation/benchmarks/eval_benchmarks.sh --families am
#   bash evaluation/benchmarks/eval_benchmarks.sh --sizes 100 --datasets Dumas
#   bash evaluation/benchmarks/eval_benchmarks.sh --modes best last
#
# The decode/aug knobs below are copied verbatim from
# evaluation/eval_instance_results.sh so benchmark numbers use the SAME inference
# as the in-distribution table -- see that file and notes/eval_decode_settings.md
# for why each value is what it is. Keep them in sync.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root" || { echo "Error: could not cd to $repo_root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $(pwd)/.venv"
    exit 1
fi

mkdir -p logs
log_file="logs/$(date +%Y%m%d_%H%M%S)_benchmarks.log"
echo "Logging to $log_file"
python -u evaluation/benchmarks/eval_benchmarks.py \
  --device cuda \
  --pomo_batch_size 128 \
  --aug_factor 8 \
  --am_width 1280 \
  --am_max_calc_batch_size 1280 \
  --am_batch_size 8 \
  "$@" 2>&1 | tee "$log_file"
exit "${PIPESTATUS[0]}"
