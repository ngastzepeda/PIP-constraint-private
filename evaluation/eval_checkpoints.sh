#!/bin/bash

# Workstation wrapper for evaluation/eval_checkpoints.py: activates the
# repo-root venv, runs the eval, and logs stdout+stderr to a timestamped file
# under logs/. Extra args are forwarded (e.g. --modes best --sizes n20 n50);
# a later flag of the same name overrides the default set below.
#
#   bash evaluation/eval_checkpoints.sh
#   bash evaluation/eval_checkpoints.sh --modes best --sizes n20 n50
#   bash evaluation/eval_checkpoints.sh --device mps
#
# The knobs below are chosen to stay as close as possible to the values each
# vendored repo (AM+PIP, POMO+PIP) uses in its own eval/test script, deviating
# only where the upstream default causes OOM on our GPU, or where a value has
# zero effect on results and can safely match our other (RL4CO) experiments.
# See notes/eval_decode_settings.md for the full writeup and evidence.

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
python -u evaluation/eval_checkpoints.py \
  --device cuda \
  \
  `# POMO+PIP: test_batch_size has zero effect on feas/cost/gap (POMO+PIP uses` \
  `# InstanceNorm, verified batch-size-invariant) -- pure compute chunking, so` \
  `# free to match our other RL4CO experiments' batch size instead of either` \
  `# upstream's own default (POMO+PIP/test.py: 2500) or this repo's prior` \
  `# undocumented choice (500).` \
  --pomo_batch_size 128 \
  `# aug_factor=8 matches BOTH POMO+PIP/test.py's own default AND the exact` \
  `# training-time validation call that selected best.ckpt (Trainer.py:922,` \
  `# _val_one_batch(..., aug_factor=8)) -- do not change.` \
  --aug_factor 8 \
  \
  `# AM+PIP: width and max_calc_batch_size are upstream AM+PIP/eval.py's own` \
  `# defaults (both 1280), kept equal on purpose -- their ratio (not their` \
  `# absolute value) controls chunking in AM+PIP/utils/functions.py::` \
  `# sample_many, and chunking (max_calc_batch_size < width) hits a` \
  `# pre-existing vendored bug (per-chunk infeasibility mask isn't` \
  `# accumulated, so the 2nd+ chunk crashes with a shape mismatch). Keeping` \
  `# them equal makes sample_many run its internal loop exactly once, so the` \
  `# bug path is never reached -- do not lower max_calc_batch_size below` \
  `# width.` \
  --am_width 1280 \
  --am_max_calc_batch_size 1280 \
  `# eval_batch_size is the one AM+PIP knob that is NOT free to change:` \
  `# AM+PIP's encoder uses BatchNorm1d(track_running_stats=False), which` \
  `# normalizes on live batch statistics even at eval time, so batch` \
  `# composition measurably changes feas/cost (verified empirically, see` \
  `# notes/eval_decode_settings.md) -- unlike POMO+PIP, this is not a free` \
  `# knob. Upstream's own default (16) OOMs on am_pipd at n100 on our GPU` \
  `# (needed ~1.65GB more than was free); 8 halves peak memory` \
  `# (width * eval_batch_size parallel sequences) for a comfortable margin` \
  `# while staying as close as possible to upstream. Drop to 4 if n100` \
  `# am_pipd still OOMs. Whatever value you land on, hold it fixed across` \
  `# the whole table and report it -- it is a real methodology choice here,` \
  `# not just a speed knob.` \
  --am_batch_size 8 \
  "$@" 2>&1 | tee "$log_file"
exit "${PIPESTATUS[0]}"
