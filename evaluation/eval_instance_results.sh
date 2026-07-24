#!/bin/bash

# Workstation wrapper for evaluation/eval_instance_results.py: activates the
# repo-root venv, runs the per-instance eval, and logs stdout+stderr to a
# timestamped file under logs/. Extra args are forwarded; a later flag of the
# same name overrides the default set below.
#
#   bash evaluation/eval_instance_results.sh                          # best, all
#   bash evaluation/eval_instance_results.sh --sizes n20 n50
#   bash evaluation/eval_instance_results.sh --modes best last        # also last
#   bash evaluation/eval_instance_results.sh --device mps
#
# Runs the SAME inference as eval_checkpoints.sh (identical knobs below, so the
# numbers match), but records results per test instance to
# evaluation/instance_results/{size}/{variant}_{mode}.csv instead of only the
# aggregate row. Defaults to --modes best (leave out "last" for now); pass
# --modes best last to record both. --sizes/--variants filter as usual.
#
# The knobs below are copied verbatim from evaluation/eval_checkpoints.sh and
# must stay in sync with it -- they are chosen to match each vendored repo's own
# eval/test script except where the upstream default OOMs on our GPU or has zero
# effect on results. See notes/eval_decode_settings.md for the full writeup.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || { echo "Error: could not cd to $repo_root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $(pwd)/.venv"
    exit 1
fi

mkdir -p logs
log_file="logs/$(date +%Y%m%d_%H%M%S)_instance.log"
echo "Logging to $log_file"
python -u evaluation/eval_instance_results.py \
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
