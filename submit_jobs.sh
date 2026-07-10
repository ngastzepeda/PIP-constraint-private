#!/bin/bash

# Submit the PIP-constraint baseline trainings (wraps submit_jobs.py).
#
# Default: reconcile the full 24-job grid ({pomo, pip, pipd, am, am_pip, am_pipd} x
# {n20, n50, n100_sw, n100_mw}) against existing checkpoints and squeue -- resumes
# crashed runs, starts missing ones, skips complete/queued ones. Safe to re-run any time.
#
# Example calls (same filter convention as the AMAI/skillvrp submit scripts):
#   bash submit_jobs.sh                                  # everything (reconcile)
#   bash submit_jobs.sh --dry_run                        # show what would happen
#   bash submit_jobs.sh --sizes 50 --variants pipd       # a single job
#   bash submit_jobs.sh --sizes 100 --tws sw             # n100 small windows only
#   bash submit_jobs.sh --resume                         # only resume crashed runs
#   bash submit_jobs.sh --fresh --sizes 20               # force new n20 runs

mkdir -p out/experiments

module load lang/Python/3.12.3-GCCcore-13.2.0
python "submit_jobs.py" "$@"
