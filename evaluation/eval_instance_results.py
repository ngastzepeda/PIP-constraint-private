"""Evaluate the gathered PIP checkpoints and record results *per instance*.

This is the per-instance sibling of eval_checkpoints.py. It runs the exact same
inference (same workers, same settings, so the numbers match), but instead of
only writing the aggregated feas_count/cost/gap_bks row per checkpoint, it
writes one csv per checkpoint with a row per test instance:

    evaluation/instance_results/{size}/{variant}_{mode}.csv

  size    : n20, n50, n100_sw, n100_mw
  variant : am, pomo, am_pip, pomo_pip, am_pipd, pomo_pipd
  columns : instance, feasibility (1/0), objective (real 0..100 units, blank if
            infeasible), runtime (per-instance inference seconds)

With these files the whole eval pipeline no longer has to be re-run whenever a
new baseline (BKS) arrives: the per-instance gap to BKS can be computed and
aggregated directly from the objective column (instances align with the BKS
csv's `instance` column). See common.write_instance_csv for the exact column
semantics (notably: runtime is amortized per instance, batched inference has no
true per-instance clock).

Like eval_checkpoints.py, each family runs in its own subprocess (POMO+PIP and
AM+PIP have colliding top-level module names). Run from the repo root's venv
(workstation only, no cluster wrapper):

    python evaluation/eval_instance_results.py                 # best, all sizes/variants
    python evaluation/eval_instance_results.py --sizes n20 n50
    python evaluation/eval_instance_results.py --families am --am_decode greedy
    python evaluation/eval_instance_results.py --modes best last   # also record last
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from submit_jobs import DATASETS, VARIANTS

WORKER = {"pomo": "eval_pomo.py", "am": "eval_am.py"}
EVAL_DIR = Path(__file__).resolve().parent


def sort_key(job):
    return (list(DATASETS).index(job["size"]), list(VARIANTS).index(job["variant"]))


def run_family(family, jobs, args, staging):
    """Launch the family worker as a subprocess with --instance_csv, so it
    writes instance_results/{size}/{variant}_{mode}.csv for each job directly.
    (--out is still required by the worker; we point it at a throwaway file and
    ignore the aggregated rows.)"""
    jobs_file = staging / f"jobs_{family}.json"
    out_file = staging / f"rows_{family}.json"
    with open(jobs_file, "w") as f:
        json.dump(jobs, f)

    cmd = [
        sys.executable, str(EVAL_DIR / WORKER[family]),
        "--jobs", str(jobs_file), "--out", str(out_file), "--instance_csv",
        "--device", args.device, "--bks_csv", args.bks_csv,
        "--test_episodes", str(args.test_episodes), "--seed", str(args.seed),
    ]
    if family == "pomo":
        cmd += ["--test_batch_size", str(args.pomo_batch_size),
                "--aug_factor", str(args.aug_factor)]
    else:
        cmd += ["--decode", args.am_decode, "--width", str(args.am_width),
                "--eval_batch_size", str(args.am_batch_size),
                "--softmax_temp", str(args.am_softmax_temp),
                "--max_calc_batch_size", str(args.am_max_calc_batch_size)]

    print(f"\n=== {family} family: {len(jobs)} checkpoint(s) ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {family} worker exited with {result.returncode}", flush=True)


def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--modes", nargs="+", default=["best"],
                   choices=["best", "last"],
                   help="checkpoint slots to record (default: best only)")
    p.add_argument("--families", nargs="+", default=["pomo", "am"],
                   choices=["pomo", "am"])
    p.add_argument("--sizes", nargs="+", default=None, choices=list(DATASETS),
                   help="default: all sizes present in checkpoints/")
    p.add_argument("--variants", nargs="+", default=None, choices=list(VARIANTS))
    p.add_argument("--bks_csv", default=str(common.DEFAULT_BKS_CSV),
                   help="AMAI eval_baselines_instance_results.csv (only used for "
                        "the console feas/gap summary; the csv itself does not "
                        "depend on it)")
    p.add_argument("--test_episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2024)
    # POMO knobs
    p.add_argument("--pomo_batch_size", type=int, default=500)
    p.add_argument("--aug_factor", type=int, default=8, choices=[1, 8])
    # AM knobs
    p.add_argument("--am_decode", default="sampling", choices=["sampling", "greedy"])
    p.add_argument("--am_width", type=int, default=1280)
    p.add_argument("--am_batch_size", type=int, default=16)
    p.add_argument("--am_softmax_temp", type=float, default=1.0)
    p.add_argument("--am_max_calc_batch_size", type=int, default=1280,
                   help="cap on parallel sampling rollouts per forward pass "
                        "(width*eval_batch_size, chunked above this); lower to "
                        "trade speed for less GPU memory on am_pip/am_pipd")
    return p


def main():
    args = get_parser().parse_args()
    args.device = common.pick_device(args.device)
    print(f"Using device: {args.device}", flush=True)

    for mode in args.modes:
        jobs = common.discover_jobs([mode], sizes=args.sizes, variants=args.variants)
        jobs = [j for j in jobs if j["family"] in args.families]
        jobs.sort(key=sort_key)
        print(f"\n##### mode={mode}: {len(jobs)} checkpoint slot(s) #####", flush=True)
        if not jobs:
            continue

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for family in args.families:
                fam_jobs = [j for j in jobs if j["family"] == family]
                if fam_jobs:
                    run_family(family, fam_jobs, args, staging)

        print(f"\nmode={mode}: per-instance csvs written under "
              f"{common.INSTANCE_RESULTS_DIR.relative_to(common.REPO_ROOT)}/"
              f"{{size}}/{{variant}}_{mode}.csv", flush=True)


if __name__ == "__main__":
    main()
