"""Evaluate the gathered PIP checkpoints (checkpoints/) on the AMAI TSPTW test
instances, producing the *same metrics* as
AMAI2025/source/evaluation/eval_checkpoints.py so the two repos are directly
comparable.

For each checkpoint slot ({size}_{variant}_{best,last}.pt) it reports, per
mode, feas_count / cost / gap_bks / inference_time in the exact AMAI txt+csv
format (results/eval_checkpoints_{best,last}.{txt,csv}). Instances are
data/feas_tsptw/{folder}/test_1k.npz (identical to the AMAI tsptw test sets);
the per-instance BKS for the gap comes from the AMAI baseline csv.

Because POMO+PIP and AM+PIP have colliding top-level module names, each family
is evaluated in its own subprocess (eval_pomo.py / eval_am.py). This script is
the orchestrator: it discovers the jobs, dispatches them per family, then
collects and writes the results.

Run directly from the repo root's venv (workstation only, no cluster wrapper):

    python evaluation/eval_checkpoints.py                      # auto device: cuda > mps > cpu
    python evaluation/eval_checkpoints.py --modes best --sizes n20 n50
    python evaluation/eval_checkpoints.py --families am --am_decode greedy
    python evaluation/eval_checkpoints.py --device cpu          # force CPU
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
    """Launch the family worker as a subprocess (same interpreter/venv) and
    return its result rows."""
    jobs_file = staging / f"jobs_{family}.json"
    out_file = staging / f"rows_{family}.json"
    with open(jobs_file, "w") as f:
        json.dump(jobs, f)

    cmd = [
        sys.executable, str(EVAL_DIR / WORKER[family]),
        "--jobs", str(jobs_file), "--out", str(out_file),
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
    if not out_file.exists():
        return []
    with open(out_file) as f:
        return json.load(f)


def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--modes", nargs="+", default=["best", "last"],
                   choices=["best", "last"])
    p.add_argument("--families", nargs="+", default=["pomo", "am"],
                   choices=["pomo", "am"])
    p.add_argument("--sizes", nargs="+", default=None, choices=list(DATASETS),
                   help="default: all sizes present in checkpoints/")
    p.add_argument("--variants", nargs="+", default=None, choices=list(VARIANTS))
    p.add_argument("--bks_csv", default=str(common.DEFAULT_BKS_CSV),
                   help="AMAI eval_baselines_instance_results.csv (for gap_bks)")
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

        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for family in args.families:
                fam_jobs = [j for j in jobs if j["family"] == family]
                if fam_jobs:
                    rows += run_family(family, fam_jobs, args, staging)

        rows.sort(key=lambda r: (list(DATASETS).index(r["size"]),
                                 list(VARIANTS).index(r["model"])))
        common.write_new_run_header(mode)
        for row in rows:
            common.append_txt_row(mode, row)
        common.write_csv(mode, rows)
        print(f"\nmode={mode}: wrote {len(rows)} rows to "
              f"{common.txt_path(mode).relative_to(common.REPO_ROOT)} and "
              f"{common.csv_path(mode).relative_to(common.REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
