"""Submit the PIP-constraint baseline trainings on the AMAI TSPTW instances.

Grid: {POMO*, POMO*+PIP, POMO*+PIP-D, AM*, AM*+PIP, AM*+PIP-D} x {n20, n50, n100_sw,
n100_mw}  =>  24 jobs. POMO variants run POMO+PIP/train.py, AM variants AM+PIP/run.py,
both on the fixed AMAI datasets (data/feas_tsptw/...) with AMAI/LMask-consistent
semantics (fixed train set, service times as node feature, depot-return check).

Epoch/validation cadence, matched to the AMAI runs (max_epochs x 100k samples there,
validation every epoch = every 100k samples): both families use one epoch = one full
pass over the fixed 100k (POMO: --train_episodes 100000), so epochs = AMAI max_epochs
(100 for n20, 500 for n50/n100) with validation + checkpoint every epoch. The lr is
constant 1e-4 (no scheduling, as in the AMAI runs); POMO's epoch-denominated PIP-D
schedules are rescaled /10 to the same sample positions.
Both keep only the latest full checkpoint plus the best-by-validation model
(feasibility rate first, cost tie-break), mirroring the LMask last.ckpt/best.ckpt setup.

Submission modes (logic mirrors AMAI2025/submit_jobs_skillvrp.py): by default the
script RECONCILES the selected grid against what already ran -- jobs with an
epoch-*.pt in a run dir under their job-specific log root are resumed (PIP-D
additionally restores the auxiliary decoder), jobs without any checkpoint are
submitted fresh, and jobs whose checkpoint already reached the target epochs are
skipped. --resume restricts this to resuming crashed runs only; --fresh submits every
selected job as a brand-new run. In reconcile and resume mode, jobs whose name is
already pending/running in slurm (per squeue) are skipped, so re-running the script
never duplicates work. Submit via submit_jobs.sh.
"""

import re
import subprocess
from pathlib import Path

POMO_DIR = Path("POMO+PIP")
AM_DIR = Path("AM+PIP")

default_seed = 2023  # PIP's default

# problem_size counts the depot (AMAI's "n50" = depot + 50 customers => 51);
# pomo_size = number of customers (rollouts per instance, POMO only).
# epochs = AMAI max_epochs; one epoch = one pass over the fixed 100k in both families.
# No lr scheduling (constant 1e-4), consistent with the AMAI experiments: AM's default
# is already constant; POMO's default milestone (epoch 9001) never fires within <=500 epochs.
# n20/n50 have a single dataset (window type None); n100 has small/medium windows (sw/mw).
DATASETS = {
    "n20": dict(folder="n20_amai", size=20, tws=None, problem_size=21, pomo_size=20, epochs=100),
    "n50": dict(folder="n50_amai", size=50, tws=None, problem_size=51, pomo_size=50, epochs=500),
    "n100_sw": dict(folder="n100_amai_sw", size=100, tws="sw", problem_size=101, pomo_size=100,
                    epochs=500, big=True),
    "n100_mw": dict(folder="n100_amai_mw", size=100, tws="mw", problem_size=101, pomo_size=100,
                    epochs=500, big=True),
}
sizes = [20, 50, 100]

VARIANTS = {
    # POMO+PIP/train.py (POMO* has the Lagrangian timeout reward on by default)
    "pomo": dict(family="pomo", extra=[]),
    "pip": dict(family="pomo", extra=["--generate_PI_mask"]),
    "pipd": dict(family="pomo", extra=["--generate_PI_mask", "--pip_decoder"]),
    # AM+PIP/run.py
    "am": dict(family="am", extra=[]),
    "am_pip": dict(family="am", extra=["--generate_PI_mask"]),
    "am_pipd": dict(family="am", extra=["--generate_PI_mask", "--pip_decoder"]),
}


def job_tag(ds_key, variant):
    return f"{ds_key}_{variant}"


def get_job_name(ds_key, variant):
    return f"pip_{job_tag(ds_key, variant)}"


def target_epochs(ds_key, variant):
    return DATASETS[ds_key]["epochs"]


def queued_job_names():
    """Names of this user's pending/running slurm jobs; None if squeue is unavailable."""
    try:
        out = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%j"],
            capture_output=True, text=True, check=True,
        ).stdout
        return {line.strip() for line in out.splitlines() if line.strip()}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def build_params(ds_key, variant, seed):
    ds = DATASETS[ds_key]
    data_dir = f"../data/feas_tsptw/{ds['folder']}"
    tag = job_tag(ds_key, variant)
    if VARIANTS[variant]["family"] == "pomo":
        params = [
            "--problem", "TSPTW",
            "--problem_size", str(ds["problem_size"]),
            "--pomo_size", str(ds["pomo_size"]),
            "--train_set_path", f"{data_dir}/train_100k.npz",
            "--val_dataset", f"{data_dir}/val_1k.npz",
            "--val_episodes", "1000",
            "--check_depot_return",
            "--include_service_time",
            # one epoch = one full pass over the fixed 100k (same epoch counter as AMAI/AM)
            "--train_episodes", "100000",
            "--epochs", str(ds["epochs"]),
            # validate + checkpoint every epoch = every 100k samples = AMAI's cadence
            "--validation_interval", "1",
            "--model_save_interval", "1",
            "--seed", str(seed),
            # one log root per job so runs never collide and resume detection is unambiguous
            "--log_dir", f"./results/{tag}",
        ] + VARIANTS[variant]["extra"]
        if variant == "pipd":
            # PIP's PIP-D schedules are epoch-denominated defaults for 10k-sample epochs
            # (200/1000/50/50; N=100: 100/./20/.); rescaled /10 for 100k-sample epochs
            if ds.get("big"):
                params += ["--simulation_stop_epoch", "10", "--pip_update_interval", "100",
                           "--pip_update_epoch", "2", "--pip_last_growup", "5"]
            else:
                params += ["--simulation_stop_epoch", "20", "--pip_update_interval", "100",
                           "--pip_update_epoch", "5", "--pip_last_growup", "5"]
    else:
        params = [
            "--problem", "tsptw",
            "--graph_size", str(ds["problem_size"]),
            "--train_set_path", f"{data_dir}/train_100k.npz",
            "--val_dataset", f"{data_dir}/val_1k.npz",
            "--val_size", "1000",
            "--include_service_time",
            "--n_epochs", str(ds["epochs"]),
            "--seed", str(seed),
            "--output_dir", f"outputs/{tag}",
            "--log_dir", f"logs/{tag}",
            "--no_progress_bar",
        ] + VARIANTS[variant]["extra"]
        if variant == "am_pipd" and ds["epochs"] > 100:
            # AM+PIP-D defaults (10/10/2/5) are tuned for ~100-epoch runs; scale x5 for 500 epochs
            params += ["--simulation_stop_epoch", "50", "--pip_update_interval", "50",
                       "--pip_update_epoch", "10", "--pip_last_growup", "25"]
    return params


def find_resume_info(ds_key, variant):
    """Locate the checkpoint with the highest epoch for a previously started job.

    POMO run dirs: POMO+PIP/results/<tag>/{timestamp}_{run_name}/epoch-{N}.pt
    AM run dirs:   AM+PIP/outputs/<tag>/tsptw_{size}/{run_name}_{timestamp}/epoch-{N}.pt
    Only the latest epoch-N.pt is kept per run dir; we take the max epoch across run
    dirs (a crashed resume leaves several dirs). Paths in the result are relative to
    the implementation directory (the entry script's working directory).
    Returns None if no checkpoint exists.
    """
    family = VARIANTS[variant]["family"]
    base = POMO_DIR if family == "pomo" else AM_DIR
    run_root = base / ("results" if family == "pomo" else "outputs") / job_tag(ds_key, variant)
    if not run_root.is_dir():
        return None
    best = None
    for ckpt in run_root.rglob("epoch-*.pt"):
        m = re.fullmatch(r"epoch-(\d+)\.pt", ckpt.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), ckpt)
    if best is None:
        return None
    epoch, ckpt = best
    ts_dir = ckpt.parent
    info = dict(
        epoch=epoch,
        run_dir=f"./{ts_dir.relative_to(base)}",
        ckpt=f"./{ckpt.relative_to(base)}",
        pip_ckpt=None,
    )
    # PIP-D: the auxiliary decoder is restored separately (default load_which_pip=train_fsb_bsf)
    if variant.endswith("pipd") and (ts_dir / "fsb_accuracy_bsf.pt").is_file():
        info["pip_ckpt"] = f"./{(ts_dir / 'fsb_accuracy_bsf.pt').relative_to(base)}"
    return info


def resume_params(variant, info):
    if VARIANTS[variant]["family"] == "pomo":
        params = ["--resume_path", info["run_dir"], "--checkpoint", info["ckpt"]]
    else:
        # AM: --resume restores model/optimizer/rng and continues at epoch+1 (n_epochs is
        # the absolute target, see run.py); logs go to a fresh timestamped save_dir
        params = ["--resume", info["ckpt"]]
    if info["pip_ckpt"]:
        params += ["--pip_checkpoint", info["pip_ckpt"]]
    return params


def is_complete(ds_key, variant, info):
    # POMO epochs are 1-based (last checkpoint: epoch-<target>.pt),
    # AM epochs are 0-based (last checkpoint: epoch-<target-1>.pt)
    target = target_epochs(ds_key, variant)
    last = target if VARIANTS[variant]["family"] == "pomo" else target - 1
    return info["epoch"] >= last


def submit_experiment(ds_key, variant, seed, dry_run, extra_params=None):
    job_name = get_job_name(ds_key, variant)
    family = VARIANTS[variant]["family"]
    program = ["POMO+PIP", "train.py"] if family == "pomo" else ["AM+PIP", "run.py"]
    params = build_params(ds_key, variant, seed) + list(extra_params or [])
    cmd = ["sbatch", f"--job-name={job_name}", "start_job.sh"] + program + params
    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return None
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  -> {result.stdout.strip() or result.stderr.strip()}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=sizes, choices=sizes,
                        help="Problem sizes to submit (same convention as the AMAI/skillvrp scripts).")
    parser.add_argument("--tws", type=str, nargs="+", default=None, choices=["sw", "mw"],
                        help="Filter n100 window types (no effect on n20/n50).")
    parser.add_argument("--variants", type=str, nargs="+",
                        default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print sbatch commands without submitting.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--resume", action="store_true",
                            help="Only resume crashed runs (highest epoch-*.pt); jobs "
                                 "without a checkpoint or already at the target epochs "
                                 "are skipped. Default (no flag) is reconcile: resume "
                                 "crashed, start missing, skip complete.")
    mode_group.add_argument("--fresh", action="store_true",
                            help="Submit every selected job as a brand-new run, "
                                 "ignoring existing checkpoints entirely.")
    args = parser.parse_args()

    selected = [k for k, ds in DATASETS.items()
                if ds["size"] in args.sizes and (ds["tws"] is None or args.tws is None or ds["tws"] in args.tws)]
    jobs = [(d, v) for d in selected for v in args.variants]
    mode = "resume" if args.resume else "fresh" if args.fresh else "reconcile"
    print(f"Total jobs: {len(jobs)} (sizes={args.sizes}, tws={args.tws or 'all'}, "
          f"variants={args.variants}, mode={mode})\n")

    queued = None
    if not args.fresh:
        queued = queued_job_names()
        if queued is None:
            print("WARNING: squeue not available -- cannot detect jobs that are "
                  "already queued/running; they would be submitted again!\n")

    for ds_key, variant in jobs:
        tag = job_tag(ds_key, variant)
        extra_params = []
        if not args.fresh:
            if queued and get_job_name(ds_key, variant) in queued:
                print(f"[{mode}] SKIP {tag}: already queued/running in slurm")
                continue
            info = find_resume_info(ds_key, variant)
            if info is None:
                if args.resume:
                    print(f"[{mode}] SKIP {tag}: no checkpoint found")
                    continue
                print(f"[{mode}] {tag}: no checkpoint found -- submitting fresh")
            else:
                if is_complete(ds_key, variant, info):
                    print(f"[{mode}] SKIP {tag}: already complete (epoch {info['epoch']})")
                    continue
                if variant.endswith("pipd") and info["pip_ckpt"] is None:
                    print(f"[{mode}] SKIP {tag}: epoch-{info['epoch']}.pt found but no "
                          f"fsb_accuracy_bsf.pt in {info['run_dir']} -- resume manually")
                    continue
                print(f"[{mode}] {tag}: resume from epoch {info['epoch']} "
                      f"({info['run_dir']})")
                extra_params += resume_params(variant, info)
        submit_experiment(ds_key, variant, args.seed, args.dry_run, extra_params)
