"""Shared helpers for evaluating the gathered PIP checkpoints (checkpoints/)
on the AMAI TSPTW test instances, producing the *same metrics* as
AMAI2025/source/evaluation/eval_checkpoints.py so the two repos' numbers are
directly comparable.

This module is intentionally torch-free and import-collision-free: it only
handles checkpoint discovery, the AMAI best-known-solution (BKS) lookup, the
AMAI-style metric aggregation and the txt/csv writers. The heavy, framework-
specific inference lives in eval_pomo.py / eval_am.py (each run in its own
subprocess because POMO+PIP and AM+PIP have colliding top-level module names,
e.g. `utils`).

Metric definitions (mirroring AMAI get_test_metrics / attach_bks):
  * a per-instance solution is FEASIBLE if the model found >=1 feasible tour
    for it (any augmentation / start);
  * the per-instance COST is the best (min) feasible tour length, in the
    model's normalized [0,1]-coord units, multiplied by domain_size (=max_loc,
    =100) to recover real 0-100 distances;
  * feas_count = number of feasible instances (out of 1000);
  * cost       = mean feasible cost over feasible instances;
  * gap_bks    = mean over feasible instances of (cost-bks)/bks*100, using the
    AMAI Gurobi/pyvrp BKS (identical instances) -> NaN if the BKS csv is absent.
"""

import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = REPO_ROOT / "checkpoints"
DATA_ROOT = REPO_ROOT / "data/feas_tsptw"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"

# The AMAI baseline instance results, whose `bks` column gives the per-instance
# best-known objective (Gurobi/pyvrp) on the *same* test instances. Override
# with --bks_csv if the sibling repo lives elsewhere.
DEFAULT_BKS_CSV = (
    REPO_ROOT.parent
    / "AMAI2025/source/evaluation/results/eval_baselines_instance_results.csv"
)

# Single source of truth for size->folder and variant->family: reuse the
# training submit script's grid definition.
sys.path.insert(0, str(REPO_ROOT))
from submit_jobs import DATASETS, VARIANTS  # noqa: E402

KINDS = ("best", "last", "pip")  # checkpoint suffixes; "pip" is the PIP-D aux head
# longest size keys first so 'n100_sw_am_pip' matches 'n100_sw', not 'n100'
_SIZES_BY_LEN = sorted(DATASETS, key=len, reverse=True)


def pick_device(requested="auto"):
    """Resolve --device: 'auto' probes cuda > mps > cpu, warning if it has to
    land on cpu. An explicit cpu/cuda/mps request is honored if available,
    else falls back through the same cuda > mps > cpu chain (also warning
    before landing on cpu). Imports torch lazily so this module stays
    torch-free at import time for callers that don't need it."""
    import torch  # noqa: PLC0415 (deferred: see module docstring)

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if requested != "auto" and available[requested]:
        return requested
    if requested != "auto":
        print(f"WARNING: --device {requested} requested but not available on "
              f"this machine; falling back.", file=sys.stderr)
    for name in ("cuda", "mps"):
        if available[name]:
            return name
    print("WARNING: no CUDA or MPS device found; falling back to CPU. "
          "Inference will be much slower.", file=sys.stderr)
    return "cpu"


def bks_size_key(size):
    """DATASETS folder ('n20_amai', 'n100_amai_sw') -> AMAI csv `size` key
    ('20_amai', '100_amai_sw'): the csv strips the leading 'n'."""
    folder = DATASETS[size]["folder"]
    return folder[1:] if folder.startswith("n") else folder


def test_file(size):
    return DATA_ROOT / DATASETS[size]["folder"] / "test_1k.npz"


def domain_size(size):
    """max_loc of the test instances (=100 for all AMAI TSPTW sets)."""
    with np.load(test_file(size)) as d:
        return int(np.max(d["max_loc"]))


def parse_ckpt_name(name):
    """'n100_sw_am_pip_best' -> ('n100_sw', 'am_pip', 'best'), or None."""
    for kind in KINDS:
        if name.endswith(f"_{kind}"):
            rest = name[: -(len(kind) + 1)]
            break
    else:
        return None
    for size in _SIZES_BY_LEN:
        if rest == size or rest.startswith(size + "_"):
            variant = rest[len(size) + 1 :]
            if variant in VARIANTS:
                return size, variant, kind
    return None


def label(size, variant):
    """AMAI-parallel label: experiment/problem/size/model."""
    return f"pip/tsptw/{size}/{variant}"


def discover_jobs(modes, sizes=None, variants=None, ckpt_root=CKPT_ROOT):
    """One job per checkpoint slot ({size}_{variant}_{best,last}.pt) present on
    disk. The PIP-D auxiliary head ('..._pip.pt') is not a job itself; it is
    attached to the matching pipd job as `pip_ckpt` (available if a variant
    ever wants to load a separate lazy decoder)."""
    jobs = []
    for ckpt in sorted(ckpt_root.glob("*.pt")):
        parsed = parse_ckpt_name(ckpt.stem)
        if parsed is None:
            continue
        size, variant, kind = parsed
        if kind not in modes:
            continue
        if sizes and size not in sizes:
            continue
        if variants and variant not in variants:
            continue
        pip_ckpt = ckpt_root / f"{size}_{variant}_pip.pt"
        jobs.append(
            dict(
                size=size,
                variant=variant,
                family=VARIANTS[variant]["family"],
                kind=kind,
                ckpt=str(ckpt),
                pip_ckpt=str(pip_ckpt) if pip_ckpt.exists() else None,
                test_file=str(test_file(size)),
                bks_size=bks_size_key(size),
                label=label(size, variant),
            )
        )
    return jobs


def load_bks(size, bks_csv, n_instances):
    """Per-instance BKS array (len n_instances) from the AMAI csv, or None if
    the file/rows are missing or the count mismatches (-> gap NaN, no crash).
    Mirrors AMAI attach_bks: filter env==tsptw & size, sort by instance."""
    bks_csv = Path(bks_csv)
    if not bks_csv.exists():
        return None
    key = bks_size_key(size)
    rows = []
    with open(bks_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["env"] == "tsptw" and r["size"] == key:
                rows.append((int(r["instance"]), float(r["bks"])))
    if len(rows) != n_instances:
        return None
    rows.sort()  # by instance index, to align with the test set's row order
    return np.array([b for _, b in rows], dtype=float)


def aggregate(feas, cost_norm, bks, dom):
    """Per-instance (feasible bool, normalized best-feasible cost) -> AMAI row
    fields. `cost_norm` may be NaN/inf where infeasible; only feasible entries
    are used."""
    feas = np.asarray(feas, dtype=bool)
    cost_norm = np.asarray(cost_norm, dtype=float)
    # real-unit cost, NaN where infeasible (matches AMAI reported_cost)
    cost_real = np.where(feas, cost_norm * dom, np.nan)
    feas_count = int(feas.sum())
    with warnings.catch_warnings():  # all-infeasible -> NaN, not a crash
        warnings.simplefilter("ignore", RuntimeWarning)
        cost = float(np.nanmean(cost_real))
        if bks is not None:
            gap = float(np.nanmean((cost_real - bks) / bks * 100.0))
        else:
            gap = float("nan")
    return dict(
        feas_count=feas_count,
        cost=round(cost, 4),
        gap_bks=round(gap, 4),
    )


def make_row(job, metrics, inference_time):
    return {
        "experiment": "pip",
        "problem": "tsptw",
        "size": job["size"],
        "model": job["variant"],
        "family": job["family"],
        "mode": job["kind"],
        "feas_count": metrics["feas_count"],
        "cost": metrics["cost"],
        "gap_bks": metrics["gap_bks"],
        "inference_time": round(inference_time, 3),
        "ckpt_path": os.path.relpath(job["ckpt"], REPO_ROOT),
        "label": job["label"],
    }


# ------------------------------- output IO -------------------------------- #

def txt_path(mode):
    return RESULTS_DIR / f"eval_checkpoints_{mode}.txt"


def csv_path(mode):
    return RESULTS_DIR / f"eval_checkpoints_{mode}.csv"


def write_new_run_header(mode):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(txt_path(mode), "a") as f:
        f.write("##### ----- NEW EVALUATION RUN ----- #####\n")


def append_txt_row(mode, row):
    """One AMAI-format line: '<ts> | <label> (<mode>):\\t <feas> & <cost> &
    <gap> & <time> | <ckpt>'."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(txt_path(mode), "a") as f:
        f.write(
            f"{ts} | {row['label']} ({row['mode']}):\t {row['feas_count']} & "
            f"{row['cost']:.3f} & {row['gap_bks']:.3f} & "
            f"{row['inference_time']:.3f} | {row['ckpt_path']}\n"
        )


def write_csv(mode, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(csv_path(mode), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
