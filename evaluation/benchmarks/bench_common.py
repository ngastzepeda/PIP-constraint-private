"""Shared, torch-free helpers for evaluating the gathered PIP checkpoints on the
classic TSPTW benchmark libraries (data/benchmarks/tsptw/) as an out-of-
distribution test.

Why this is its own pipeline (not eval_checkpoints.py):

  * The benchmark instances (Dumas, GendreauDumasExtended, Langevin,
    OhlmannThomas) are distributed as an explicit n x n DISTANCE MATRIX + time
    windows, with NO coordinates. The POMO+PIP / AM+PIP models and the AMAI
    checker are coordinate/Euclidean based, so there is nothing to embed.

  * We therefore recover 2D coordinates from the distance matrix via classical
    MDS and feed those to the model (the model only needs coordinates to pick a
    visit ORDER). The reported feasibility + objective, however, are computed by
    check_tsptw_matrix() using the ORIGINAL, EXACT distance matrix -- so the gap
    to the literature best-known values is measured on the true distances, not
    the (approximate) MDS embedding. See --instance semantics in eval_benchmarks.

Data conventions mirror data/feas_tsptw/*/test_1k.npz (see evaluation/common.py):
node 0 is the depot; customers are nodes 1..n-1; travel time == distance;
service times are 0 for these libraries; the depot due date in the file is the
binding makespan bound (kept as the depot time-window upper bound).

This module is intentionally torch-free (numpy only) and import-collision-free:
it handles benchmark parsing, MDS, the model-input npz build, the exact-distance
feasibility check and the csv writers. The framework-specific inference lives in
eval_am_bench.py / eval_pomo_bench.py (each run in its own subprocess, as in the
in-distribution pipeline). The orchestrator (eval_benchmarks.py) ties them
together.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_ROOT = REPO_ROOT / "data/benchmarks/tsptw"
RESULTS_DIR = REPO_ROOT / "evaluation" / "benchmarks" / "results"

# The AMAI benchmark manifest is the source of truth for which nominal-size
# groups each checkpoint size is evaluated on (exact-size match: n20->size 20,
# n100->size 100, n50->nothing, so every model is compared on identical
# instances). Override with --manifest.
DEFAULT_MANIFEST = (
    REPO_ROOT.parent / "AMAI2025/source/benchmarks/tsptw_benchmark_manifest.json"
)
# manifest checkpoint key -> this repo's DATASETS key(s). n100 has two window
# variants (single-/multi-window); BOTH are evaluated by default -- each on the
# same size-100 instances -- mirroring AMAI's separate n100_sw & n100_mw benchmark
# result files. --ckpt_mw restricts the n100 group to n100_mw; --ckpt_size picks one.
MANIFEST_CKPT_MAP = {"n20": ["n20"], "n50": ["n50"], "n100": ["n100_sw", "n100_mw"]}

# arrival/return comparison tolerance in real time units (matches AMAI
# EPS_DEFAULT); benchmark distances are integer, so this only absorbs float
# drift accumulated along the MDS-independent exact replay.
EPS_DEFAULT = 1e-2


# ------------------------------- parsing ---------------------------------- #

def _numbers(line):
    return [float(x) for x in line.split()]


def parse_instance(path):
    """Parse a lopez-ibanez-format TSPTW file.

    Layout (all four libraries share it): line 1 = n (nodes incl. depot); next n
    lines = the n x n distance matrix; final n lines = time windows, one
    '<ready> <due>' pair per node (depot first). Service times are 0.

    -> dict(name, dataset, size, window, dist(n,n float64), tw(n,2 float64),
            service(n,) float64=0)
    """
    path = Path(path)
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(line)
    n = int(float(tokens[0].split()[0]))
    matrix_rows = tokens[1 : 1 + n]
    tw_rows = tokens[1 + n : 1 + 2 * n]
    if len(matrix_rows) != n or len(tw_rows) != n:
        raise ValueError(
            f"{path.name}: expected n={n} matrix rows and n tw rows, got "
            f"{len(matrix_rows)} / {len(tw_rows)}"
        )
    dist = np.array([_numbers(r) for r in matrix_rows], dtype=np.float64)
    if dist.shape != (n, n):
        raise ValueError(f"{path.name}: distance matrix is {dist.shape}, expected {(n, n)}")
    tw = np.array([_numbers(r)[:2] for r in tw_rows], dtype=np.float64)
    service = np.zeros(n, dtype=np.float64)
    dataset, size, window = parse_name(path)
    return dict(name=f"{dataset}/{path.stem}", dataset=dataset, size=size,
                window=window, dist=dist, tw=tw, service=service, n=n)


def parse_name(path):
    """(dataset_folder, size_int, window_str) from a benchmark file path.
    size = first integer after n/N in the stem (customers, depot excluded);
    window = the 'w<NN>' class if present, else the non-numeric class tag."""
    path = Path(path)
    dataset = path.parent.name
    stem = path.stem
    m = re.search(r"[nN](\d+)", stem)
    size = int(m.group(1)) if m else -1
    w = re.search(r"[wW](\d+)", stem)
    if w:
        window = f"w{w.group(1)}"
    else:  # Langevin 'N20ft301' -> class 'ft'
        wm = re.search(r"[a-zA-Z]+", stem[m.end():]) if m else None
        window = wm.group(0) if wm else ""
    return dataset, size, window


def node_count(path):
    """Node count (depot + customers) of a benchmark file = its first token,
    without parsing the whole matrix. Used to group a nominal-size manifest group
    by ACTUAL node count (Langevin's N20/N40/N60 have 19/39/59 customers, so a
    nominal-size group is not uniform)."""
    with open(path) as f:
        for line in f:
            if line.strip():
                return int(float(line.split()[0]))
    raise ValueError(f"{path}: empty file")


def load_manifest(path=DEFAULT_MANIFEST):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def manifest_plan(manifest, ckpt_sizes=None, mw=False):
    """-> list of (datasets_key, [nominal_sizes]) for each AMAI checkpoint size
    the manifest matches to a non-empty set of benchmark size groups. `ckpt_sizes`
    (DATASETS keys) narrows which checkpoints run. The n100 group maps to BOTH
    n100_sw and n100_mw by default (each evaluated on the same size-100 instances,
    mirroring AMAI's two n100 result files); `mw=True` restricts it to n100_mw."""
    plan = []
    for mkey, sizes in manifest["amai_checkpoint_size_match"].items():
        if not sizes:
            continue
        dkeys = list(MANIFEST_CKPT_MAP[mkey])
        if mkey == "n100" and mw:
            dkeys = ["n100_mw"]
        for dkey in dkeys:
            if ckpt_sizes and dkey not in ckpt_sizes:
                continue
            plan.append((dkey, list(sizes)))
    return plan


def manifest_instances(manifest, nominal_sizes, datasets=None, windows=None):
    """Instance records for the given nominal size groups, filtered by
    dataset/window. Paths are resolved against this repo's root (the manifest
    stores repo-relative 'data/benchmarks/tsptw/...' paths)."""
    recs = []
    for s in nominal_sizes:
        grp = manifest["sizes"].get(str(s))
        if not grp:
            continue
        for it in grp["instances"]:
            path = REPO_ROOT / it["path"]
            _dataset, _size, window = parse_name(path)
            if datasets and it["family"] not in datasets:
                continue
            if windows and window not in windows:
                continue
            recs.append(dict(name=it["name"], dataset=it["family"], size=s,
                             window=window, path=str(path),
                             customers=it.get("customers")))
    recs.sort(key=lambda r: r["name"])
    return recs


def discover_instances(datasets=None, sizes=None, windows=None, bench_root=BENCH_ROOT):
    """All benchmark files matching the filters, as parsed-name records
    (name, dataset, size, window, path). sizes filters on customer count
    (default: caller supplies, e.g. [100]); datasets/windows are optional."""
    recs = []
    for path in sorted(bench_root.rglob("*")):
        if path.suffix.lower() not in (".txt", ".dat") or not path.is_file():
            continue
        dataset, size, window = parse_name(path)
        if datasets and dataset not in datasets:
            continue
        if sizes and size not in sizes:
            continue
        if windows and window not in windows:
            continue
        recs.append(dict(name=f"{dataset}/{path.stem}", dataset=dataset,
                         size=size, window=window, path=str(path)))
    recs.sort(key=lambda r: r["name"])
    return recs


# --------------------------------- MDS ------------------------------------ #

def classical_mds(dist, dims=2):
    """Classical (Torgerson) MDS: n x n distance matrix -> (n, dims) coords
    whose pairwise Euclidean distances best reproduce `dist`. The matrix is
    symmetrized first (benchmark matrices are symmetric up to rounding).
    Coordinates are only used to give the model a visit ORDER; the reported
    objective is recomputed from the exact matrix, so small embedding error is
    harmless."""
    dist = np.asarray(dist, dtype=np.float64)
    dist = 0.5 * (dist + dist.T)
    n = dist.shape[0]
    d2 = dist ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    B = 0.5 * (B + B.T)  # enforce symmetry for eigh
    eigvals, eigvecs = np.linalg.eigh(B)  # ascending
    order = np.argsort(eigvals)[::-1][:dims]
    L = np.clip(eigvals[order], 0.0, None)
    coords = eigvecs[:, order] * np.sqrt(L)
    return coords  # (n, dims)


# ---------------------------- model-input npz ----------------------------- #

def build_model_npz(records, out_npz):
    """Parse every record, MDS-embed it, and stack into ONE model-input npz
    (all instances must share the same node count n). Coordinates are shifted to
    the positive quadrant and scaled so the bounding box is [0, DOMAIN]; the time
    windows and service times are scaled by the SAME factor so the model sees an
    in-distribution scale while the travel-time/time-window ratio is preserved
    (the loaders divide everything by max_loc again -> [0,1]-ish coords). Node 0
    stays the depot.

    Returns (names, exact_dist, exact_tw, exact_service): the per-instance name
    order and the ORIGINAL (unscaled) distance/time-window/service arrays used by
    check_tsptw_matrix. Writes `out_npz` (model input) as a side effect.
    """
    DOMAIN = 100.0  # training coords live in [0, 100]; keep the model in-range
    insts = [parse_instance(r["path"]) for r in records]
    n_nodes = insts[0]["n"]
    for it in insts:
        if it["n"] != n_nodes:
            raise ValueError(
                f"build_model_npz needs a uniform node count; {it['name']} has "
                f"n={it['n']} != {n_nodes}. Group instances by size first.")

    N = len(insts)
    locs = np.zeros((N, n_nodes, 2), dtype=np.float32)
    time_windows = np.zeros((N, n_nodes, 2), dtype=np.float32)
    service_times = np.zeros((N, n_nodes), dtype=np.float32)
    names, exact_dist, exact_tw, exact_service = [], [], [], []

    for b, it in enumerate(insts):
        coords = classical_mds(it["dist"], dims=2)
        coords = coords - coords.min(axis=0, keepdims=True)  # -> positive quadrant
        scale = DOMAIN / max(float(coords.max()), 1e-9)
        locs[b] = (coords * scale).astype(np.float32)
        time_windows[b] = (it["tw"] * scale).astype(np.float32)
        service_times[b] = (it["service"] * scale).astype(np.float32)
        names.append(it["name"])
        exact_dist.append(it["dist"])
        exact_tw.append(it["tw"])
        exact_service.append(it["service"])

    max_loc = np.full((N, 1), int(DOMAIN), dtype=np.int64)
    n_vehicles = np.ones((N, 1), dtype=np.int64)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        locs=locs,
        time_windows=time_windows,
        service_times=service_times,
        max_loc=max_loc,
        n_vehicles=n_vehicles,
        num_routes=np.ones((N, 1), dtype=np.int64),
        travel_cost=np.ones((N, 1, 1), dtype=np.float32),
        current_node=np.zeros((N, 1), dtype=np.int64),
        current_time=np.zeros((N, 1), dtype=np.float32),
    )
    return names, exact_dist, exact_tw, exact_service


# ------------------------- exact-distance checker ------------------------- #

def check_tsptw_matrix(dist, tw, service, tour, eps=EPS_DEFAULT):
    """Exact-distance TSPTW feasibility + objective for one proposed tour.

    MIRROR of AMAI2025 source/checkers/solution_checker/checker.py
    ::check_tsptw_matrix (that repo is not importable from this venv -- it needs
    pyrootutils; evaluation/common.py mirrors AMAI logic for the same reason).
    KEEP THE TWO IN SYNC.

    Unlike the AMAI Euclidean checker, travel time between nodes a and b is taken
    straight from the distance matrix (dist[a, b]) rather than recomputed from
    coordinates -- this is the whole point: the objective feeding the gap uses
    the benchmark's true distances.

    `tour` is the visit order of CUSTOMER node ids (1..n-1), depot excluded.
    Semantics match the AMAI/LMask TSPTW env: start at depot ready time; arrival
    must be <= tw_late (+eps); wait to tw_early; add service; return to depot by
    its tw_late. Returns dict(feasible, objective, max_violation, errors).
    """
    dist = np.asarray(dist, dtype=np.float64)
    tw = np.asarray(tw, dtype=np.float64)
    service = np.asarray(service, dtype=np.float64)
    n = dist.shape[0]
    depot = 0
    errors = []

    tour = list(int(i) for i in tour)
    if sorted(tour) != list(range(1, n)):
        errors.append(f"tour is not a permutation of customers 1..{n - 1} "
                      f"(len {len(tour)}, {len(set(tour))} unique)")

    t = float(tw[depot, 0])
    prev = depot
    distance = 0.0
    max_viol = 0.0
    for i in tour:
        if not (0 <= i < n):
            errors.append(f"node id {i} out of range")
            continue
        leg = float(dist[prev, i])
        distance += leg
        t += leg
        late = float(tw[i, 1])
        if t > late + eps:
            over = t - late
            max_viol = max(max_viol, over)
            errors.append(f"cust {i}: arrival {t:.3f} > tw_late {late:.3f} (+{over:.3f})")
        t = max(t, float(tw[i, 0]))
        t += float(service[i])
        prev = i
    leg = float(dist[prev, depot])
    distance += leg
    t += leg
    dlate = float(tw[depot, 1])
    if t > dlate + eps:
        over = t - dlate
        max_viol = max(max_viol, over)
        errors.append(f"depot return {t:.3f} > tw_late {dlate:.3f} (+{over:.3f})")

    return dict(feasible=not errors, objective=round(distance, 4),
                max_violation=round(max_viol, 4), errors=errors)


# -------------------------------- output IO ------------------------------- #

def instance_csv_path(variant_label, ckpt_size, mode):
    return RESULTS_DIR / f"{variant_label}_{ckpt_size}_{mode}.csv"


def write_instance_csv(variant_label, ckpt_size, mode, rows):
    """One csv per (variant, checkpoint size): columns instance, feasible (1/0),
    objective (blank if infeasible), runtime (seconds). Named
    ``{variant}_{ckpt_size}_{mode}.csv`` to mirror the AMAI/LMask benchmark layout
    (``{model}_{size}_{mode}.csv``) so n20 / n100_sw / n100_mw land in separate
    files. `rows` is a list of dicts with keys name/feasible/objective/runtime."""
    path = instance_csv_path(variant_label, ckpt_size, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "feasible", "objective", "runtime"])
        for r in rows:
            obj = "" if not r["feasible"] else round(float(r["objective"]), 6)
            w.writerow([r["name"], int(r["feasible"]), obj, round(float(r["runtime"]), 6)])
    return path


def summary_csv_path(mode):
    return RESULTS_DIR / f"summary_{mode}.csv"


def write_summary(mode, summary_rows):
    """Optional grouped view: one row per (variant, dataset, size, window) with
    feasibility rate, mean feasible objective and mean runtime -- a compact table
    when the per-instance files have too many rows for a paper."""
    path = summary_csv_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "ckpt_size", "dataset", "size", "window", "n_instances",
              "feas_count", "feas_rate", "mean_objective", "mean_runtime"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    return path


def summarize(variant_label, ckpt_size, rows):
    """Collapse one (variant, checkpoint size) block's per-instance rows into
    per-(dataset,size,window) summary rows (mean objective over FEASIBLE instances
    only). `ckpt_size` keeps the n100_sw / n100_mw rows distinct."""
    groups = {}
    for r in rows:
        dataset = r["name"].split("/", 1)[0]
        key = (dataset, r["size"], r["window"])
        groups.setdefault(key, []).append(r)
    out = []
    for (dataset, size, window), grp in sorted(groups.items()):
        feas = [g for g in grp if g["feasible"]]
        mean_obj = (round(float(np.mean([g["objective"] for g in feas])), 4)
                    if feas else "")
        out.append(dict(
            variant=variant_label, ckpt_size=ckpt_size, dataset=dataset,
            size=size, window=window,
            n_instances=len(grp), feas_count=len(feas),
            feas_rate=round(len(feas) / len(grp), 4),
            mean_objective=mean_obj,
            mean_runtime=round(float(np.mean([g["runtime"] for g in grp])), 6),
        ))
    return out
