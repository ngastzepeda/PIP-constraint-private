"""Evaluate the gathered PIP checkpoints on the classic TSPTW benchmark
libraries (data/benchmarks/tsptw/) as an out-of-distribution test, writing one
per-instance csv per checkpoint:

    evaluation/benchmarks/results/{variant_label}_{mode}.csv
      columns: instance, feasible (1/0), objective (blank if infeasible),
               runtime (per-instance inference seconds)

plus a compact grouped view (summary_{mode}.csv, one row per
variant x dataset x size x window).

Pipeline (see bench_common.py for the why):
  1. read the AMAI benchmark manifest (--manifest), which maps each checkpoint
     size to its exact-match benchmark size group(s); select + parse those
     instances (--datasets / --windows / --sizes / --ckpt_size narrow further);
  2. per checkpoint size, sub-group its instances by ACTUAL node count and
     recover 2D coordinates via classical MDS into one model-input npz per group;
  3. run each family's checkpoints in its own subprocess (eval_am_bench.py /
     eval_pomo_bench.py) to get, per instance, the TOUR the model proposes;
  4. recompute feasibility + objective from the EXACT distance matrix
     (bench_common.check_tsptw_matrix) and write one accumulated csv per variant.

The model only picks a visit order from the (approximate) MDS coordinates; every
reported number uses the benchmark's true distances, so the gap to the
literature best-known values is exact.

Run from the repo root's venv (workstation only, no cluster wrapper):

    python evaluation/benchmarks/eval_benchmarks.py                    # manifest sizes, best, all variants
    python evaluation/benchmarks/eval_benchmarks.py --ckpt_size n100_sw   # size-100 group only
    python evaluation/benchmarks/eval_benchmarks.py --families am --datasets Dumas
    python evaluation/benchmarks/eval_benchmarks.py --no_manifest --sizes 100 --device cpu
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # evaluation/
sys.path.insert(0, str(Path(__file__).resolve().parent))         # evaluation/benchmarks/
import common
import bench_common as bc
from submit_jobs import DATASETS, VARIANTS

BENCH_DIR = Path(__file__).resolve().parent
WORKER = {"pomo": "eval_pomo_bench.py", "am": "eval_am_bench.py"}


def nearest_ckpt_size(n_customers, prefer_sw=True):
    """Benchmark customer count -> the DATASETS checkpoint key whose trained size
    is closest (ties -> the larger). n100 resolves to the single-window (sw)
    checkpoint by default, since the benchmark time windows are single-window."""
    def customers(key):
        return DATASETS[key]["size"]
    keys = sorted(DATASETS, key=lambda k: (abs(customers(k) - n_customers), -customers(k)))
    best = customers(keys[0])
    tied = [k for k in DATASETS if customers(k) == best]
    if prefer_sw:
        sw = [k for k in tied if not k.endswith("_mw")]
        if sw:
            return sw[0]
    return tied[0]


def run_family(family, jobs, npz, names, problem_size, args, staging):
    """Launch the family worker subprocess; return its {variant_label: {...}} map
    of per-instance tours."""
    jobs_file = staging / f"jobs_{family}.json"
    out_file = staging / f"tours_{family}.json"
    with open(jobs_file, "w") as f:
        json.dump(jobs, f)

    cmd = [sys.executable, str(BENCH_DIR / WORKER[family]),
           "--npz", str(npz), "--jobs", str(jobs_file), "--out", str(out_file),
           "--device", args.device, "--seed", str(args.seed),
           "--test_episodes", str(len(names))]
    if family == "pomo":
        cmd += ["--problem_size", str(problem_size),
                "--test_batch_size", str(args.pomo_batch_size),
                "--aug_factor", str(args.aug_factor)]
    else:
        cmd += ["--decode", args.am_decode, "--width", str(args.am_width),
                "--eval_batch_size", str(args.am_batch_size),
                "--softmax_temp", str(args.am_softmax_temp),
                "--max_calc_batch_size", str(args.am_max_calc_batch_size)]

    print(f"\n=== {family} family: {len(jobs)} checkpoint(s) on "
          f"{len(names)} instance(s) ===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {family} worker exited with {result.returncode}", flush=True)
    if not out_file.exists():
        return {}
    with open(out_file) as f:
        return json.load(f)


def score(variant_label, res, recs, exact_dist, exact_tw, exact_service, args):
    """Turn one checkpoint's tours into per-instance rows (exact-distance check).
    Returns the rows (accumulated by the caller, one csv per variant)."""
    tours, runtimes = res["tours"], res["runtime"]
    rows = []
    for i, rec in enumerate(recs):
        chk = bc.check_tsptw_matrix(exact_dist[i], exact_tw[i], exact_service[i],
                                    tours[i], eps=args.eps)
        rows.append(dict(name=rec["name"], size=rec["size"], window=rec["window"],
                         feasible=chk["feasible"], objective=chk["objective"],
                         runtime=runtimes[i]))
    n_feas = sum(1 for r in rows if r["feasible"])
    print(f"[{variant_label}] feasible {n_feas}/{len(rows)} "
          f"(size-{recs[0]['size']} group)", flush=True)
    return rows


def run_ckpt_size(dkey, recs, mode, args, rows_by_variant):
    """Evaluate one checkpoint size on its matched instances. Instances are
    grouped by ACTUAL node count (a nominal-size group is not uniform -- Langevin
    N20/N40/N60 have 19/39/59 customers), one model-input npz per node count.
    Rows are appended to rows_by_variant[variant_label] so each variant's csv
    accumulates every instance it is size-matched to, across size groups."""
    if not recs:
        return
    jobs = common.discover_jobs([mode], sizes=[dkey], variants=args.variants)
    jobs = [j for j in jobs if j["family"] in args.families]
    if not jobs:
        print(f"  no checkpoints for ckpt_size={dkey}; skipping "
              f"{len(recs)} instance(s).", flush=True)
        return

    by_n = {}
    for r in recs:
        by_n.setdefault(bc.node_count(r["path"]), []).append(r)

    for n_nodes in sorted(by_n):
        grp = by_n[n_nodes]
        print(f"\n----- {dkey} checkpoints on {len(grp)} instance(s), "
              f"{n_nodes} nodes -----", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            npz = staging / f"bench_{dkey}_n{n_nodes}.npz"
            names, exact_dist, exact_tw, exact_service = bc.build_model_npz(grp, npz)

            results = {}
            for family in args.families:
                fam_jobs = [{"variant": j["variant"], "kind": j["kind"],
                             "ckpt": j["ckpt"], "size": dkey}
                            for j in jobs if j["family"] == family]
                if fam_jobs:
                    results.update(run_family(family, fam_jobs, npz, names,
                                              n_nodes, args, staging))

            for variant_label, res in results.items():
                rows = score(variant_label, res, grp, exact_dist, exact_tw,
                             exact_service, args)
                rows_by_variant.setdefault(variant_label, []).extend(rows)


def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--modes", nargs="+", default=["best"], choices=["best", "last"])
    p.add_argument("--families", nargs="+", default=["pomo", "am"], choices=["pomo", "am"])
    p.add_argument("--variants", nargs="+", default=None, choices=list(VARIANTS))
    p.add_argument("--manifest", default=str(bc.DEFAULT_MANIFEST),
                   help="AMAI benchmark manifest defining which checkpoint size "
                        "evaluates which nominal benchmark sizes (source of truth)")
    p.add_argument("--no_manifest", action="store_true",
                   help="ignore the manifest; discover instances directly and use "
                        "the nearest-size checkpoint per benchmark size instead")
    p.add_argument("--sizes", nargs="+", type=int, default=None,
                   help="restrict to these nominal benchmark sizes (default: all "
                        "the manifest matches, e.g. 20 and 100)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="benchmark folders to include (e.g. Dumas GendreauDumasExtended)")
    p.add_argument("--windows", nargs="+", default=None,
                   help="window classes to include (e.g. w20 w40)")
    p.add_argument("--ckpt_size", default=None, choices=list(DATASETS),
                   help="restrict to a single checkpoint size (manifest mode) / "
                        "force it (no-manifest mode)")
    p.add_argument("--ckpt_mw", action="store_true",
                   help="use the multi-window n100 checkpoints (n100_mw) instead "
                        "of single-window (n100_sw) for the size-100 group")
    p.add_argument("--eps", type=float, default=bc.EPS_DEFAULT)
    p.add_argument("--seed", type=int, default=2024)
    # POMO knobs
    p.add_argument("--pomo_batch_size", type=int, default=500)
    p.add_argument("--aug_factor", type=int, default=8, choices=[1, 8])
    # AM knobs
    p.add_argument("--am_decode", default="sampling", choices=["sampling", "greedy"])
    p.add_argument("--am_width", type=int, default=1280)
    p.add_argument("--am_batch_size", type=int, default=16)
    p.add_argument("--am_softmax_temp", type=float, default=1.0)
    p.add_argument("--am_max_calc_batch_size", type=int, default=1280)
    return p


def build_plan(args):
    """-> (manifest_or_None, list of (dkey, [recs])): what to run, per checkpoint
    size. Manifest mode (default) uses the AMAI exact-size matching; --no_manifest
    falls back to direct discovery + nearest-size checkpoint."""
    ckpt_sizes = [args.ckpt_size] if args.ckpt_size else None
    manifest = None if args.no_manifest else bc.load_manifest(args.manifest)
    plan = []
    if manifest is not None:
        for dkey, nominal_sizes in bc.manifest_plan(manifest, ckpt_sizes, args.ckpt_mw):
            if args.sizes:
                nominal_sizes = [s for s in nominal_sizes if s in args.sizes]
            recs = bc.manifest_instances(manifest, nominal_sizes, args.datasets, args.windows)
            if recs:
                plan.append((dkey, recs))
    else:
        recs_all = bc.discover_instances(args.datasets, args.sizes, args.windows)
        by_size = {}
        for r in recs_all:
            by_size.setdefault(r["size"], []).append(r)
        for size in sorted(by_size):
            dkey = args.ckpt_size or nearest_ckpt_size(size)
            plan.append((dkey, by_size[size]))
    return manifest, plan


def main():
    args = get_parser().parse_args()
    args.device = common.pick_device(args.device)
    print(f"Using device: {args.device}", flush=True)

    manifest, plan = build_plan(args)
    print(f"Using manifest: {args.manifest if manifest is not None else 'no (nearest-size)'}",
          flush=True)
    if not plan:
        print("No benchmark instances / checkpoints match the filters.", flush=True)
        return

    for mode in args.modes:
        print(f"\n##### mode={mode} #####", flush=True)
        rows_by_variant = {}  # variant_label -> accumulated per-instance rows
        for dkey, recs in plan:
            run_ckpt_size(dkey, recs, mode, args, rows_by_variant)

        summary_rows = []
        for variant_label in sorted(rows_by_variant):
            rows = sorted(rows_by_variant[variant_label], key=lambda r: r["name"])
            path = bc.write_instance_csv(variant_label, mode, rows)
            n_feas = sum(1 for r in rows if r["feasible"])
            print(f"\n{variant_label}_{mode}.csv: {len(rows)} instances, "
                  f"{n_feas} feasible -> {path.relative_to(bc.REPO_ROOT)}", flush=True)
            summary_rows += bc.summarize(variant_label, rows)

        if summary_rows:
            spath = bc.write_summary(mode, summary_rows)
            print(f"mode={mode}: grouped summary -> {spath.relative_to(bc.REPO_ROOT)}",
                  flush=True)


if __name__ == "__main__":
    main()
