#!/usr/bin/env python3
"""Check whether any run's val-best model was lost to a pre-fix resume.

Background (see summary.md): before the val-best tracker fix, the in-memory
best-so-far reset on every resume. POMO overwrites trained_model_val_best.pt
in place, so a pre-resume best that is never beaten again is unrecoverable;
AM writes one val_best.pt per attempt dir, so its best is always recoverable
and this script just reports which attempt's file holds it.

For every selected run the script pulls the full validation history from
wandb (all resume attempts share one run id) and computes the global best
under the training-time selection rule (instance-level feasibility rate
first, feasible cost tie-break). Verdicts:

  OK        the current val-best file holds (or, for AM, some attempt's file
            holds) the global best. For POMO this means the global best falls
            after the last resume (whose start time is read from the newest
            wandb/run-<ts>-<id> dir inside the run dir), the run was never
            resumed, or the run has a val_best_meta.json matching the global
            best (post-fix runs).
  AT RISK   POMO only: the global best predates the last resume - those
            weights are overwritten. If the run is still training it may yet
            re-attain the value; re-check after completion.
  UNKNOWN   wandb history unavailable (no run id, network, deleted run, ...).

Runs short of their target epoch are marked PROVISIONAL. Needs network access
to wandb; run on the cluster login node via the wrapper:

    bash check_val_best.sh                              # full grid
    bash check_val_best.sh --sizes 100 --variants pomo pip pipd

Exit code: 1 if any run is AT RISK (or UNKNOWN), else 0.
"""

import argparse
import sys
import time

# progress marker before the heavy imports: gather_checkpoints pulls in torch,
# which can take minutes from .venv when the parallel filesystem is slow
print(
    "loading libraries (torch import - can take a while on the cluster filesystem)...",
    flush=True,
)
from gather_checkpoints import (
    am_lineage,
    epoch_ckpts,
    pomo_lineage,
    read_meta,
    wandb_run_id,
)
from submit_jobs import AM_DIR, DATASETS, POMO_DIR, VARIANTS, job_tag

WANDB_PROJECT = "pip_feasible"  # submit_jobs.py --wandb_project
VAL_NAME = "val_1k"  # POMO metric suffix = val dataset basename

_api = None


def api():
    global _api
    if _api is None:
        print("connecting to the wandb API...", flush=True)
        import wandb

        # timeout so a blocked/throttled connection errors out (-> UNKNOWN
        # verdicts) instead of hanging forever
        _api = wandb.Api(timeout=60)
    return _api


def better(cand, best):
    """Strict improvement of (feas, score), as in training; None never wins."""
    if cand is None:
        return False
    return (
        best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1])
    )


def as_score(x):
    """Feasible cost from a wandb cell; NaN/None (no feasible sol) -> inf."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("inf")
    return x if x == x else float("inf")


def wandb_history(run_id, keys):
    return list(
        api()
        .run(f"{api().default_entity}/{WANDB_PROJECT}/{run_id}")
        .scan_history(keys=keys)
    )


def progress(lineage, target_last):
    """(last saved epoch, is_complete) of a lineage."""
    eps = [e for d in lineage for e, _ in epoch_ckpts(d)]
    last = max(eps) if eps else None
    return last, last is not None and last >= target_last


def check_pomo(tag, lineage, target):
    """POMO: one dir per lineage; resumes overwrite trained_model_val_best.pt
    in place, so the file only covers validations after the last resume."""
    d = lineage[0]
    run_id = wandb_run_id(d)
    if run_id is None:
        return "UNKNOWN", "no wandb run id in run dir"

    feas_key, score_key = (f"val_ins_infsb_rate/{VAL_NAME}", f"val_score/{VAL_NAME}")
    try:
        feas_rows = wandb_history(run_id, [feas_key])
        score_rows = wandb_history(run_id, [score_key])
    except Exception as exc:
        return "UNKNOWN", f"wandb lookup failed: {exc}"
    if not feas_rows:
        return "UNKNOWN", "no validation rows in wandb"

    points = []  # (feas, score, timestamp-or-None, index)
    for i, row in enumerate(feas_rows):
        score = (
            as_score(score_rows[i][score_key]) if i < len(score_rows) else float("inf")
        )
        points.append((100.0 - float(row[feas_key]), score, row.get("_timestamp"), i))
    gbest = None
    for p in points:
        if better(p[:2], gbest and gbest[:2]):
            gbest = p
    desc = f"global best: feas {gbest[0]:.2f}% score {gbest[1]:.4f} (~epoch {gbest[3] + 1})"

    # post-fix runs record the file's content exactly - trust the meta
    meta = read_meta(d)
    if meta is not None and not better(
        gbest[:2], (meta["feas"], as_score(meta["score"]))
    ):
        return "OK", f"{desc}; val_best_meta.json matches"

    # last resume start = newest wandb attach dir (run-<YYYYmmdd_HHMMSS>-<id>);
    # dir names use cluster-local time, so mktime converts correctly on the cluster
    attaches = sorted(p.name for p in (d / "wandb").glob("run-*"))
    if len(attaches) <= 1:
        return "OK", f"{desc}; never resumed"
    boundary = (
        time.mktime(time.strptime(attaches[-1].split("-")[1], "%Y%m%d_%H%M%S")) - 60
    )
    post = None
    for p in points:
        # missing timestamp -> treat as pre-resume
        if p[2] is not None and p[2] >= boundary and better(p[:2], post and post[:2]):
            post = p
    if not better(gbest[:2], post and post[:2]):
        return (
            "OK",
            f"{desc}; re-attained after the last resume ({len(attaches)} attaches)",
        )
    lost = (
        f"only feas {post[0]:.2f}% score {post[1]:.4f} since the last resume"
        if post
        else "no validation since the last resume yet"
    )
    return "AT RISK", f"{desc} predates the last resume; {lost}"


def check_am(tag, lineage, target):
    """AM: every attempt keeps its own val_best.pt, so the global best is
    always recoverable - report which attempt's file holds it."""
    run_id = wandb_run_id(lineage[-1])
    if run_id is None:
        return "UNKNOWN", "no wandb run id in run dir"
    try:
        rows = wandb_history(run_id, ["epoch", "val/infsb_rate", "val/feasible_cost"])
    except Exception as exc:
        return "UNKNOWN", f"wandb lookup failed: {exc}"
    if not rows:
        return "UNKNOWN", "no validation rows in wandb"
    gbest = None  # (feas, cost, epoch)
    for r in rows:
        cand = (
            100.0 - float(r["val/infsb_rate"]),
            as_score(r["val/feasible_cost"]),
            int(r["epoch"]),
        )
        if better(cand[:2], gbest and gbest[:2]):
            gbest = cand
    desc = f"global best: feas {gbest[0]:.2f}% cost {gbest[1]:.4f} (epoch {gbest[2]})"
    if len(lineage) == 1:
        return "OK", f"{desc}; single attempt"
    # attempt k covers epochs (last epoch of k-1, last epoch of k]
    holder = next(
        (
            d
            for d in lineage
            if epoch_ckpts(d)
            and epoch_ckpts(d)[-1][0] >= gbest[2]
            and (d / "val_best.pt").is_file()
        ),
        None,
    )
    if holder is None:
        return "OK", f"{desc}; holder attempt unclear - gather resolves via wandb"
    return "OK", f"{desc}; held by {holder / 'val_best.pt'}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[20, 50, 100], choices=[20, 50, 100]
    )
    ap.add_argument(
        "--tws",
        type=str,
        nargs="+",
        default=None,
        choices=["sw", "mw"],
        help="filter n100 window types",
    )
    ap.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
    )
    args = ap.parse_args()

    jobs = [
        (ds_key, variant)
        for ds_key, ds in DATASETS.items()
        if ds["size"] in args.sizes
        and (ds["tws"] is None or args.tws is None or ds["tws"] in args.tws)
        for variant in args.variants
    ]
    print(
        f"Checking {len(jobs)} runs (1-2 wandb history downloads each, "
        "typically a few seconds per run)...",
        flush=True,
    )

    counts = {"OK": 0, "AT RISK": 0, "UNKNOWN": 0}
    for i, (ds_key, variant) in enumerate(jobs, 1):
        ds = DATASETS[ds_key]
        tag = job_tag(ds_key, variant)
        prefix = f"[{i:2}/{len(jobs)}]"
        family = VARIANTS[variant]["family"]
        base = POMO_DIR if family == "pomo" else AM_DIR
        run_root = base / ("results" if family == "pomo" else "outputs") / tag
        if not run_root.is_dir():
            print(f"{prefix} {tag:18} -       : no run dir -> skip", flush=True)
            continue
        lineage = pomo_lineage(run_root) if family == "pomo" else am_lineage(run_root)
        if not lineage:
            print(f"{prefix} {tag:18} -       : no checkpoints -> skip", flush=True)
            continue

        print(f"{prefix} {tag:18} fetching wandb history...", flush=True)
        started = time.monotonic()
        # POMO epochs are 1-based (last ckpt: epoch-<target>), AM 0-based
        target = ds["epochs"] if family == "pomo" else ds["epochs"] - 1
        last, complete = progress(lineage, target)
        check = check_pomo if family == "pomo" else check_am
        verdict, detail = check(tag, lineage, target)
        counts[verdict] += 1
        prov = "" if complete else f" [PROVISIONAL - at epoch {last}/{target}]"
        print(
            f"{prefix} {tag:18} {verdict:7} ({time.monotonic() - started:.1f}s): "
            f"{detail}{prov}",
            flush=True,
        )

    print(
        f"\nSummary: {counts['OK']} OK, {counts['AT RISK']} AT RISK, "
        f"{counts['UNKNOWN']} UNKNOWN"
    )
    if counts["AT RISK"]:
        print(
            "AT RISK runs: the overwritten best may still be re-attained if the "
            "run is unfinished - re-check after completion before re-running "
            "anything."
        )
    sys.exit(1 if counts["AT RISK"] or counts["UNKNOWN"] else 0)


if __name__ == "__main__":
    main()
