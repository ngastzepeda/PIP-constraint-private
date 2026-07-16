#!/usr/bin/env python3
"""Gather each run's best/last checkpoint into a dedicated folder and git-add it.

Runs on the cluster from the repo root, ALWAYS via the bash wrapper (which
loads the Python module and sources the venv):

    bash gather_checkpoints.sh                # copy into checkpoints/ (no git)
    bash gather_checkpoints.sh --git-add      # ...and git add the folder
    bash gather_checkpoints.sh --dest somedir # different destination folder
    bash gather_checkpoints.sh --sizes 100 --tws mw --variants pipd  # subset

Run-dir layouts (see submit_jobs.py):

  POMO: POMO+PIP/results/<tag>/<YYYYmmdd_HHMMSS><run_name>/
        epoch-N.pt (only the latest is kept), trained_model_val_best.pt,
        and for PIP-D fsb_accuracy_bsf.pt. Resumes REUSE the same dir
        (train.py sets log_path = resume_path), so the current lineage is a
        single dir: the one holding the highest epoch-N.pt (same rule as
        submit_jobs.find_resume_info). Other dirs under the tag are stale
        fresh restarts and are ignored.

  AM:   AM+PIP/outputs/<tag>/tsptw_<size>/<run_name>_<YYYYmmddTHHMMSS>/
        epoch-N.pt (only the latest is kept), val_best.pt, and for PIP-D
        fsb_accuracy_bsf.pt. Resumes create a FRESH dir but keep the wandb
        run id (submit_jobs passes --wandb_id), so the lineage is the newest
        dir's wandb id plus every older dir sharing it; a fresh restart gets
        a new id, which cuts stale dirs off (same idea as LMask's script).

What is gathered per tag:

  last: the epoch-N.pt with the highest N across the lineage (epoch read from
        the filename; POMO epochs are 1-based, AM 0-based).
  best: the val-best model (POMO trained_model_val_best.pt / AM val_best.pt,
        monitor: instance feasibility rate first, feasible cost tie-break).
        Runs trained after the tracker fix write a val_best_meta.json
        (epoch/feas/score) next to the file and restore the tracker on resume,
        which makes selection exact and fills the manifest fields. For runs
        from BEFORE the fix the files are bare state_dicts and the tracker
        RESET on every resume: POMO overwrites its single file in place, so it
        holds the best since the last resume - a better pre-resume model is
        unrecoverable (flagged in the manifest). AM keeps one val_best.pt per
        attempt dir; when several exist without metas, the true best is
        resolved from the run's wandb history (val/infsb_rate, then
        val/feasible_cost; the best epoch maps to an attempt through the
        attempts' epoch-N.pt boundaries). If wandb is unreachable, the newest
        attempt's file is taken and flagged in the manifest.
  pip:  PIP-D variants only: fsb_accuracy_bsf.pt, the auxiliary decoder used
        at test time (default load_which_pip=train_fsb_bsf). Its stored
        fsb_accuracy IS carried across resumes, so ranking by it is exact
        (max across lineage dirs; ties resolve to the newest).

Copies go to flat files <dest>/<tag>_{best,last,pip}.pt plus a manifest.csv
recording source path, epoch, and score for provenance. Run dirs are
gitignored, so copying out is required for the checkpoints to be committable.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from submit_jobs import AM_DIR, DATASETS, POMO_DIR, VARIANTS, job_tag


def epoch_ckpts(run_dir):
    """[(epoch, path)] of the epoch-N.pt files in one run dir, ascending."""
    out = []
    for f in run_dir.iterdir():
        m = re.fullmatch(r"epoch-(\d+)\.pt", f.name)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def wandb_run_id(run_dir):
    """Extract the wandb run id from <run_dir>/wandb/run-<datetime>-<id>/."""
    wdir = run_dir / "wandb"
    if not wdir.is_dir():
        return None
    latest = wdir / "latest-run"
    cand = latest.resolve().name if latest.exists() else None
    if not cand or not cand.startswith("run-"):
        runs = sorted(wdir.glob("run-*"))
        cand = runs[-1].name if runs else None
    if cand and cand.startswith("run-"):
        return cand.rsplit("-", 1)[-1]
    return None


def pomo_lineage(run_root):
    """POMO resumes reuse their dir -> the lineage is the single dir holding
    the highest epoch-N.pt; other dirs are stale fresh restarts."""
    best = None
    dirs = sorted(d for d in run_root.iterdir() if d.is_dir())
    for d in dirs:
        eps = epoch_ckpts(d)
        if eps and (best is None or eps[-1][0] > best[0]):
            best = (eps[-1][0], d)
    if best is None:
        return []
    stale = [d.name for d in dirs if d != best[1]]
    if stale:
        print(
            f"  (ignoring stale run dir(s) of {run_root.name}: {', '.join(stale)})",
            file=sys.stderr,
        )
    return [best[1]]


def am_lineage(run_root):
    """AM attempt dirs of the current lineage, oldest -> newest (shared wandb
    id; the trailing _YYYYmmddTHHMMSS makes the name sort chronological)."""
    atts = sorted(
        (d for d in run_root.glob("*/*") if d.is_dir()),
        key=lambda d: d.name.rsplit("_", 1)[-1],
    )
    ids = {d: wandb_run_id(d) for d in atts}
    current = next((ids[d] for d in reversed(atts) if ids[d]), None)
    if current is None:
        return atts
    kept = [d for d in atts if ids[d] == current]
    stale = [d.name for d in atts if d not in kept]
    if stale:
        print(
            f"  (ignoring stale attempt(s) of {run_root.name}: "
            f"{', '.join(stale)} - different wandb id than current lineage "
            f"'{current}')",
            file=sys.stderr,
        )
    return kept


def read_meta(run_dir):
    """dict from <run_dir>/val_best_meta.json (epoch/feas/score-or-cost), or None."""
    p = run_dir / "val_best_meta.json"
    if not p.is_file():
        return None
    try:
        m = json.loads(p.read_text())
        return dict(
            epoch=m.get("epoch"),
            feas=m.get("feas"),
            score=m.get("score", m.get("cost")),
        )
    except Exception as exc:
        print(f"  ! unreadable {p}: {exc}", file=sys.stderr)
        return None


def am_wandb_history(newest_dir):
    """Validation history rows of the lineage's wandb run, or None."""
    run_id = wandb_run_id(newest_dir)
    if run_id is None:
        return None
    project = "pip_feasible"
    args_json = newest_dir / "args.json"
    if args_json.is_file():
        project = json.loads(args_json.read_text()).get("wandb_project") or project
    try:
        import wandb

        api = wandb.Api()
        rows = list(
            api.run(f"{api.default_entity}/{project}/{run_id}").scan_history(
                keys=["epoch", "val/infsb_rate", "val/feasible_cost"]
            )
        )
        return rows or None
    except Exception as exc:
        print(f"  ! wandb lookup for run '{run_id}' failed: {exc}", file=sys.stderr)
        return None


def am_pick_best(lineage):
    """The val_best.pt of the lineage as dict(path, epoch, feas, score, note).

    One candidate is unambiguous. With several (i.e. runs resumed before the
    tracker fix), each file is only the best of its own attempt window: if all
    candidates carry a val_best_meta.json, the max over the metas is exact;
    otherwise the global best epoch is recovered from the wandb history (same
    comparison as training: feasibility desc, cost asc, strict improvement)
    and mapped to the attempt whose epoch range covers it.
    """
    cands = [d for d in lineage if (d / "val_best.pt").is_file()]
    if not cands:
        return None
    metas = {d: read_meta(d) for d in cands}
    if len(cands) == 1:
        m = metas[cands[0]] or dict(epoch=None, feas=None, score=None)
        return dict(path=cands[0] / "val_best.pt", note="", **m)
    if all(metas.values()):
        # newest-first so exact ties resolve to the newest attempt's file
        chosen = max(
            reversed(cands), key=lambda d: (metas[d]["feas"], -metas[d]["score"])
        )
        return dict(
            path=chosen / "val_best.pt",
            note=f"resolved among {len(cands)} candidates via val_best_meta.json",
            **metas[chosen],
        )
    rows = am_wandb_history(lineage[-1])
    if rows is None:
        return dict(
            path=cands[-1] / "val_best.pt",
            epoch=None,
            feas=None,
            score=None,
            note=f"AMBIGUOUS: {len(cands)} val_best.pt candidates, wandb "
            "unreachable - took the newest attempt's file",
        )
    best = None  # (feas, cost, epoch)
    for r in rows:
        feas, cost = 100.0 - r["val/infsb_rate"], r["val/feasible_cost"]
        ep = int(r["epoch"])
        if best is None or feas > best[0] or (feas == best[0] and cost < best[1]):
            best = (feas, cost, ep)
    feas, cost, ep = best
    # attempts' epoch windows: attempt k covers (last epoch of k-1, last epoch of k]
    chosen = next(
        (d for d in cands if epoch_ckpts(d) and epoch_ckpts(d)[-1][0] >= ep), cands[-1]
    )
    return dict(
        path=chosen / "val_best.pt",
        epoch=ep,
        feas=feas,
        score=cost,
        note=f"resolved among {len(cands)} candidates via wandb history",
    )


def pick_fsb(lineage):
    """Best fsb_accuracy_bsf.pt of the lineage (highest stored fsb_accuracy;
    the bsf value is restored across resumes, so this ranking is exact)."""
    best = None  # (rank, path, epoch, acc)
    for d in lineage:
        p = d / "fsb_accuracy_bsf.pt"
        if not p.is_file():
            continue
        epoch = acc = None
        try:
            blob = torch.load(p, map_location="cpu", weights_only=False)
            epoch = blob.get("epoch")
            acc = blob.get("fsb_accuracy")
            acc = float(acc) if acc is not None else None
        except Exception as exc:
            print(f"  ! unreadable pip checkpoint {p}: {exc}", file=sys.stderr)
        rank = acc if acc is not None else float("-inf")
        if best is None or rank >= best[0]:
            best = (rank, p, epoch, acc)
    if best is None:
        return None
    _, p, epoch, acc = best
    return dict(
        path=p,
        epoch=epoch,
        feas=None,
        score=acc,
        note="score = pip decoder fsb_accuracy",
    )


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
    ap.add_argument("--dest", default="checkpoints", help="destination folder")
    ap.add_argument(
        "--git-add",
        action="store_true",
        help="git add the destination folder after copying",
    )
    args = ap.parse_args()

    dest = Path(args.dest)
    rows = []
    for ds_key, ds in DATASETS.items():
        if ds["size"] not in args.sizes:
            continue
        if ds["tws"] is not None and args.tws is not None and ds["tws"] not in args.tws:
            continue
        for variant in args.variants:
            tag = job_tag(ds_key, variant)
            family = VARIANTS[variant]["family"]
            base = POMO_DIR if family == "pomo" else AM_DIR
            run_root = base / ("results" if family == "pomo" else "outputs") / tag
            if not run_root.is_dir():
                print(f"- {tag}: no run dir ({run_root}) -> skip", file=sys.stderr)
                continue

            lineage = (
                pomo_lineage(run_root) if family == "pomo" else am_lineage(run_root)
            )
            if not lineage:
                print(
                    f"- {tag}: no checkpoints in any run dir -> skip", file=sys.stderr
                )
                continue

            # last: highest epoch-N.pt across the lineage (epoch from filename)
            eps = [e for d in lineage for e in epoch_ckpts(d)]
            last = None
            if eps:
                epoch, path = max(eps)
                last = dict(path=path, epoch=epoch, feas=None, score=None, note="")

            # best: val-best state_dict (see docstring for the resume caveat)
            if family == "pomo":
                p = lineage[0] / "trained_model_val_best.pt"
                best = None
                if p.is_file():
                    m = read_meta(lineage[0])
                    best = (
                        dict(path=p, note="", **m)
                        if m
                        else dict(
                            path=p,
                            epoch=None,
                            feas=None,
                            score=None,
                            note="best since last resume (pre-fix run, "
                            "POMO overwrites in place)",
                        )
                    )
            else:
                best = am_pick_best(lineage)

            picks = [("best", best), ("last", last)]
            if variant.endswith("pipd"):
                picks.append(("pip", pick_fsb(lineage)))

            dest.mkdir(parents=True, exist_ok=True)
            for kind, cand in picks:
                if cand is None:
                    print(f"- {tag}: no {kind} checkpoint found", file=sys.stderr)
                    continue
                dst = dest / f"{tag}_{kind}.pt"
                shutil.copy2(cand["path"], dst)
                rows.append(
                    dict(
                        run=tag,
                        kind=kind,
                        source=str(cand["path"]),
                        epoch=cand["epoch"],
                        feas=cand["feas"],
                        score=cand["score"],
                        note=cand["note"],
                    )
                )
                print(
                    f"{tag:18} {kind:4}: epoch={cand['epoch']} feas={cand['feas']} "
                    f"score={cand['score']}  <- {cand['path']}"
                )

    if not rows:
        sys.exit("Nothing gathered.")

    manifest = dest / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["run", "kind", "source", "epoch", "feas", "score", "note"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote manifest: {manifest}")

    if args.git_add:
        subprocess.run(["git", "add", str(dest)], check=True)
        print(f"git add {dest} done -- review with 'git status', then commit.")


if __name__ == "__main__":
    main()
