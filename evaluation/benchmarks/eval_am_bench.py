"""AM+PIP benchmark-evaluation worker.

Runs the gathered AM-family checkpoints on a prebuilt benchmark model-input npz
(built by bench_common.build_model_npz from the TSPTW libraries) and, per
instance, extracts the TOUR the model proposes. It does NOT judge feasibility or
objective itself -- the orchestrator recomputes both from the exact distance
matrix (bench_common.check_tsptw_matrix). This mirrors eval_am.py's inference
stack (same model build/weights) but returns node orderings instead of the
model's own Euclidean costs.

Tour selection per instance (`sample_best`): among the `width` sampled rollouts,
prefer the min-cost one the model considered feasible under its (MDS-Euclidean)
masking; if the model found none feasible, fall back to the min-cost rollout
overall -- so there is always a concrete, model-chosen tour to check. `greedy`
decode returns the single greedy rollout.

Launched by eval_benchmarks.py in its own subprocess (POMO+PIP and AM+PIP share
top-level module names).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))               # evaluation/  -> common, eval_am
sys.path.insert(0, str(Path(__file__).resolve().parent))  # evaluation/benchmarks -> bench_common
import common  # noqa: E402
import eval_am  # noqa: E402  (reuse build_model / load_weights / VARIANT_CFG)

AM_DIR = common.REPO_ROOT / "AM+PIP"


def _strip_depot(pi_row):
    """(seq,) tensor of node ids -> python list of customer ids (depot 0 removed,
    order preserved)."""
    return [int(x) for x in pi_row.tolist() if int(x) != 0]


def sample_best(model, problem, do_batch_rep, F, batch, decode, batch_rep, iter_rep):
    """-> (tours[list[list[int]]], model_feasible[list[bool]]) for one batch.

    Mirrors utils.functions.sample_many's rollout/reshape, but selects per
    instance the min-cost FEASIBLE rollout when one exists, else the min-cost
    rollout overall (sample_many instead returns a meaningless identity tour when
    nothing is feasible)."""
    if decode == "greedy":
        with torch.no_grad():
            _cost, _ll, pi = model(batch, return_pi=True)
        return [_strip_depot(pi[i]) for i in range(pi.size(0))], None

    embeddings = model.embedder(model._init_embed(batch))[0]
    inner = (batch, embeddings)
    inner = do_batch_rep(inner, batch_rep)

    costs, pis, masks = [], [], []
    for _ in range(iter_rep):
        _log_p, pi = model._inner(*inner)
        cost, mask = problem.get_costs(inner[0], pi)
        if isinstance(cost, list):  # tsptw returns [cost, timeout, timeout_nodes]
            cost = cost[0]
        costs.append(cost.view(batch_rep, -1).t())                 # (B, batch_rep)
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))  # (B, batch_rep, seq)
        masks.append(mask.sum(-1).view(batch_rep, -1).t())         # (B, batch_rep); 0 == feasible
    max_len = max(p.size(-1) for p in pis)
    pis = torch.cat([F.pad(p, (0, max_len - p.size(-1))) for p in pis], 1)  # (B, S, seq)
    costs = torch.cat(costs, 1)   # (B, S)
    masks = torch.cat(masks, 1)   # (B, S)

    tours, model_feasible = [], []
    for i in range(costs.size(0)):
        feas = masks[i] == 0
        if feas.any():
            j = costs[i][feas].argmin()
            sel = pis[i][feas][j]
            model_feasible.append(True)
        else:
            j = costs[i].argmin()
            sel = pis[i][j]
            model_feasible.append(False)
        tours.append(_strip_depot(sel))
    return tours, model_feasible


def evaluate_job(model, problem, move_to, DataLoader, do_batch_rep, F, npz, device,
                 decode, width, eval_batch_size, max_calc_batch_size, softmax_temp,
                 test_episodes):
    model.eval()
    model.set_decode_type("greedy" if decode == "greedy" else "sampling", temp=softmax_temp)
    if decode == "greedy":
        batch_rep, iter_rep = 1, 1
    elif width * eval_batch_size > max_calc_batch_size:
        batch_rep, iter_rep = max_calc_batch_size, width // max_calc_batch_size
    else:
        batch_rep, iter_rep = width, 1

    dataset = problem.make_dataset(filename=npz, num_samples=test_episodes, offset=0)
    loader = DataLoader(dataset, batch_size=eval_batch_size)

    tours, feas_flags, runtimes = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_to(batch, device)
            bs = batch["loc"].size(0) if isinstance(batch, dict) else batch.size(0)
            t0 = time.time()
            btours, bfeas = sample_best(
                model, problem, do_batch_rep, F, batch, decode, batch_rep, iter_rep)
            if device.type != "cpu":
                torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
            dt = time.time() - t0
            tours.extend(btours)
            feas_flags.extend(bfeas if bfeas is not None else [None] * bs)
            runtimes.extend([dt / bs] * bs)
    return tours, feas_flags, runtimes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="benchmark model-input npz")
    ap.add_argument("--jobs", required=True, help="JSON: list of {variant, kind, ckpt}")
    ap.add_argument("--out", required=True, help="JSON to write per-variant tours")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--decode", default="sampling", choices=["sampling", "greedy"])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--max_calc_batch_size", type=int, default=1280)
    ap.add_argument("--softmax_temp", type=float, default=1.0)
    ap.add_argument("--test_episodes", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    with open(args.jobs) as f:
        jobs = json.load(f)

    os.chdir(AM_DIR)
    sys.path.insert(0, str(AM_DIR))
    from nets.attention_model import AttentionModel  # noqa: E402
    from utils.functions import load_problem, move_to, seed_everything, do_batch_rep  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402
    import torch.nn.functional as F  # noqa: E402

    resolved = common.pick_device(args.device)
    device = torch.device({"cuda": "cuda:0", "mps": "mps", "cpu": "cpu"}[resolved])
    problem = load_problem("tsptw")
    width = 0 if args.decode == "greedy" else args.width

    out = {}
    for job in jobs:
        seed_everything(args.seed)
        variant, kind = job["variant"], job["kind"]
        label = f"{eval_am.common.VARIANTS[variant]['label']}/{kind}"
        try:
            model = eval_am.build_model(AttentionModel, problem, variant, device)
            eval_am.load_weights(model, job["ckpt"], device)
            tours, feas, runtimes = evaluate_job(
                model, problem, move_to, DataLoader, do_batch_rep, F, args.npz, device,
                args.decode, width, args.eval_batch_size, args.max_calc_batch_size,
                args.softmax_temp, args.test_episodes)
        except Exception as exc:  # a bad ckpt must not kill the whole run
            print(f"[eval_am_bench] FAILED {label}: {exc}", file=sys.stderr, flush=True)
            import traceback; traceback.print_exc()
            continue
        out[eval_am.common.VARIANTS[variant]["label"]] = dict(
            variant=variant, mode=kind, tours=tours,
            model_feasible=feas, runtime=runtimes)
        print(f"[eval_am_bench] {label}: {len(tours)} tours "
              f"(model-feasible {sum(1 for x in feas if x)}/{len(tours)})", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
