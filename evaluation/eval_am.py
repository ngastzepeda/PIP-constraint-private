"""AM+PIP evaluation worker.

Runs the gathered AM-family checkpoints (variants: am / am_pip / am_pipd)
through the AM+PIP inference stack on the AMAI TSPTW test instances and
returns, per instance, whether a feasible tour was found and its best feasible
length. Launched by eval_checkpoints.py in its own subprocess (POMO+PIP and
AM+PIP share top-level module names).

The gathered checkpoints are flat weight files with no run-dir args.json, so
the model is constructed directly from the deterministic per-variant training
flags (submit_jobs.py) instead of AM+PIP's load_model. All models were trained
with --include_service_time and the repo defaults (embedding_dim/hidden_dim
128, 3 encoder layers, batch norm, tanh clipping 10, BCEWithLogitsLoss ->
sigmoid=False).

Masking at inference (nets/attention_model._get_log_p): when generate_PI_mask
is set the analytic 1-step preventative-infeasibility mask is applied. For
am_pipd the PIP-decoder SL head is built (so its weights load) and evaluated,
but sample_many decodes with the analytic mask — the SL head does not change
decoding — so am_pip and am_pipd use the same masking, matching AM+PIP's own
eval.py.

Decode: sampling with `width` samples per instance (AM+PIP eval.py default
1280); the best feasible sample per instance is reported (sample_many). Use
--decode greedy for a single-shot greedy rollout instead.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

AM_DIR = common.REPO_ROOT / "AM+PIP"

# generate_PI_mask, pip_decoder per variant
VARIANT_CFG = {
    "am": dict(gen=False, pipd=False),
    "am_pip": dict(gen=True, pipd=False),
    "am_pipd": dict(gen=True, pipd=True),
}


def build_model(AttentionModel, problem, variant, device):
    cfg = VARIANT_CFG[variant]
    model = AttentionModel(
        128,  # embedding_dim
        128,  # hidden_dim
        problem,
        generate_PI_mask=cfg["gen"],
        pip_decoder=cfg["pipd"],
        sigmoid=False,  # sl_loss=BCEWithLogitsLoss
        n_encode_layers=3,
        decision_boundary=0.5,
        mask_inner=True,
        mask_logits=True,
        normalization="batch",
        tanh_clipping=10.0,
        checkpoint_encoder=False,
        shrink_size=None,
        include_service_time=True,
    ).to(device)
    return model


def load_weights(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:  # epoch-N.pt full checkpoint
        sd = sd["model"]
    # val_best.pt is already a bare state_dict; strict=False tolerates the aux
    # baseline/decoder tensors that differ between save formats.
    model.load_state_dict({**model.state_dict(), **sd}, strict=False)


def evaluate_job(
    AttentionModel, problem, move_to, DataLoader, job, device,
    decode, width, eval_batch_size, max_calc_batch_size, softmax_temp,
    test_episodes,
):
    """-> (feas[bool, N], cost_norm[float, N]) for one checkpoint."""
    model = build_model(AttentionModel, problem, job["variant"], device)
    load_weights(model, job["ckpt"], device)
    model.eval()
    model.set_decode_type("greedy" if decode == "greedy" else "sampling", temp=softmax_temp)

    if decode == "greedy":
        batch_rep, iter_rep = 1, 1
    elif width * eval_batch_size > max_calc_batch_size:
        batch_rep, iter_rep = max_calc_batch_size, width // max_calc_batch_size
    else:
        batch_rep, iter_rep = width, 1

    dataset = problem.make_dataset(
        filename=job["test_file"], num_samples=test_episodes, offset=0
    )
    loader = DataLoader(dataset, batch_size=eval_batch_size)

    feas_parts, cost_parts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_to(batch, device)
            _seq, costs, ins_infsb, _sol_infsb = model.sample_many(
                batch, batch_rep=batch_rep, iter_rep=iter_rep
            )
            costs = costs.detach().cpu().numpy().astype(float)
            feas = ins_infsb.detach().cpu().numpy().astype(int) == 0
            cost = np.where(feas, costs, np.nan)  # costs is inf where infeasible
            feas_parts.append(feas)
            cost_parts.append(cost)
    return np.concatenate(feas_parts), np.concatenate(cost_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--bks_csv", default=str(common.DEFAULT_BKS_CSV))
    ap.add_argument("--decode", default="sampling", choices=["sampling", "greedy"])
    ap.add_argument("--width", type=int, default=1280, help="samples/instance (sampling)")
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--max_calc_batch_size", type=int, default=1280)
    ap.add_argument("--softmax_temp", type=float, default=1.0)
    ap.add_argument("--test_episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    if args.decode == "sampling" and args.max_calc_batch_size < args.width:
        print(
            f"WARNING: --max_calc_batch_size ({args.max_calc_batch_size}) < "
            f"--width ({args.width}) makes AM+PIP/utils/functions.py::sample_many "
            f"chunk (iter_rep>1). That path has a pre-existing vendored bug: the "
            f"per-chunk infeasibility mask isn't accumulated across chunks (only "
            f"costs/pis are), so the 2nd+ chunk crashes with a shape-mismatch "
            f"IndexError. Keep --max_calc_batch_size >= --width to avoid it "
            f"(reduce --eval_batch_size instead to lower peak memory).",
            file=sys.stderr,
        )

    with open(args.jobs) as f:
        jobs = json.load(f)

    os.chdir(AM_DIR)
    sys.path.insert(0, str(AM_DIR))
    from nets.attention_model import AttentionModel  # noqa: E402
    from utils.functions import load_problem, move_to, seed_everything  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402

    resolved = common.pick_device(args.device)
    device = torch.device({"cuda": "cuda:0", "mps": "mps", "cpu": "cpu"}[resolved])
    problem = load_problem("tsptw")

    width = 0 if args.decode == "greedy" else args.width
    rows = []
    for job in jobs:
        seed_everything(args.seed)
        label = job["label"]
        try:
            start = time.time()
            feas, cost_norm = evaluate_job(
                AttentionModel, problem, move_to, DataLoader, job, device,
                args.decode, width, args.eval_batch_size,
                args.max_calc_batch_size, args.softmax_temp, args.test_episodes,
            )
            inference_time = time.time() - start
        except Exception as exc:
            print(f"[eval_am] FAILED {label}: {exc}", file=sys.stderr, flush=True)
            continue
        bks = common.load_bks(job["size"], args.bks_csv, len(feas))
        dom = common.domain_size(job["size"])
        metrics = common.aggregate(feas, cost_norm, bks, dom)
        row = common.make_row(job, metrics, inference_time)
        rows.append(row)
        print(
            f"[eval_am] {label} ({job['kind']}): feas={row['feas_count']} "
            f"cost={row['cost']:.3f} gap={row['gap_bks']:.3f}% "
            f"t={row['inference_time']:.1f}s",
            flush=True,
        )

    with open(args.out, "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    main()
