"""POMO+PIP evaluation worker.

Runs the gathered POMO-family checkpoints (variants: pomo / pip / pipd) through
the POMO+PIP inference stack on the AMAI TSPTW test instances and returns, per
instance, whether a feasible tour was found and its best feasible length. The
orchestrator (eval_checkpoints.py) launches this in its own subprocess because
POMO+PIP and AM+PIP share top-level module names (e.g. `utils`).

Inference settings reproduce the training-time validation that selected
best.ckpt (Trainer._val_one_batch): aug_factor=8, eval_type="argmax",
pomo_start=False. With pomo_start off and argmax decoding the pomo rollouts are
identical copies, so pomo_size=1 gives the same feasibility/cost as the full
multistart at a fraction of the cost. include_service_time and
check_depot_return match the AMAI/LMask semantics the models were trained with.

Per-variant masking (see summary.md; matches Trainer validation):
  pomo : no PIP mask.
  pip  : env 1-step preventative-infeasibility mask (generate_PI_mask).
  pipd : generate_PI_mask + the model's own PIP-decoder head predicting the
         mask (pip_decoder, use_predicted_PI_mask; W_kv_sl/W_out_sl so the SL
         weights load). The separate '..._pip.pt' aux checkpoint is not needed
         here — the SL head lives in the main model.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# common.py (torch-free) imported before we put POMO+PIP on the path, so its
# submit_jobs import resolves against the repo root, not a POMO module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from submit_jobs import DATASETS  # noqa: E402

POMO_DIR = common.REPO_ROOT / "POMO+PIP"

# generate_PI_mask, pip_decoder (+ SL weights), use_predicted_PI_mask per variant
VARIANT_CFG = {
    "pomo": dict(gen=False, pipd=False, sl=False, use_pred=False),
    "pip": dict(gen=True, pipd=False, sl=False, use_pred=False),
    "pipd": dict(gen=True, pipd=True, sl=True, use_pred=True),
}


def _model_params(variant, device):
    cfg = VARIANT_CFG[variant]
    return {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": 6,
        "decoder_layer_num": 1,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "norm": "instance",
        "norm_loc": "norm_last",
        "eval_type": "argmax",
        "problem": "TSPTW",
        "include_service_time": True,
        "pip_decoder": cfg["pipd"],
        "tw_normalize": True,
        "decision_boundary": 0.5,
        "detach_from_encoder": False,
        "W_q_sl": False,
        "W_out_sl": cfg["sl"],
        "W_kv_sl": cfg["sl"],
        "use_ninf_mask_in_sl_MHA": False,
        "generate_PI_mask": cfg["gen"],
        "device": device,
    }


def _env_params(size, device):
    return {
        "problem_size": DATASETS[size]["problem_size"],
        "pomo_size": 1,  # redundant under argmax + pomo_start=False
        "hardness": "hard",  # unused for .npz test sets
        "pomo_start": False,
        "k_sparse": 500,
        "check_depot_return": True,
        "device": device,
    }


def _tester_params(job, batch_size, aug_factor, test_episodes):
    cfg = VARIANT_CFG[job["variant"]]
    return {
        "checkpoint": job["ckpt"],
        "test_episodes": test_episodes,
        "test_batch_size": batch_size,
        "sample_size": 1,
        "aug_factor": aug_factor,
        "aug_batch_size": batch_size,
        "test_set_path": job["test_file"],
        "test_set_opt_sol_path": None,
        "fsb_dist_only": True,
        "use_predicted_PI_mask": cfg["use_pred"],
        "lazy_pip_model": False,
        "pip_step": 1,
        "k_sparse": 500,
        "output_best_tour_path": None,
    }


def evaluate_job(Tester, job, device, batch_size, aug_factor, test_episodes):
    """-> (feas[bool, N], cost_norm[float, N]) for one checkpoint."""
    env_params = _env_params(job["size"], device)
    model_params = _model_params(job["variant"], device)
    tester_params = _tester_params(job, batch_size, aug_factor, test_episodes)
    args = SimpleNamespace(
        problem="TSPTW", checkpoint=job["ckpt"], device=device, pip_checkpoint=None
    )
    tester = Tester(
        args=args,
        env_params=env_params,
        model_params=model_params,
        tester_params=tester_params,
    )
    # Under cuda/cpu the default tensor type (set in setup_device) already
    # places model params correctly at construction time; under mps there is
    # no such trick, so move explicitly (a no-op when already on device).
    tester.model = tester.model.to(device)
    env = tester.envs[0](**env_params)

    feas_parts, cost_parts = [], []
    episode = 0
    while episode < test_episodes:
        bs = min(batch_size, test_episodes - episode)
        data = env.load_dataset(job["test_file"], offset=episode, num_samples=bs)
        if data[0].size(0) == 0:  # dataset exhausted early
            break
        bs = data[0].size(0)
        # _test_one_batch -> (.., .., no_aug_score, aug_score, .., .., .., aug_feasible)
        out = tester._test_one_batch(data, env)
        aug_score, aug_feasible = out[3], out[7]
        feas = aug_feasible.detach().cpu().numpy().astype(bool)
        cost = aug_score.detach().cpu().numpy().astype(float)  # -best feasible reward
        cost[~feas] = np.nan  # aug_score is a large sentinel where infeasible
        feas_parts.append(feas)
        cost_parts.append(cost)
        episode += bs
    return np.concatenate(feas_parts), np.concatenate(cost_parts)


def setup_device(device_str):
    """device_str is an already-resolved (available) choice: cpu/cuda/mps.

    Match test.py: POMO env code relies on the default tensor type for
    device-less tensor creation. There is no 'torch.mps.FloatTensor' legacy
    tensor type, so the mps branch instead relies on the explicit "device" key
    threaded into _model_params/_env_params (SINGLEModel/TSPTWEnv already
    support that key) plus the explicit .to(device) calls added around the
    model and env.load_problems() output.
    """
    if device_str == "cuda":
        torch.cuda.set_device(0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        return torch.device("cuda", 0)
    if device_str == "mps":
        # No 'torch.mps.FloatTensor' legacy type exists, so use the modern
        # replacement instead: it covers the env's device-less factory calls
        # (torch.zeros/arange/eye/...), just not the legacy torch.Tensor()
        # constructor -- that gap is closed separately (see load_problems).
        torch.set_default_device("mps")
        return torch.device("mps")
    torch.set_default_tensor_type("torch.FloatTensor")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON file: list of job dicts")
    ap.add_argument("--out", required=True, help="JSON file to write result rows")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--bks_csv", default=str(common.DEFAULT_BKS_CSV))
    ap.add_argument("--test_batch_size", type=int, default=500)
    ap.add_argument("--aug_factor", type=int, default=8, choices=[1, 8])
    ap.add_argument("--test_episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    with open(args.jobs) as f:
        jobs = json.load(f)

    os.chdir(POMO_DIR)
    sys.path.insert(0, str(POMO_DIR))
    from Tester import Tester  # noqa: E402
    from utils import seed_everything  # noqa: E402

    device = setup_device(common.pick_device(args.device))
    rows = []
    for job in jobs:
        seed_everything(args.seed)  # per-checkpoint determinism, as in test.py
        label = job["label"]
        try:
            start = time.time()
            feas, cost_norm = evaluate_job(
                Tester, job, device, args.test_batch_size,
                args.aug_factor, args.test_episodes,
            )
            inference_time = time.time() - start
        except Exception as exc:  # a bad/incompatible ckpt must not kill the run
            print(f"[eval_pomo] FAILED {label}: {exc}", file=sys.stderr, flush=True)
            continue
        bks = common.load_bks(job["size"], args.bks_csv, len(feas))
        dom = common.domain_size(job["size"])
        metrics = common.aggregate(feas, cost_norm, bks, dom)
        row = common.make_row(job, metrics, inference_time)
        rows.append(row)
        print(
            f"[eval_pomo] {label} ({job['kind']}): feas={row['feas_count']} "
            f"cost={row['cost']:.3f} gap={row['gap_bks']:.3f}% "
            f"t={row['inference_time']:.1f}s",
            flush=True,
        )

    with open(args.out, "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    main()
