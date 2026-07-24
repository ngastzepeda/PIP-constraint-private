"""POMO+PIP benchmark-evaluation worker.

Runs the gathered POMO-family checkpoints on a prebuilt benchmark model-input
npz (bench_common.build_model_npz) and, per instance, extracts the TOUR the
model proposes. Feasibility/objective are recomputed downstream from the exact
distance matrix (bench_common.check_tsptw_matrix), so this worker only produces
node orderings. It reproduces eval_pomo.py's inference settings (aug_factor=8,
argmax, pomo_start=False) and the tour-selection logic from
Tester._test_one_batch's output_best_tour branch.

Tour selection per instance: the best augmentation among those the model
considered feasible (max masked reward); if the model found none feasible, the
min-cost augmentation overall -- so there is always a concrete tour to check.

Launched by eval_benchmarks.py in its own subprocess (POMO+PIP and AM+PIP share
top-level module names).
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

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))               # evaluation/  -> common, eval_pomo
sys.path.insert(0, str(Path(__file__).resolve().parent))  # evaluation/benchmarks
import common  # noqa: E402
import eval_pomo  # noqa: E402  (reuse param builders / setup_device / VARIANT_CFG)
from submit_jobs import DATASETS, VARIANTS  # noqa: E402

POMO_DIR = common.REPO_ROOT / "POMO+PIP"


def rollout_tours(tester, env, data, aug_factor):
    """Full POMO rollout over one batch -> list of tours (customer id order,
    depot stripped), one per instance. Mirrors Tester._test_one_batch up to the
    reward/feasibility reshape, then reuses its output_best_tour selection."""
    model = tester.model
    batch_size = data[-1].size(0)
    model.eval()
    with torch.no_grad():
        env.load_problems(batch_size, problems=data, aug_factor=aug_factor, normalize=True)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            use_pred = bool(tester.model_params["pip_decoder"]
                            and tester.tester_params["use_predicted_PI_mask"])
            selected, _prob = model(
                state, pomo=tester.env_params["pomo_start"],
                tw_end=env.node_tw_end, use_predicted_PI_mask=use_pred, no_sigmoid=True)
            state, reward, done, infeasible = env.step(
                selected, generate_PI_mask=tester.model_params["generate_PI_mask"],
                pip_step=tester.tester_params["pip_step"])

    # (aug, batch, pomo); pomo_size == 1 here
    aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
    infeasible = infeasible.reshape(aug_factor, batch_size, env.pomo_size)
    aug_feasible = (infeasible == False).any(dim=0).any(dim=-1)          # (batch,)

    reward_masked = aug_reward.masked_fill(infeasible, -1e10)
    # best augmentation index per instance: best feasible reward, or (if none
    # feasible) the min-cost (=max raw reward) augmentation.
    best_feas_idx = reward_masked.max(dim=0)[1].squeeze(-1)              # (batch,)
    best_raw_idx = aug_reward.max(dim=0)[1].squeeze(-1)                  # (batch,)
    best_idx = torch.where(aug_feasible, best_feas_idx, best_raw_idx)    # (batch,)

    route_list = env.selected_node_list.reshape(aug_factor, batch_size, -1)
    route_list = route_list[best_idx, torch.arange(batch_size)]          # (batch, steps)

    tours, feas_flags = [], []
    for b in range(batch_size):
        tours.append([int(x) for x in route_list[b].tolist() if int(x) != 0])
        feas_flags.append(bool(aug_feasible[b]))
    return tours, feas_flags


def evaluate_job(Tester, job, size, device, batch_size, aug_factor, test_episodes,
                 npz, problem_size):
    env_params = eval_pomo._env_params(size, device)
    # The env builds problem_size-shaped tensors (eye matrices, neighbour index),
    # so problem_size must equal the benchmark instance's NODE count, not the
    # checkpoint's trained size. The attention weights themselves are size-
    # agnostic, so a checkpoint trained at one size runs on another.
    env_params["problem_size"] = problem_size
    model_params = eval_pomo._model_params(job["variant"], device)
    tester_params = eval_pomo._tester_params(
        {"variant": job["variant"], "ckpt": job["ckpt"], "test_file": npz},
        batch_size, aug_factor, test_episodes)
    args = SimpleNamespace(problem="TSPTW", checkpoint=job["ckpt"], device=device,
                           pip_checkpoint=None)
    tester = Tester(args=args, env_params=env_params, model_params=model_params,
                    tester_params=tester_params)
    tester.model = tester.model.to(device)
    env = tester.envs[0](**env_params)

    tours, feas_flags, runtimes = [], [], []
    episode = 0
    while episode < test_episodes:
        bs = min(batch_size, test_episodes - episode)
        data = env.load_dataset(npz, offset=episode, num_samples=bs)
        if data[0].size(0) == 0:
            break
        bs = data[0].size(0)
        t0 = time.time()
        btours, bfeas = rollout_tours(tester, env, data, aug_factor)
        dt = time.time() - t0
        tours.extend(btours)
        feas_flags.extend(bfeas)
        runtimes.extend([dt / bs] * bs)
        episode += bs
    return tours, feas_flags, runtimes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jobs", required=True, help="JSON: list of {variant, kind, ckpt, size}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--problem_size", type=int, required=True,
                    help="node count of the benchmark instances (depot + customers)")
    ap.add_argument("--test_batch_size", type=int, default=500)
    ap.add_argument("--aug_factor", type=int, default=8, choices=[1, 8])
    ap.add_argument("--test_episodes", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    with open(args.jobs) as f:
        jobs = json.load(f)

    os.chdir(POMO_DIR)
    sys.path.insert(0, str(POMO_DIR))
    from Tester import Tester  # noqa: E402
    from utils import seed_everything  # noqa: E402

    device = eval_pomo.setup_device(common.pick_device(args.device))

    out = {}
    for job in jobs:
        seed_everything(args.seed)
        variant, kind, size = job["variant"], job["kind"], job["size"]
        label = f"{VARIANTS[variant]['label']}/{kind}"
        try:
            tours, feas, runtimes = evaluate_job(
                Tester, job, size, device, args.test_batch_size, args.aug_factor,
                args.test_episodes, args.npz, args.problem_size)
        except Exception as exc:
            print(f"[eval_pomo_bench] FAILED {label}: {exc}", file=sys.stderr, flush=True)
            import traceback; traceback.print_exc()
            continue
        out[VARIANTS[variant]["label"]] = dict(
            variant=variant, mode=kind, tours=tours,
            model_feasible=feas, runtime=runtimes)
        print(f"[eval_pomo_bench] {label}: {len(tours)} tours "
              f"(model-feasible {sum(1 for x in feas if x)}/{len(tours)})", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
