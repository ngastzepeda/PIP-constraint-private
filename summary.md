# PIP-constraint as a baseline on the AMAI TSPTW instances

This fork adapts the NeurIPS'24 PIP-constraint repository (Bi et al., *Learning to
Handle Complex Constraints for Vehicle Routing Problems*) so that two of its PIP
implementations — **POMO+PIP** and **AM+PIP** — can be trained and evaluated on the
AMAI2025 TSPTW instances (`data/feas_tsptw/`), under semantics consistent with the
AMAI2025 and LMask codebases. (The upstream repo also ships a third implementation,
`GFACS+PIP/`, merged from upstream July 2026 but left unmodified and not used as a
baseline.) This document records every deviation from the
upstream repo and the experiment settings, as a basis for the paper's baseline
description.

## 1. Data

Instances are the AMAI `.npz` files (identical to `AMAI2025/data/instances/tsptw/`):

| dataset | nodes (incl. depot) | files |
|---|---|---|
| `n20_amai` | 21 | `train_100k.npz`, `val_1k.npz`, `test_1k.npz` |
| `n50_amai` | 51 | " |
| `n100_amai_sw` | 101 (small windows) | " |
| `n100_amai_mw` | 101 (medium windows) | " |

Instance-defining arrays: `locs` (coords in [0, 100], depot = node 0 at (50, 50)),
`time_windows` (minutes, depot [0, 1440]), `service_times` (non-zero: 5/10/20 min),
`max_loc` (= 100), `n_vehicles` (= 1). The remaining arrays (`visited`,
`current_time`, ... ) are initial RL-state fields and are ignored.

**Normalization** (identical to AMAI/LMask): `locs`, `time_windows` **and**
`service_times` are all divided by `max_loc`, so coordinates and time share one unit;
travel time = Euclidean distance at speed 1. The depot time window from the data is
kept (upstream PIP recomputes/loosens it — see §3).

## 2. Problem semantics (aligned with AMAI/LMask)

* Arrival at node *j*: `arrival = max(t + dist(i,j), tw_start_j)`; a time-window
  violation is measured **on arrival, before service**; then
  `t = arrival + service_j`.
* The vehicle must be **back at the depot before the depot's `tw_end`**; a late
  return counts as one more violated node and makes the solution infeasible
  (upstream PIP never checks the return leg).
* Feasibility, reward shaping (Lagrangian timeout penalties) and PIP's preventative
  infeasibility masking all operate on these dynamics, with service times included
  in every lookahead.

**Verification**: 200 known-feasible Gurobi tours (from
`AMAI2025/data/baselines/gurobi/tsptw/n50_amai_test_1k_feasibility/`) were replayed
through both adapted environments: 200/200 feasible (zero timeout) and tour lengths
match Gurobi's ObjVal/100 to ≤ 2e-6 — in the POMO env (`TSPTWEnv.step`, with
`check_depot_return`) and in the AM cost function (`TSPTW.get_costs`).

## 3. Changes vs. the upstream repository

Upstream behavior is preserved when the new flags are not passed, with the
exceptions marked *(default changed)*.

### Both implementations
* **`.npz` dataset support** with the AMAI schema and ÷`max_loc` normalization
  (POMO: `TSPTWEnv.load_dataset`; AM: `TSPTWDataset`). Pre-normalized data
  (coords in [0, 1]) skips the repo's own ÷100 normalization *and* its depot-window
  recomputation, so the data's depot window is binding.
* **Fixed-dataset training** (`--train_set_path`): batches are drawn from the fixed
  training file (shuffled pass, reshuffle when exhausted) instead of the repos'
  on-the-fly instance generation. Upstream trains on a fresh instance stream and
  ignores any training file.
* **Service times** as part of the dynamics (upstream POMO carries them but its
  generator forces them to 0; upstream AM had **no service-time concept at all** —
  the dataset row was `[x, y, tw_start, tw_end]`).
* **`--include_service_time`**: adds service time as a 5th node input feature to the
  policy network (both models previously embedded only `x, y, tw_start, tw_end`,
  i.e. were blind to service times; the AMAI models see them). Off by default so
  upstream pretrained checkpoints still load.
* **Bounded checkpointing** *(default changed)*: only the latest full
  `epoch-N.pt` (model+optimizer state, the resume anchor) is kept — older ones are
  deleted after each save; `--keep_all_checkpoints` restores upstream behavior
  (which kept every checkpoint). Caveat: PIP-D's non-default
  `--load_which_pip last_epoch` mode would need the deleted files; the default
  (`train_fsb_bsf`) is unaffected.
* **Best-model selection** *(default changed)*: the best-by-validation model
  (`trained_model_val_best.pt` / `val_best.pt`, weights only) is chosen by
  **instance-level feasibility rate first, feasible cost as tie-break** — matching
  LMask's `monitor: val/ins_feas_rate`. Upstream POMO used feasible cost only.
* Robustness fixes: CPU fallback when CUDA is unavailable; graceful stop when
  val/test files are smaller than the requested episode count; optimality-gap
  computation only runs when a solution file actually exists.

### POMO+PIP specifics
* `--check_depot_return` (env flag): return-leg timeout + infeasibility at rollout
  end (`TSPTWEnv.step`).
* `--val_dataset` accepts direct file paths; `test.py`/`Tester.py` accept `.npz`
  test sets; `problem_size` counts the depot (AMAI "n50" → `--problem_size 51`),
  `pomo_size` = number of customers.

### AM+PIP specifics
* Service times added to `get_costs`, `StateTSPTWInt.update`, and the PI-mask
  two-step lookahead; the dataset row is now `[x, y, tw_start, tw_end, service]`
  (5 columns; zero service for upstream pkl/generated data → identical behavior).
* Depot-return timeout added **unconditionally** in `get_costs` (a no-op for
  upstream data, whose recomputed depot window is non-binding for
  otherwise-feasible tours).
* The rollout baseline's comparison dataset is drawn from the fixed training file
  (random slice) instead of the random generator when `--train_set_path` is set.
* `--n_epochs` is now the **absolute** target epoch *(default changed for resumed
  runs only)*: a resumed run continues up to `n_epochs` instead of training
  `n_epochs` further, so PIP-D's epoch schedules stay consistent across restarts.
* Bug fixes: rollout-baseline resume crashed for tsptw (dict-indexing a tensor);
  `torch.load` needed `weights_only=False` on torch ≥ 2.6; two hardcoded `.cuda()`
  calls broke CPU execution; `--val_dataset`/`--val_solution_path` were
  unconditionally overwritten by the hardness defaults.

## 4. Experiment settings

Grid: **6 variants × 4 datasets = 24 jobs**
(variants: POMO\*, POMO\*+PIP, POMO\*+PIP-D, AM\*, AM\*+PIP, AM\*+PIP-D;
POMO\*/AM\* = base models with the Lagrangian timeout reward, i.e. the repo's
starred baselines).

Budgets and epoch counters are matched to the AMAI experiments: in every codebase
one epoch = one full pass over the fixed 100k training instances (POMO:
`--train_episodes 100000`), trained for **100 epochs (n20)** and **500 epochs
(n50, n100)** — i.e. 10M / 50M training instances total.

| | n20 | n50 / n100 |
|---|---|---|
| epochs (both families, = AMAI max_epochs) | 100 | 500 |
| total training instances | 10M | 50M |

Common settings:
* **Constant lr 1e-4, no scheduling** — consistent with the AMAI experiments (which
  used no lr scheduler). AM's default is already constant (`lr_decay 1.0`); POMO's
  default MultiStepLR milestone (epoch 9001, i.e. 90% of upstream's 10000-epoch
  schedule) never fires within ≤500 epochs, so no override is needed. Deviation
  from upstream PIP, which decays ×0.1 at 90% of training.
* Validation on `val_1k.npz` (1000 instances) every epoch — same cadence as the
  AMAI runs (POMO: `--validation_interval 1`; AM: built-in).
* Checkpoint every epoch (POMO: `--model_save_interval 1`;
  AM: `--checkpoint_epochs 1`), keeping only the latest + best.
* `--include_service_time`, `--check_depot_return` (POMO; built-in for AM),
  seed 2023 (PIP's default), all other hyperparameters at the repos' defaults.
* POMO: `--problem_size` = customers+1 (21/51/101), `--pomo_size` = customers
  (20/50/100). AM: `--graph_size` = total nodes (21/51/101), batch size 512
  (AM default).
* PIP-D schedules (all epoch-denominated, rescaled to sample-equivalent positions):
  POMO's defaults are stated in 10k-sample epochs (simulation stop / update interval /
  update epochs / grow-up = 200/1000/50/50; N=100 per the README: 100/1000/20/50)
  and are divided by 10 for the 100k-sample epochs used here → n20/n50:
  20/100/5/5, n100: 10/100/2/5. AM's PIP-D defaults (10/10/2/5) are tuned for
  ~100-epoch runs and are scaled ×5 for the 500-epoch runs (50/50/10/25).

Budget note for the paper: upstream PIP trains on ~100M (POMO, 10000×10k) /
~128M (AM, 100×1.28M) **freshly generated** instances; here both are pinned to the
fixed 100k set at the AMAI budget for like-for-like comparability with the other
neural baselines. (Optional ablation: rerun one or two configurations at ~2× budget
to confirm the validation metrics have plateaued.)

## 5. Running / resuming

Environment: the upstream repo ships no package metadata (`pip install -e .` is not
applicable); dependencies are pinned in `requirements.txt` (added in this fork,
trimmed from the README's install line — no torchvision/torchaudio/tensorflow
needed; versions verified on python 3.12). On the cluster
(`module load lang/Python/3.12.3-GCCcore-13.2.0`, assumed by `start_job.sh` and
`submit_jobs.sh`), create `.venv` at the repo root (sourced by `start_job.sh`):
`python -m venv .venv && pip install -r requirements.txt`, then record the full
environment with `pip freeze > requirements.lock.txt`.

Logging: all jobs log to **wandb project `pip_feasible`** (one dedicated project
for this baseline, same convention as LMask's `lmask_feasible`), run names spell
out base model + PIP variant + dataset:
`{pomo|am}[_pip|_pipd]_tsptw_{n20|n50|n100_sw|n100_mw}` (e.g.
`pomo_pipd_tsptw_n100_sw`; all base models are the paper's starred versions, i.e.
with the Lagrangian timeout reward); tensorboard files are additionally written
into each run dir. Epoch-level validation metrics only (feasible cost, instance
infeasibility rate, timeout stats) — no per-batch wandb logging. The wandb
integration was fixed/added in this fork: upstream POMO's `wandb.init` crashed on
resumed runs and upstream AM had no wandb support; runs now place their wandb
folder inside the run dir so `submit_jobs.py` recovers the run id on resume and
the wandb curve continues instead of starting a new run.

```shell
bash submit_jobs.sh                                # reconcile all 24 jobs
bash submit_jobs.sh --dry_run                      # show what would happen
bash submit_jobs.sh --sizes 50 --variants pipd am_pipd
bash submit_jobs.sh --sizes 100 --tws sw           # n100 small windows only
bash submit_jobs.sh --resume                       # only resume crashed runs
bash submit_jobs.sh --fresh --sizes 20             # force new runs
```

Filter convention matches the AMAI/skillvrp submit scripts: `--sizes {20,50,100}`
plus `--tws {sw,mw}` for the n100 window types.

`submit_jobs.py` reconciles the grid against existing checkpoints and squeue:
crashed runs are resumed from the highest `epoch-N.pt` (POMO:
`--resume_path` + `--checkpoint`; AM: `--resume`; PIP-D additionally
`--pip_checkpoint <run>/fsb_accuracy_bsf.pt`), completed runs (target epoch
reached) and already-queued jobs are skipped. Each job has its own log root
(`POMO+PIP/results/<dataset>_<variant>/`, `AM+PIP/outputs/<dataset>_<variant>/`).
`start_job.sh` is the slurm wrapper (a100, repo-root `.venv`, runs the entry
script from inside its implementation directory).

## 6. Evaluation (after training)

* POMO: `python test.py --test_set_path ../data/feas_tsptw/<ds>/test_1k.npz
  --test_episodes 1000 --checkpoint <...> --problem_size <N+1> --include_service_time
  --check_depot_return [--generate_PI_mask]`.
* AM: `python eval.py --datasets <test file> --model <run dir>` (model settings are
  restored from the run dir's `args.json`).
* Checkpoints to evaluate: `trained_model_val_best.pt` (POMO) / `val_best.pt` (AM),
  selected by validation feasibility rate; the latest `epoch-N.pt` is the "last"
  model.

## 7. Open items

* **Optimality gap**: convert the Gurobi solutions
  (`AMAI2025/data/baselines/gurobi/tsptw/`) into the repos' opt-sol pkl format
  (`[(cost, route), ...]`, costs in raw 0–100 units — the testers divide by 100) and
  pass via `--test_set_opt_sol_path` / `--val_solution_path`. Only needed at
  evaluation time; training and validation run without it. Alternatively (and for
  the feasibility-rate metrics in any case), export tours and evaluate with the
  AMAI checker for a single shared evaluation pipeline.
* Decide on the optional 2× budget ablation (see §4).
* The `lkh_*.pkl` files in `data/TSPTW|TSPDL/` are LKH3 reference solutions for the
  *original* PIP datasets (gap computation only) — not used for the AMAI experiments.
