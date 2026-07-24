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
* **Val-best tracker survives resumes** *(fix)*: each val-best save also writes a
  `val_best_meta.json` (epoch, feas, score/cost) next to the file, the tracker
  values go into the epoch checkpoints, and resuming restores them (meta file
  preferred — the epoch checkpoint is written *before* that epoch's validation, so
  its tracker can lag one validation behind; pre-fix POMO checkpoints fall back to
  the `result_log` history). Without this, the in-memory best-so-far reset on every
  resume, so a resumed run could overwrite (POMO, single in-place file) or shadow
  (AM, one `val_best.pt` per attempt dir) a better pre-resume model. Runs resumed
  before the fix keep that caveat; `gather_checkpoints.py` flags them and resolves
  AM lineages via the wandb validation history.
* Robustness fixes: CPU fallback when CUDA is unavailable; graceful stop when
  val/test files are smaller than the requested episode count; optimality-gap
  computation only runs when a solution file actually exists.
* **Data files are read once per process and served from memory afterwards**
  (POMO: validation/opt-sol cache in `Trainer`; AM: fixed-train-set cache shared
  between `train_epoch` and the rollout baseline). Upstream re-opened these files
  during training (POMO: every validation batch; AM: every rollout-baseline
  update), so a data file becoming unavailable on the cluster's scratch
  filesystem killed 15h runs mid-training (observed July 2026: a Lustre metadata
  failure made `data/feas_tsptw/` unstat-able and crashed the running jobs at
  their next file access).

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
  (random slice) instead of the random generator when `--train_set_path` is set;
  baseline updates slice the in-memory copy (`SubsetDataset` in
  `reinforce_baselines.py`, items cloned so epoch checkpoints stay small — see
  the read-once bullet under "Both implementations").
* `--n_epochs` is now the **absolute** target epoch *(default changed for resumed
  runs only)*: a resumed run continues up to `n_epochs` instead of training
  `n_epochs` further, so PIP-D's epoch schedules stay consistent across restarts.
* Bug fixes: rollout-baseline resume crashed for tsptw (dict-indexing a tensor);
  rollout-baseline resume crashed for PIP-D runs (upstream bug: the baseline model
  is checkpointed as the `[am_model, pip_model]` list for PIP-D, but
  `RolloutBaseline.load_state_dict` assumed a single module — only triggered on
  the first-ever `am_pipd` resume; the loader now takes the AM half, matching
  fresh-run construction); resumed AM PIP-D runs crashed at the next lazy-PIP
  load epoch because AM (unlike POMO, which resumes into the old run dir)
  creates a fresh `save_dir` on resume that has no `fsb_accuracy_bsf.pt` yet —
  `run.py` now falls back to the `--pip_checkpoint` directory, and
  `submit_jobs.py` likewise falls back to an earlier run dir's fsb checkpoint
  when the newest run dir lacks one;
  `torch.load` needed `weights_only=False` on torch ≥ 2.6 (also applied to all
  checkpoint/dataset loads in POMO+PIP and GFACS+PIP — the PIP-D checkpoints
  store accuracies as numpy scalars, which the torch ≥ 2.6 weights-only
  unpickler rejects, so resuming `pipd` runs crashed); three hardcoded `.cuda()`
  calls broke CPU execution (two in training plus one in `sample_many`'s
  no-feasible-solution branch, `utils/functions.py`, which broke CPU *evaluation*
  — the `evaluation/` pipeline in §6 exercises this path); `--val_dataset`/`--val_solution_path` were
  unconditionally overwritten by the hardness defaults.

## 4. Experiment settings

For the paper-facing writeup of how AM\*/POMO\* differ from vanilla AM/POMO
(if used as baselines alongside separately-trained models), see
`notes/training.md`.

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

The gathered checkpoints are evaluated with a dedicated pipeline in `evaluation/`
that reports the **same metrics as the AMAI eval**
(`AMAI2025/source/evaluation/eval_checkpoints.py`) in the same txt/csv format, so
the two repos' numbers are directly comparable.

* Entry point: `bash evaluation/eval_checkpoints.sh` (workstation-only, no
  slurm — activates the repo-root `.venv`, runs with `--device cuda`, logs
  stdout+stderr to `logs/<timestamp>.log`; extra args forwarded, e.g. `--modes
  best --sizes n20 n50`, and a later `--device` overrides the script's
  default). Can also be run directly as `python evaluation/eval_checkpoints.py`
  (no logging then); `--device` there defaults to `auto`, which probes
  `cuda > mps > cpu` and prints a `WARNING:` if it has to land on `cpu`
  (`common.pick_device`). Filters mirror the other scripts: `--modes
  {best,last}`, `--families {pomo,am}`, `--sizes {n20,n50,n100_sw,n100_mw}`,
  `--variants ...`.
  MPS required real fixes, not just accepting the string: POMO+PIP relied on
  `torch.set_default_tensor_type('torch.cuda.FloatTensor')` (no MPS analogue)
  and `SINGLEModel`/`TSPTWEnv` silently fell back to `cuda-if-available-else-cpu`
  when no device was threaded in. Fixed by passing `device` through
  `_model_params`/`_env_params` in `eval_pomo.py`, moving the model explicitly,
  using `torch.set_default_device('mps')` for the MPS branch, and patching
  `POMO+PIP/envs/TSPTWEnv.py:load_problems` to move the loaded-dataset tensors
  onto `self.device` (the legacy `torch.Tensor(...)` constructor used in
  `load_dataset`/`load_npz_dataset` ignores `set_default_device`, so those
  would otherwise stay on cpu and crash against mps tensors elsewhere).
  AM+PIP needed no fix (already device-agnostic via `.to(device)`). Verified:
  MPS and CPU produce identical cost numbers on the same checkpoint/instances.
* Output: `evaluation/results/eval_checkpoints_{best,last}.{txt,csv}`, one row per
  checkpoint slot with `feas_count` (feasible instances / 1000), `cost` (mean
  feasible tour length in real 0–100 units = normalized length × `max_loc`),
  `gap_bks` (mean % gap over feasible instances to the AMAI per-instance BKS) and
  `inference_time`. Test instances: `data/feas_tsptw/<folder>/test_1k.npz`
  (identical to the AMAI tsptw test sets); the BKS comes from the AMAI baseline
  csv (`--bks_csv`, default the sibling repo's
  `eval_baselines_instance_results.csv`; absent → `gap_bks` NaN, no crash).
* Because POMO+PIP and AM+PIP share top-level module names (`utils`, …), each
  family is evaluated in its own subprocess (`evaluation/eval_pomo.py`,
  `evaluation/eval_am.py`); `evaluation/common.py` (torch-free) holds checkpoint
  discovery, the BKS lookup, the AMAI-style aggregation and the txt/csv writers.
  The flat gathered checkpoints carry no run-dir context (`args.json` / CLI
  flags), so each model's hyperparameters are reconstructed deterministically
  from its variant (all trained with `--include_service_time`; `--generate_PI_mask`
  for pip/am_pip, `+ --pip_decoder` for pipd/am_pipd), and the checkpoint's
  `{best,last}` weights are loaded onto that model.
* **Batch size is not just a speed knob for AM+PIP**: it changes results
  (`feas`/`cost`), not only runtime — `AM+PIP/nets/graph_encoder.py` uses
  `BatchNorm1d(track_running_stats=False)`, which normalizes on live batch
  statistics even at eval time, so batch composition changes encoded
  representations. POMO+PIP (`InstanceNorm1d`) is unaffected (verified
  bit-identical cost across batch sizes). `--am_max_calc_batch_size` is now
  wired through `eval_checkpoints.py` (previously `eval_am.py`-only) to trade
  wall-clock for peak GPU memory on `am_pip`/`am_pipd` without touching
  `--am_batch_size`. This surfaced a pre-existing vendored bug in
  `AM+PIP/utils/functions.py::sample_many` (upstream
  [jieyibi/PIP-constraint](https://github.com/jieyibi/PIP-constraint), not
  introduced here): chunked calls (`max_calc_batch_size < width`) crash
  because the per-instance infeasibility mask isn't accumulated across chunks
  like `costs`/`pis` are. Left the vendored logic untouched by choice and
  instead sidestepped it: `evaluation/eval_checkpoints.sh` pins `--am_width`
  and `--am_max_calc_batch_size` equal (1280, upstream's own defaults) so the
  broken chunked path is never reached, and controls peak memory only via
  `--am_batch_size` (defaulted there to 8, down from upstream's 16, to avoid
  the observed n100 `am_pipd` OOM); `eval_am.py` warns if `max_calc_batch_size
  < width` is ever set anyway. `--pomo_batch_size` defaults to 128 there too
  (matches the other RL4CO experiments' batch size — free to pick, since
  POMO+PIP's batch size has no effect on results). Full detail + evidence in
  `notes/eval_decode_settings.md`.
* **Decode settings** (per family, all CLI-configurable; full reference in
  `notes/eval_decode_settings.md`):
  * POMO (pomo/pip/pipd): `aug_factor=8`, `eval_type=argmax`, `pomo_start=False`
    — reproducing the training-time validation that selected `best.ckpt`
    (`Trainer._val_one_batch`); with `pomo_start` off + argmax the pomo rollouts
    are identical copies, so `pomo_size=1` gives the same feasibility/cost cheaply.
    pip adds the env 1-step PI mask; pipd additionally uses the model's own
    PIP-decoder head to predict the mask (`use_predicted_PI_mask`).
  * AM (am/am_pip/am_pipd): sampling with `--am_width` samples/instance (default
    1280, AM+PIP's `eval.py` default; `--am_decode greedy` for a single greedy
    rollout). `generate_PI_mask` supplies the analytic 1-step mask for
    am_pip/am_pipd; the PIP-decoder SL head is built (weights load) but does not
    change decoding under `sample_many`, matching AM+PIP's own `eval.py` — so the
    separate `..._pip.pt` aux head is not needed at eval time.
* Feasibility and cost use each repo's real env dynamics (service times +
  depot-return check), so they are consistent with training/validation and the
  AMAI semantics (§2). The per-repo `test.py` / `eval.py` remain usable directly
  for gap-to-LKH on the *original* PIP datasets, but the AMAI-comparable numbers
  come from this pipeline.
* **Per-instance results** (`evaluation/eval_instance_results.py`): a second
  entry point that runs the *same* inference (same workers/settings, so numbers
  match `eval_checkpoints.py`) but records results **per instance** instead of
  only the aggregate row. It writes one csv per checkpoint at
  `evaluation/instance_results/{size}/{variant}/{mode}.csv` — sizes `n20`, `n50`,
  `n100_sw`, `n100_mw`; variants `am`, `pomo`, `am_pip`, `pomo_pip`, `am_pipd`,
  `pomo_pipd` (i.e. `VARIANTS[variant]['label']`); columns `instance`,
  `feasibility` (1/0), `objective` (best feasible tour length in real 0–100
  units, blank if infeasible), `runtime` (per-instance inference seconds). The
  point is decoupling from the BKS: with these files a new baseline no longer
  forces a full re-run — the per-instance gap to BKS can be computed and
  aggregated directly from `objective`, since rows are ordered by instance index
  and align with the BKS csv's `instance` column. Defaults to `best` only (pass
  `--modes best last` for both). Entry point `bash
  evaluation/eval_instance_results.sh` (workstation-only), whose pinned
  decode/batch knobs are copied verbatim from `eval_checkpoints.sh` and must
  stay in sync with it, so the two pipelines' numbers match. Implemented by
  adding an `--instance_csv` flag to the
  workers (which now also time each batch and split the wall-clock evenly over
  the batch's instances — batched inference has no true per-instance clock, so
  `runtime` is amortized and rises with more rollouts per instance, e.g. AM
  sampling `width>1`) plus `common.write_instance_csv`; the aggregate pipeline is
  unchanged. Its `inference_time` column stays a single total-wall-clock number
  per checkpoint.
* Checkpoints evaluated: `trained_model_val_best.pt` (POMO) / `val_best.pt` (AM)
  gathered as `<tag>_best.pt`, selected by validation feasibility rate; the latest
  `epoch-N.pt` gathered as `<tag>_last.pt` is the "last" model.
* `bash gather_checkpoints.sh [--git-add]` (cluster, repo root) collects each run's
  checkpoints into flat `checkpoints/<tag>_{best,last,pip}.pt` files plus a
  `manifest.csv` (source path, epoch, score) — `pip` is the PIP-D auxiliary decoder
  (`fsb_accuracy_bsf.pt`, needed at test time). By default only **finished** runs
  are gathered (last `epoch-N.pt` ≥ `submit_jobs.target_epochs`; POMO's last ckpt is
  `epoch-<target>`, AM's `epoch-<target-1>`); unfinished runs are skipped with a note.
  Pass `--include_all` to gather still-training runs too. Lineage handling: POMO resumes reuse
  their run dir (stale fresh-restart dirs are ignored); AM resumes open a new dir
  with the same wandb id. The `best` fields (epoch/feas/score) come from
  `val_best_meta.json` where available (runs trained after the tracker fix); for
  pre-fix AM lineages with several `val_best.pt` candidates the true best is
  resolved from the run's wandb validation history, and for pre-fix POMO runs a
  pre-resume best that was never beaten again is overwritten in place and
  unrecoverable (the manifest notes this).
* `bash check_val_best.sh` (cluster login node, repo root) audits exactly that
  caveat run by run: it pulls each run's full validation history from wandb and
  reports OK (the global best under the feasibility-first rule is held by an
  existing val-best file — for POMO because it falls after the last resume, whose
  start is read from the newest `wandb/run-<ts>-<id>` attach dir) or AT RISK (POMO
  only: the global best predates the last resume and its weights were overwritten —
  decide between reporting the post-resume best and re-running). Unfinished runs
  are marked PROVISIONAL; re-check after completion.

## 7. Open items

* **Optimality gap**: the `evaluation/` pipeline (§6) reports the gap to the AMAI
  per-instance BKS directly, so the AMAI-comparable numbers need no LKH opt-sol
  files. (The per-repo `test.py`/`eval.py` still expect LKH pkls for their own
  gap-to-LKH on the original PIP datasets; converting the AMAI Gurobi solutions to
  that pkl format — `[(cost, route), ...]`, raw 0–100 units, divided by 100 by the
  testers — remains optional and is only needed if using those scripts directly.)
* Decide on the optional 2× budget ablation (see §4).
* The `lkh_*.pkl` files in `data/TSPTW|TSPDL/` are LKH3 reference solutions for the
  *original* PIP datasets (gap computation only) — not used for the AMAI experiments.
