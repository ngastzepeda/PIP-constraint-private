# Evaluation decode settings for the PIP checkpoints

Reference for the inference/decode configuration used by the `evaluation/`
pipeline (`eval_checkpoints.py` → `eval_pomo.py` / `eval_am.py`) when scoring the
gathered PIP checkpoints on the AMAI TSPTW test sets. This documents *what* each
model does at test time and *why*, so the baseline numbers are reproducible and
the paper's methods section can cite the exact settings. See `summary.md` §6 for
how this fits the overall pipeline; this file is the detailed decode reference.

All settings are **CLI-configurable** (defaults below); every checkpoint is
scored on `data/feas_tsptw/<folder>/test_1k.npz` (1000 instances, identical to
the AMAI tsptw test sets).

## Reported metrics (identical to AMAI, decode-independent)

For every checkpoint the pipeline reports, per instance:
* **feasible** iff the model found ≥ 1 feasible tour (over all augmentations /
  starts / samples);
* **cost** = the best (min) feasible tour length, in normalized [0,1]-coordinate
  units, then × `max_loc` (= 100) to recover real 0–100 distances.

Aggregated exactly as AMAI's `get_test_metrics`: `feas_count` = feasible
instances / 1000; `cost` = mean feasible cost over feasible instances; `gap_bks`
= mean % gap to the AMAI per-instance BKS over feasible instances. The *decode
strategy* only changes how the per-instance best-feasible tour is found; the
metric definitions are fixed and shared with AMAI.

## POMO family (`pomo`, `pip`, `pipd`)

| knob | value | source |
|---|---|---|
| `aug_factor` | **8** (dihedral ×8) | `--aug_factor` |
| `eval_type` | **argmax** (greedy per rollout) | fixed |
| `pomo_start` | **False** | fixed |
| `pomo_size` | **1** | fixed (see below) |
| `include_service_time` | True | variant |
| `check_depot_return` | True | variant |

**Rationale.** These exactly reproduce the training-time validation that
selected `best.ckpt` (`POMO+PIP/Trainer.py::_val_one_batch`, called with
`aug_factor=8, eval_type="argmax"`), so the reported test metric is the same
quantity the model was checkpointed on, evaluated on the test split. The ×8
dihedral augmentation is POMO's standard test-time boost and mirrors AMAI's
augmented evaluation, keeping the two repos comparable.

**Why `pomo_size=1` is exact, not an approximation.** POMO's multi-start only
diversifies rollouts when `pomo_start=True` (forced distinct 2nd node) or under
softmax sampling. These models trained with `pomo_start=False`, and validation
decodes with `argmax`, so the `pomo_size` parallel rollouts are *identical
copies* — `pomo_size=1` yields the same feasibility/cost as the full multistart
at a fraction of the compute. (This matches how the val metric was computed:
`aug_feasible` / best-feasible-cost reduce over the pomo axis, which is constant
here.)

**Per-variant masking.**
* `pomo` — no PIP mask.
* `pip` — env 1-step preventative-infeasibility mask (`generate_PI_mask`), the
  analytic look-ahead computed in `TSPTWEnv.step`.
* `pipd` — `generate_PI_mask` **plus** the model's own PIP-decoder head predicting
  the mask (`pip_decoder` + `use_predicted_PI_mask`, with `W_kv_sl`/`W_out_sl` so
  the SL weights load). This reproduces the pipd validation path
  (`Trainer._val_one_batch` sets `generate_PI_mask=True` and queries the SL head
  for pipd). The separate `<tag>_pip.pt` auxiliary head is **not** needed — the
  SL head lives inside the main model.

## AM family (`am`, `am_pip`, `am_pipd`)

| knob | value | source |
|---|---|---|
| `decode` | **sampling** | `--am_decode` (`greedy` also available) |
| `width` | **1280** samples / instance | `--am_width` |
| `softmax_temp` | 1.0 | `--am_softmax_temp` |
| `eval_batch_size` | 16 | `--am_batch_size` |
| model | 128 emb / 128 hid / 3 enc layers / batch-norm / tanh-clip 10 | fixed |
| `include_service_time` | True | variant |

The best feasible sample per instance is reported (`sample_many`).

**Rationale.** `width=1280` sampling is AM+PIP's own `eval.py` default and the
model's standard strong inference — the "best-of-many-feasible" analogue of
POMO's ×8 augmentation, so the two families are reported on comparable
best-of-many footing (and comparable to AMAI's augmented AM). `--am_decode
greedy` runs a single greedy rollout instead (much lower feasibility; this is
what AM's *in-training* validation used to select `best`).

**Per-variant masking** (`nets/attention_model.py::_get_log_p`).
* `am` — no PIP mask.
* `am_pip` — analytic 1-step mask via `generate_PI_mask`.
* `am_pipd` — the PIP-decoder SL head is built so its weights load and it is
  evaluated, but `sample_many` decodes with the analytic mask (`generate_PI_mask`)
  — the SL head does **not** change decoding. So `am_pip` and `am_pipd` use the
  same masking at test time, matching AM+PIP's own `eval.py`. The separate
  `<tag>_pip.pt` head is not used at eval.

## Batch size: NOT purely a performance knob for AM+PIP

`--am_batch_size`/`eval_batch_size` and `--am_max_calc_batch_size` are **not**
result-invariant the way batch size is for POMO+PIP. Empirically verified
(same checkpoint, same instances, greedy/deterministic decode): `eval_batch_size
16` vs `4` gave different `feas`/`cost` on an identical 32-instance slice.
Root cause: `AM+PIP/nets/graph_encoder.py:132` uses
`nn.BatchNorm1d(..., track_running_stats=False)`, which computes *live* batch
statistics even in `model.eval()` mode — so which instances happen to share a
batch changes their encoded representations, hence their decoded tours.
POMO+PIP has no such issue (`POMO+PIP/models/SINGLEModel.py` uses
`InstanceNorm1d`, per-instance; verified bit-identical cost across
`test_batch_size` 20 vs 4). **Implication:** changing `--am_batch_size` for
"consistency" with the other RL4CO models' batch size 128 is a real
methodology choice for AM+PIP, not a free/neutral one — pick it once and hold
it fixed across the whole AM+PIP table, and note it in the paper if it differs
from the other models' batch size.

`--am_max_calc_batch_size` (added to `eval_checkpoints.py` — previously only
settable by calling `eval_am.py` directly) chunks the *sampling* axis (same
`width` total samples/instance, computed in smaller sequential pieces) when
set below `width`, trading wall-clock for peak GPU memory.

**Found, but deliberately did not fix, an upstream `sample_many` bug** while
exercising this: `AM+PIP/utils/functions.py::sample_many` only chunks
(`iter_rep>1`) when `max_calc_batch_size < width` — a code path nothing in
this repo had exercised before (the inherited default keeps `max_calc_batch_size
== width` always, so `iter_rep` is always exactly 1). It turned out broken:
the per-instance infeasibility `mask` is computed fresh every `iter_rep` loop
iteration but never accumulated (unlike `costs`/`pis`, which are correctly
`torch.cat`'d across iterations) — so only the *last* chunk's mask survives,
sized to one chunk while `costs`/`pis` are sized to the full `width`, crashing
with a shape-mismatch `IndexError` as soon as chunking actually triggers.
Pre-existing in the vendored code (present since the original "upload AM+PIP"
commit — this is a fork of
[jieyibi/PIP-constraint](https://github.com/jieyibi/PIP-constraint) — not
introduced by anything in `evaluation/`) and generic to all three AM variants,
not PIP-specific; upstream's own `eval.py` would hit it identically if run
with `max_calc_batch_size < width * eval_batch_size` far enough to make
`iter_rep>=2`.

Decision: **do not patch vendored AM+PIP logic.** Instead, sidestep the bug
entirely by never triggering the chunked path — `evaluation/eval_checkpoints.sh`
now pins `--am_width` and `--am_max_calc_batch_size` equal (both 1280,
upstream's own defaults), which forces `iter_rep=1` unconditionally regardless
of `eval_batch_size`, so the broken multi-chunk branch is structurally
unreachable. `evaluation/eval_am.py` prints a `WARNING:` if `max_calc_batch_size
< width` is ever set, explaining the bug and pointing at `eval_batch_size` as
the safe lever instead. Peak memory (`width * eval_batch_size`) is instead
controlled via `--am_batch_size` (see above — not a free knob either, but at
least a documented, deliberate one).

## Open decision (paper)

For **AM**, the default is sampling-1280 (strong, comparable to AMAI's augmented
inference), which differs from the greedy decode AM used to *select* its `best`
checkpoint. Options, all one CLI flag away:
* keep **sampling-1280** (recommended for AMAI comparability),
* switch to **greedy** (`--am_decode greedy`, matches AM's selection-time metric),
* report **both**.

POMO has no such tension: its evaluation decode (`aug8`/argmax) is the same as its
selection decode.

## Not tunable by design

* Feasibility and cost always use each repo's **real env dynamics** (service times
  included, depot-return timeout checked), consistent with training/validation and
  the AMAI/LMask semantics (`summary.md` §2) — independent of decode strategy.
* Metric definitions and the BKS gap are fixed to match AMAI; only the search
  breadth (aug / width) is a knob.
