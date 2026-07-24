# Training summary: how PIP-constraint's AM*/POMO* differ from vanilla AM/POMO

Reference for the paper's methods section if this repo's gathered checkpoints
(`checkpoints/`) are included as baselines alongside separately-trained (e.g.
RL4CO) AM/POMO models. The short version: **the non-PIP variants gathered here
(`am`, `pomo`) are not vanilla AM/POMO** — they are this repo's own AM\*/POMO\*
"starred baselines" (terminology from the upstream
[jieyibi/PIP-constraint](https://github.com/jieyibi/PIP-constraint) paper),
plus further environment/training changes from this fork. See
`notes/summary.md` (§2–4) for the full implementation writeup this is
condensed from.

## Experiment grid

6 variants × 4 instance sets = 24 jobs:

| variant | family | PIP mask | notes |
|---|---|---|---|
| `pomo` | POMO\* | none | base model + Lagrangian timeout reward |
| `pip` | POMO\*+PIP | analytic 1-step | env lookahead mask |
| `pipd` | POMO\*+PIP-D | analytic + learned | learned SL head predicts the mask |
| `am` | AM\* | none | base model + Lagrangian timeout reward |
| `am_pip` | AM\*+PIP | analytic 1-step | env lookahead mask |
| `am_pipd` | AM\*+PIP-D | analytic + learned | learned SL head predicts the mask |

Instance sets: n20, n50, n100_sw, n100_mw (`data/feas_tsptw/`, AMAI-identical
test sets). As of 2026-07-23, `n100_{sw,mw}_{pip,pipd}` are the only gaps
(still training / just finished, not yet gathered) — every other cell is
populated in `checkpoints/`.

Budgets/epochs matched to the AMAI experiments (not upstream PIP's own
budget): one epoch = one full pass over a **fixed** 100k-instance training
set (not on-the-fly generation), **100 epochs for n20**, **500 epochs for
n50/n100** — 10M / 50M total training instances. Upstream PIP itself trains
on ~100M (POMO)/~128M (AM) **freshly generated** instances; that was
deliberately not reproduced here, to keep the training budget comparable to
the other neural baselines instead.

## Why AM\*/POMO\* are not "plain" AM/POMO

**1. Lagrangian timeout-penalty reward, always on — including for the
no-mask `am`/`pomo` variants.** Verified in code:
`POMO+PIP/envs/TSPTWEnv.py:400`, `total_timeout_reward =
-self.timeout_list.sum(dim=-1)`, computed unconditionally as part of the
reward regardless of `--generate_PI_mask`. This matches the repo's own
naming convention (`submit_jobs.py:72`: "POMO\* has the Lagrangian timeout
reward on by default"). This reward shaping is part of the **original
upstream PIP-constraint paper's baseline design**, not something this fork
added — but it is a real difference from a plain AM/POMO that only optimizes
negative tour length (e.g. enforcing feasibility purely through masking, with
no soft timeout penalty in the reward).

**2. Environment/training changes from this fork, applying to every variant**
(full detail: `notes/summary.md` §3):
* Service times modeled as a 5th node input feature and included in the
  dynamics/cost (`--include_service_time`) — **upstream AM had no
  service-time concept at all** (dataset row was `[x, y, tw_start, tw_end]`).
* Depot return-leg timeout check added unconditionally (AM) / via
  `--check_depot_return` (POMO) — upstream PIP never checked the return leg.
* Fixed-dataset training (shuffled passes over the same 100k instances)
  instead of upstream's on-the-fly instance generation.
* Best-checkpoint selection by **instance-level feasibility rate first,
  feasible cost as tie-break** (matching LMask's `val/ins_feas_rate` monitor)
  — upstream POMO selected by feasible cost only.
* AMAI-aligned problem semantics throughout (arrival measured before service,
  depot-`tw_end` return constraint) — verified against 200 known-feasible
  Gurobi tours, exact match to ≤ 2e-6.

## Concrete hyperparameter comparison

Per the paper's Models subsection (batch size 128 for train/val/test across
AM, POMO, MVMoE-POMO, PolyNet, SymNCO; lr 0.0001; RL4CO framework defaults),
checked directly against this repo's actual settings:

| setting | this repo (AM\*/POMO\*) | your RL4CO experiments | match? |
|---|---|---|---|
| training batch size (AM) | **512** (`AM+PIP/options.py:13`, upstream's own unchanged default) | 128 | no |
| training batch size (POMO) | **64** (`POMO+PIP/train.py:122`, upstream's own unchanged default) | 128 | no |
| learning rate | constant **1e-4**, no scheduler | 0.0001 | yes |
| seed | 2023 (PIP's own default) | — | not verified |

**Open question, not verifiable from this repo** — needs checking against the
RL4CO training code directly:
* Do your RL4CO AM/POMO baselines model **service times** at all? If not,
  they're solving a subtly different problem than this repo's `am`/`pomo`
  (which explicitly add service-time dynamics) — a problem-definition
  mismatch, not just a hyperparameter difference, and would need flagging
  prominently rather than folded into a hyperparameter-differences footnote.
* Do they use any timeout-penalty reward shaping, or enforce feasibility
  purely via masking with a plain negative-tour-length reward?

## Recommendation for the paper

If `am`/`pomo` (or `am_pip`/`am_pipd`/`pip`/`pipd`) are included in the same
results table as your own AM/POMO/MVMoE-POMO/PolyNet/SymNCO runs, describe
them as this repo's AM\*/POMO\* baselines (Lagrangian timeout reward,
service-time-aware, fixed-100k-instance training budget) rather than as "the
same AM/POMO" trained in a different framework — and state the batch-size and
budget differences explicitly, since they are not free/neutral choices here
(unlike at inference time, where batch size is close to neutral for POMO but
not for AM — see `notes/eval_decode_settings.md`).
