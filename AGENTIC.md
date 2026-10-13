# Agentic BO — integrating SARA

Branch: `agentic-bo`. Status: **run.** 392 tests pass, the headroom gate has a
verdict, and both tiers are swept. The report is
[`results/ladder_report.html`](results/ladder_report.html). See
[Results](#results) and [What actually ran](#what-actually-ran).

## Results

**It is the BO.** Both tiers agree.

| contrast | CHEMBL203 (20 seeds) | PMO (9 tasks) |
|---|---|---|
| BO's own contribution (`bo_ei` &minus; `random`) | **+0.0757**, p < 0.0001 | **+0.1179**, p = 0.004 |
| the LLM alone (`llm_only` &minus; `random`) | +0.0205, p = 0.008 | +0.0016, p = 0.91 |
| the LLM on top of BO (`bo_llm_select` &minus; `bo_random_select`) | +0.0141, p = 0.15 | +0.0052, p = 0.027 |
| what BO adds under the model | +0.0657, p < 0.0001 | +0.1181, p = 0.004 |
| LLM constraint vs plain BO | not run on this tier | &minus;0.0435, p = 0.055 |
| LLM policy vs plain BO | not run on this tier | +0.0004, p = 0.50 |
| **LLM policy vs `ScriptedPolicy`** | not run on this tier | **&minus;0.0002, p = 0.73** |

Four things follow, and each needed an arm that exists only to make it sayable:

1. **BO carries every arm**, by an order of magnitude over anything the model contributes.
2. **The model's edge does not generalise.** It beats a uniform draw on EGFR
   potency (+0.0205, p = 0.008) and not at all across nine PMO objectives
   (+0.0016, p = 0.91). That edge was knowledge *about the target*, not the
   ability to read a molecule.
3. **On top of BO the model adds almost nothing** --- ns on ChEMBL, and 23x
   smaller than BO's own contribution where it is significant on PMO.
4. **The agentic coupling contributes nothing at all.** `bo_policy_llm` ties
   plain BO (p = 0.50) and ties four napkin rules reading the same diagnostics
   (p = 0.73), winning 0 of 9 tasks. The policy log says why: across 98 rounds
   and 5 seeds it chose `ucb` **every single time**, 0 switches, `beta` never
   off 2.0 --- with 10 independent decisions per run and 10 distinct rationales.
   It read the diagnostics and always reached the same conclusion. The arm
   degenerated to a fixed acquisition.

**The baseline decides the headline.** On ChEMBL203, `bo_llm_select` beats
`bo_ei` (+0.0106, p = 0.0153) and ties its own control `bo_random_select`
(+0.0141, p = 0.15). Same arm, same data; the difference is whether the
shortlist is held fixed. Reporting the first number alone would be the exact
mistake `bo_diverse_warmstart` was built to catch.

The headroom gate predicted the agentic result before a single model call was
spent: the hindsight ceiling sits inside twice the SEM of the best fixed
acquisition at both budgets.

### Caveats

- PMO model arms cover **9 of 23** tasks; controls cover 19 informative ones.
- **Three PMO oracles are broken in this environment** and two fail silently
  (below). Any PMO number touching `drd2`, `gsk3b` or `jnk3` is invalid.
- `bo_policy_llm` never ran on the ChEMBL tier, so the agentic result rests on
  PMO alone.
- n = 9 tasks is thin: `+0.0052, p = 0.027` would not survive much multiplicity
  correction.

This wires [SARA](https://github.com/uta-smile/AgenticBo) (*Surrogate-Augmented
Autoresearch Agent*, arXiv 2608.00316) into this project's harness, and adds the
control arms that would make a SARA result mean something here.

## Why this is not just "run SARA on EGFR"

SARA's pitch is against fixed roles: rather than inserting a language model at
one point in the loop, the agent configures the problem, queries the backend,
selects evaluations, and adjusts strategy. But two of those actions are arms this
project already ran, in isolation, with controls:

| SARA action | this repo | result |
|---|---|---|
| modify search bounds | `bo_llm_constraint` | **−0.030, p = 0.006** — significantly worse |
| select evaluations | `bo_llm_select` | neutral; diagnosed as positional bias, not chemistry |
| **switch acquisition function** | — | **never tested** |
| **inspect surrogate state** | — | **never tested** |
| reframe the objective | — | no scope on single-objective potency |

So the experiment worth running is the two bold rows. Everything in
`src/molbo/agentic/` implements exactly those.

The second row may matter more than the first. In all three existing arms the
model was asked a *chemistry* question and answered it by misnaming the
chemotype — it proposed a "pyrazolo[3,4-d]pyrimidine core" for a library that is
100% anilinoquinazoline, and on the warm-start arm invented two scaffold classes
that do not exist. It was **never once shown the surrogate's state**: not the
fitted noise, not the posterior spread, not how much of the library it had
explored. Whether it can read a GP is a different question from whether it can
read a molecule, and this project has no evidence on it either way.

## Run this first

```bash
python scripts/acquisition_headroom.py --dataset data/chembl/egfr_anilinoquinazoline.csv
```

Needs no endpoint, no API key, no GPU. It runs every fixed acquisition alone,
then a hindsight ceiling that switches acquisition every round to whichever one
reaches the best true value, and reports the gap.

**Headroom is the ceiling minus the best fixed acquisition, not minus `ei`.** If
`ucb` alone beats `ei` alone, that is a better fixed policy, not evidence for
adaptivity. Crediting it to an agent would repeat the mistake the warm-start
control was built to catch.

If the headroom falls inside twice the standard error of the best fixed policy,
the script says so and you should stop: the action space is inert on this
library, and no controller can demonstrate anything. That is a conclusion about
the testbed, not about the agent — and it costs seconds to reach here instead of
a week after wiring up a model.

This mirrors `constraint_headroom.py`, which exists for the same reason and
found that a tight budget against a loose constraint is the worst place to look
for a constraint effect.

## The confound that must be controlled

Every LLM number in this repo is **Qwen3.5-4B**. SARA runs against a hosted
endpoint, presumably a frontier model. Compare the two as published and you
cannot separate *agentic architecture* from *model scale*.

`SaraBackend` implements the existing `molbo.llm.Backend` protocol precisely so
this costs no new code:

```python
from molbo.agentic.sara import SaraBackend
from molbo.llm import LLMConstraintProposer, ResponseCache

proposer = LLMConstraintProposer(
    backend=SaraBackend.from_env(),        # SARA_BASE_URL / SARA_MODEL / SARA_API_KEY
    cache=ResponseCache("data/cache/llm_sara"),
)
```

Re-run the existing three arms with that and nothing else changed. If a frontier
model beats the non-LLM controls where Qwen-4B did not, the project's headline
finding was about model size rather than coupling design — a correction worth
publishing on its own. If it still loses, the architecture question is isolated.

**Nothing downstream is interpretable until this runs.**

## The arm ladder

`ArmConfig.policy_source` is the new hook. It chooses *how the next candidate is
scored* — acquisition function and hyperparameters — once per round. It cannot
choose a molecule; that is `selector`, and keeping them apart is what makes each
a clean test of one idea.

| policy | role |
|---|---|
| `FixedPolicy` | null — reproduces plain BO exactly, pinned by a test |
| `RandomPolicy` | control: does *any* switching help? |
| `RoundRobinPolicy` | control: does switching *unpredictably* help? |
| `ScriptedPolicy` | **the control that matters** — napkin rules on identical inputs |
| `OraclePolicy` | ceiling, not a method; reads the answer key |
| `LLMPolicy` | the arm under test |

`ScriptedPolicy` is the `bo_diverse_warmstart` of this study. Four rules:
endgame → `greedy`; stalled → `ucb` with wider `beta`; posterior collapsed →
`thompson`; otherwise `ei`. Being obvious is the point. If a model cannot beat
four napkin rules reading the same numbers, the agency is not what is helping —
which is exactly what happened when a free MaxMin heuristic beat the LLM warm
start at its own stated strategy.

## What the controller sees

`Diagnostics` (in `agentic/diagnostics.py`) is the channel the existing arms
never had:

```
round 4  calls 30/100  observed 30  candidates 1871
library explored      1.44%
best observed         9.4000
rounds since improve  3
posterior mean        max 0.9000  mean 0.5000
posterior sd          max 0.2000  mean 0.1167
GP noise ratio (eta)  0.0300
GP log marg lik       -12.5000
acquisition in use    ei
```

No unobserved library value appears, in any form — no true optimum, no rank, no
count of good compounds remaining. A test asserts that. `OraclePolicy` is the
one labelled exception, following `constraints.oracle_constraint`.

The model is shown **no SMILES here**, deliberately. The existing arms all asked
it a chemistry question; this one asks whether it can read a surrogate, and
mixing the two would leave the result uninterpretable.

## What this preserves

Every rule the rest of the project enforces still holds. The policy hook runs
before scoring and cannot touch the candidate pool, so:

- the initial design stays arm-independent, and arms stay paired on the seed
- budget accounting is untouched; a policy cannot spend a call
- no molecule can be re-proposed
- `FixedPolicy` produces a byte-identical run to `policy_source=None` —
  `test_fixed_policy_matches_no_policy` pins it, and if it ever fails every
  paired comparison in this package is contaminated

## Known differences from upstream AgenticBO

**Discrete, not continuous.** `lenz` is BoTorch, built for box-bounded problems
(Branin-2D, Ackley-10D/20D). This search space is a fully enumerated set of 2,086
molecules, one categorical index, effectively 1-D, with the acquisition
maximised by exhaustive scan. "Modify the bounds" has no direct analogue; its
closest relative is the constraint arm, which already ran. This is the main
reason to measure headroom before building further.

**Single decision per round, not multi-turn tool calling.** `LLMPolicy` makes
one schema-constrained call per round rather than an open tool-calling loop. It
is agentic in the sense that matters for the hypothesis — the model sees
surrogate state and chooses strategy — but it is not the full agent. Extending
to multi-turn is the obvious next step and would need trajectory recording
designed in, not retrofitted.

**Caching degrades.** The existing arms' prompts are a function of
`(seed, round, observations)`, so re-running both budgets made 0 model calls and
40 cache hits. These prompts vary with the GP fit, so hit rates will be far
lower. `LLMPolicy.every` is the cost dial.

**Seed counts.** AgenticBO's protein runs default to 20 evaluations. This
project's discipline is 20 paired seeds because a single BO curve is noise.
Whatever SARA's published numbers are, a comparison here needs this seed count.

## What actually ran

The environment now has rdkit, scipy and pytest, so everything below was
executed rather than reasoned about.

**Five defects execution surfaced**, none of which a re-read would have caught:

1. `agentic/diagnostics.py` spelled Thompson sampling `"thompson"`; the scorer
   in `molbo/acquisition.py` spells it `"ts"`. `PolicyDecision.validated()`
   waved the name through and `acquisition_scores` raised mid-campaign, so
   `ScriptedPolicy` rule 3, `OraclePolicy` and any `LLMPolicy` taking the option
   **its own prompt offered it** crashed the run. `acquisition_headroom.py`
   could not run at all. Fixed to `ts`, with
   `test_acquisition_names_match_the_scorer` pinning the two tuples equal.
2. `run_pmo_experiment.py` saved results only at the very end, and `jnk3`'s TDC
   oracle is a scikit-learn pickle predating sklearn 1.3's
   `missing_go_to_left` tree field, so it raises on a current sklearn. The
   23-task sweep died at task 10 and discarded ten completed tasks. Oracles that
   will not load are now skipped and recorded in `unavailable_tasks`, and
   results checkpoint after every task.
3. TDC's `drd2` and `gsk3b` oracles **load without complaint on a current
   scikit-learn and then score every molecule identically** --- raw TDC scores
   haloperidol, a known D2 ligand, at 0.0. Every arm ties at exactly 0.0000,
   which in a results table is indistinguishable from "the methods are
   equivalent", and averaging such a task in drags every mean toward zero while
   compressing the gaps between arms. `probe_oracle()` now scores five
   unrelated molecules before any budget is spent and separates three
   outcomes: **broken** (skip and record), **flat** (`valsartan_smarts` ---
   run it, it is a real task whose SMARTS matches nothing, but mark it), and
   **ok**. Dropping a legitimate task and silently keeping a dead one are both
   ways to get a wrong mean, in opposite directions.
4. The ladder did not pass `mine_sample` to `LLMConstraintProposer`, so on PMO
   the substructure menu would have been mined from all 249,455 ZINC molecules.
   The PMO runner now passes 3,000, as its own constraint arm always did.
5. The report matched the headroom ceiling row on `"ceiling"` where the
   instrument writes `"ceiling (hindsight)"`, rendering an empty table --- which
   reads as "no effect" rather than "the data was not found". The build now
   fails loudly on an empty `<tbody>`.

**The arm ladder.** `src/molbo/ladder.py` is shared by both runners so the
ChEMBL and PMO tiers are read against the identical set of methods. Its
categories answer one question --- *is it the BO, the LLM, or the two
together?*

| category | arms |
|---|---|
| `none` | `random` |
| `bo` | `bo_ei`, `bo_ucb`, `bo_greedy` |
| `llm` | `llm_only` |
| `bo+llm` | `bo_random_select`, `bo_llm_select`, `bo_llm_constraint` |
| `agentic` | `bo_policy_{scripted,random,roundrobin,llm}` |
| `ceiling` | `bo_policy_oracle` |

**A new strategy.** `ArmConfig.strategy="llm"` is the arm the project never had:
a model choosing molecules from a uniform random shortlist with **no surrogate
underneath it**. Its control is `random` --- the identical draw with a uniform
pick --- and `shortlist_k=1` collapses it to `random` exactly, pinned by
`test_shortlist_k_one_reproduces_random`. Without this arm a BO+LLM number
cannot be attributed: "beat random" is compatible with the model contributing
nothing and the GP carrying the arm. `build_select_prompt` drops the prediction
column when there is no posterior, because a placeholder number is
indistinguishable from a real prediction inside a prompt.

**Serving a reasoning model.** Qwen3.5-4B needs `--reasoning-budget 0`, which
this repo already documented. Granite 4.2 8B needs the *opposite*: that exact
flag makes every schema-constrained call fail with
`Unexpected empty grammar stack`, because budget 0 forces a think block open,
the reply begins with `</think>`, and the JSON grammar applies from token 0 so
that tag can never be produced. Granite works with
`--reasoning off --reasoning-format deepseek`. Throughput is comparable
(~38 vs ~40 calls/min), so the arms here use Qwen, which keeps every LLM number
in this project on one model and removes a scale confound.

**Cost, and why PMO's model arms cover a subset.** `bo_llm_select` and
`llm_only` fire once per *round*. On PMO at budget 1,000 with batch 10 that is
99 calls per seed per task --- 11,385 calls per arm across 23 tasks, about ten
hours of decode for the pair. The control arms cover the 22 tasks whose oracle
loads (`jnk3` does not); the model arms cover **9**, spanning every PMO family.
Every table states per-arm coverage,
because a mean over each arm's own task list compares arms on different
problems and is the easiest way to manufacture a winner.

## Reproducing

```bash
python -m pytest tests/ -q                      # 392 tests
python scripts/acquisition_headroom.py     --dataset data/chembl/egfr_anilinoquinazoline.csv     --out results/acquisition_headroom.json     # the gate; no endpoint, no GPU
llama serve -hf unsloth/Qwen3.5-4B-GGUF:Q8_0 --port 8080     --ctx-size 16384 --n-gpu-layers all --reasoning-budget 0
python scripts/run_lookup_experiment.py --dataset data/chembl/egfr_all.csv     --ladder --llm-backend llamacpp --llm-model qwen3.5-4b     --budget 50 --seeds 20 --out results/chembl203_ladder_b50
python scripts/run_pmo_experiment.py --ladder --seeds 5 --budget 1000     --llm-backend llamacpp --llm-model qwen3.5-4b --policy-every 10     --out results/pmo_ladder_llm
python scripts/build_ladder_report.py           # results/ladder_report.html
```

The regression to watch is still `test_fixed_policy_matches_no_policy`. If it
fails, the hook is perturbing the run and every paired comparison in this
package is contaminated.
