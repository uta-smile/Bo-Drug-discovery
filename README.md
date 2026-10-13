# molbo — sample-efficient molecular BO with LLM-derived constraints

## RECOMB auxiliary-data diagnostics

The current manuscript studies auxiliary-data gains in molecular BO. The verified
historical-response permutation and crossed response/fitting experiment are
documented in [the crossed-control protocol](results/crossed_auxiliary_v1/README.md).
It includes the six-family run commands, the independent result audit, and the
limits on interpreting sequential and matched-history contrasts. Earlier LLM
experiments described below remain separate from this manuscript study.

Testing one claim: **can an LLM, shown the SAR observed so far, narrow a
virtual library usefully enough to make Bayesian optimisation find potent
compounds sooner?**

The design problem is that a single BO curve is noisy enough that any two
methods can look different by luck, so the claim needs 20+ seeds per arm. That
is only affordable if the oracle is cheap, which is why the project is built on
three oracle tiers rather than one.

## The three tiers

| Tier | Oracle | Cost | Role |
|---|---|---|---|
| **Primary** | ChEMBL measured pIC50, dictionary lookup | free, instant | where the method is developed; 20+ seeds per arm |
| **Secondary** | PMO tasks at a 1,000-call budget | seconds | comparability — buys 25 published baselines |
| **Realism** | Boltz-2 co-folding affinity | ~20 s/ligand, GPU | shows the method survives an expensive structure-based oracle |

The primary tier is the only one with **real ground truth**. Boltz-2 returns a
prediction; the assay value is the actual answer. It is also the only tier
where the true optimum is known, so simple regret and top-1% recall are
available there and nowhere else.

The realism tier is the differentiator. PMO deliberately excluded docking
scores as *"more costly than the other oracles while still being coarse
estimates that don't reflect true bioactivity"*, leaving them to future work.
Boltz-2's affinity head is a step past docking, so sample-efficient molecular
BO against a co-folding oracle sits in the gap the benchmark community punted
on.

## Quickstart

```bash
# Build a congeneric series from ChEMBL (already done for EGFR; see data/chembl)
python scripts/build_chembl_series.py --target CHEMBL203 --out data/chembl/egfr

# Before running any arm: how much room is there for a constraint to help?
python scripts/constraint_headroom.py --dataset data/chembl/egfr_anilinoquinazoline.csv

# The sweep: every arm, every seed
python scripts/run_lookup_experiment.py \
    --dataset data/chembl/egfr_anilinoquinazoline.csv \
    --budget 100 --seeds 20 --constraint-fraction 0.05 \
    --out results/egfr_congeneric_b100
```

A 20-seed × 5-arm sweep at a 100-call budget takes about **20 seconds**. That
is the entire point of the primary tier: the same experiment against Boltz-2
would be roughly 90 GPU-hours, and would have to be redone every time a prompt
changed.

## The report

`results/report.html` is a single self-contained page -- no CDN, no web fonts, no sibling
assets -- holding the chart, every arm table, the full hyperparameter ranges, what BO
fits and suggests round by round, the search-space geometry, and the PMO comparison.
Open it straight off disk.

```bash
python scripts/build_report.py     # regenerates it from whatever is in results/
```

Every number in it is read from the run artifacts rather than typed in, so the page
cannot drift from the data it describes.

## Measure headroom before running an arm

`scripts/constraint_headroom.py` compares plain BO against a **ceiling arm**
restricted to the library's true best *k*% — the constraint you would write if
you already knew every answer. No real constraint source can beat that line, so
the gap bounds what any arm could possibly demonstrate.

EGFR anilinoquinazolines (2,103 compounds, 10 seeds):

| budget | allowed fraction | plain BO | ceiling | headroom |
|---|---|---|---|---|
| 50 | 0.25 | 0.6654 | 0.7113 | +0.046 |
| 50 | 0.05 | 0.6654 | 0.7594 | **+0.094** |
| 100 | 0.05 | 0.7582 | 0.8398 | +0.082 |
| 200 | 0.25 | 0.8250 | 0.8563 | +0.031 |

Headroom roughly triples between a quarter of the library and a twentieth, and
grows as the budget shrinks. **A tight budget against a loose constraint is the
worst place to look for a constraint effect** — and is the configuration one
reaches for first. Report headroom next to any effect size: +0.01 AUC means
something very different against +0.09 of headroom than against +0.02.

## Arms

| Arm | What it is |
|---|---|
| `random` | Floor. If an arm cannot beat this, nothing else about it matters. |
| `bo_ei` | Plain BO. **The reference** every other arm is tested against — beating random is a smoke test, not a result. |
| `bo_random_constraint` | BO restricted to a random *k*% of the library. Separates "the LLM knows chemistry" from "any restriction helps". |
| `bo_data_driven_constraint` | Descriptor ranges of the best-observed compounds, by statistics, no model. The sharpest control. |
| `bo_oracle_constraint` | Ceiling. Not a method — a reference line. |
| `bo_llm_constraint` | The arm under test. Local Qwen3.5-4B via llama.cpp. |

### Results

EGFR anilinoquinazolines, 2,086 compounds (heavy atoms 10-55), 20 paired seeds,
paired Wilcoxon signed-rank vs `bo_ei`. ECFP4 **with chirality** (`v2-chiral`).
Report: `results/report.html`

**Budget 50 unique calls**

| arm | AUC | best pIC50 | regret | top-1% recall | dAUC | p |
|---|---|---|---|---|---|---|
| `random` | 0.6040 +/- 0.0062 | 9.85 | 1.150 | 2.4% | -0.0741 | <0.001 |
| `bo_random_select` | 0.6551 +/- 0.0079 | 10.13 | 0.856 | 15.7% | -0.0229 | <0.001 |
| `bo_llm_select` | 0.6600 +/- 0.0090 | 10.09 | 0.896 | 15.5% | -0.0180 | 0.15 |
| `bo_llm_constraint` | 0.6648 +/- 0.0085 | 10.03 | 0.966 | 17.1% | -0.0132 | 0.26 |
| `bo_ei` (reference) | 0.6780 +/- 0.0095 | 10.19 | 0.815 | 22.6% | -- | -- |
| `bo_llm_warmstart` | 0.6953 +/- 0.0099 | 10.19 | 0.808 | 25.7% | +0.0173 | 0.23 |
| `bo_diverse_warmstart` | 0.7002 +/- 0.0117 | 10.42 | 0.583 | 30.0% | +0.0221 | 0.11 |
| `bo_oracle_constraint` (ceiling) | 0.7576 +/- 0.0059 | 10.75 | 0.253 | 63.6% | +0.0795 | <0.001 |

**Budget 100 unique calls**

| arm | AUC | dAUC | p |
|---|---|---|---|
| `random` | 0.6694 +/- 0.0044 | -0.0964 | <0.001 |
| `bo_random_constraint` | 0.6834 +/- 0.0069 | -0.0823 | <0.001 |
| `bo_llm_constraint` | 0.7359 +/- 0.0101 | -0.0298 | 0.006 |
| `bo_data_driven_constraint` | 0.7655 +/- 0.0094 | -0.0003 | 0.84 |
| `bo_ei` (reference) | 0.7658 +/- 0.0089 | -- | -- |
| `bo_llm_warmstart` | 0.7760 +/- 0.0093 | +0.0102 | 0.62 |
| `bo_diverse_warmstart` | 0.7897 +/- 0.0083 | +0.0239 | 0.070 |
| `bo_oracle_constraint` (ceiling) | 0.8382 +/- 0.0031 | +0.0724 | <0.001 |

**No LLM arm beats plain BO.** The constraint arm is significantly *worse* at
100 calls; the selection arm is neutral; the warm start gains ground but does
not reach significance, and its free non-LLM control gains more. Three things
the controls settle:

1. **There is real headroom** (+0.080 at 50 calls, +0.072 at 100). A good
   constraint wins comfortably here, so a null result is not "the task is
   impossible" -- this model just does not find one.
2. **"Any restriction helps" is ruled out.** A random mask of the same
   tightness significantly *hurts* (-0.082).
3. **The statistical baseline is neutral** (-0.0003, p = 0.84), so a win could
   not have been dismissed as descriptor-summarising.

> **Correction.** Earlier runs used stereochemistry-blind fingerprints (RDKit's
> default) and reported the LLM warm start as a significant win (+0.030,
> p = 0.024). After fixing the representation the effect is no longer
> significant and the non-LLM control overtakes it. See
> [A measured limitation, fixed](#a-measured-limitation-fixed). Every number
> above was regenerated.

## Three ways to pair an LLM with BO

The constraint arm is one bet. Two more were run, chosen to attack its specific
failure mode, and each has a non-LLM control that isolates what the model
actually contributed.

### 1. Constraint -- the model narrows the library

Significantly **worse** than plain BO at a 100-call budget (-0.030, p = 0.006).
A bad mask is *absorbing*: it fences off the optimum for the rest of the run and
BO cannot recover. It excluded the single best compound in 19 of 20 proposals.

### 2. Selection -- BO shortlists, the model picks

`bo_llm_select`: each round BO ranks the candidates, the top `--shortlist-k` go
to the model, and it chooses which one costs an oracle call. A bad pick here
costs exactly one evaluation, so unlike a constraint it is recoverable. One call
per *evaluation* rather than per run, so ~40x the cost.

`shortlist_k` is the dial: **k=1 is exactly plain BO** (pinned by a test,
including on a flat acquisition surface, which is why ranking is shared between
the two paths), and k = all candidates is pure selection.

The result is neutral, and the diagnostic is the interesting part. Over 800
decisions on a 15-long shortlist the mean chosen rank is 7.5 against 7.0 for
uniform -- but the distribution is strongly non-uniform (chi-square vs uniform,
p < 1e-4) and the shape is **positional**, not chemical:

```
rank:  0    1   2   3   4   5   6   7   8   9  10  11  12  13  14
picks 118   9  56  34  37  52  36  46  25  45  50  55  56  75 106
```

Peaks at the first and last entries, a trough at the second: textbook
primacy/recency bias. Presented in acquisition order, "picks position 0" and
"agrees with BO's top candidate" are the *same event*, so position bias
masquerades as judgment. `LLMSelector.shuffle` (default on) now presents the
shortlist in a seeded random order and maps the answer back, which decouples
them -- without it the arm is uninterpretable.

### 3. Warm start -- the model picks the initial design

`bo_llm_warmstart`: with nothing yet measured, the model chooses the first
`n_init` compounds from a seeded 40-molecule sample. This is the best case a
language model gets -- there is no data, so the surrogate has nothing to offer
and prior knowledge is all there is -- and it costs one call per run.

**It works.** At a 100-call budget, +0.0239 AUC over plain BO (p = 0.027).

**But the LLM is not why.** `bo_diverse_warmstart` picks the same number of
compounds from the *same* sample by greedy MaxMin Tanimoto distance, with no
model involved, and scores +0.0234 (p = 0.064). Head to head:

| comparison | budget | dAUC | p | seeds ahead |
|---|---|---|---|---|
| LLM warm start vs plain BO | 50 | +0.0173 | 0.23 | 11/20 |
| diversity warm start vs plain BO | 50 | +0.0221 | 0.11 | 15/20 |
| **LLM vs diversity warm start** | 50 | **-0.0049** | **0.78** | 9/20 |
| LLM warm start vs plain BO | 100 | +0.0102 | 0.62 | 10/20 |
| diversity warm start vs plain BO | 100 | +0.0239 | 0.070 | 15/20 |
| **LLM vs diversity warm start** | 100 | **-0.0137** | **0.29** | 9/20 |

The model is behind its own free control at both budgets. And it says so itself -- its
recorded rationales are explicitly diversity arguments ("Core Scaffold
Diversity", "distinct structural scaffolds", "maximize information gain"). It is
doing MaxMin selection in prose, and a free heuristic does it equally well.

### What this adds up to

Across three quite different couplings, the LLM never beat its non-LLM control:

| pairing | vs plain BO | vs its control | mechanism |
|---|---|---|---|
| constraint | **-0.030** (p=0.006) | worse than statistical mask | masks are absorbing; excluded the optimum 19/20 |
| selection | -0.018 (ns) | ~= random shortlist pick | a noisy copy of the acquisition function |
| warm start | +0.017 (ns) | **-0.005 (p=0.78)** vs diversity | it is doing diversity selection |

The one thing that helps at all -- warm-starting BO with a structurally spread
initial design -- is worth keeping, and needs no LLM. The MaxMin heuristic is
ahead of the model on 15 of 20 seeds against plain BO, where the model manages
10 or 11.

**Scope.** This is Qwen3.5-4B. It misidentifies the series' chemotype in every
prompt format tried, calling anilinoquinazolines a "pyrazolo[3,4-d]pyrimidine
core", so these are results about a small local model, not about language models
in general. The harness is backend-agnostic (`AnthropicBackend` is one flag
away) and independently validated on PMO, so a frontier model can be dropped in
and the same sweep re-run in minutes.

## Do the pairings transfer to PMO?

Every LLM number on the ChEMBL tier is from a library we assembled. The same
arms were run on PMO's tasks over the ZINC-250k pool -- a structurally diverse
pool with objectives that are not potency at all, which is a harsher test of a
method tuned on a congeneric series.

```bash
python scripts/run_pmo_experiment.py --seeds 5 --budget 1000     --tasks albuterol_similarity celecoxib_rediscovery deco_hop median1             osimertinib_mpo --warmstart-arms --constraint-arm
```

**Warm start, dAUC against plain BO on matched seeds:**

| task | plain BO | diversity | LLM | LLM-BO | LLM-diversity |
|---|---|---|---|---|---|
| albuterol_similarity | 0.5505 | 0.5556 | 0.5579 | +0.0074 | +0.0023 |
| celecoxib_rediscovery | 0.3969 | 0.4047 | 0.4010 | +0.0041 | -0.0037 |
| deco_hop | 0.6253 | 0.6197 | 0.6148 | -0.0105 | -0.0049 |
| median1 | 0.2598 | 0.2593 | 0.2579 | -0.0019 | -0.0014 |
| osimertinib_mpo | 0.7533 | 0.7532 | 0.7529 | -0.0003 | -0.0003 |
| **mean** | | | | **-0.0002** | **-0.0016** |

Nothing transfers. The LLM warm start beat plain BO on 2 of 5 tasks and its own
free control on 1 of 5 -- coin flips. That is what the budget arithmetic
predicts: the initial design is 2% of a 1,000-call budget here against 20% of a
50-call budget on the ChEMBL tier, so there is far less for it to move.

> **These are directional, not significant -- and cannot be.** PMO's convention
> is 5 independent runs, and a two-sided paired Wilcoxon on 5 seeds has a floor
> of **p = 0.0625**: even a perfectly consistent effect cannot reach p < 0.05.
> Six seeds is the minimum that can. The 20-seed ChEMBL tier is where effects
> are resolved; PMO is here for comparability against published baselines.

### Why the selection arm was not run here

One model call per *evaluation*, at a 1,000-call budget, 5 seeds, 5 tasks is
~25,000 calls -- roughly 15 hours. The ChEMBL selector run was 800 calls in 20
minutes. It is affordable at a 50-call budget and not at 1,000; that is a
property of the arm, and it is worth stating plainly rather than leaving as a
gap in the table.

## A measured limitation, fixed

Maximum *off-diagonal* Tanimoto in this library was **1.0** -- some distinct
library entries were **identical** to the GP. RDKit's Morgan generator ignores
stereochemistry by default, so E/Z and R/S isomers collapsed onto the same bit
vector while their measured activities did not:

```
CN(/N=N/c1ccc2ncnc(Nc3cccc(Cl)c3)c2c1)C(=O)OCCl    pIC50 6.41
CN(/N=N\c1ccc2ncnc(Nc3cccc(Cl)c3)c2c1)C(=O)OCCl    pIC50 9.41
```

A GP handed two identical inputs with different targets can only average them,
which puts an irreducible noise floor under every prediction and shows up as a
large fitted `eta` early in a run.

| ECFP4 setting | identical pairs | molecules | worst pIC50 gap | pairs > 0.5 log |
|---|---|---|---|---|
| `includeChirality=False` (RDKit default) | 133 | 204 of 2,086 | 3.00 | 42 |
| **`includeChirality=True`** (in use, `v2-chiral`) | **65** | **94** | **2.03** | **15** |

The remaining 65 pairs are regioisomers ECFP4 at radius 2 cannot separate
either; those need a larger radius or a different descriptor, not a flag.
Density (3.04%) and effective rank (10.1) are unchanged -- chirality alters
*which* bits are set, not how many -- so this is a targeted correction rather
than a different representation.

`chem.FINGERPRINT_VERSION` records the recipe and is part of the pooled
fingerprint cache filename, so a representation change can never be silently
mixed with stale data. `scripts/analyze_search_space.py` regenerates the
measurement.

**This mattered.** The LLM warm start was significant under the old
fingerprint (+0.030, p = 0.024) and is not under the corrected one (+0.017,
p = 0.23). The constraint and selection conclusions were unaffected.

## Search space and dimensionality

Not continuous box-bounded BO. The space is a discrete, fully enumerated set of
molecules; "bounds" means membership rules, not intervals, and the acquisition
function is maximised by **exhaustive scan** -- global by construction, no
inner optimiser.

| | |
|---|---|
| Search space | discrete, 2,086 molecules (PMO tier: 249,455) |
| Decision variable | one categorical index -- effectively 1-D |
| Membership bounds | anilinoquinazoline core; 10-55 heavy atoms; pIC50 in [4.00, 11.00] |
| Nominal feature dim | 2,048 ECFP4 bits (1,803 ever set, 1,800 informative) |
| Density | 3.04% -- 62 bits set per molecule |
| **Effective rank** | **10.1** (participation ratio of the Tanimoto Gram) |

2,048 nominal against ~10 effective is the whole argument for a Tanimoto kernel
over a squared-exponential one: Tanimoto has no lengthscale to fit, so the
curse of dimensionality never arises. Observed pairwise similarity is mean
0.30, median 0.28, 95th percentile 0.47 -- congeneric enough to generalise,
not so tight as to be degenerate.

## Secondary tier: PMO, for comparability

The primary oracle is one we built, which is exactly what reviewers distrust.
PMO (Gao et al., NeurIPS 2022) fixed 23 tasks, capped the oracle budget and
re-ran 25 algorithms, finding that most "state-of-the-art" methods stop beating
their predecessors once the budget is controlled. Running a subset of it is what
puts our numbers next to published ones.

```bash
python scripts/run_pmo_experiment.py     --tasks albuterol_similarity celecoxib_rediscovery deco_hop median1             osimertinib_mpo --seeds 5 --budget 1000
```

Search space is the **ZINC-250k pool**, so the directly comparable PMO baselines
are the library-based ones: `screening` (random sampling) and `mol_pal` (BO over
the pool). The generative baselines search unrestricted chemical space and will
score higher partly for that reason.

**AUC top-10. Ours: 1,000 calls, 5 seeds. Published: 10,000 calls, 5 runs.**

| task | ours `random` | ours `bo_ei` | `screening` | `MolPAL` |
|---|---|---|---|---|
| albuterol_similarity | 0.4133 | **0.5505** | 0.489 | 0.609 |
| celecoxib_rediscovery | 0.2853 | **0.3969** | 0.354 | 0.415 |
| deco_hop | 0.5723 | **0.6253** | 0.589 | 0.643 |
| median1 | 0.1620 | **0.2598** | 0.202 | 0.249 |
| osimertinib_mpo | 0.6984 | 0.7534 | 0.771 | 0.779 |

**Plain BO at 1,000 calls beats `screening` at 10,000 calls on 4 of 5 tasks**,
and beats `MolPAL` at 10,000 on `median1`. It is behind MolPAL by 0.015-0.048 on
the other four. With a 10x smaller budget that is a good showing, and it is the
point of this tier: it says the surrogate and the harness are sound, so a null
result on the LLM arms is the arm's fault and not the scaffolding's.

`zaleplon_mpo` was run but is **excluded from the comparison**: TDC's current
implementation uses a soft isomer term (a random ZINC molecule with formula
C20H20FNO2 scores 0.20, not 0), which cannot reproduce the published
`screening` value of 0.039. Our 0.4091 and their 0.039 are not measuring the
same function. Two other caveats worth stating whenever these numbers are
quoted:

* **The budgets differ by 10x** and are not interchangeable. An exact GP is
  O(n^3) per fit, so 10,000 observations is out of reach for this surrogate;
  1,000 is also the regime an expensive oracle actually lives in.
* The GP scores `--candidate-subset` (default 2,000) randomly drawn unevaluated
  molecules per round rather than all 250k. The *search space* is still the
  whole pool; only the per-round shortlist is sampled.

### Installing the PMO tier

PyTDC pulls `tiledbsoma` (single-cell data, needs a build toolchain) and its
oracle module still does `from rdkit.six import iteritems`, which modern RDKit
dropped -- TDC hides the resulting ImportError behind a misleading "install
rdkit" message. Both are worked around:

```bash
uv pip install --no-deps PyTDC==0.4.17
uv pip install fuzzywuzzy seaborn "setuptools<81" mygene huggingface_hub networkx biopython
```

`molbo.oracles.pmo.install_rdkit_six_shim()` supplies the one function TDC
actually uses (`iteritems`, called once inside the synthetic-accessibility
score, and identical to `dict.items()` on Python 3). No oracle arithmetic is
touched. 22 of 23 tasks load; `jnk3` fails on an sklearn pickle dtype.

## Oracle plumbing

The rules that make budget numbers comparable, enforced in
`src/molbo/oracles/base.py`:

- **Memoise on canonical SMILES; count unique calls only.** BO over a discrete
  library re-proposes the same compound constantly. Counting repeats makes
  budget numbers meaningless and incomparable to PMO. Two caches are kept
  deliberately separate: the *run memo* is what the budget is spent against (a
  molecule the method has not seen is charged even if an earlier run computed
  it — the method did not know), and the *value cache* (SQLite, cross-run) is
  purely a compute saver.
- **Invalid proposals still cost budget.** A generative arm must not be able to
  dodge the budget by emitting garbage.
- **A campaign may never re-propose a molecule.** The oracle would not charge
  for it, so it would be a stealth budget extension.
- **The initial design is arm-independent.** Seed *s* hands every arm the same
  starting molecules, so arms can be compared with a paired test. This matters
  more than it sounds: unpaired, seed-to-seed spread swamps the between-arm
  difference.

### Boltz-2 determinism

Boltz-2 is **nondeterministic by default** and this oracle pins it two
independent ways, because one is not enough:

- **Per-ligand seed.** `boltz predict --seed` defaults to `None` — no seeding at
  all. The seed is derived from `sha256(canonical_smiles)`, so a ligand's value
  depends on the ligand and nothing else: not on when it was proposed, not on
  what it was batched with. (`hash()` is deliberately avoided — Python
  randomises string hashing per process.)
- **`gamma_0 = 0`.** The seed alone only makes a run *repeatable*; the sampler
  is still a stochastic SDE. In Boltz-2's sampler:

  ```python
  gammas    = torch.where(sigmas > gamma_min, gamma_0, 0.0)
  t_hat     = sigma_tm * (1 + gamma)
  noise_var = noise_scale**2 * (t_hat**2 - sigma_tm**2)   # == 0 when gamma == 0
  ```

  so `gamma_0 = 0` makes `t_hat == sigma_tm`, the injected-noise variance
  collapses to exactly zero, and sampling becomes the deterministic
  probability-flow ODE. Boltz-2's defaults are `gamma_0 = 0.8`,
  `gamma_min = 1.0`.

  `gamma_0`/`gamma_min` are **not** CLI flags — only `step_scale` is — so they
  are applied by patching `Boltz2DiffusionParams` in the launched process.

Both are in the cache key, so changing either invalidates cached values rather
than blending two samplers' numbers into one table. To study noise
deliberately, set `seed_salt` or a non-zero `gamma_0` — an explicit arm, not an
accident.

`affinity_pred_value` is returned **negated**: Boltz reports log10(IC50) in µM
where lower is more potent, and every tier here maximises.

### LLM determinism

The default backend is a **local llama.cpp server** (`LlamaCppBackend`), which is
the better scientific instrument: `temperature=0` with `top_k=1` is greedy
decoding and the seed is fixed, so the model is deterministic on its own, and
llama.cpp compiles the JSON schema to a GBNF grammar, so a schema-valid response
is guaranteed by construction. It is also free.

```bash
llama serve -hf unsloth/Qwen3.5-4B-GGUF:Q8_0 --port 8080     --ctx-size 16384 --n-gpu-layers all --reasoning-budget 0
```

`--reasoning-budget 0` matters: Qwen3.5 is a hybrid reasoning model and will
otherwise spend its budget thinking before emitting the JSON.

`AnthropicBackend` remains available. Claude Opus 5 **removed**
`temperature`/`top_p`/`top_k` -- sending them is a 400 -- so there is no "set
temperature to zero" option and the model is *not* reproducible on its own.

Either way, every response is cached on disk keyed by a hash of the exact
request **including the backend's sampling settings**. Re-running both budgets
made 0 model calls and 40 cache hits.

### Substructures are chosen, not written

Asked to write SMARTS, Qwen3.5-4B produced an invalid pattern **every single
time** (unbalanced ring closures); the constraint collapsed to a molecular-weight
range and the arm kept 89% of the library. That measures typing ability, not
chemistry. So `molbo.fragments` mines a menu of substructures that actually occur
in the library and the model returns indices into it. With the menu, **0 of 20
proposals were malformed**.

Each entry is counted with the *same query used at match time*, so the coverage
figure shown to the model is true by construction. This caught two real bugs: the
anilinoquinazoline core was being advertised as matching 25.3% of the library when
it actually matches 100% (the library was selected by that core), and generic
Murcko scaffolds were advertised at 50% coverage while matching nothing at all
(`MakeScaffoldGeneric` maps benzene to `C1CCCCC1`, which only matches aliphatic
rings).

## Metrics

The headline number is the **AUC of the top-*k* mean against cumulative unique
oracle calls**, which rewards finding good molecules early rather than
eventually.

`metrics.top_auc` is a faithful port of PMO's implementation, kept bug-for-bug
so numbers line up with its published baselines. `metrics.auc_top_k` computes
the same quantity exactly. They are not interchangeable: PMO samples the curve
every `freq_log=100` calls and starts the trapezoid at zero, which
**systematically understates AUC by roughly 4 points** at a 1,000-call budget —
larger than the between-arm differences this project is trying to resolve.
Use `auc_top_k` for internal comparisons and `top_auc` when quoting against
PMO's tables, and always state the budget.

Raw pIC50 must be scaled with `normalize_values` using the **library's** range,
not the observed range — otherwise the scaling depends on what each arm
happened to find and the arms stop being comparable.

## Layout

```
src/molbo/
  chem.py          canonical SMILES — the single memoisation key for everything
  surrogate.py     exact Tanimoto-kernel GP (numpy; no torch) + LibraryKernel
  acquisition.py   EI / UCB / PI / greedy / Thompson, batch selection
  constraints.py   SMARTS + descriptor constraints, degradation ladder, controls
  llm.py           the arm under test: Claude proposes a constraint; cached
  campaign.py      one run: an arm, a seed, a budget
  experiment.py    sweeps, paired comparisons, result files
  metrics.py       PMO AUC, top-k curves, effect sizes
  oracles/
    base.py        memoisation, unique-call budgets, traces
    lookup.py      PRIMARY: measured pIC50 by dictionary lookup
    boltz2.py      REALISM: co-folding affinity, made deterministic
scripts/
  build_chembl_series.py    ChEMBL -> congeneric series
  constraint_headroom.py    run this BEFORE an arm
  run_lookup_experiment.py  the sweep
```

## Agentic BO (branch `agentic-bo`)

A fourth coupling is under construction: an LLM steering the optimiser itself
rather than advising it about chemistry, following
[SARA](https://github.com/uta-smile/AgenticBo) (arXiv 2608.00316).

Two of SARA's actions are arms this project already ran -- "modify search
bounds" is `bo_llm_constraint` (-0.030, p = 0.006) and "select evaluations" is
`bo_llm_select` (neutral). What is untested is whether a model can choose the
*acquisition function* from the surrogate's own diagnostics, which it has never
been shown. `ArmConfig.policy_source` is that hook, with the usual ladder of
controls and a hindsight ceiling.

Measure the headroom before spending a model call:

```bash
python scripts/acquisition_headroom.py --dataset data/chembl/egfr_anilinoquinazoline.csv
```

See [AGENTIC.md](AGENTIC.md). Written but **not yet run**.

## Status

Done: the three-tier oracle layer, the GP surrogate and acquisition functions,
constraints with controls and a ceiling, the campaign loop, the sweep driver
with paired statistics, the headroom instrument, and the Boltz-2 oracle's
determinism plumbing. 221 tests pass.

Not yet run: the LLM arm itself (needs an API key — the plumbing and its cache
are in place and tested), the PMO tier, and any actual Boltz-2 prediction —
Boltz is not installed here and needs a GPU, so its flags, output paths and
dataclass names were verified against the Boltz source rather than by running
it.
