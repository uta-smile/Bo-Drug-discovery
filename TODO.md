# Handoff — multi-task BO for molecular screening

Last updated 2026-10-07. Detail for everything below is in `results/FINDINGS.md`
§§14–35; the write-up is `report/multitask_bo.tex`, which still states the
retired unanimity claim and needs §34 folded in.

**S40-S45 are written.** The short version: the primary history design is now
**uniform**, not top-k-withheld; `pooled` is out of every headline table; the
headline is **+11.49 +- 3.25 recall points, 6/6 families, p = 0.031 (the floor
at n = 6)**; and any transfer gain measured under EI against plain single-task
BO is inflated by an untuned exploration level (on JAK, +7.66 becomes +5.31
against an uncertainty-matched baseline). `report/paper.tex` carries all of
this. S44 adds the out-of-sample validation and is a negative: a rho-based
chooser does not beat always taking the better fixed option, in either design.
**S45 is the first advisor review** -- eight claims cut back, the paper
narrowed around the evaluation question, and one number that changed when it was
tested at the right unit. **S46 is the second**, which caught a wrong equation
S45 introduced and a significance test that was reading three blocks while
claiming eighteen. Neither changed a conclusion; both changed what may be
written.

**Two runs have results but no written section.** `results/auxiliary_candidates*.json`
(the co-measurement gate, §36) and `results/kiba_*.json` (the kinome panels, §37)
exist only as JSON. FINDINGS stops at §35. Nothing downstream should treat them
as established until they are written — see "The KIBA excursion" below.

## Getting oriented (do this before reading source)

The repo has a **graphify** knowledge graph at `graphify-out/`, refreshed
2026-09-29 and covering everything below. It returns a scoped subgraph, which is
far cheaper than grepping or reading files:

```bash
graphify query   "how does the campaign loop select compounds"
graphify explain "CoregionalisedGP"        # the multi-task surrogate
graphify explain "drop_censored"           # the censoring filter
graphify path    "Oracle" "run_campaign"   # how the oracle seam reaches the loop
```

Useful starting points: `dense_block_campaign.py` (the contribution),
`CoregionalisedGP` (the surrogate), `Objective` (the free/assayed split),
`Oracle` (the seam a prospective run would replace).

**Run `graphify update .` after editing code** — several entries below depend on
the graph being current. `graphify-out/wiki/index.md` is the broad-navigation
view; read `GRAPH_REPORT.md` only for whole-architecture questions.

---

## Where the research stands

**The claim that survives every control** (restated after §34 — the
unanimity clause is gone and the effect size is now conditional):

> Given free historical assay data on a **co-measured target family**, a
> coregionalised GP improves top-10 recovery over single-task BO when screening
> **22–33% of the library**. Across **six protein families spanning six
> mechanistic classes** the mean gain is positive in 12/12 window cells, and
> ΔAUC is positive in 26/26 cells of the full grid. It beats a model-free
> similarity baseline by ~14 points. Auxiliary assays are worth **~1/5** of a
> primary assay, so this depends on the history being already paid for.
>
> **How large the gain is depends on two measurable things** (§34): the task
> correlation between the cold target and its relatives (partial ρ = +0.55),
> and how much headroom single-task BO has left (partial ρ = −0.60). It also
> scales with **what share of the library the free history covers** — at
> matched 25% coverage the per-family spread collapses from +0.6…+20.0 to
> −2.8…+12.8 recall points.

**What §34 retired.** The earlier version of this claim said "unanimously across
all four held-out targets on two independent protein families". **Do not quote
that.** Over six families unanimity holds in 7/12 window cells, and the verdict
**flips in 5 of 12 cells** when the repeats go from 8 to 32 — including on JAK
at both window budgets. Unanimity is a minimum over four noisy per-target
estimates and does not converge; ΔAUC does (rank agreement ρ = +0.916 between
the two runs). The **leave-one-target-out self-check** that §25 recommended is
retired with it: its answer depends on which target you held out *and* on how
many seeds you ran.

| hypothesis | verdict |
|---|---|
| H1-split — spread a *fresh* budget across coupled targets | **rejected**, DOCKSTRING (0/21) and both measured blocks |
| H1-cold, separate per-target libraries | **no effect** (ChEMBL), cause **unknown** |
| H1-cold, co-measured panel | **holds** on 6/6 families, survives to campaign level (§34) |
| H1-cold gain is *unanimous* over held-out targets | **rejected** (§34) — 7/12 cells, verdict flips in 5/12 when repeats go 8 → 32 |

**Operating window** (fraction of library screened, six families):
<11% does nothing or costs you · **22–33% is where the gain is** · >50% decays
on recall as single-task saturates, while ΔAUC stays positive.

The window is a *proxy*. The direct quantity is **how much headroom single-task
BO has left**: "single-task recall in 40–85%" separates the strong cells slightly
better than "22–33% of library" (§34). Two measurable things set the size of the
gain — task correlation with the relatives, and the **share of the library the
free history covers**. The six families were not comparable on the second until
§34 controlled for it.

---

## The KIBA excursion (2026-10-07) — unwritten, and what it is worth

**Why it was run.** §36's gate found **no co-measured auxiliary outside a
target's own paralogue group anywhere in BindingDB**. That is a fact about how
med-chem data is generated — a team makes fifty analogues, runs them against its
own target and the three paralogues it needs selectivity over, and publishes — so
the blocks come out family-shaped because the experiments were. Kinome profiling
panels are generated the other way round, one compound set pushed across a slice
of the kinome, so if cross-family co-measurement exists anywhere it is there.
`scripts/fetch_kinase_panels.py` pulls both public ones into the dump's
five-column shape, which is why `find_dense_blocks` and `auxiliary_candidates`
read them through `--source` with no other change.

* **DAVIS is excluded by measurement, not by assertion.** 68 compounds × 379
  kinases, the matrix *completely* observed — the cleanest geometry in the field
  — and still unusable, because the protocol needs 160 compounds on the compound
  axis and this has 68. Fetched anyway so the sentence is a measurement.
* **KIBA clears the floor**: 2,068 × 229, ~25% observed.

**What came out** (`results/dense_blocks_kiba.json`, `kiba_block_rho.json`, three
campaigns at ~22% of library, 8 repeats × 4 targets, audit clean):

| block | targets | median ρ | ΔAUC mt | Δ pooled |
|---|---|---|---|---|
| k01 `kiba_prkc` | PRKD3/PRKCG/PRKCZ/PRKCD | 0.99 | +0.250 ± 0.053 | +0.230 |
| k02 `kiba_cdk5` | PIM1/PRKACA/CDK5/DYRK1A | 0.78 | +0.104 ± 0.032 | +0.126 |
| k10 `kiba_lck` | LCK/MAPK8/MAPK9/GSK3B | 0.48 | +0.036 ± 0.046 | **−0.156** |

**This is the first direct test of priority #2 below.** §34 could only measure
relatedness as a partial ρ = +0.55 *across six families*, each with its own
library, assay history and coverage. These three blocks are one panel, one
protocol, one compound axis — so the monotone fall of the gain with task
correlation, and its disappearance at the cross-family block, is relatedness
with the confounds held fixed. Cross-family transfer does not fail for want of
co-measured data: here the data exists and the transfer still dies.

**Three things to fix before any of it is quotable.**

1. **k01 is near-tautological.** ρ = 0.99 means the four columns are nearly one
   column; +0.250 is an upper reference line, not a result, and should be
   labelled the way `--acquisition oracle` is.
2. **The censoring gate was bypassed.** 14 of 15 blocks are `runnable: false`
   (n_clean 25–99 from ~900 raw). The campaigns ran at n_pool = n_raw, i.e. with
   `--keep-censored`. Defensible — KIBA's pile-ups are not BindingDB's reporting
   ceilings — but it has to be argued in §37, not left implicit.
3. **8 repeats is the count §34 showed is unreliable.** Quote ΔAUC only. Do not
   quote `frac_better`; it is the statistic that flipped in 5 of 12 cells going
   8 → 32.

The scale was **not** a problem: `--potency identity` exists for exactly this and
was used (the block CSVs hold raw KIBA scores ~11.5, not 9 − log₁₀). The seam was
already there; `build_block`'s docstring records the +0.855 Spearman check
against this dump's own pIC50 on shared JAK2 compounds.

**Next on this line**, in order: fix (1) and (2) by labelling, re-run the three
at 32 repeats, then extend from three points to a curve — every block over the
size floor, ΔAUC against median ρ, to locate the ρ where the gain crosses zero.
That threshold is the deliverable; three points are not.

### The threshold, confirmed on real weak relatives (`results/cross_family_scan.json`)

The ladder degraded auxiliaries *uniformly at random*; real weak coupling is
structured. Until a scan over genuinely weak relatives agreed, the number was a
bound and not an answer. It agrees.

**Why a new search was needed.** `find_dense_blocks.search` grows a block by
*maximum shared compounds*. Kinases that were profiled together are relatives —
which is exactly why all 15 blocks it returned sit at rho >= 0.277 and why the
sweep crossing was an extrapolation. **Searching for size finds families.**
`cross_family_scan.py` searches for coupling instead: random 4-subsets of the
173 KIBA targets with >= 160 measurements, keep the 7,210 whose fully observed
block has >= 300 compounds, compute median Spearman on the **raw** block,
stratify into six bins (four of them below 0.28), and within a bin take the
blocks whose size is **closest to the global median** — not the largest, because
size would otherwise drift with coupling and confound the fit.

Result: 18 blocks, 428–433 compounds, rho from −0.015 to +0.433, size and
coupling independent (rho_s = −0.058, p = 0.82). Genuinely cross-family: the
weakest is {ERBB2, LCK, AKT1, FLT1} — an RTK, a Src-family kinase, an AGC
kinase and a VEGF receptor.

| arm | fit (x) | rule | rho_s(x, gain) | p | crossing | 95% CI |
|---|---|---|---|---|---|---|
| stl | block rho (raw) | `aux_top` | +0.725 | 0.00066 | +0.090 | [0.01, 0.15] |
| stl | effective rho | `aux_top` | **+0.759** | **0.00032** | **+0.081** | [0.01, 0.13] |
| stl | effective rho | `aux_scaffold_nn` | +0.459 | 0.055 | +0.030 | [−0.16, 0.16] |
| mt | block rho (raw) | `aux_top` | +0.884 | <1e−4 | +0.068 | [−0.01, 0.12] |
| mt | effective rho | `aux_top` | **+0.897** | **<1e−4** | **+0.058** | [−0.01, 0.11] |

**The two instruments agree, on both arms.** stl: crossing +0.081, CI
[0.01, 0.13], contains the ladder's +0.062. mt: crossing +0.058, CI
[−0.01, 0.11], contains the ladder's +0.013. **Four estimates, two mechanisms,
two surrogates — all between 0.01 and 0.09.**

The mt relation is the *stronger* of the two (rho_s = +0.897 vs +0.759) while
its level is lower (+0.056 ± 0.019, positive 14/18, p = 0.0056 vs
+0.062 ± 0.026, 12/18, p = 0.048). The coregionalised surrogate spends some of
the same information, which flattens the level; it does not move where the level
reaches zero. The threshold is a property of
the coupling, not of how the coupling was produced. That is the claim the ladder
alone was not entitled to make, and priority #0 is now closed.

**Visible without a fit.** Split at rho = 0.28, the floor of everything the
size-greedy search returned:

| | blocks | stl mean | pos | mt mean | pos |
|---|---|---|---|---|---|
| rho < 0.28 | 12 | −0.004 | 6/12 | +0.012 | 8/12 |
| rho >= 0.28 | 6 | **+0.195** | **6/6** | **+0.144** | **6/6** |

Above the line it pays on every block; below it the gain is zero on average and
a coin flip block by block. Pooled: +0.062 ± 0.026, positive on 12/18,
signed-rank p = 0.048 — far below the +0.262 of the within-family sweep, which
is the point. That sweep sampled only relatives, and most kinase pairs are not
relatives.

**`maxmin` replicates its failure** on a completely different block set:
negative on 16/18 under stl (−0.140 ± 0.025, p = 0.0001) and 15/18 under mt
(−0.086 ± 0.018, p = 0.0003). Now measured on 15 within-family and 18
cross-family blocks across both surrogates — **the most reliably signed effect
in the project, and it is a negative one about the textbook default.**

**The gate flips, and this is the actionable finding.** It says **room on
18/18 cross-family blocks under stl and 16/18 under mt** — 34 of 36 cells —
against INERT on most within-family ones and 5/6 from a bad start. Where relatives are distant the free rules have little to work
with, so the gap between the ceiling (mean +0.458) and the best free rule stays
open.

> **Build the rule where the relatives are close; build the chooser where they
> are not.** An agent should be tested on cross-family cells and on
> `kiba_lck`-shaped blocks, with `aux_top` as its control on identical inputs.
> On a within-family panel it is being asked to beat something free that already
> reaches the ceiling.

**A defect fixed on the way.** `rho_eff` guarded only the auxiliary side of the
Spearman. A block whose *primary* is constant on its history rows (k08, DAPK3 /
SRC / PIM3 / TSSK2) yields NaN, and one NaN makes `np.median` NaN, silently
poisoning every fit using that axis as x. Now guarded on both sides, with
`nanmedian` as a second line of defence. Verified non-disruptive: a stored
ladder cell reproduces to 1e−12 with rho_eff and the triple count unchanged.
Regression test in `tests/test_bad_start.py`.

**Still open:** both instruments run on KIBA. What is shown is that the
threshold does not depend on *how the coupling was weakened*, not that it is
panel-independent. Davis is registered in `PANELS` and needs no new code.
Cross-family blocks are also smaller (~430 vs ~900), so history coverage is 35%
vs 17% — size is constant *across bins* so it cannot drive the slope, but the
absolute gains are not comparable with the sweep's.

### The bad start, closed (`results/bad_start_gate.json`)

Akke et al. (2025, SPIGM@NeurIPS, OpenReview `GExQdGDD6g`) compare reasoning
LLMs, agentic workflows and classical BO on molecular and protein design. Their
clearest molecular win for the agent is **not** on a normal campaign — it is on
one *initialised in a bad local minimum*. So the value of a chooser is
concentrated in a regime the **initial design** created. Their four campaigns
are single-target, so they have no relative to borrow a start from and cannot
ask whether the regime survives a free good start. We can.

`init_headroom.py --start bad` replaces the baseline designs with a low-scoring
compound plus its nearest Tanimoto neighbours — low *and* chemically tight. The
free rules are untouched. 12 cells, audit clean on all 12.

| block | arm | bad start costs | `aux_top` vs random | vs bad | recovered | gate rnd | gate bad |
|---|---|---|---|---|---|---|---|
| prkc | stl | −0.752 ± 0.060 | +0.652 | **+1.455** | 1.94× | INERT | INERT |
| prkc | mt | −0.205 ± 0.039 | +0.273 | +0.459 | 2.24× | INERT | INERT |
| cdk5 | stl | −0.416 ± 0.068 | +0.347 | +0.685 | 1.65× | room | INERT |
| cdk5 | mt | −0.189 ± 0.102 | +0.169 | +0.318 | 1.68× | INERT | INERT |
| lck | stl | −0.383 ± 0.065 | −0.012 | +0.303 | 0.79× | room | **room** |
| lck | mt | −0.415 ± 0.056 | +0.020 | +0.344 | 0.83× | room | INERT |

**The rule is immune, not remedial.** On prkc/stl: gain over random (+0.652) plus
the cost (0.752) = 1.404, against a measured +1.455 over the bad start. The rule
does not escape the local minimum — it never enters it.

**The gate fires *more* from a bad start: INERT on 5/6 vs 3/6 from random.**
Damaging the start opens room for the **free rule**, not for a chooser. On
prkc/stl `aux_top` ends up **+0.513 ± 0.079 above the hindsight ceiling over bad
starts** — the best reachable choice inside the bad regime is worse than not
being in it.

**Where room survives:** one cell of six, `kiba_lck` stl, +0.261 ± 0.095 over
`aux_scaffold_nn`. The same block that never paid from a random start and showed
no ladder. That is the cell shape to test an agent on, with `aux_top` as its
control on identical inputs — and it is a minority of the panel.

**Correction to an earlier read of ours.** I had argued the LLM warm-start
negative result was unfair because the agent was never shown the history. Half
right. Akke et al.'s `SIMPLEAGENT` finding is that **removing** sample-level data
(SMILES + scores) from the prompt and keeping only a natural-language cycle
summary makes the model *better* — with the raw table it copy-pastes substrings
instead of forming chemical hypotheses. So the fix is not "put the family history
table in the prompt"; it is a natural-language summary plus tools, with `aux_top`
as a tool rather than a table. Any agent experiment on this line must be built
that way.

### What was built for it (2026-10-07)

Additive only — no stored result changes, no existing numeric path touched, full
suite green.

- `dense_block_campaign.PANELS` — each kinome panel paired with the `--potency`
  it requires. The two settings are not symmetric in failure: `nanomolar` on a
  KIBA score silently **inverts** the objective and optimises for the weakest
  binders, and no audit invariant catches it.
- `dense_block_campaign.PANEL_FAMILIES` — the three KIBA blocks pinned by
  UniProt as `kiba_prkc`, `kiba_cdk5`, `kiba_lck`, with `--source` and
  `--potency` filled in from the prefix. They had been reached as `--blocks-from
  results/dense_blocks_kiba.json` + `k01`/`k02`/`k10`, and **those labels are
  positions in a scan, not names** — re-run `find_dense_blocks` at a different
  `--min-per-target` and `k10` is different proteins, with nothing in the stored
  results to show they had moved. `--blocks-from` still works and is still how to
  reach the other twelve.
- `dense_block_campaign.block_provenance` — a new `provenance` key on the report:
  source, potency, blocks-from, target labels, block CSV, size before and after
  filters, and **whether the censoring filter ran at all**. None of that was
  stored, so a KIBA run and a BindingDB run were indistinguishable in `results/`
  once the scrollback was gone. Written with `getattr` so `dense_block_h1` and
  `selectivity_campaign` can call it rather than grow a second copy.
- `tests/test_block_provenance.py` (11) — including a tie from `PANEL_FAMILIES`
  back to the gene labels in `results/kiba_block_rho.json`, so the pinned set
  cannot drift from the artefacts that cite it.

The three stored `results/kiba_*.json` are **left as they are**. They predate the
field and backfilling provenance into a stored artefact is inventing a record of
a run nobody watched; the reconstruction is above, in prose, where it is visibly
a reconstruction. Re-runs will carry it for real.

## History-aware seeding (2026-10-07) — the ρ prediction, tested and held

**The question.** Can the free history be spent on *where to start*, not just on
the surrogate? The initial design is the one hook a language model has already
been given here, and the one it lost on: `bo_llm_warmstart` beat plain BO
(+0.0239, p = 0.027 at budget 100) and then lost to `bo_diverse_warmstart`, a
free MaxMin rule drawing from the same sample, at both budgets (−0.0049,
−0.0137). But that arm's prompt says literally *"Nothing has been measured
yet"* — it was **never shown the auxiliary history**. So that result bounds the
chemistry question, not this one.

**The gate.** `scripts/init_headroom.py`, built to AGENTIC.md's standing rule:
score candidate initial designs on an otherwise identical run, take the max over
12 random draws as the hindsight ceiling, and report **ceiling minus the best
free rule** — never ceiling minus random, because a free rule that beats random
is a better fixed policy and crediting it to an agent is exactly the mistake
`bo_diverse_warmstart` was built to catch. All runs `--history-excludes-top`,
audit clean (960 checks, 32 cells each), ~3 s per cell-grid.

**The ρ axis had to be recomputed.** `find_dense_blocks --labelled` measures task
correlation on the *censored-filtered* block, and on KIBA that filter leaves
25–99 of ~900 rows, while the campaigns run on the raw block with
`--keep-censored`. The stored `median_rho` is therefore a correlation over a few
dozen compounds and the gain is measured over nine hundred. Ordering survives,
values do not: prkc 0.991 → **0.866**, cdk5 0.781 → **0.505**, lck 0.482 →
**0.277**. Every ρ below is measured on the block the campaign actually saw.

| block | median ρ (campaign's block) | `aux_top` ΔAUC vs random (stl) | | headroom over best free rule |
|---|---|---|---|---|
| `kiba_prkc` | 0.866 | **+0.652 ± 0.068** | 9.6 SEM | +0.068 ± 0.078 → **inert** |
| `jak` (BindingDB) | 0.541 | **+0.143 ± 0.030** | 4.8 SEM | +0.196 ± 0.037 |
| `kiba_cdk5` | 0.505 | **+0.347 ± 0.054** | 6.4 SEM | +0.176 ± 0.061 (stl); **inert** on mt |
| `kiba_lck` | 0.277 | −0.012 ± 0.095 | 0.1 SEM | +0.631 ± 0.105, unreachable |

The two blocks that pay sit at ρ = 0.51 and 0.87; the one that does not sits at
0.28. **§8's independently derived prior threshold is ρ ≈ 0.37**, and it falls
between them. Two routes to the same number — a prior-mean bar derived from
campaign AUC on single-task BO, and a seeding gain measured on co-measured
panels — is worth more than either on its own, and it was not designed for.
`jak` at 0.54 breaks the monotonicity in gain (+0.143 below cdk5's +0.347 at
ρ = 0.51), so ρ is not the only term; `jak` is also the only non-KIBA block here
and the only one whose campaign ran *with* the censoring filter.

**Four findings.**

1. **Spending the history on the seed set works, and it is worth a lot** — up to
   +0.65 AUC against a random initial design at 9.6 SEM, from a rule with no
   model in it. This is the first thing in the repo that makes the free history
   pay *before* the first batch is bought.
2. **It falls with task correlation — but far more weakly than four blocks
   suggested.** See the correction below; the four-block version of this line
   claimed a monotone law and a locatable threshold, and the 15-block sweep
   supports neither.
3. **No leakage.** `aux_top` seeds **0.00** of the true top-10 outright on every
   block. The gain is transfer, not the history handing over the answer.
   (`aux_scaffold_nn` seeds 0.09–0.34, which is why its number is the weaker of
   the two and is reported next to it.)
4. **MaxMin — the rule that beat the LLM — is negative on every dense block**
   (−0.06 to −0.78). Diversity seeding is a good control on one congeneric
   ChEMBL library and actively harmful on a co-measured panel. Do not carry that
   result across.

**What this means for an agent.** The gate says *no* on three of six cells and
the three where it says yes are the ones where a free rule already took most of
the ceiling. In particular, on `kiba_cdk5` with the multi-task surrogate — the
configuration the project actually recommends — the hook is **inert**: the
coregionalised GP was already spending the same history, and seeding is a second
channel for it rather than new information. The honest reading: **build the free
rule, not the agent.** An agent on this hook would have to beat `aux_top` on
identical inputs, and `aux_top` is six lines.

**Where an agent is still unbeaten-but-untested:** `kiba_lck` at ρ = 0.48 has a
+0.63 ceiling that *no* free rule reaches and the history cannot touch. That is
the only cell in the grid with room and no incumbent. It is also the cell where
the history is worthless, so whatever fills it will not be transfer.

### Correction, after the 15-block sweep (`results/seed_rho_curve.json`)

The four-block table above was built on blocks picked for spread, and §34's
lesson repeated itself almost exactly. Run over **all 15 KIBA blocks × 2 arms**
(audit clean on 30/30 cells, `scripts/seed_rho_curve.py`); the figure is
`report/fig_seed_rho.pdf`.

| claim | four blocks | fifteen blocks |
|---|---|---|
| seeding beats a random start | yes, 4/4 | **yes, 13/15 stl and 14/15 mt** — mean +0.262 ± 0.053 and +0.154 ± 0.026, signed-rank **p = 0.0009** / **0.0003** |
| the gain grows with ρ | "monotone" | ρs = **+0.579 (p = 0.024)** stl; **+0.400 (p = 0.140, n.s.)** mt |
| …and survives dropping the one high-ρ block | not checked | **it does not**, for `aux_top`: +0.481, **p = 0.081**. It does for `aux_scaffold_nn`: +0.618, p = 0.019 |
| the ρ crossing can be located | "bracketed 0.28–0.51" | **no.** Fitted at 0.153, CI [0.09, 0.32] — below *every* block measured |

**What survives is the level, not the slope.** Seeding the initial design from a
relative's measured data is worth +0.26 AUC on average and beats a random start
on 13 of 15 panels at p < 0.001. That is the result. The dose-response in ρ is
real but weak, is carried by one leverage block (14 of 15 blocks sit in
ρ 0.28–0.51; `kiba_prkc` alone sits at 0.87), and is not significant at all under
the multi-task surrogate.

**The crossing is not locatable from KIBA**, and the earlier wording was wrong to
imply otherwise. Every block sits at ρ ≥ 0.277; the fitted zero crossing is an
extrapolation *below the data*. Locating it needs blocks at ρ ≈ 0.1–0.25, which
KIBA does not contain — those must come from a cross-family scan or from the
synthetic `--aux-quality` ladder, and the second is already built.

**ρ is not a per-block decision rule.** In the band ρ = 0.33–0.39 the gain runs
from **−0.215** (k12) to **+0.405** (k14), both at ~0.07 SEM. A practitioner
cannot read ρ off their own panel and predict what seeding will buy — the same
per-target lottery §24 found, one level up.

**What is unambiguous:** `maxmin` is negative on **15/15** blocks under stl
(mean −0.283, signed-rank p = 0.0001) and 13/15 under mt. The free diversity rule
that beat the LLM warm start on one congeneric ChEMBL library is harmful on every
co-measured panel tested.

### The threshold, located (`results/seed_quality_ladder.json`)

The 15-block sweep could not place the crossing because the panel has no weak
blocks. The `--aux-quality` ladder makes weak ones: it degrades every auxiliary
to a requested Spearman with its own true values, leaving the primary's column
and the scoring untouched, so the coupling can be walked down through the region
KIBA does not cover. Three blocks × eight rungs × both arms, 8 repeats, audit
clean on 48/48 runs. Figure: `report/fig_quality_ladder.pdf`.

**The x axis is measured, not requested.** Every run records Spearman between
the values a rule actually sees and the primary's true values on those same rows,
and reports the median over (target, repeat, auxiliary) triples. `q = 0.3` on
`kiba_cdk5` is an *effective* ρ of 0.132, and that is the number a threshold can
be stated in.

| arm / rule | pooled fit | crossing | 95% CI | inside the data? |
|---|---|---|---|---|
| `stl` / `aux_top` | ρs = **+0.757**, p < 1e-4 | **+0.062** | [−0.00, 0.11] | **yes** |
| `stl` / `aux_scaffold_nn` | ρs = **+0.837**, p < 1e-4 | +0.035 | [−0.03, 0.09] | **yes** |
| `stl` / `maxmin` | ρs = −0.229, p = 0.28 | — | — | no |
| `mt` / `aux_top` | ρs = **+0.762**, p < 1e-4 | **+0.013** | [−0.11, 0.10] | **yes** |
| `mt` / `aux_scaffold_nn` | ρs = **+0.785**, p < 1e-4 | +0.022 | [−0.15, 0.13] | **yes** |
| `mt` / `maxmin` | ρs = +0.757, p < 1e-4 | +0.762 | [0.49, 1.00] | yes — see limit 4 |

**`mt` replicates the shape and halves the size.** The slope is the same
(+0.762 vs +0.757) and the crossing lands in the same place, but the gain at the
top rung drops from +0.652 to +0.273 on `kiba_prkc` and +0.347 to +0.169 on
`kiba_cdk5`. That is the expected direction and the reason it matters: the
coregionalised GP already spends the auxiliary columns, so seeding from them is
partly redundant with the surrogate. Roughly **half** the stl gain is something
`mt` was already getting for free. The other half is not, and `mt` is the
configuration this project recommends.

**Within a block, where the ladder is the only thing that moved** — this is the
causal estimate; the pooled fit above mixes it with between-block differences:

| block | ρs(coupling, gain) | p | crossing |
|---|---|---|---|
| `kiba_prkc` | **+0.976** | <1e-4 | **+0.118, inside the data** |
| `kiba_cdk5` | **+0.881** | 0.0039 | −0.161 (still positive at zero coupling) |
| `kiba_lck` | +0.381 | 0.35 (n.s.) | flat at zero throughout |

Under `mt` the same three, weaker and noisier: `kiba_prkc` ρs = +0.857
(p = 0.0065), `kiba_cdk5` +0.619 (p = 0.10, n.s.), `kiba_lck` +0.690 (p = 0.058).
Only `kiba_prkc` is significant on its own, which is consistent with a halved
effect at unchanged repeats rather than with the effect disappearing — but it is
the reason the *pooled* mt fit should not be quoted without the within-block
row beside it.

**The null control passes**, and it is what makes the rest readable: at requested
`q = 0` (effective ρ +0.023) the gains are, under `stl`, `aux_top` −0.027,
`aux_scaffold_nn` −0.058, `maxmin` −0.454; under `mt`, +0.007, −0.064, −0.393.
A pure-noise auxiliary buys nothing on either arm. This is §32's control, re-run
on this hook.

**The headline the ladder adds:** seeding keeps paying down to a coupling of
**ρ ≈ 0.06–0.12** (`stl`; `mt` crosses at +0.013 with a CI that straddles zero,
so quote the stl number as the conservative one), which is *far* below §8's
prior-mean bar of **0.37**. The two
hooks do not have the same requirement, and the cheaper one tolerates much
weaker relatives. That is the most useful number this line has produced: it says
a relative worth nothing to a prior can still be worth something to a seed.

**Three honest limits.**

1. **`kiba_cdk5` does not reach zero.** It sits at +0.062 at effective ρ = −0.005.
   Either the pool restriction itself helps on that block independently of
   coupling, or the null has ±0.1 of block-level spread. The three-block mean is
   −0.027, so the crossing estimate carries that uncertainty and the CI's lower
   bound touching zero is honest rather than conservative.
2. **`kiba_lck` has no ladder at all** — flat at zero from ρ 0.29 down. A block
   that was never paying cannot show a crossing, and it is one of the three.
3. **A synthetic ladder degrades uniformly at random.** Real weak coupling is
   structured — a relative that ranks one chemotype well and another not at all.
   So this brackets the crossing under the *most favourable* kind of weakness.
   A cross-family scan is still owed, and the two must be checked against each
   other before the number goes in a paper.
4. **The `mt` ladder is a joint intervention, not a clean one.** `--aux-quality`
   degrades the auxiliary columns *before anything reads them*, and under `mt`
   the coregionalised surrogate reads them too — for the seeded design and for
   the random baseline alike. So an mt rung varies what the seeder sees **and**
   what the model sees, while an stl rung varies only the first. The tell is in
   the table: `maxmin` uses no history at all, yet its mt slope is +0.757, which
   can only be the random baseline moving under a degraded surrogate. The stl
   ladder is therefore the clean causal instrument and the mt one is a
   replication under a confound — a benign one, since the mt null at q = 0 is
   +0.007, so the confound is not manufacturing the gain at the bottom rung.

### What was built for it

Additive only — `--seed-pool random` is the default and reproduces every stored
number; full suite green (463).

- `dense_block_campaign.history_seed_pool` — `aux_top` (each auxiliary's
  best-scoring history compounds, z-scored per target) and `aux_scaffold_nn` (a
  deliberate superset: those, plus Bemis-Murcko scaffold-mates, plus each one's
  Tanimoto neighbours). Reads no primary value; with `--aux-quality` it seeds off
  the degraded numbers, which is what a computed auxiliary would really give.
- `--seed-pool {random,scaffold_split,aux_top,aux_scaffold_nn}` on the campaign,
  plus `--seed-top`/`--seed-nn`. **A property of the cell, not of an arm** — every
  arm in a run shares the pool, so `audit.same_across(..., "initial design")`
  still holds and the comparison against `random` is between runs, paired on
  `--seed`, exactly as `--scaffold-split` always was. `--scaffold-split` keeps
  working as a spelling of one mode; conflicting with `--seed-pool` is a
  fail-fast error raised before the block is cut.
- `run_campaign(init_idx=...)` — hands the initial design in instead of drawing
  it, so several candidates share one rng stream and differ *only* in where they
  started. `None` is the default and byte-identical to before; a test pins that.
- `murcko_labels` factored out of `scaffold_disjoint_pool`, which now takes a
  precomputed `scaf` — the RDKit parse is the expensive part of all three rules.
- `scripts/init_headroom.py` — the gate. Reuses `build_block`, `draw_history`,
  `run_campaign`, `history_seed_pool` and `block_provenance` rather than
  reimplementing a campaign; the only thing it adds is `maxmin_design`.
- `scripts/seed_rho_curve.py` — the gate over a whole scan, with the task
  correlation recomputed on the block the campaign sees (the stored `median_rho`
  is measured on the censored-filtered block, which on KIBA is 25-99 of ~900
  rows). Carries `robustness`: signed-rank on the level, and a
  drop-the-leverage-block refit on the slope.
- `scripts/figure_seed_rho.py` — `report/fig_seed_rho.{pdf,png}`, plus the table
  view its palette's relief rule requires.
- `scripts/seed_quality_ladder.py` + `scripts/figure_quality_ladder.py` —
  `report/fig_quality_ladder.{pdf,png}`, the figure that can carry a threshold
  because the coupling is varied rather than observed. Carries `within_block`
  fits and `null_control`.
- `init_headroom.py --aux-quality`, and `rho_eff` on every report: the coupling
  a rule actually had, median over (target, repeat, auxiliary) triples.
- `init_headroom.py --blocks-from` — reaches the 12 blocks nothing is pinned to.
- `tests/test_seed_pool.py` (9), `results/init_headroom_*.json` (7),
  `results/seed_rho/*.json` (30), `results/seed_rho_curve.json`.

### Next on this line

0. ~~**Get blocks below ρ = 0.28**~~ — **CLOSED, both halves.** Synthetic:
   `seed_quality_ladder.py`, 48 runs, crossing +0.062 (stl) / +0.013 (mt),
   inside the data, null clean on both arms. Real: `cross_family_scan.py`, 18
   genuinely cross-family blocks spanning ρ = −0.015 to +0.433, crossing
   **+0.081, CI [0.01, 0.13]**, which **contains** the ladder's number. The
   threshold is a property of the coupling, not of how it was weakened, and can
   now be quoted plainly. Remaining caveat: both run on KIBA, so panel
   independence is untested — Davis is registered and needs no new code.
1. **Spend the budget, not just the seed.** n_init is 10 of 60 here, so the hook
   can only move ~17% of a campaign. `aux_top` as a *prior mean* on the GP, or as
   a standing admission filter, reaches the other 83% — and §9 says a prior and
   an acquisition cannot be chosen independently, so run it under UCB as well
   as EI.
2. **Locate the ρ crossing.** Four points bracket it between 0.48 and 0.78. Run
   every KIBA block over the size floor and fit it; that threshold is the
   deliverable a practitioner can act on before committing an assay.
3. **Only then an agent**, on `kiba_lck`-shaped cells, with `aux_top` as its
   control on identical inputs.

## What changed this session

### New source
- `src/molbo/surrogate.py::CoregionalisedGP` — ICM over the Tanimoto kernel,
  `B=(1-r)I+r11ᵀ`, `r` by exact LML. `structure="free"` fits a full matrix
  (retired, see below). Per-task z-scoring; **predictions return on the
  caller's scale** — an earlier version didn't, which silently broke EI.
- `src/molbo/dockstring.py::Objective` — owns the free/assayed split
  (`assayed()`, `free()`, `admits()`, `reweighted()`) plus `*_potency` variants.

### New scripts
| script | what it answers |
|---|---|
| `transfer_headroom.py` | H1-split, ceilings, joint-vs-separated, DOCKSTRING cold-target |
| `objective_floor.py` | how much of an objective is free, as a function of QED weight |
| `cold_target_chembl.py` | H1-cold on separate ChEMBL libraries |
| `dense_block_h1.py` | BindingDB dense blocks; censoring + drug-likeness filters; free `B` |
| `dense_block_campaign.py` | campaigns, baselines, budget/reach sweeps, multi-objective |

### Tests
`tests/test_coregionalised.py` (11), `tests/test_dockstring.py` (12). Full suite green.

### Data-quality findings that protect other people's work
- **Published DOCKSTRING objectives are ~93% a free descriptor.** QED-only ranking
  recovers 48.1% of the top-50 with zero assays; 80 assays reach 50.6%.
- **The BindingDB CYP block is 77% censored** — fixed screening doses (10/30/50 µM)
  stored as potencies; 40.3% of compounds identical across all four targets.
  `--drop-censored` keeps 22.8% of CYP, 57.8% of JAK.

### Hypotheses we tested and **falsified** (don't redo these)
- **Free `B` beats one shared `r`** — loses on both blocks (−0.003 to −0.029).
  Six correlations from 10–40 observations is overfitting. Retired.
- **Chemical overlap explains the ChEMBL null** — gain is *flat* from reach 0.96
  to 0.44, and real libraries sit at 0.53. Withdrawn.
- **Regression gain predicts campaign gain** — opposite sign. Campaign gain
  *grows* with budget; regression gain shrinks.
- **Drug-likeness as a Pareto objective** — fronts are 2–10 compounds, QED-only
  recovers 20–50% of them with zero assays. Retired behind `--multiobjective`.
- **Post-hoc drug-likeness scoring** — costs 13–17 recall points vs filtering.

### Where a free descriptor belongs (settled)
score term ✗ · Pareto objective ✗ · scoring-time filter ✗ · **constraint on
spending ✓**. `--report-druglike 0.5 --restrict-selection` (budget confined to
eligible compounds, free history spans the whole library) is best at low budget.

### Verification infrastructure
`--audit` on both `dense_block_campaign.py` and `dense_block_h1.py`. Checks equal
initial design across arms, shared history, exact budget, no repeat picks, no
eligibility escapes, no top-k leaking into withheld history, no auxiliary column
equal to the primary's values, evaluation/training disjointness. **Both validated
by deliberate injection.** Runs are byte-identical across invocations.

---

### The section-32 artefact, separated (`results/FINDINGS.md` S40)

Two new flags on `dense_block_campaign.py`, both no-ops by default:
`--sd-scale` (multiply the posterior sd before the acquisition reads it) and
`--aux-shuffle` (permute an auxiliary column instead of corrupting it).
`--sd-scale 1.0` reproduces the stored `auxq_jak_0.0` single-task arm to the
last digit.

**S32 split in two.** On JAK its explanation closes exactly: a single-task arm
with **no auxiliary data at all**, told only to be 20% less uncertain, gains
**+2.34 +- 0.73** recall points (p = 0.0022) against the noise multi-task arm's
+2.03 +- 0.86. The 0.80 was predicted beforehand by `null_mechanism.py`, which
measures posterior sd and never sees a campaign.

On CYP it fails. No uniform sd scale **in a 57-fold range, both directions**,
reproduces more than +0.08 of pooling's +10.55.

**What is underneath on CYP is a leak in the experimental design.** Under
`pooled` (r = 1) every history compound is treated as already observed and gets
near-zero EI, so the arm avoids the history -- and `--history-excludes-top`
guarantees the answer is *outside* the history. Removing that flag flips the
sign almost exactly: **+22.97 -> -23.05** at H = 100. The effect collapses onto
one axis, **what fraction of the library the history covers** (9% -> -1.2,
17% -> +0.6, 22% -> +2.3, 45% -> +10.6, 90% -> +23.0, across two blocks).

**What survives.** `mt` passes its null control at *both* settings of the flag
(-0.47 and -2.73, both n.s. or near) while its real-panel gain is large at both
(+9.69 and +11.25, p < 0.0001). On CYP the coregionalisation's advantage is
entirely transfer -- **stronger** than S32 said. Both nulls (corrupt and
permute) agree in every cell, so none of it is the auxiliary column's shape.

**The correction that touches every table.** Read against an
uncertainty-matched baseline rather than plain single-task BO, transfer on JAK
is **+5.31 +- 0.91** rather than +7.66 -- real, p < 0.0001, and ~31% smaller
than this package has been quoting since S18.

Two things this creates:

0. **S34's coverage moderator needs re-reading.** S34 found that controlling for
   "what share of the library the free history covers" collapsed the per-family
   spread from +0.6...+20.0 to -2.8...+12.8. That is the *same axis* the S40
   artefact runs on. S34's numbers are `mt`, which is far less exposed, but the
   coincidence is close enough that it should not be quoted as guidance until a
   null arm has been run at matched coverage.
1. **`pooled` is retired for null work.** S31 was already retracted on other
   grounds; a result that changes sign with a protocol flag is a second,
   independent reason.

### The controls, finished (`results/FINDINGS.md` S41-S43)

Target is **UAI**; no LLM experiment in this paper. `QUESTIONS.md` records the
decisions and what each one changed.

**S41 -- the mechanism, run down.** Both explanations the real blocks could
offer are dead. The GP is *under*-dispersed on both (std(z) = 1.4-4.8), so the
"EI over-explores because the error bars are too wide" story fails in direction
and magnitude and does not separate JAK from CYP. And the compounds the two
exploration levels buy are statistically identical (isolation p = 0.29, value
p = 0.51 on JAK). What survives is an accounting identity: of the 52 picks per
campaign that change, the top-10 hit rate doubles from 0.5% to 1.0%, which is
**+2.61 predicted against +2.34 measured**. The gain is a small-probability tail
shift, not a change in search behaviour.

**S42 -- the headline under the honest design.** All twelve window cells re-run
with a **uniform history**, which is now the primary design. `mt` *improves*:
+9.37 -> **+11.49** recall points, 11/12 -> **12/12** cells, 6/6 families
(p = 0.031, the floor at n = 6). The cell that had been negative, pi3k at
-0.23, becomes +4.30. `pooled` collapses: +8.89 -> +3.16, 10/12 -> 7/12,
4/6 families, **p = 0.44** -- under the honest design it has no result, and it
is out of every headline table.

**S43 -- the controlled simulation.** On synthetic data where surrogate quality
is a dial, the optimal sd scale falls as quality rises:
**rho = -0.546, p = 0.043**, confirmed on fresh libraries at 14 quality levels,
independently on a smooth landscape and one with activity cliffs. A statistic
pre-registered from the pilot (gain at a fixed x0.6) **did not replicate**
(p = 0.16 / 0.76, and the mean gain there is negative) -- recorded because
pre-registering is only worth doing if the failure is reported.

**Three new tools, all reading stored artefacts:**

* `scripts/leak_audit.py` -- which stored results are exposed to the design
  leak, and how far. **122 artefacts carry a `pooled` arm under the flag**, at
  coverages from 14% to 90%.
* `scripts/block_level.py` -- restates any sweep at campaign / cell / block
  level, with the **signed-rank floor**. At six blocks that floor is 0.031, so a
  p at the floor is a unanimity statement and carries no effect-size
  information. The campaign-level p on the same contrast is 2e-91 and must not
  be quoted.
* `scripts/sd_scale_simulation.py` -- the S43 simulation, driving `run_campaign`
  unchanged.

**Two new campaign flags, both no-ops by default and verified against stored
artefacts:** `--sd-scale` (multiply the posterior sd before the acquisition
reads it) and `--aux-shuffle` (permute an auxiliary column instead of corrupting
it; it agrees with `--aux-quality 0` in every cell, so none of this is the
column's shape).

**Standing rule this adds.** Three mechanisms can make auxiliary data look
useful, and each needs its own control: information (`--aux-quality 0`),
exploration level (`--sd-scale` at the measured ratio), and the history design
(cross `--history-excludes-top`). `mt` passes all three on both blocks; `pooled`
fails the third.

### Deferred, with the reason (not for want of effort)

* **E1, the blinded time-split.** The BindingDB block dumps carry no date column
  at all (`smiles,ABCC3,ABCC4,...`), and the TDC source is the five-column
  ID/SMILES/Y shape. Only the ChEMBL cache has `document_chembl_id`, and that is
  the line of work where the effect did not replicate anyway (S17). Doing this
  properly means re-fetching BindingDB with deposition dates, or resolving
  ChEMBL document years over the network, and then re-cutting blocks. It is
  worth doing and it is a day, not an evening.
* **C2(ii), a rule on *predicted* coupling.** Needs the sequence-identity
  baseline, which is cheap but was not reached. C2(i) -- the rule on measured
  coupling, validated leave-one-block-out -- is built
  (`scripts/transfer_rule.py`, tested) and running.

### The decision rule, validated out of sample (S44) -- a negative

72 cells: three uses of a free history (abstain / seed / transfer) crossed with
both history designs, on the 18 cross-family blocks, 32 repeats, all audited
clean. `scripts/run_xfam_options.py` runs them; `scripts/transfer_rule.py` fits
the rule on 17 blocks and spends the held-out one.

**Which fixed policy wins depends on the design, not the chemistry.** Uniform
history: seed +0.1293 (17/18) beats transfer +0.1089 (12/18). Withheld:
transfer +0.0494 (12/18) beats seed +0.0350 (10/18). The cause is measured --
the 30-compound `aux_top` pool holds **1.73 of the true top-10** under a uniform
history (0.70 expected at random) and **0.00** when the top-k is withheld.

**A rho-based rule beats neither.** Uniform: -0.0048 +- 0.0027 against always
seeding, better on **0/18** blocks, p = 0.11. Withheld: +0.0055 +- 0.0040
against always transferring, 4/18, p = 0.17. Not a fit failure -- coupling
predicts *both* options almost identically (rho_s = +0.849 and +0.880 under
uniform), so their **difference** carries no signal (-0.278, p = 0.26). There is
nothing for a threshold on rho to separate.

**But the decision problem is real.** The per-block ceiling is 10% above the
best fixed policy under uniform and 34% under withheld. Something predicts the
right choice; rho does not. That is the first place in this package where a
richer feature is justified by a *measured* gap rather than by assumption, and
it is exactly the slot C2(ii) describes -- a feature computable before any
paired assay data exists, since S36 found such data does not exist outside a
paralogue group.

**This retires a standing line.** After S34 the guidance was *"build the rule
where the relatives are close; build the chooser where they are not."* Under the
uniform design the seeding rule is better on **both** sides of rho = 0.28:
+0.0805 against +0.0538 below, +0.2269 against +0.2189 above. **Do not quote the
split.**

### The advisor review, applied (S45)

The UAI draft came back with a list of places where the claims outran the
evidence. All eight are applied in `report/paper.tex`; S45 records what each
one was and why.

**One of them was not a wording fix.** The between-design change was tested
over the *twelve budget cells*, and two budgets on one family are not two
independent units -- the same argument S41 makes one level down, not applied
here. At the family level:

| contrast | cells (12) | **families (6)** |
|---|---|---|
| `mt`, uniform - withheld | +2.12, p = 0.052 | **+2.12 +- 1.04, 4/6, p = 0.16** |
| `pooled`, uniform - withheld | -5.73, p = 0.0034 | **-5.73 +- 2.01, 0/6, p = 0.031** |

So the uniform design is **not** shown to improve `mt`; it is shown not to cost
it anything, which is all the paper needs. `pooled`'s decline is unanimous.
**Do not quote the cell-level p for either.**

The other seven, in one line each: "reproduces the whole effect" is now "a gain
of the same size"; the `Not transfer` 27%/60% column is gone (a ratio of two
arms that differ in more than one thing is not an identified fraction);
`pooled` gets an interval instead of "entirely"; the simulation correlation is
out of the abstract and labelled exploratory; `D_q` is written out from
`degrade_to_rho`; "at r = 0 it reduces to single-task GPs" is qualified, since
S40's own diagnostic contradicts it; and the per-block ceiling is now "best per
block in hindsight", with no claim that something predicts the right choice.

**The paper is narrowed.** Controls, mechanism diagnostics and the corrected
cross-family evaluation are the main text. The seeding-threshold study (the old
`sec:res-sweep` and `sec:res-rule`) moved to *Secondary and supporting
analyses*, labels intact. New title: *When Auxiliary Data Help Molecular
Bayesian Optimization: Transfer, Exploration, and History-Selection Bias.*

Code added: `block_level.py --against`, which pairs two sweeps cell by cell and
runs the existing three-level aggregation on the difference. Default path
unchanged; four tests pin it.

### Advisor round two: two real defects, both mine (S46)

S45 fixed the overclaims and **introduced a wrong equation**. The second review
caught it. Two of the six remaining items were defects, not wording.

**1. `q` is a mixing parameter, not an achieved correlation.** The noise in
`degrade_to_rho` is standardised but **not orthogonalised** against the signal,
so `a = Corr(z, e)` is O(1/sqrt(n)), the standard deviation is not preserved and
the sample Pearson correlation is not exactly `q`. The correct closed forms are
`Var(y~)/s^2 = 1 + 2q*qbar*a` and `Corr(y~,y) = (q + qbar*a)/sqrt(1 + 2q*qbar*a)`.
`scripts/degradation_audit.py` confirms them against the implementation to 1e-13
and measures the cost: unbiased to 0.002, draw-to-draw sd up to 0.055, variance
off by at most 0.0018. **Both terms carry `q*qbar`, so they vanish at q = 0 and
q = 1 -- where every control sits.** The controls are fine; the description was
not.

**2. The rule-vs-fixed test saw three blocks, not eighteen.** The rule picks the
fixed policy on 15 of 18 blocks, so the difference is exactly zero there.
`scipy.wilcoxon` drops zero pairs and, because zeros are present, also drops its
exact path. The quoted `p = 0.11` / `p = 0.17` were normal approximations over 3
and 6 pairs. Correct and exact on the discordant blocks: **p = 0.25 and
p = 0.22**. Conclusion unchanged, and better supported.

**S45's "every signed-rank p is exact, since n <= 18" was wrong for the same
reason.** Audited: the family-level tests (n = 6) are exact, verified against
`method="exact"` directly; the campaign level has **481 zeros and 1037 ties**
among 1536 pairs and was always an approximation. `block_level.report()` now
records zeros, ties and the method, and a floor is only labelled on an exact
test.

**Counts resolved from the run records**, since Methods and captions disagreed:
`auxq_*` is 8 seeds x 4 targets = **32 campaigns**; `sdscale_*`, `win32_*`,
`winleak_*`, `xfam_*` are 32 x 4 = **128**. The null table mixes the two, which
is why its matched-uncertainty rows have smaller SEMs than the q = 0 rows they
are read against.

**Standing instruction: do not write a formula into the paper that has not been
run against the code it describes.** Both defects came from describing an
operator or a test from memory. The audit script and the zeros/ties fields exist
so the next description can be checked.

### Advisor round three: the numbers are generated now (S47)

Two defects again, and the second is the one that mattered: **S46's own
"correction" to the campaign counts was wrong, in the opposite direction.**
Three rounds of prose gave three statements of one count. The fix is not a
fourth round of prose.

**1. A mean quoted as a maximum.** S46 said the variance ratio departs from 1
"by at most 0.0018". That is the deviation of the *mean*. The per-draw sd is
**0.055** at q = 1/sqrt(2) -- where the deviation equals `a` exactly -- with a
max of **0.17** over 400 draws. Understated by ~105x.

**2. The null table comes from `sdref_*`, not `auxq_*`.** 32 seeds x 4 targets,
**128 campaigns, every row**. The original caption was right and S46 moved a
correct number to an incorrect one. The reason the inventory did not catch it:
I wrote it from a **hand-written list of groups**, and `sdref` was not on it.

**The inventory now discovers groups by scanning `results/`.** Nine tests pin
it, including the exact failure mode -- a new prefix must be picked up with
nobody editing the script. An inventory that can omit an experiment is not an
inventory.

**Third appearance of the unaudited-run mistake.** Once the inventory could see
everything, it found `sdref_*` and `sdscale_*` ran without `--audit` -- the
artefacts behind the null table *and* the matched-uncertainty claim.
`rerun_audited.py` rebuilds each command line from the artefact's own recorded
parameters, re-runs with the audit on into `results/audited/`, and compares.
**sdscale: 20/20 clean and byte-exact.** sdref: clean and exact so far.

**The evidence package** (`report/paper.tex`, `app:repro`):

| script | establishes |
|---|---|
| `degradation_audit.py` | the closed forms match the implementation (8e-14); what the operator delivers, as mean **and** spread |
| `experiment_inventory.py` | every group's protocol, discovered by scanning; fails on conflict; checks seeds x targets == stored array length |
| `regenerate_tables.py` | all three main tables recomputed from artefacts; zeros/ties/exactness per test; `--check` verifies the numbers are in the manuscript |

`regenerate_tables.py --check report/paper.tex` passes; all eight published
null-control values reproduce exactly.

**Standing instruction, strengthened.** S46: do not write a formula into the
paper that has not been run against the code. **S47 adds: do not write a number
into the paper that was not generated from the run records.** Both defects this
round were citing from memory -- once an operator's properties, once an
experiment group.

### Preprocessing audited, paper swept before external review (S48)

`scripts/preprocessing_audit.py` recomputes every quantitative statement in the
Data section from `data/bindingdb/*.csv` and exits nonzero on a mismatch.
**19 of 20 matched exactly** -- compound counts, survival rates, modal shares,
the identical-on-all-four fractions, and all nine block n/rho values. The five
pIC50 spikes the paper names are exactly the fixed doses 5/10/30/50/100 uM.

**The one mismatch:** the KIBA surviving-row range, quoted as 25-99. It is
**32-263** on the three pinned blocks -- the quoted range excluded `kiba_prkc`
(263 of 904), one of the three blocks the sentence names. Corrected. It
slightly *weakens* the paper's own argument, which is the right direction for a
correction to go.

**Two claims narrowed in the sweep.** The permutation null was run on **CYP
only**, not "every cell" -- rewritten to say so, and it gained the
uniform-design figure (`pooled` **-19.69**) that reproduces the design-leak sign
reversal with a null preserving the marginal exactly. And `shuf_cyp_*` had also
run without `--audit`; re-run clean and exact (2/2).

**The inventory now splits on history design** rather than calling a deliberate
both-designs pairing a conflict, while still failing on a real one (two repeat
counts sharing a name).

**Checked and found correct:** all ten headline quantities agree across
abstract/results/discussion/conclusion; every DOCKSTRING/QED number traces to
S14 (sd ratio 2.02-2.65, rho +0.929, rho(obj,-JAK2) +0.007, QED-only 48.1%,
best arm 50.6%, free share 95% -> 33%); the permutation figures trace to
`shuf_cyp_excl`; block provenance matches the artefacts.

**The evidence package is now four scripts**: `preprocessing_audit.py`,
`experiment_inventory.py`, `regenerate_tables.py`, `rerun_audited.py`, all
wired into `app:repro` in the paper.

### Advisor round four: the evidence package replaces the assertions (S49)

The advisor's real point was item 4: *"the reproducibility claims still require
evidence. I cannot independently confirm those checks from their
descriptions."* Correct, and the fix is a package, not a sentence.

**`scripts/build_evidence.py`** builds a directory and a **1.7 MB zip**:
scripts (67), `src/molbo` (28), every cited artefact (170, including
`results/audited/` and its 26 logs), the block CSVs (53), the captured stdout
and exit status of six checks, and `MANIFEST.sha256` over all 335 files.
**The outputs are produced by running the checks inside the package at build
time**, not transcribed, and the manifest is written last so it covers them.
**A failing check fails the build**; six tests pin that, including a
deliberately failing check. All six checks exit 0.

**A third maths error.** S47 said both deviations carry `q*qbar`. True of the
variance ratio, **false of the correlation**, whose deviation is `qbar^3 * a`
to first order -- **`a` at q = 0**, zero only at q = 1. S47's own next
paragraph said this correctly, so the paper contradicted itself across two
sentences. Now generated by `degradation_audit.py` with five parametrised
tests. Third maths error in four rounds, each found by someone reading the
prose rather than by me running the code.

Two more wording fixes: the noise-independence sentence contradicted itself
(now: the underlying Gaussian draws are independent; sample standardisation
induces dependence within a column), and the permutation null was oversold
(it preserves the empirical marginal exactly; its achieved correlation stays
random).

**Inventory terminology.** "Campaigns/cell" collided with the Methods' use of
*cell* for one compound-target assay -> **campaigns per configuration**. While
renaming I printed `blocks x budgets x campaigns` as the group total, and that
identity is false: `win32` has 6 blocks and 12 distinct budgets but **12
configurations**, not 72. Now **total = configurations x campaigns per
configuration**, exact for every group. **16,128 campaigns in total.**

**The audit column counts re-runs.** It read 0/4 for `sdref`, contradicting the
appendix. It now looks for the audited twin in `results/audited/`. Final tally:
**26/26 clean and reproducing exactly.**

## Next — in priority order

### 1. ~~Close the audit gap~~ — **done**, and it found something (§30)
Both scripts now have `--audit`. The checks live in `src/molbo/audit.py` and are
unit-tested (`tests/test_audit.py`, 24 tests, each firing a violation on purpose
with a clean twin). Checks A and B came back clean. Check C and
`cold_target_chembl.py` failed one invariant in **every cell**: `H` was in the
RNG seed, so the `H = 0` control and every `H > 0` arm were fitted on different
primary rows and scored on different held-out rows. Not a leak — the gain stays
unbiased — but the comparison was never paired, which is the same defect
`dense_block_h1.py`'s `--holdout` was added to fix and which was never carried
across.

`--c-paired --c-holdout 0.4` (and `--paired --holdout 0.4`) bind them. Both
default off so the stored tables stay reproducible. Re-run paired:

* **§16 holds**, break-even still H = 5–10 and 4/4 targets, but the gain shrinks
  in 15 of 18 cells — headline +0.445 → **+0.427**.
* **§17's null holds and tightens**: SEMs fall ~40%, largest gain in the grid
  halves to +0.008 ± 0.007.
* New number: cross-library overlap exposes only **0.09%** of the primary's
  evaluation compounds to an auxiliary measurement — §17's overlap argument now
  rests on a measurement.

**Quote the paired numbers.** Anything from the unpaired tables is superseded.

While checking provenance for the above, two more things turned up and are fixed:

* **FINDINGS had no §25.** It jumped 24 → 26, and §26 cited "the 4/4 unanimity
  of §25". The project's headline claim — the 22–33% operating window — had
  artefacts and a paper table but no written section. **Now written** from the
  existing `hb_*` artefacts, re-run audited as `results/win_{jak,cyp}_*.json`.
* **`report/multitask_bo.tex` table `tab:h1cold` was wrong**, independently of
  the pairing fix: it mixed H = 200 (DOCKSTRING/ChEMBL) with H = 50 (blocks)
  under a caption claiming H = 200 for both, and printed an n = 80 ChEMBL result
  in the n = 40 column. Rebuilt at H = 50 for every row. The "roughly double"
  claim in the surrounding prose was an artefact of that mismatch — at matched
  n and H it is 1.2x to 3x.
* **`dense_block_campaign.py` now writes its audit verdict into the JSON**, as
  the other two do. It printed to the terminal and nowhere else, which is how
  the window table went into a paper unaudited without leaving a trace.

### 2. Why does cross-family transfer fail?  *(first direct evidence, 2026-10-07)*
Chemical overlap is excluded (§23). **Target relatedness is the surviving
candidate**, and the KIBA blocks test it with the confounds held fixed — one
panel, one protocol, one compound axis, ρ = 0.99 / 0.78 / 0.48 — with the gain
falling monotonically and vanishing at the cross-family block. See "The KIBA
excursion" above for the three fixes this needs before it is quotable, and for
why the deliverable is the ρ threshold rather than the three points. The
paragraph below is the original framing, kept because it is what the block
cutter was built to do and still the way to reach a new block: ChEMBL used four different kinase *families*; the
blocks are one family each. Need a block spanning families with measured
cross-correlations. Screen `results/h1_dense_blocks.json` candidates for censoring
first — `graphify explain "build_block"` shows the cutter, which takes any
UniProt set via `--uniprot P12345=LABEL ...`.

### 3. A pre-flight predictor  *(what makes it deployable)*
Unanimity at 22–33% means a leave-one-target-out self-check is *valid* there, but
nothing predicts **which** targets gain most (spread +1% to +15%). Candidate
features computable before committing: the cold target's library tightness, where
its actives sit relative to the history. If none predict, say so — "works in
expectation over a portfolio, not on the single target in front of you" is an
honest and useful claim.

### 4. Firm up two soft results
- **`restricted` vs `pre-filter`**: +10.0 vs +4.4 at budget 62, but reverses at
  187, on only 8 repeats × 4 targets. Needs more repeats before it's a recommendation.
- **Batch size is a confound across sweeps.** Batch 5 and batch 10 runs are not
  comparable. Pin one and note it.

### 5. ~~Selectivity as a contrast objective~~ — **done (§33)**, and it is the biggest real-world result here
`scripts/selectivity_campaign.py`. Two findings, opposite directions:

* **The objective choice is worth +39.7 recall points (p < 0.0001)** on JAK —
  regress the selectivity contrast instead of potency, at identical cost and on
  identical data. Nothing else in this repo is worth as much. The selectivity
  top-10 and the potency top-10 are **disjoint** on 3 of 4 JAK targets.
* **The multi-task surrogate is worth nothing, even here.** `compose`
  (coregionalised per task, then compose) ties `contrast` (a plain single-task
  GP on `y_sel`): −2.2 and +1.6 points, both n.s. This was the objective where
  ICM *should* win structurally. It does not — and §32's variance artifact
  *flatters* `compose`, which still does not win.
* **CYP: nothing works.** No arm separates from potency.
* Integrity checked, not assumed: `integrity()` reports the binding anti-target
  values of the top-k — 3.70–7.82, near-unique, so real chemistry rather than a
  reporting floor. One contaminated compound found and reported
  (`[4.7, 4.7, 9.7, 4.7]`, in JAK3's top-10); CYP3A4 sits at 20%.

**Next on this line, and now the highest-value direction:** a selectivity
*window* (threshold not margin), a weighted anti-target panel, and
potency-under-a-selectivity-constraint. Those are the shapes a programme
actually specifies, and the lesson of §33 is that the target variable buys far
more than the model does.

### 6. Escape the retrospective frame
Cheapest first:
- **Blinded time-split** — split BindingDB/ChEMBL by deposit date; pre-cutoff is
  history, post-cutoff is unseen. Free, attacks the library pre-enrichment
  optimism directly. *Do this one.*
- **Prospective against a compute oracle** — point the loop at a purchasable
  catalogue subset with no precomputed values, let Boltz-2 compute on demand. The
  `Oracle._evaluate_batch` seam already supports it. First test under genuine
  uncertainty.
- **Wet lab** — one cold kinase with a family panel, ~200 assays split between
  `mt` and `stl`, threshold-hit metric. Only thing that changes the work's category.

### 7. LLM controller  *(the original goal — now unblocked)*
Surrogate, baselines and operating window are fixed, so an LLM that adapts
acquisition settings / exploration weight / target allocation can be measured
against a stable reference. **Required control:** a non-LLM adaptive heuristic
with the same action space, or "LLM adaptive > fixed" proves only that adaptation
helps.

---

## Housekeeping / traps

- **`report/multitask_bo.tex` still carries the unpaired §16/§17 numbers.** §30
  supersedes them; the paired tables are in `results/cold_target_audit_paired.json`
  and `results/cold_target_chembl_paired.json`. Regenerate before submitting.
- **`--c-paired`/`--paired` and `--c-holdout`/`--holdout` default off**, like
  `--holdout` in `dense_block_h1.py`, so §§16–17 stay reproducible. Flip all four
  defaults once the old tables are regenerated — same decision, same reason.
- **`report/multitask_bo.tex` has never been compiled** and there is no TeX
  engine on this machine (`pdflatex`/`xelatex`/`latexmk`/`tectonic` all absent;
  `uv tool install tectonic` does not resolve). `tools/texcheck.py` is the
  stand-in: environments, labels/refs, table cell counts, cmidrule and
  multicolumn spans, brace balance, math-mode delimiters, non-ASCII, and
  `\label` before `\caption` inside floats. It passes. That is not a
  substitute for a compile --- run `pdflatex` twice before trusting layout.
- **(superseded) `report/multitask_bo.tex` has never been compiled** — no TeX toolchain on this
  machine. Structure-checked only (7 tables, cell counts verified). Run `pdflatex`
  twice before trusting layout.
- **`--holdout` defaults to 0**, kept so §§14–18 stay reproducible. It varies the
  test set with `H`. **Use `--holdout 0.4` for anything quoted.** Flip the default
  once the old tables are regenerated.
- **`--drop-censored` defaults off** and 77% of the CYP block is junk without it.
  Consider defaulting on.
- The filtered **CYP** comparison in §26 is confounded (its `H` had to drop from
  50 to 20 for a 192-compound pool) and is not reported.
- **The unanimity result is demonstrated at N ≈ 900.** Do not assume it holds at
  arbitrary library size — a random cut to 566 degrades it (§26).
- **§24 is batch 5, §25 is batch 10.** The window table and the assay-budget
  sweep are not comparable cell by cell. Still unresolved (item 4 below); §25
  states the window in its own units and says so.
- **Read §25's regime 3 on recall, not AUC.** ΔAUC stays above 3 SEM at every
  JAK budget including 66%; it is final recall that converges. "The method stops
  working past 50%" would be wrong — it still gets there sooner, there is just
  nothing left to win.
- PowerShell `Select-Object -First N` closes the pipe and kills a running script
  (exit 255). Write to a file, then read. Cost two re-runs this session.

---

## Standing rules earned the hard way

1. **Budget is cells, not molecules.** Comparing at equal compounds hands the
   multi-task arm more data.
2. **Always run the pooling control** (`r` fixed at 1). On JAK it is *worse* than
   using no auxiliary data at all; on CYP it ties the fitted model.
3. **Report a floor, never zero** — chance for a filtered objective, the free
   ranking for a weighted one, and the zero-assay recall of every free objective
   in a multi-objective benchmark.
4. **Report per held-out target, not just pooled.** At small budgets one target
   gained 15 points while another lost 15, pooling to a meaningless "+3.7%".
5. **A regression curve does not predict the campaign curve, even in sign.**
   Drive anything new through a campaign, not just a ρ.
6. **Run the control before writing the number.** Every headline this session
   shrank when its control was run; none reversed in sign. Seven for seven
   including §30.
7. **A gain is a paired difference or it is not a gain.** Whatever varies
   between the arms — `H`, budget, surrogate — must not also vary the split they
   are differenced on. `--audit` checks this now; it found the defect in 240 of
   240 cells across two scripts, and nothing crashed.
8. **The error bar is across held-out targets, not repeats.** So pairing tightens
   it only where there is no real between-target signal. In §30 that asymmetry
   was itself the evidence: the ChEMBL SEM collapsed under pairing and the
   DOCKSTRING one did not.
