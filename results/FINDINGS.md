# Where is there room for a learned chooser in low-budget molecular BO?

A ceiling analysis across four intervention points, and a quantified answer to
why learned region selection does not pay.

> **Headline.** A learned prior must reach **Spearman ρ ≈ 0.37** with the true
> objective before it improves campaign AUC at all. Below that it actively
> hurts. Cold-target affinity prediction — the only kind available for a new
> target — measures **ρ ≈ 0.22**. That single number explains every negative
> result below without appealing to model quality: the prior route is closed
> not because the models tried were weak, but because the threshold is above
> what cold-target prediction can currently deliver. See section 8.
>
> **Second finding.** *How* a prior is applied matters more than its accuracy,
> and Expected Improvement wastes it entirely. A **perfect** prior scores
> −0.019 to −0.144 against plain BO under EI, and +0.057 to +0.113 under UCB.
> A prior and an acquisition function cannot be chosen independently.
>
> **Third finding.** Every chooser here is *forced to act*, and that is the
> expensive part. Allowed to **decline**, the same region prior at the same
> accuracy goes from −0.027 to **+0.0059** against a free bandit (p < 1e-4,
> **all 411 eligible targets**, and unchanged under UCB), and the break-even
> accuracy falls 70% → 20% as the gate's AUROC rises from 0.50 to 0.80. On BO's shortlist, where declining costs
> exactly nothing, break-even falls from 20% to chance the moment any gate
> exists. **Calibration substitutes for accuracy at about 3:1, and it is worth
> most exactly where accuracy is worst.** But **no free signal reaches the
> required gate quality**: 0.80 is needed at the prior's measured accuracy, a
> free signal reaches 0.581 (p < 0.001) and a gate *trained* on the campaign
> outcome reaches 0.590 (p = 0.005). A measured gap of ~0.21, not a null.
> See sections 12-13.

> **Correction, after the 24-target sweep.** An earlier version of this
> document reported the region hook as having "the largest measured headroom",
> on the strength of two hand-picked targets (+0.061, +0.077). Run over 24
> *randomly drawn* held-out targets, the hindsight ceiling averages **+0.0064**
> and is negative on 46% of them. The effect is real but **strongly
> conditional on how concentrated a target's actives are**, and the two
> original targets were unrepresentative. Section 3 is the corrected account;
> the two-target numbers are kept because the contrast is the finding.

> **Correction, after the six-family sweep (§34).** The same failure mode, in
> the multi-task half of this document. §25 built the central practical claim on
> two hand-picked protein families and reported the gain as **unanimous over all
> four held-out targets, eight times out of eight**. Run on six families chosen
> by an unbiased search over BindingDB, unanimity holds in **7 of 12** window
> cells; re-run at four times the repeats, the verdict **flips in 5 of 12,
> including on JAK itself**. The mean gain replicates everywhere — positive in
> 12/12 window cells, ΔAUC positive in 26/26 cells of the full grid — so the
> effect is real and the *unanimity* was the artefact. §34 also shows the
> JAK-vs-CYP effect-size gap was history coverage rather than the blocks, and
> that §31's verdict on naive pooling does not hold outside those two families.

Every number below is from `results/*.json`, 200 paired seeds unless stated.
Reference arm is plain BO (Tanimoto GP on ECFP4, Expected Improvement) on a
fixed library with a measured-pIC50 lookup oracle.

---

## 1. The four hooks, ranked by ceiling

A hook's *ceiling* is what a hindsight-optimal chooser achieves — it reads the
answer key, so it is a bound on every possible method, learned or not.

| hook | what is chosen | ceiling (ΔAUC) | best free control takes |
|---|---|---|---|
| acquisition function | 1 of 5, per round | **+0.0018** (p=0.90) single-task; **+0.0100 ± 0.0091** multi-task (§35) | round-robin and random both **0/12** |
| shortlist selector | 1 of BO's top 15 | **+0.031** (dz 0.89) | **0%** |
| region of chemical space, **random target** | 1 of k clusters | **+0.0064**, negative on 46% | — |
| region of chemical space, **concentrated target** | 1 of k clusters | +0.038 … +0.087 (dz 0.76–0.97) | 0–49% |
| library constraint | a sub-library, once | +0.094 (one target) | not re-measured here |

The acquisition hook was retired by a single script with no GPU — the ceiling
could not beat the best fixed acquisition.

> **§35 re-gates this row on the multi-task blocks**, because the number above
> is single-task and §9 says a prior and an acquisition cannot be chosen
> separately. The hook stays retired — 12-cell mean +0.0100 ± 0.0091, 1.1 SEM,
> p = 0.299 — but the *diagnosis* changes. The portfolio is degenerate: EI, UCB
> and PI agree at ρ ≥ 0.97 and share 67–79% of their picks, while `greedy` and
> `ts` are strictly worse in every cell. A near-zero ceiling over three
> interchangeable arms and two bad ones is a fact about the action space, not
> about adaptivity. §35 says what a non-degenerate portfolio would contain.
> It also confirms §34's EI-only choice cost nothing: EI is the best fixed
> acquisition in 11 of 12 multi-task cells, having been best in *none*
> single-task.

The region hook looks like the
largest headroom **only on targets whose actives are concentrated**; averaged
over 24 random targets it is +0.0064 and negative half the time (section 3).
No hook tested here clears its own free control on a target picked at random.

## 2. Region headroom is strongly budget-dependent

`results/cluster_curve_{cruzipain,usp7}.json`, k=10, 200 seeds. "Bandit" is
UCB over regions — no model, regret guarantees.

| budget | cruzipain ceiling | bandit share | usp7 ceiling | bandit share |
|---|---|---|---|---|
| 20 | +0.0379 | **−86%** | +0.0413 | **−40%** |
| 30 | +0.0610 | 2% | +0.0765 | 22% |
| 40 | +0.0639 | 23% | +0.0870 | 42% |
| 60 | +0.0540 | 39% | +0.0757 | 49% |
| 120 | +0.0319 | 48% | +0.0427 | 46% |

Monotone on both targets. **At budget 20 the bandit is worse than not
restricting at all** — with 10 regions and 20 calls it spends the entire budget
on first visits and never exploits. The mechanism is simple: *a bandit must pay
samples to learn which region is good; a prior does not.* So the room for a
prior is largest exactly where the budget is tightest.

This is the regime that matters: an assay that costs weeks is the reason
sample efficiency is studied at all.

## 3. On a random target, region restriction is worth nothing

`results/prior_sweep.json` — 24 held-out BindingDB targets drawn at random
from the 411 eligible, budget 30, 200 seeds each, one ridge fitted on the 583
targets not under test.

| hindsight ceiling over 24 random targets | |
|---|---|
| mean | **+0.0064** |
| median | +0.0057 |
| positive on | **54%** of targets |
| range | −0.0736 … +0.0708 |

**Even reading the answer key, restricting to the best region is worth
approximately nothing on a target picked at random**, and actively harmful on
nearly half of them.

What separates the targets where it works:

| predictor of the ceiling | Spearman |
|---|---|
| **best-region lift** (concentration of the top 1%) | **+0.602** |
| library size | −0.052 |

| subset | mean ceiling |
|---|---|
| all 24 random targets | +0.0064 |
| sweep targets with lift ≥ 10 (n=5) | **+0.0355** |
| cruzipain (lift 22) | +0.0610 |
| USP7 (lift 15) | +0.0765 |

So the hook's value is governed by **how concentrated the actives are**, not by
library size or budget alone. cruzipain and USP7 were chosen for being
undrugged and low-data — properties that happen to correlate with
concentration — which is why they looked far better than a random draw.

The awkward part: **lift is a hindsight quantity.** Deciding whether to use
region restriction requires knowing which compounds are best, which is the
thing being searched for. A method that helps on concentrated targets and
hurts on the rest needs to know which it faces, and nothing here can tell it.

## 4. Why a learned prior still does not pay — the asymmetry

We trained a region prior three ways and ran the two where it mattered.

**Prediction accuracy**, 120–152 held-out targets from BindingDB (607 targets,
543k measurements, sequences included), 10 regions each:

| chooser | top-1 | coverage |
|---|---|---|
| learned per-target prior (ESM2 × centroid ECFP4) | 0.08 – 0.19 | 0.15 – 0.25 |
| per-molecule ridge (ESM2 ⊕ ECFP4 → pIC50, 150k pairs) | 0.142 | 0.267 |
| **pick the largest region** | **0.248** | **0.377** |
| random | 0.100 | 0.147 |

**Campaign outcome**, same prior wired into the bandit as a warm start:

| | budget 20 | 30 | 60 |
|---|---|---|---|
| USP7 — prior **hit** the best region | −0.012 vs −0.017 | +0.025 vs +0.017 | +0.046 vs +0.037 |
| cruzipain — prior **missed** | −0.059 vs −0.033 | −0.093 vs +0.001 | −0.104 vs +0.021 |

*(prior vs cold bandit)*

**The payoff is catastrophically asymmetric.** Being right buys ≈ +0.008 over
the bandit; being wrong costs ≈ −0.094, because committing to a bad region is
absorbing while the bandit's exploration escapes it.

| budget | gain if right | loss if wrong | break-even accuracy |
|---|---|---|---|
| 20 | +0.0041 | −0.0259 | 86.5% |
| 30 | +0.0083 | −0.0940 | 91.9% |
| 60 | +0.0085 | −0.1243 | 93.6% |

### The 24-target sweep says it is worse than that

Those three rows extrapolate from one hit and one miss. Measured properly over
24 random held-out targets at budget 30:

| | value |
|---|---|
| prior top-1 accuracy | **29.2%** (random 10%) — nearly 3× chance |
| mean prior − bandit **when right** (n=7) | **−0.0016** |
| mean prior − bandit **when wrong** (n=17) | −0.0257 |
| **overall mean prior − bandit** | **−0.0186** |
| break-even accuracy implied | **>100%** |

The model does learn something — 29% against a 10% floor. It does not matter.
**Even on the targets where it picked the right region, it did not beat the
free bandit** (−0.0016). Break-even is above 100%, meaning *no* accuracy
suffices under this mechanism: a prior-anchored bandit gives up the ability to
walk away from a region cheaply, and that costs more than picking correctly
saves.

This is a stronger negative than the two-target extrapolation suggested, and
it points somewhere different: the problem is not the chooser's accuracy but
**committing at all**.

## 4. Offline proxy metrics invert against campaign outcome — twice

Both times, the arm that won an offline metric lost badly in a campaign.

| | offline metric | campaign ΔAUC |
|---|---|---|
| `max_mu` (buy highest posterior mean) | gain +1.053, **3.4× BO's own pick** | **−0.004** (p=0.18, null) |
| `largest region` | top-1 0.248, **2.5× random** | **−0.165** (cruzipain, b30) |

Offline shortlist metrics are single-round and greedy; campaign AUC rewards
the optimiser continuing to learn. They are not interchangeable, and in this
project the offline metric was actively misleading both times it was used to
choose between methods.

## 5. Twenty seeds does not resolve effects of this size

The identical `bo_maxmu_select` arm, on library definitions differing by **17
of 2,103 compounds** (none in the true top 1%):

| | 20 seeds | 200 seeds |
|---|---|---|
| filtered library | −0.0195 (p=0.011, starred) | −0.0039 (p=0.18) |
| unfiltered library | +0.0087 (p=0.19) | +0.0051 (p=0.006) |

Opposite signs, one of them "significant". Selector-scale effects here have
dz 0.1–0.5 and **20 seeds resolves neither their magnitude nor their sign.**
All instruments in this work default to 200.

## 6. A 4.7B language model reads molecules worse than 2,048 bits

Ridge from each representation to measured pIC50, within-target, Murcko
scaffold split, three targets (`results/probe_*.json`, deleted and
regenerable):

| representation | params | Spearman | R² |
|---|---|---|---|
| **Morgan fingerprint** | **0** | **0.64 – 0.74** | **0.37 – 0.51** |
| MoLFormer-XL | 47M | 0.34 – 0.50 | ≈ 0 |
| ChemBERTa (3 variants) | 5–44M | 0.20 – 0.50 | ≈ 0 |
| Qwen3.5-4B, last-token | 4,660M | 0.27 | −0.06 |
| Qwen3.5-4B, mean-pooled | 4,660M | 0.22 | −0.06 |

Mean pooling does not rescue it, so the failure is not the pooling — a general
language model does not encode chemistry. Every neural representation has
R² ≈ 0: they rank weakly and predict nothing.

## 7. Target choice decides whether any of this is measurable

| library | n | diversity D | % active (pIC50≥6) | best-region lift |
|---|---|---|---|---|
| EGFR anilinoquinazoline | 2,103 | 0.700 | 89% | — |
| EGFR whole-target | 9,153 | 0.846 | 74% | 1.3 – 1.6 |
| **cruzipain** (Chagas, no approved drug) | 504 | **0.873** | **41%** | 9 – 10 |
| USP7 | 539 | 0.803 | 55% | 15 – 20 |

EGFR — six approved drugs, 74% of its library already active — shows almost no
region concentration. The undrugged low-data targets show a great deal. Across
607 BindingDB targets the median best-region lift is 5.4, with 91% above 2.0,
so concentration is general rather than a quirk of these two.

---

## What this adds up to

The honest summary is that **region selection has the most headroom of any hook
tested, and it is still not exploitable by a learned chooser** — not because
the models tried were bad, but because the decision's payoff asymmetry sets a
break-even accuracy near 90% that cold-target prediction is nowhere near.

The three things worth carrying forward:

1. **Run the ceiling before the method.** Two lines of work were retired here
   for the cost of one script each, with no GPU and no training.
2. **Do not trust an offline proxy.** It inverted against campaign outcome both
   times it was used.
3. **Report the break-even, not just the accuracy.** A chooser at 25% accuracy
   sounds like progress over a 10% floor; against a 92% break-even it is not.

## Reproducing

```bash
python scripts/cluster_headroom.py --dataset data/chembl/cruzipain.csv \
    --k 10 --budgets 20 30 60 --seeds 200 --min-candidates 10
python scripts/build_prior_data.py --out data/prior/clusters.npz
python scripts/train_prior.py --data data/prior/clusters.npz --exclude P25779
python scripts/score_regions.py --dataset data/chembl/cruzipain.csv \
    --sequence data/targets/cruzipain.txt --exclude P25779 \
    --out data/prior/regions_cruzipain.npy
python scripts/cluster_headroom.py --dataset data/chembl/cruzipain.csv \
    --k 10 --budgets 20 30 60 --seeds 200 --min-candidates 10 \
    --prior-scores data/prior/regions_cruzipain.npy
```

Sections 12-13 (the abstention axis) need no GPU and no trained chooser:

```bash
# four independent target draws -> 121 unique targets after dedup.
# ~25 s per target; run them in parallel, they share nothing.
python scripts/selective_headroom.py --n-targets 24 --seeds 200 --seed 0 \
    --out results/selective_headroom.json
python scripts/selective_headroom.py --n-targets 24 --seeds 200 --seed 1 \
    --out results/selective_headroom_s1.json
python scripts/selective_headroom.py --n-targets 45 --seeds 100 --seed 2 \
    --out results/selective_headroom_s2.json
python scripts/selective_headroom.py --n-targets 45 --seeds 100 --seed 3 \
    --out results/selective_headroom_s3.json
# merge on target_id, then re-analyse. Every policy is algebra over the
# stored per-seed AUC, so pooling a new draw -- or adding a policy to a
# finished sweep -- costs seconds rather than a re-run.
python scripts/selective_headroom.py --replay results/selective_headroom_all.json \
    --out results/selective_headroom_all.json

python scripts/selective_selector.py --budgets 30 60 --seeds 200 --datasets \
    data/chembl/{cruzipain,usp7,cblb,hpk1,egfr_anilinoquinazoline}.csv
python scripts/gated_selector.py --budgets 30 60 --seeds 200 --datasets \
    data/chembl/{cruzipain,usp7,cblb,hpk1,egfr_anilinoquinazoline}.csv
```

## Known limits

- **Two campaign targets.** The asymmetry is measured on one hit and one miss.
  The break-even arithmetic is sound but the gain/loss magnitudes come from a
  single example each; more targets would tighten them.
- **k-means regions.** Chemically coherent clustering (scaffold, sphere
  exclusion) shatters real libraries into hundreds of mostly-singleton groups,
  which no bandit can search at these budgets. k-means gives choosable regions
  that are not chemically principled. That tension is unresolved.
- **Retrospective libraries.** Even cruzipain at 41% active is far denser than
  a real screening deck at under 1%.
- **The prior was never the limiting factor.** Break-even near 90% means no
  model in this family would have paid; a stronger predictor is not the fix.

---

## 8. The prior-quality threshold — how good would a model have to be?

`results/prior_quality_threshold.json`. Rather than train another model, this
sweeps prior *quality* directly: for each library a synthetic prior is built
with a controlled rank correlation to the true objective and used as the GP's
mean function. The question is where the curve crosses zero.

| prior quality (achieved ρ) | ΔAUC vs plain BO | libraries improved |
|---|---|---|
| 0.02 (noise) | −0.0485 | 0% |
| 0.18 | −0.0423 | 0% |
| 0.35 | **+0.0061** | 67% |
| 0.71 | +0.0461 | 100% |
| 1.00 | +0.0936 | 100% |

**Crossover: ρ ≈ 0.37.**

For comparison, the predictors actually available for a target with no
measurements:

| predictor | cold-target ρ |
|---|---|
| ridge on (ESM2-35M ⊕ ECFP4), 150k pairs, held-out targets | **0.22** |
| published cold-target DTI models | broadly comparable |

**The available prior sits below the threshold.** This is the cleanest
statement of why everything in sections 3–5 failed, and it is a statement
about the *problem*, not about the models tried: any prior below ρ ≈ 0.37
makes low-budget BO worse, and a prior of exactly zero quality costs −0.049.

It also says what would change the answer. Not a bigger model — a jump in
cold-target accuracy from 0.22 to beyond 0.37, which is a substantial advance
in the DTI literature rather than an engineering detail.

## 9. A prior and an acquisition function cannot be chosen separately

Same three libraries, prior held fixed, acquisition varied:

| acquisition | perfect prior (ρ=1) | realistic prior (ρ=0.20) |
|---|---|---|
| **Expected Improvement** | **−0.019 … −0.144** | −0.008 … −0.051 |
| UCB | +0.057 … +0.113 | −0.004 … −0.042 |
| greedy | +0.059 … +0.120 | −0.008 … −0.024 |

**EI throws away a perfect prior.** The mechanism is exact: with a confident
prior the posterior sd collapses, and once the incumbent is the library
maximum every remaining candidate has μ < best, so EI is zero for all of them.
The arm ties and picks at random for the rest of its budget. Verified
directly — the EI arm with a perfect prior finds the library maximum on every
seed (best 9.699 of 9.699) and still loses on AUC.

Any work adding a learned prior to BO should report which acquisition it used,
and should not use EI. Section 12 adds the counterpart: a *gate* is
acquisition-invariant here (EI and UCB agree within SEM on every quantity), so
the reporting requirement applies to priors and not to gates.

## 10. The threshold moves with budget — and there is a regime where it pays

The threshold is not a constant. It tracks the GP's own accuracy, which grows
with every observation, so a prior's window closes as the campaign runs.

Measured GP accuracy (Spearman of the posterior mean against truth, on
unobserved compounds, 30 repeats, 4 libraries):

| observations | 5 | 10 | 15 | 20 | 30 | 50 | 80 |
|---|---|---|---|---|---|---|---|
| GP Spearman | 0.186 | 0.299 | 0.355 | 0.405 | 0.462 | 0.496 | 0.542 |

And the measured prior-quality threshold, same libraries, UCB:

| budget | threshold ρ | a ρ = 0.27 prior scores |
|---|---|---|
| **10** | **0.18** | **+0.0310, better on 100% of libraries** |
| 30 | 0.38 | −0.0074 |

**The threshold is the GP's own accuracy at that budget.** A prior pays only
while it is more accurate than what the surrogate already knows, and the
surrogate catches up fast — by 15 observations it is at ρ 0.355, past what
cold-target prediction delivers.

### The consequence: the prior route is closed at realistic budgets

Full curve, **8 randomly drawn libraries** (not selected), UCB, 200 seeds:

| budget | threshold rho | a rho = 0.19 prior scores | libraries improved |
|---|---|---|---|
| 8  | 0.13 | **+0.0226** | 75% |
| 10 | 0.15 | **+0.0207** | 75% |
| **12** | **0.20** | +0.0059 | 62% |
| 15 | 0.32 | -0.0081 | 50% |
| 20 | 0.34 | -0.0177 | 38% |
| 30 | 0.39 | -0.0253 | 38% |
| 40 | 0.37 | -0.0244 | 25% |

Cold-target prediction reaches rho ~ 0.22, so the crossover sits at **about 12
oracle calls**. Below it a cold-target prior genuinely helps; above it the same
prior actively hurts, and the damage grows.

**Twelve measurements is not a drug discovery campaign.** Real programmes screen
in plates -- tens to hundreds of compounds at a time -- and a lead-optimisation
series runs to hundreds. So this is a positive result in a regime nobody
operates in, which makes it a curiosity rather than a contribution.

The honest reading: **a learned prior that competes with the surrogate is closed
for molecular BO.** Not because the models tried were weak, but because the GP
reaches rho 0.46 by 30 observations and cold-target prediction is at 0.22. The
fix is not a bigger model; it would take cold-target accuracy roughly doubling,
which is a major advance in the DTI literature rather than an engineering
detail.

What this does *not* close is a signal about regions the surrogate has no
information on at all. That is section 11.

## 11. Scaffold hopping: the gap BO cannot close with more budget

Sections 8-10 close off priors that *compete* with the GP: they must beat the
GP's own accuracy, which grows with every observation, and cold-target
prediction does not. That argument has a hole in it, and the hole is the
interesting part.

**A GP on Tanimoto similarity has zero information about a scaffold it has
never sampled.** There is nothing to beat there. And it turns out BO barely
samples the library at all.

### BO stays in one basin

30 seeds per cell, Murcko scaffolds, EI:

| library | compounds | scaffolds | budget | scaffolds seen | top-10 in unseen scaffolds | regret (log units) |
|---|---|---|---|---|---|---|
| Q96LD8 | 1075 | 775 | 30 | **3.5%** | **9.4 / 10** | **1.60** |
| Q96LD8 | 1075 | 775 | 120 | 14.1% | 6.6 / 10 | 0.88 |
| P24530 | 1038 | 325 | 30 | 6.2% | 7.1 / 10 | 0.89 |
| O00408 | 1033 | 307 | 30 | 6.3% | 6.9 / 10 | 0.81 |
| O94782 | 543 | 180 | 30 | 11.4% | 7.4 / 10 | 0.88 |

Quadrupling the budget on Q96LD8 (30 to 120) still leaves 86% of scaffolds
untouched and 0.88 log units of regret. **This gap does not close with more
evaluations** -- BO deepens the basin it is in.

### Directed hopping recovers it; random hopping does not

Every fourth round the candidate pool is restricted to one scaffold the run
has not sampled. 100 seeds, paired against plain BO.

| library | budget | **directed (oracle)** | random | regret: plain → directed |
|---|---|---|---|---|
| O94782 | 60 | **+0.0687** (p<1e-4) | +0.0028 (p=0.20) | 0.70 → **0.00** |
| O94782 | 30 | **+0.0463** (p<1e-4) | −0.0098 | 1.14 → **0.00** |
| Q96LD8 | 60 | **+0.0458** (p<1e-4) | −0.0086 | 1.20 → 0.90 |
| Q96LD8 | 30 | +0.0355 (p<1e-4) | −0.0085 | 1.60 → 1.17 |
| O00408 | 60 | +0.0186 (p<1e-4) | −0.0034 | 0.41 → 0.42 |
| P24530 | 60 | +0.0130 (p=0.001) | −0.0216 | 0.48 → 0.44 |

Three properties that none of the earlier interventions had:

1. **Random hopping loses** (−0.003 to −0.022 on 5 of 6 cells). The gain is
   not from exploring more; it is from knowing *which* chemotype. The
   oracle-minus-random gap, +0.02 to +0.08 AUC, is exactly what a model would
   have to supply.
2. **The effect grows with budget** (+0.046 → +0.069 on O94782). Every prior
   in sections 8-10 *shrank* with budget because the GP caught up. This one
   does not, because the GP never acquires information about regions it does
   not sample.
3. **The regret reductions are chemically meaningful.** O94782 goes from 0.70
   log units short of the optimum to finding it on every seed -- a 5x more
   potent compound, not a decimal place on an aggregate.

### Why this is not blocked by the prior-quality threshold

The threshold (section 8) governs a signal that ranks compounds the GP can
already reason about, and there the GP sets a bar of rho 0.46 by 30
observations. In an unsampled scaffold the GP's accuracy is **zero**. A model
does not have to be good in absolute terms; it has to be better than nothing,
about regions where nothing is what the surrogate has.

That is the first mechanism found here where a cross-target model has a
structural advantage rather than a shrinking one.

---


## 12. The axis that was never varied: a chooser allowed to decline

Section 4 closes the region-prior line on a number — break-even accuracy above
100%, so no accuracy suffices — and then gives a mechanism that is not about
accuracy at all:

> the problem is not the chooser's accuracy but **committing at all**

Every chooser in this project is *forced to act*. Asked which region, it names
one, including on the 46% of targets where restricting to any region is worth
nothing (section 3). "Not committing" was only ever tested in one form: a
bandit that can **escape** a bad region mid-campaign. The other form —
**abstaining**, on a confidence signal, before paying for the decision — has
never been measured here, and it is the lever a calibrated decision layer
actually offers.

`scripts/selective_headroom.py`, **all 411 eligible held-out targets** — the
entire BindingDB pool with 250–1200 measured compounds and a sequence under
1022 residues, covered exhaustively rather than sampled, so there is no target
draw left to be lucky in. k=10, budget 30, 100–200 seeds. Campaigns run once
per (target, region); every policy below is algebra over the stored per-seed
AUC, so adding a policy costs seconds, not an hour.

*(Section 4 used 24 of these, and the numbers below supersede every earlier
version of this section. The progression is itself reported in "Correction"
below, because it is the most useful thing in it.)*

### Gating flips the sign

Reference is the free UCB bandit over regions — the bar section 4 set.

| policy | ΔAUC vs bandit | 95% CI | p | acts on |
|---|---|---|---|---|
| `commit(prior)` | **−0.0266** | [−0.0322, −0.0212] | <1e-4 | 100% |
| `gate(correctness)` | **+0.0059** | [+0.0040, +0.0079] | **<1e-4** | 16% |
| `gate(outcome)` — **ceiling** | **+0.0107** | [+0.0089, +0.0127] | **<1e-4** | 36% |
| `commit(best region)` (answer key) | +0.0309 | [+0.0278, +0.0338] | <1e-4 | 100% |
| `gate(outcome)` on the best region | +0.0339 | — | <1e-4 | 87% |

`gate(correctness)` is the **same prior, at the same accuracy**, allowed to
decline when it is wrong. It goes from −0.027 to **+0.0059 over a free
bandit** — the first positive region-hook number in this project, and it
required no better model, only permission to say nothing.

`gate(outcome)` acts exactly when acting would have beaten the bandit. It reads
the campaign answer key, so it **bounds every possible abstention rule**,
calibrated or not, at this accuracy: **+0.0107**. That is the entire budget the
calibration axis has to spend on this hook.

**The permuted-label control.** A gate that declines on 84% of targets acts
less, and acting less is worth something on its own when the action has
negative expectation — so the row above proves nothing until the gate is
compared with one that abstains equally often for no reason. Shuffling the
correctness labels and keeping the coverage:

| | ΔAUC vs bandit |
|---|---|
| `gate(correctness)` | **+0.0059** |
| same gate, **labels permuted**, same 16% coverage | −0.0042 [−0.0062, −0.0022] |
| **knowing *which* targets** | **+0.0101**, one-sided p < 1e-5 (20k permutations) |

So the gain is not the abstention rate; it is the placement. This is the same
control shape as directed-vs-random hopping in section 11, and the same
conclusion: exploring less is free, knowing where is the finding.

*(`commit(prior)` here is a hard restriction for the whole campaign, so it is
more negative than section 4's −0.0186, which used a prior-warmed bandit that
could still escape. Both are in the table on purpose: **escape** recovers about
+0.007 of the −0.025, **abstention** about +0.031. They are different repairs
for the same wound and the second is four times the first.)*

### Correction: the prior's 29% top-1 accuracy does not replicate

Section 4 reports the prior at **29.2%** top-1 against a 10% floor and calls it
"nearly 3× chance". On a second, independent draw of 24 targets the identical
pipeline measures **8.3%** — *below* the random floor.

| draw | targets | prior top-1 | `gate(correctness)` | `gate(outcome)` |
|---|---|---|---|---|
| seed 0 (section 4's) | 24 | **29.2%** | +0.0086 | +0.0113 |
| seed 1 | 24 | **8.3%** | +0.0011 | +0.0026 |
| seed 2 | 45 | 22.2% | — | — |
| seed 3 | 45 | 17.8% | — | — |
| pooled of the four draws | 121 | 19.8% | +0.0057 | +0.0117 |
| **the whole eligible pool** | **411** | **15.8%** | **+0.0059** | **+0.0107** |

This is section 5's lesson one level up. There, 20 seeds could not resolve an
arm's sign; here, **24 targets cannot resolve a chooser's accuracy** — two
draws of the same size differ by 3.5×, and the flattering one is the one that
got published. Section 4's accuracy figure should be read as **15.8%**, not
29.2%: half again above chance, not three times it.

The two *policy* columns are the reassuring part. The chooser's accuracy moved
by a factor of 3.5 across draws while `gate(outcome)` moved from +0.0113 to
+0.0107 — because a ceiling over an abstention rule is an average of a
one-sided quantity over targets, and is far better behaved than a top-1 rate.
**The thing this document is measuring is more stable than the thing it was
measuring before.** Nothing below depends on a target draw: the pool is
covered exhaustively, so there is no sampling left to be lucky in.

### The break-even moves, and which question you ask decides how far

A synthetic chooser with accuracy α and a gate whose confidence separates its
own cases at AUROC δ. δ = 0.50 is an uncalibrated chooser — which is every
chooser this project has built.

**Lowest accuracy that pays**, by gate quality:

| δ (gate AUROC) | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|
| gate on **correctness** — *"is my pick the best region?"* | 70% | 50% | 30% | 20% | 10% | 10% |
| gate on **outcome** — *"will acting beat the bandit?"* | 70% | 50% | **20%** | **10%** | 10% | 10% |

Two things fall out.

1. **Calibration substitutes for accuracy, at roughly 3.5:1 here.** An
   uncalibrated chooser needs 70% top-1; at δ = 0.8 it needs 20%. The prior
   measures 15.8%. (This surface is the stable part of the section: it is
   identical at n = 48, n = 121 and n = 411.) So the question
   stops being *"can cold-target prediction
   reach 70% top-1"* — it cannot, and section 8 explains why — and becomes
   *"can a gate reach δ ≈ 0.7"*. That is a different and much smaller ask.
2. **The two gates are not the same question.** *"Will acting help"* has the
   lower break-even at every δ, and it never requires knowing *which* region is
   best. It is a yes/no on the decision's value rather than a choice among
   regions, and the cheaper question is also the better one.

### The signal is real at full power, and it is not enough

Five confidence signals, computable from the prior's own output with no answer
key and no training, over all 411 targets:

| signal | AUROC vs *"acting pays"* | perm p | AUROC vs *"pick is right"* |
|---|---|---|---|
| `n_pred_top_in_pick` | **0.581** | **<0.001** | **0.619** |
| `margin` (top-1 minus top-2) | 0.568 | **0.035** | 0.492 |
| `pred_lift` — how concentrated the model thinks this target is | 0.555 | 0.085 | 0.443 |
| `top1_score` | 0.548 | 0.095 | 0.442 |
| `entropy` over region scores | 0.473 | 0.37 | 0.469 |
| `best_lift` — *reads the answer key* | 0.508 | 0.82 | 0.589 |

**This overturns the n = 121 version of this section**, which concluded that
nothing was distinguishable from a coin flip. At full power two signals are:
`n_pred_top_in_pick` at 0.581 (p < 0.001) and `margin` at 0.568 (p = 0.035).
The gate is not reading noise.

It is still not enough, and the gap is now a number rather than a verdict. At
the prior's measured 15.8% accuracy the break-even surface requires **δ ≥ 0.80**
for either gate target. The best free signal delivers **0.581**.

| | required | measured | shortfall |
|---|---|---|---|
| outcome gate, at α = 15.8% | 0.80 | 0.581 | **0.22** |
| correctness gate, at α = 15.8% | 0.80 | 0.619 | **0.18** |

### What moving from 24 to 411 targets did to each claim

Worth tabulating, because three of the four statements this section made at
n = 24 did not survive, and the one that did is the one the section is about.

| claim | n = 24 | n = 121 | **n = 411** | survived? |
|---|---|---|---|---|
| `gate(outcome)` ceiling | +0.0113 | +0.0117 | **+0.0107** | **yes — moved 5%** |
| break-even surface | 70→20% | 70→20% | **70→20%** | **yes — unchanged** |
| prior top-1 accuracy | 29.2% | 19.8% | **15.8%** | no — halved |
| `pred_lift` AUROC | 0.719 | 0.529 | **0.555** | no — inflated at small n |
| "no signal reaches chance" | — | asserted | **false** | no — reversed |

The ceiling and the surface are averages of one-sided quantities over targets
and they barely moved. Everything that is a *maximum* — a top-1 rate, an AUROC
picked as the best of five, a threshold swept for the best gain — moved a lot,
and always downward from the small-sample value. That is the whole
methodological content of this section and it is mechanical, not surprising.

`pred_lift` is the cautionary case: 0.719 at n = 24 with a mechanism attached
that made it easy to believe (the answer-key-free twin of section 3's
`best_lift`), then 0.529 at n = 121, then 0.555 at 411. It settles slightly
above chance but never near its bar, and the n = 24 value was inflated by a
factor that a permutation null would have exposed at the time.

Two observations that do survive at full power:

- `best_lift` **with the answer key** reaches only 0.508 against *"acting
  pays"*. Even a perfect concentration predictor would not gate this decision:
  the gate's target is a genuinely different quantity from concentration, so a
  model built to predict concentration is building the wrong thing.
- The two signals that express confidence in the *pick itself* — `pred_lift`
  and `top1_score` — sit at **0.443 and 0.442** against correctness, below
  chance at n = 411 as they were at n = 121. Meanwhile `n_pred_top_in_pick`,
  which is a raw count rather than a normalised confidence, reaches 0.619. The
  prior's *self-reported* confidence is anti-correlated with being right; a
  crude count of how much of its predicted top 1% landed in the region it chose
  is the best predictor available. That is a statement about this prior's
  calibration, not about the hook.

Thresholding those signals, in-sample and leave-one-out:

| signal | in-sample ΔAUC (coverage) | leave-one-out ΔAUC | p |
|---|---|---|---|
| `n_pred_top_in_pick` | +0.0003 (2%) | **+0.0002** | 0.40 |
| `pred_lift` | +0.0003 (2%) | −0.0003 | 0.86 |
| `top1_score` | +0.0003 (2%) | −0.0004 | 0.78 |
| `margin` | +0.0002 (3%) | −0.0006 | 0.31 |
| `entropy` | −0.0001 (0%) | −0.0003 | 0.18 |

**The in-sample column is worth less than it looks, and the amount is
measurable.** Sweeping every threshold and both signs and keeping the best is
the estimator that CSPI-MT (arXiv:2408.12004, KDD '25) shows adopts a
worse-than-baseline policy on a null DGP more than 80% of the time at n = 2000.
Run that null here — a pure-noise signal, the identical sweep, 2000 draws:

| in-sample best-threshold gain, **pure noise** | n = 121 | n = 411 |
|---|---|---|
| mean | +0.0004 | +0.0001 |
| 95th percentile | +0.0014 | +0.0004 |
| **P(reports a positive gain)** | **75.7%** | **77.3%** |

Three targets in four report a win when there is nothing there, and **more data
does not fix it** — the bias shrinks in magnitude but its *frequency* does not
move at all, because the sweep always finds some threshold. Against that null
no signal's in-sample number clears the 95th percentile at n = 411 (best:
`pred_lift` and `n_pred_top_in_pick` at the 92nd), and every leave-one-out
number is inside noise. A gate statistic that maximises over a threshold has to
be reported against its own permuted null, never against zero.

### The acquisition ablation: the gate's value does not depend on it

Section 9 established that a prior and an acquisition function cannot be chosen
independently — a **perfect** prior scores −0.019 to −0.144 under EI and +0.057
to +0.113 under UCB, because a confident prior collapses the posterior sd and
EI then ties across the library. Every number above is under EI, so the obvious
question is whether the abstention result is an artefact of the worst possible
acquisition for a prior.

It is not. The whole instrument re-run under UCB, same 411 targets, same k,
same budget:

| policy | **EI** | **UCB** |
|---|---|---|
| `commit(prior)` | −0.0266 | −0.0242 |
| `gate(correctness)` | **+0.0059** | **+0.0055** |
| `gate(outcome)` — ceiling | **+0.0107** | **+0.0100** |
| `commit(best region)` (answer key) | +0.0309 | +0.0309 |
| prior top-1 accuracy | 15.8% | 15.3% |
| best free signal, AUROC vs *"acting pays"* | 0.581 | 0.586 |
| gate **trained** on the outcome, OOF AUROC | 0.590 (p = 0.005) | 0.582 (p = 0.015) |
| certified 95% LCB | −0.0060 | −0.0065 |

Break-even surface, both acquisitions:

| δ (gate AUROC) | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|
| EI, outcome gate | 70% | 50% | 30% | **10%** | 10% | 10% |
| UCB, outcome gate | 60% | 40% | 20% | **10%** | 10% | 10% |

Every quantity agrees to within its own standard error, and the verdict is
identical: the ceiling is real, the trained gate carries real signal, and it is
~0.21 short of the required δ.

**Why this differs from section 9, which is the interesting part.** Section 9's
mechanism is specific to how EI *scores* a candidate: a confident prior mean
shrinks the posterior sd, EI goes to zero once the incumbent is the library
maximum, and the arm ties and picks at random. That is a statement about the
acquisition formula. A gate does not touch the formula. It decides whether the
candidate *pool* is restricted at all, and when it declines, the run is the
free bandit under whichever acquisition is in use. So the two levers sit at
different places in the loop:

| lever | acts on | acquisition-sensitive? |
|---|---|---|
| a prior as a GP mean (section 9) | how candidates are **scored** | **violently** — sign flips |
| a gate on region restriction (here) | which candidates **exist** | **no** — within SEM |

This is worth stating as a design rule, because it is the opposite of
section 9's: *"any work adding a learned prior to BO should report which
acquisition it used"* still holds, and **"any work adding a gate need not"**
now holds alongside it. A gate is the more portable intervention, which is a
point in its favour independent of whether one can currently be built.

### Where this leaves the region hook

| | value |
|---|---|
| what abstention is worth at most, on this hook | **+0.0107** EI / **+0.0100** UCB (p < 1e-4, n = 411) |
| gate quality needed at the measured accuracy | **δ ≈ 0.80** |
| best free signal measured | **0.581** (p < 0.001) — real, and short by 0.22 |
| gate **trained** on the campaign outcome (`gate_fit.py`) | **0.590** EI / **0.582** UCB (perm p = 0.005 / 0.015) — short by ~0.21 |
| what would close the gap | features independent of the prior and the surrogate |

The ceiling is firmly established, the signal is real, and both a free signal
and a fitted one land ~0.21 short of the bar. Note also why the ceiling is not
larger: at 15.8% top-1 the prior's pick beats the bandit on only **36%** of
targets, so a perfect gate has a third of the campaigns to work with. The
*best* region beats the bandit on **87%** of them and by +0.031, which is why
gating a perfect chooser is worth +0.034 — three times gating this one. **A
gate recovers downside; the upside still has to come from the pick.**

And the fitted gate's failure is specific rather than general. Every feature it
was given — the prior's region scores, its margin, its entropy, the surrogate's
posterior — is a function of one of the two models whose failure the gate is
supposed to detect. A gate reading only its patient's self-report is the
simplest explanation for a real-but-insufficient 0.59, and it is consistent
with the calibration result above: this prior's *self-reported* confidence runs
**below chance** against its own correctness (0.443), while a crude count it
does not present as confidence reaches 0.619.

That, and the fact that declining here still costs a bandit's exploration, is
the argument for moving the gate to a hook where the ceiling is live and
declining is genuinely free — section 13.

---

## 13. Where abstention is structurally cheap: BO's shortlist

Section 12 bounds the gate on the region hook at +0.0107, and the bound is set
by the hook, not by the gate: region restriction is worth little on a random
target, so there is little downside to recover and less upside to protect. The
shortlist hook is the opposite shape on all three counts that matter.

| | region hook | shortlist hook |
|---|---|---|
| cost of declining | a free bandit's exploration | **nothing — position 0 is plain BO** |
| ceiling | +0.025 over the bandit | **+0.036** over plain BO |
| what the best free control takes of it | 40–50% (a ten-line bandit) | **0%** (section 1) |
| decisions per campaign | 1 | **20–50** |

`scripts/selective_selector.py`. Five libraries (cruzipain, USP7, CBL-B, HPK1,
EGFR anilinoquinazoline), budgets 30 and 60, shortlist 15, 200 seeds, paired
against plain BO. The chooser is two numbers and nothing else:

* **α** — P(it names the shortlist's truly best compound).
* **γ** — P(act | right) − P(act | wrong), with P(act | right) held at 1. This
  is Youden's J for the gate, so the gate's AUROC against its own correctness
  is exactly **(1 + γ)/2**. γ = 0 is the committing chooser — the only column
  this project has ever measured.

Declining returns position 0, so a declined round is plain BO exactly.

The free random control is **on the grid**, at α = 1/15 with γ = 0: picking
uniformly from the shortlist *is* a chooser with chance accuracy and no gate.
Everything is read against that cell, not against plain BO, because any
deviation from the EI argmax adds exploration and at these budgets exploration
alone can win.

### ΔAUC vs plain BO, pooled over 5 libraries × 2 budgets

| α ↓ γ → | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| **0.067** (= the free control) | **−0.0021** | +0.0009 | +0.0040 | +0.0055 | **+0.0065** |
| 0.20 | +0.0099 | +0.0124 | +0.0146 | +0.0146 | +0.0163 |
| 0.40 | +0.0221 | +0.0237 | +0.0254 | +0.0260 | +0.0269 |
| 0.60 | +0.0290 | +0.0300 | +0.0316 | +0.0322 | +0.0327 |
| 0.80 | +0.0337 | +0.0343 | +0.0350 | +0.0353 | +0.0357 |
| 1.00 (= `sel_oracle`) | +0.0358 | +0.0358 | +0.0358 | +0.0358 | +0.0358 |

**Break-even accuracy against the free control**, pooled and on every one of
the ten (library, budget) cells individually:

| γ | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| lowest α that beats a uniform draw | **20%** | **7%** | 7% | 7% | 7% |

Three readings.

1. **Any gate at all collapses the accuracy requirement to the floor.** A
   committing chooser needs 20% top-1 on a 15-long shortlist to be worth
   running; with γ = 0.25 — a gate at AUROC **0.625**, which is a weak gate —
   the requirement drops to 6.7%, which is chance. γ = 0 needs α = 0.20 in
   **all ten** (library, budget) cells separately; γ = 0.25 drops it to the
   floor in **eight of ten**, and on CBL-B — the one library where a uniform
   draw already wins — it takes γ = 0.50.
2. **Calibration is worth most exactly where accuracy is worst.** At α = 0.067
   the γ axis is worth **+0.0086** (−0.0021 → +0.0065); at α = 0.80 it is worth
   +0.0020. A chooser that knows nothing about molecules but knows when it is
   guessing beats one that guesses confidently, and matching +0.0065 by
   accuracy alone would take α ≈ 0.16 — **2.4×** the accuracy.
3. **It is bounded by the hook, and the hook is bigger here.** γ = 1 at chance
   accuracy captures 18% of the oracle ceiling. The equivalent number on the
   region hook (section 12) is +0.0107 against a +0.031 commit ceiling, and
   against a *free bandit* that already takes 40–50% of it.

Per library the magnitude tracks that library's own ceiling. What the γ axis is
worth at chance accuracy, by (library, budget 30 / 60):

| | cruzipain | USP7 | CBL-B | HPK1 | EGFR anilinoquinazoline |
|---|---|---|---|---|---|
| `sel_oracle` ceiling | +0.037 … +0.052 | +0.050 … +0.052 | +0.033 | +0.015 … +0.016 | +0.035 … +0.038 |
| γ = 0 → 1 at α = 1/15 | +0.0065 / +0.0084 | +0.0101 / +0.0142 | +0.0025 / +0.0012 | +0.0032 / +0.0046 | **+0.0136 / +0.0215** |

EGFR anilinoquinazoline is the sharpest case, and the one that matters most,
because it is the library where deviating from the EI argmax is *punished*: a
uniform draw loses −0.0100 and −0.0131 there (both p < 1e-4), and the same
chooser with a perfect gate is positive (+0.0047, +0.0092). CBL-B is the mirror
— the one library where a uniform draw already wins (+0.0022, +0.0046), and the
gate accordingly adds least.

### The other half: no free signal supplies γ

`scripts/gated_selector.py`. The subject is `MaxMeanSelector` — buy the
shortlist's highest posterior mean — chosen because it is the project's
sharpest case of being *good at the wrong thing*: 3.4× plain BO's offline gain
and negative campaign AUC (section 4). Its failure has a mechanical diagnosis
("never spends a call resolving uncertainty"), and a mechanical diagnosis
implies the good rounds are distinguishable from the bad ones before the answer
is known.

Five families of gate, all computed from what the selector is already handed:
`late(f)` / `early(f)` (position in the campaign), `margin(t)` (the pick's
posterior-mean lead over rank 0, in posterior sds), `settled(t)` (the
shortlist's mean posterior sd relative to the first round's), and `rand(c)` —
**the control that decides the section**, acting on a random c of rounds.

Pooled over the same 5 libraries × 2 budgets, against plain BO, with each gate
matched to the random gate that deviates equally often:

| gate | acts on | ΔAUC vs BO | **vs `rand` at the same coverage** |
|---|---|---|---|
| `sel_oracle` (ceiling) | 100% | +0.0358 | — |
| `early(0.75)` | 53% | +0.0010 | **+0.0010** |
| `late(0.25)` | 54% | +0.0003 | +0.0002 |
| `settled(0.98)` | 24% | +0.0007 | +0.0002 |
| `always` (no gate) | 71% | +0.0012 | +0.0000 |
| `margin(0.25)` | 39% | −0.0004 | −0.0006 |
| `settled(0.9)` | 17% | −0.0003 | −0.0008 |
| `rand(0.25/0.5/0.75)` | 18/37/54% | +0.0005/+0.0003/+0.0001 | — |

**Every gate lands within ±0.0010 of a coin flip firing equally often**, on a
hook whose ceiling is +0.0358. The best of them captures under 3% of it, and
the sign is not even stable across libraries. `late` and `early` — exact
mirror images — are both slightly positive, which is the signature of a
coverage effect rather than a placement one.

So the mechanical diagnosis does not cash out into a run-time signal. Nothing
in the GP's own state at decision time (how far into the campaign, how much
posterior spread remains, how far the pick leads rank 0) knows whether this
round's exploitation is the one that hurts.

### The gate, trained: γ measured rather than dialled

The grid above dials γ. `scripts/shortlist_gate.py` fits it. The shortlist hook
admits this where the region hook does not, because **the gate's label is
observable every round**: γ is defined against *"did the chooser name the
shortlist's best compound"*, and on a retrospective library that is known the
moment the shortlist is drawn — ~20 labelled decisions per campaign instead of
one per target.

The chooser is `MaxMeanSelector`. Decisions are harvested with
`RecordingSelector`, which returns rank 0 — so the harvesting run *is* plain BO
and the shortlists match what a deployed gate would face. The gate is a
standardised logistic regression on 14 decision-time features (posterior-mean
lead over rank 0, that lead in posterior sds, shortlist sd profile, EI's own
preference, round fraction, observations so far), trained **leave-one-library-out**:
never on the library it is scored on, which is also the honest transfer question.

2,400 decisions per library, 120 harvest seeds, 200 paired evaluation seeds.

| held-out library | α (maxmu) | gate AUROC | γ | γ, labels shuffled | **γ net** |
|---|---|---|---|---|---|
| USP7 | 0.161 | 0.656 | +0.239 | +0.102 | **+0.137** |
| cruzipain | 0.134 | 0.593 | +0.219 | +0.061 | **+0.158** |
| EGFR anilinoquinazoline | 0.218 | 0.548 | +0.099 | +0.031 | +0.068 |
| HPK1 | 0.146 | 0.542 | +0.101 | +0.035 | +0.066 |
| CBL-B | 0.260 | 0.544 | +0.077 | +0.034 | +0.043 |
| **mean** | **0.184** | **0.577** | **+0.147** | **+0.053** | **+0.094** |

**The shuffled-label column is the finding, not a formality.** γ is a maximum
of Youden's J over thresholds, so it is biased upward even when the gate knows
nothing; training the identical pipeline on permuted labels and applying the
identical sweep measures that bias at **+0.053**. A third of the raw γ is the
sweep. Net γ is +0.094 against a bar of 0.25, and no library clears.

And the campaign agrees — which matters, because section 4 recorded two
occasions where an offline shortlist metric inverted against campaign outcome:

| library | maxmu | **gated** | random gate at the same rate | oracle | acts on | gated − random |
|---|---|---|---|---|---|---|
| cruzipain | +0.0050 | +0.0006 | +0.0020 | +0.0515 | 16% | −0.0014 |
| USP7 | +0.0034 | +0.0002 | +0.0009 | +0.0496 | 22% | −0.0006 |
| CBL-B | −0.0008 | +0.0009 | +0.0001 | +0.0332 | 11% | +0.0007 |
| HPK1 | +0.0001 | +0.0002 | +0.0004 | +0.0163 | 17% | −0.0003 |
| EGFR anilinoquinazoline | +0.0053 | +0.0008 | +0.0013 | +0.0380 | 21% | −0.0005 |
| **mean** | +0.0026 | +0.0005 | +0.0009 | **+0.0377** | 17% | **−0.0004** |

Against a ceiling of +0.0377, a learned gate firing on 17% of rounds is
**−0.0004 against a coin flip firing equally often** (every p ≥ 0.23). The
offline γ and the campaign agree for once, and they agree on a negative.

### A consistency check the grid did not need but passes

`α(maxmu)` was measured at harvest, not assumed: **0.184** averaged over the
five libraries (0.134–0.260). The grid says the γ = 0 break-even is α = 0.20.
So `MaxMeanSelector` sits essentially *on* the committing break-even — which is
an independent explanation, from a different instrument, of why it scores near
zero against plain BO (+0.0026 here) despite 3.4× plain BO's offline gain
(section 4). The synthetic grid and a real chooser agree on where the line is.

### Small-n inflation, three times in one night

Every gate-quality number in sections 12–13 is a **maximum over thresholds**,
and every one of them shrank toward its null as the sample grew:

| quantity | small n | large n |
|---|---|---|
| `pred_lift` AUROC vs *"acting pays"* | **0.719** (24 targets) | **0.555** (411 targets) |
| shortlist gate γ | **+0.274** (20 harvest seeds) | **+0.147** (120 seeds) |
| in-sample threshold sweep, pure noise | — | reports a **positive gain 75.7%** of the time |

These are three independent instruments producing the same artefact, and in
each case the small-*n* value was large enough to write up. The rule this
project should carry forward is narrow and mechanical:

> **A gate statistic that maximises over a threshold must be reported against
> its own permuted-label null, not against zero.** The null is not negligible
> and not guessable — it was +0.105 at 20 seeds and +0.053 at 120.

### What sections 12 and 13 say together

The two halves point in opposite directions and both are needed to state the
result.

**The axis is real and it is the cheap one.** On both hooks the break-even in
accuracy falls sharply with gate quality — 70% → 20% on regions, 20% → 7% on
the shortlist — and the gain concentrates exactly in the low-accuracy regime
that cold-target prediction actually occupies (section 8: ρ ≈ 0.22, top-1 19%).
Every previous section of this document asks a chooser to get better at
choosing. This is the first measurement saying it would pay more to get better
at *knowing when not to*.

**And nothing supplies it — free or fitted.** Four instruments, four nulls:

| instrument | what it tried | result |
|---|---|---|
| §12 free signals | five answer-key-free confidences, n = 411 | best AUROC **0.581 (p < 0.001)** vs **0.80** needed |
| §12 `gate_fit.py` | gate **trained** on the campaign-outcome label, n = 411 | OOF AUROC **0.590 (perm p = 0.005)**; policy value −0.0009, certified LCB −0.0060 — all four pre-registered criteria fail |
| §13 `gated_selector.py` | five families of hand-written run-time rule | all within ±0.0010 of a coin flip at matched coverage |
| §13 `shortlist_gate.py` | gate **trained**, leave-one-library-out | net γ **+0.094** vs 0.25 needed; campaign −0.0004 vs a random gate |

The two fitted gates are the ones that matter, because "nothing was ever
trained as a gate" was this document's own excuse for the free-signal nulls
until now. Both were trained, both were given a permuted-label null, and both
**failed their pre-registered criteria while carrying real signal** — which is
a different and more useful result than a null.

On the region hook the fitted gate's out-of-fold AUROC is **0.590 at
permutation p = 0.005**, and its power curve *rises* with target count
(0.492 → 0.555 → 0.572 → 0.578 at n = 50 → 100 → 300 → 411). An earlier version
of this section reported 0.355 with a flat curve and concluded there was no
stable direction to learn; that was measured at n = 121 and **it was wrong**.
The direction is real. It is simply not strong enough: the surface demands 0.80
and the campaign value stays at −0.0009 with a certified 95% lower bound of
−0.0060, so the gate must not be deployed.

The shortlist hook says the same thing in its own units — net γ +0.094 against
a bar of 0.25, with the campaign agreeing at −0.0004 versus a random gate. Two
hooks, two fitted gates, two measured shortfalls of roughly the same relative
size.

That combination is a well-posed problem rather than a closed one, and it is
worth saying what would settle it, because it is not what the earlier sections
would have suggested:

- **Not a bigger model.** Section 8's threshold argument (ρ ≈ 0.37 needed,
  0.22 available) governs a signal that *ranks candidates*. A gate does not
  rank anything; it answers one yes/no about whether a decision will pay, and
  the break-even tables above say a weak one — AUROC 0.62 to 0.70 — is enough.
  That experiment has now been run on both hooks, with the gate trained as a
  gate and a permuted-label null attached, and it failed on both. So the
  remaining question is not "would a gate help" (measured: yes, up to +0.0107)
  nor "has anyone built one" (measured: no) but **what a gate would have to
  see**. Every feature tried is a function of the surrogate's own posterior or
  the prior's own scores — that is, of the two models whose failure the gate is
  supposed to detect. A gate reading only its patient's self-report is the
  likeliest explanation of four nulls, and the untried alternative is features
  that are independent of both: library geometry, assay metadata, campaign
  trajectory shape.
- **Targets, not seeds.** Section 5 established that 20 seeds cannot resolve an
  arm. Section 12 adds that **24 targets cannot resolve a chooser, and cannot
  resolve a confidence signal either** — the prior measured 29.2% top-1 on one
  draw of 24 and 8.3% on another, and its best confidence signal measured AUROC
  0.719 at n = 24 and 0.555 at n = 411. Both the chooser and the gate are
  fitted *across* targets, so they need targets the way an arm needs seeds. At
  ~25 s per target this is CPU-hours, not a GPU budget, and it should be the
  default scale rather than an afterthought.
- **Report a gate statistic against its permuted null, never against zero.**
  Every gate quantity here maximises over a threshold, and all three shrank
  toward their null as the sample grew (`pred_lift` 0.719 → 0.529; shortlist γ
  +0.274 → +0.147; the in-sample threshold sweep reports a gain on pure noise
  **75.7%** of the time). The null is not negligible and not guessable — it
  measured +0.105 at 20 seeds and +0.053 at 120 on the same instrument.
- **Ask the yes/no, not the which.** The *"will acting help"* gate has the
  lower break-even at every δ (20% vs 30% at δ = 0.70) and never needs to know
  which region is best. Separately, every signal that expresses confidence in
  the pick itself lands at **AUROC 0.373 against correctness** — stably *below*
  chance at n = 121. The Choice-shaped question is the harder one, has the
  higher break-even, and is the one this model is worst at. If a decision layer
  is wired into this loop, give it the yes/no.

## 14. DOCKSTRING: H1 fails at equal cost, and the published objectives are mostly free

Every library in sections 1–13 is one target deep, which makes *"which molecule,
on which target, next"* unaskable: a retrospective benchmark can only reveal
cells it holds, and the repo's ChEMBL libraries share at most 10% of their
compounds pairwise. DOCKSTRING is 260,148 ligands × 58 targets and 99.998%
dense, so an arbitrary `(molecule, target)` is revealable. The price is that the
labels are AutoDock Vina scores, not measurements — **nothing in this section is
a claim about biology.** What it buys is a testbed where H1 (*multi-task
transfer beats independent-target modelling*) can be decided rather than
assumed.

Instruments: `scripts/dockstring_audit.py`, `scripts/transfer_headroom.py`.

### The dataset is the notebook's, and the tasks are strongly coupled

| | |
|---|---|
| rows / targets | 260,148 / 58 — matches Kubincová & Mobley exactly |
| missing cells | 339 of 15,088,584 (density 99.9978%) |
| kinase pairwise correlation | min +0.48, median **+0.75**, max +0.89 |
| JAK2 vs LCK (the objective's two targets) | **+0.80** |

So H1's precondition holds at the label level: the task covariance is strictly
between 0 and 1, which is where transfer is supposed to pay.

### It does not pay. Budget in cells, and every arm loses to spending it on one target

A **cell** is one `(molecule, target)` measurement. Single-task spends all *n*
cells on the target; multi-task buys `n / (1 + |aux|)` molecules and measures
every target on each. Comparing at equal *molecules* instead hands the
multi-task arm more data and makes any result it produces unfalsifiable.

21 kinase targets, 3 fixed auxiliaries (LCK, SRC, ABL1), pool 2,000, 20 repeats.
Spearman against truth on held-out rows:

| cells | molecules | `stl` | `mt_gp` blocked | `mt_gp` spread | `mt_free` (ceiling) |
|---|---|---|---|---|---|
| 10 | 2 | 0.194 | 0.077 | 0.120 | 0.745 |
| 20 | 5 | 0.273 | 0.135 | 0.204 | 0.772 |
| 30 | 7 | 0.326 | 0.165 | 0.255 | 0.777 |
| 50 | 12 | 0.406 | 0.228 | 0.345 | 0.781 |
| 80 | 20 | **0.471** | 0.282 | 0.415 | 0.784 |

`mt_gp` is a coregionalised Tanimoto GP, `k((m,a),(m',b)) = k_tan(m,m') B[a,b]`
with `B = (1-r)I + r11'` and *r* chosen by marginal likelihood — it selected
**r = 1.0 at every budget**, so the model did see the coupling and used it as
hard as the family allows. See "what r is, and what it is not" below.
`mt_free` is the non-deployable ceiling: the same *n* cells on the target *plus*
the auxiliary columns handed over exactly, for nothing, on every molecule.

**H1 fails: 0 of 21 targets improve, at any budget.** The best multi-task design
still loses 0.055 to the independent one. This is not the model being weak — a
two-stage ridge lands within 0.01 of the ICM GP, and the free-label ceiling is
+0.31 to +0.55, so the structure is unmistakably there.

### The mechanism: at r = 1 a second target on the same molecule is nearly a duplicate

The gap between the blocked and spread designs is the whole story. Blocked
measures all four targets on the same molecules; spread measures each target on
molecules of its own, for the same cell count.

| | 80 cells |
|---|---|
| `stl` — 20 molecules, 1 target | 0.471 |
| `mt_gp` spread — 20 molecules, 4 targets, none shared | 0.415 |
| `mt_gp` blocked — 5 molecules, 4 targets each | 0.282 |

Re-reading the same molecule on a near-perfectly coupled target buys almost nothing;
what the auxiliary cells buy is *coverage of new chemistry*, and spreading them
recovers nearly all of the loss. But even at its best, a cell spent on an
auxiliary target is worth slightly **less** than the same cell spent on the
target you care about. That is a precise, falsifiable statement about this
benchmark, and it is the control any later multi-task result has to clear:
**a multi-task gain measured without holding cells fixed is measuring the
substitution, not the transfer.**

### The published objectives are ~93% a free descriptor

Both of Kubincová & Mobley's objectives carry `-10 (1 - QED)`, and QED is
computed from the SMILES — it costs nothing and is known for every library
member before a single assay.

| objective | sd(QED term) : sd(docking term) | ρ(objective, QED term) | ρ(objective, docking term) |
|---|---|---|---|
| `jak2` | 2.65 : 1 | **+0.929** | +0.200 |
| `jak2_only` | 2.07 : 1 | +0.879 | +0.165 |
| `ppar` | 2.02 : 1 | +0.872 | +0.148 |

ρ(`jak2` objective, −JAK2) is **+0.007**. The objective barely ranks by JAK2
potency.

The consequence for measurement, on the `jak2` objective (pool 2,000, top-50
recall, 20 repeats):

| n | `qed_only` (0 assays) | `joint` | `joint_qed` | `separated` |
|---|---|---|---|---|
| 5 | 48.1% | 3.8% | 47.5% | 47.2% |
| 30 | 49.1% | 5.6% | 49.0% | 47.7% |
| 80 | **51.0%** | 7.7% | 50.4% | 49.1% |

- **Ranking by QED alone, with zero assays, recovers ~48% of the top-50.** Every
  top-k number on these objectives has to be read against that floor, not
  against zero — the same discipline section 3 applies to MaxMin.
- **80 assays buy +2.3 points over the free descriptor.**
- `joint` — regressing the scalarised objective, which is what the notebook does
  — scores **7.7%**, far *below* free QED, because it has to learn the dominant
  term from 80 fingerprints instead of being handed it.
- `separated` vs `joint_qed` (both given the free term): **no separation**,
  +0.6% ± 0.6% at best. Under a global Spearman the two are 0.933 vs 0.933 and
  the comparison is unresolvable, because ρ is pinned at the 0.929 QED floor.
  A per-target surrogate has to justify itself on **allocation**, not accuracy.

### Section 10's GP curve is library-specific

The measured Spearman-vs-observations curve replicated poorly: 0.259 on
DOCKSTRING at n = 30 against 0.462 on ChEMBL, 0/7 points within 0.05 + 3 SEM.
The obvious confound, measured rather than asserted (mean pairwise Tanimoto):

| pool | mean | median |
|---|---|---|
| `egfr_anilinoquinazoline` (congeneric) | 0.299 | 0.283 |
| `jak2_all` | 0.164 | 0.144 |
| `vegfr2_all` | 0.145 | 0.133 |
| `cdk2_all` | 0.134 | 0.120 |
| DOCKSTRING, uniform | **0.121** | 0.117 |

Diversity explains part of it and not all of it — DOCKSTRING sits only 0.013
below `cdk2_all` yet the curve is 0.20 lower. A raw docking column is also more
predictable than the composite objective (`stl` reaches 0.471 at 80 observations
where the `jak2` objective reaches 0.371). **Section 10's numbers are a property
of four target-specific ChEMBL libraries, not of molecular BO**, and the
prior-quality threshold derived from them inherits that scope.

### What this changes

1. **H1 is answered negatively on DOCKSTRING and the answer is mechanistic**, not
   a null: the coupling is real (r = 1.0 selected, ceiling +0.31 to +0.55) and
   still does not survive paying for it. Testing H1 on measured data now has a
   sharp prior and a ready-made control.
2. **Do not run a campaign on `score_jak2` as published without the QED-only
   arm.** Roughly half the top-50 is free.
3. **Cells, not molecules, is the budget unit** for anything multi-task in this
   repo.


## 15. An objective H2 and H3 can actually be measured on

Section 14 closed the published DOCKSTRING objectives for the warm-start and
controller questions: `score_jak2` hands over 95% of its own top-50 recall
before a single assay, so an LLM warm start or an adaptive controller would be
competing for the 2.6 points that 80 assays buy. The obvious repair is to turn
down `- w (1 - QED)`, and the obvious trap is picking `w` by taste.

Instrument: `scripts/objective_floor.py`. Objective family:
`molbo.dockstring.Objective`, which now owns the split — `assayed()` is what
costs measurements, `free()` is what QED supplies for nothing, `admits()` is the
library the question is asked over.

Pool 2,000, top-50 recall, 20 repeats. The reachable arm is the better of
regressing the finished objective and regressing only its assayed part and
adding the free part back, so no treatment is penalised for a naive baseline:

| treatment | admits | floor | via | n=80 recall | headroom | free share | learnable ρ |
|---|---|---|---|---|---|---|---|
| **w = 10** (published) | 2000 | 48.0% | QED | 50.6% | **+2.6%** | **95%** | 0.935 |
| w = 5 | 2000 | 28.0% | QED | 31.4% | +3.4% | 89% | 0.791 |
| w = 2 | 2000 | 10.0% | QED | 11.8% | +1.8% | 85% | 0.467 |
| w = 1 | 2000 | 0.0% | QED | 8.0% | +8.0% | 0% | 0.311 |
| w = 0 | 2000 | 2.5% | chance | 7.9% | +5.4% | 32% | 0.297 |
| **w = 0, QED ≥ 0.5** | 1345 | 3.7% | chance | 11.3% | **+7.6%** | **33%** | 0.326 |

**Free share** — floor ÷ reachable — is the decisive column. At `w = 10` the
optimiser earns 5% of its own score. Under a QED ≥ 0.5 admission filter it earns
67%, and the band to measure in is three times wider.

`w = 1` bands statistically indistinguishably (+8.0% vs +7.6%), and the filter
is taken anyway: a weight is a knob fitted to this benchmark that would have to
be re-justified on the next one, an admission filter is a statement about which
molecules the question is asked over and survives the move. It is also what a
medicinal chemist does with drug-likeness.

**The filter removes molecules rather than scoring them badly.** A reject left
in the library at a bad score is still an objective term, and the optimiser can
spend budget rediscovering it — which is the thing this section exists to stop.
`objective_table` omits them; `tests/test_dockstring.py` pins it.

### What the new objective measures

`jak2_potency` = `-JAK2 + min(LCK, -8.1)` over `QED ≥ 0.5`, same pool, top-50:

| n | chance | `joint` | `separated` |
|---|---|---|---|
| 5 | 4.0% | 4.6% | 4.9% |
| 30 | 4.0% | 9.4% | 9.5% |
| 80 | 3.7% | 11.6% | 11.9% |

80 assays take the arm to **3.2× chance**, against 1.05× on the published
objective. That is the band H2 and H3 now have to show an effect inside.

`joint` and `separated` remain indistinguishable (+0.3% ± 0.9% at n = 80) — the
same null as section 14, now on an objective where a null means something,
because it is no longer being read off a QED ranking. **A per-target surrogate
has to justify itself on allocation, not on accuracy**, on both objectives.

### Standing rules for anything run on DOCKSTRING from here

1. **Report the floor.** Chance for a filtered objective, the free ranking for a
   weighted one. Never against zero.
2. **`jak2_potency` / `ppar_potency` for H2 and H3.** The published objectives
   stay for the comparison to Kubincová & Mobley and for nothing else.
3. **Cells, not molecules** (section 14).


## 16. H1 survives, but only in the asymmetric form — *superseded by section 17*

> **Correction.** Everything below is DOCKSTRING only. Section 17 ran the same
> design on measured ChEMBL pIC50 and found no effect at any budget or any `H`.
> The +0.445 here is a property of AutoDock Vina. Read section 17 first.
>
> **Audited, and the effect sizes move.** Section 30 found that the table below
> is an *unpaired* comparison: the `H = 0` control and every `H > 0` arm were
> fitted on different primary rows and scored on different held-out rows, in
> 120 of 120 cells. Re-run paired (`--c-paired --c-holdout 0.4`) the conclusion
> is unchanged — break-even at H = 5–10, 4/4 targets at every H past it — but
> **the gain shrinks in 15 of 18 cells**, by −0.014 on average and up to −0.049,
> with the n = 10 row worst hit. The headline +0.445 becomes **+0.427**.
> **Quote the paired numbers in section 30, not the ones below.**

Section 14 rejected H1 as *budget-splitting*: spreading a fresh budget of cells
across coupled targets loses to concentrating it, 0 of 21 targets, at every
budget. That result stands. It is also not the question a real campaign asks.

The arm that won in section 14 was `mt_free`, +0.31 to +0.55, where the
auxiliary columns are already known. That is not a fantasy condition — it is the
ordinary one. A new target starts with nothing and its relatives start with
years of assays that cost *this* campaign nothing, because someone else already
paid for them. Section 14's design gave every task the same amount of data and
so never touched the case where multi-task GPs have always earned their keep.

Check C (`scripts/transfer_headroom.py --check-c`) measures it. The primary
target gets `n` observations charged to the budget; each of three auxiliary
kinases gets `H` historical observations on *different* molecules, free. The
historical rows are disjoint from the primary's training rows and from the
evaluation rows, so no arm sees a test label. `H = 0` is exactly the
single-task GP. 4 targets, 10 repeats, pool 2,000.

| n | H=0 (`stl`) | H=5 | H=10 | H=25 | H=50 | H=100 | H=200 |
|---|---|---|---|---|---|---|---|
| 10 | 0.180 | 0.257 | 0.325 | 0.437 | 0.500 | 0.579 | **0.626** |
| 20 | 0.281 | 0.313 | 0.359 | 0.453 | 0.523 | 0.589 | **0.631** |
| 80 | 0.470 | 0.478 | 0.509 | 0.533 | 0.574 | 0.612 | **0.649** |

Gain over the single-task GP:

| n | H=10 | H=25 | H=50 | H=200 |
|---|---|---|---|---|
| 10 | +0.145 ± 0.017 | +0.257 ± 0.018 | +0.320 ± 0.019 | **+0.446 ± 0.017** |
| 20 | +0.078 ± 0.010 | +0.172 ± 0.020 | +0.242 ± 0.021 | +0.350 ± 0.022 |
| 80 | +0.039 ± 0.009 | +0.063 ± 0.002 | +0.104 ± 0.012 | +0.179 ± 0.012 |

**Break-even is 5–10 historical observations per auxiliary target**, and 4 of 4
targets improve at every H past it. At n = 10 with H = 200 the cold target is
predicted better (0.626) than a single-task GP reaches at n = 80 (0.470) —
**existing data on three relatives is worth more than 8× the fresh budget.**

### The two H1s are different claims and only one is dead

| | claim | verdict |
|---|---|---|
| H1-split | spread a fresh budget across coupled targets | **rejected**, 0/21 targets (§14) |
| H1-cold | a new target borrows from existing data on relatives | **supported**, break-even at H ≈ 5–10 |

The mechanism reconciles them. At the task correlation DOCKSTRING has (r = 1.0
selected by marginal likelihood) an auxiliary observation is nearly as
informative as a primary one — which is *fatal* when you paid for it with a
molecule you could have measured on the target, and *decisive* when it was free.
Same coupling, opposite sign, and the difference is entirely who paid.

### What this means for the project

1. **Reframe H1 before running it.** "Multi-task transfer improves BO" is not
   falsifiable as stated; it is two hypotheses with opposite answers here. The
   comparison to state is: *cold target + frozen historical auxiliaries* vs
   *single-task on the same fresh budget*.
2. **Report the history, not just the budget.** An H1 result is a function of
   `H` as much as of `n`, and the table above spans +0.04 to +0.45 across `H` at
   fixed `n`. A paper that quotes one number has picked an `H`.
3. **Caveat the correlation.** r = 1.0 is a property of Vina scores, which are
   driven heavily by generic ligand size and lipophilicity. Real cross-kinase
   pIC50 is correlated far less, and heterogeneously. DOCKSTRING establishes the
   *mechanism* and the shape of the `H` curve; it cannot establish the effect
   size for measured data. The ChEMBL libraries in `data/chembl/` are the place
   to check whether break-even at H ≈ 5–10 survives a realistic r.

### What *r* is, and what it is not

`r` is the task-correlation parameter of the coregionalisation matrix, fitted by
marginal likelihood — **not** an empirical correlation between docking columns.
After per-task z-scoring, `B[a,a] = 1` and `B[a,b] = r`, so `r` is the model's
prior correlation between two targets' latent functions at the *same* molecule.
`r = 0` is exactly the independent-target GP; `r = 1` is one function wearing
four labels, with everything that distinguishes the targets pushed into the
noise term.

Three corrections to how this was first reported:

- **It is not the audit's +0.75.** The audit measures Pearson between *observed*
  columns, which carries each target's idiosyncratic part. `r` is the
  correlation of the shared smooth component after `eta` absorbs the rest, so it
  is expected to be larger. The two numbers are not independent confirmations of
  each other and must not be quoted as if they were.
- **0.97 was a wall, not an estimate.** The first `TASK_RHO_GRID` stopped at
  0.97 and every fit selected exactly 0.97. Extending it to 1.0 moved all 8
  target × budget configurations to **1.0**. Re-running both checks on the
  corrected grid changed every number in the third decimal and no conclusion:
  check A's spread design −0.055 → −0.056, check C's break-even unchanged at
  H ≈ 5–10. A selected value sitting on its own grid boundary is not a
  measurement, and the earlier "r = 0.97" should be read as "≥ 0.97".
- **The one-parameter `B` is a real restriction.** A single shared `r` forces
  JAK2–ABL1 (empirical +0.59) and JAK2–LCK (+0.80) to the same value; the
  likelihood resolves the mismatch by railing `r` high and letting per-task
  noise absorb the difference. A free PSD `B`, or low-rank plus diagonal, would
  fit that structure properly. It is unlikely to overturn either conclusion —
  a richer `B` can only help the multi-task arm, which makes check A's negative
  **conservative** and check C's positive an **understatement** — but the effect
  sizes in both are lower bounds on what a better task covariance would give.
  This is the first thing to relax if H1-cold is taken further.

## 17. H1-cold does not replicate on separate ChEMBL libraries — *confound resolved in §18*

> **Update.** The confound this section could not settle is settled in section
> 18: it was library geometry, not task correlation. On a co-measured dense
> block H1-cold holds at r = 0.60. The null below is real but is a statement
> about disjoint per-target libraries, not about measured data in general.
>
> **Audited, and the null gets stronger.** Section 30 found the same unpaired
> split here as in section 16. Re-run paired, the gain itself barely moves
> (mean −0.001, as many cells up as down) but the SEMs fall by ~40% and the
> largest gain anywhere in the grid halves, from +0.016 ± 0.026 to
> +0.008 ± 0.007. The sentence below — "this is not an underpowered null" — was
> a claim when it was written and is now measured on a tighter instrument.
> **Quote the paired numbers in section 30.**

Section 16 reported that a cold target borrowing from free historical data on
its relatives beats a single-task GP: break-even at 5–10 observations per
auxiliary, **+0.445 Spearman at H = 200**, 4/4 targets. It flagged the risk that
this was a property of Vina scores. It is.

Instrument: `scripts/cold_target_chembl.py`. The identical design — primary
target cold at `n` charged observations, three auxiliaries carrying `H` free
historical ones, `H = 0` the single-task control — on four measured ChEMBL
kinase libraries (JAK2, CDK2, VEGFR2, EGFR; 1,000 compounds each, 3,970 unique).

| n | H=0 (`stl`) | H=10 | H=25 | H=50 | H=100 | H=200 |
|---|---|---|---|---|---|---|
| 10 | 0.168 | +0.014 | −0.007 | +0.016 | +0.007 | +0.011 |
| 20 | 0.282 | −0.028 | −0.041 | −0.026 | −0.013 | −0.008 |
| 80 | 0.476 | +0.001 | −0.016 | +0.004 | −0.009 | −0.006 |

**Nothing.** Every cell is inside noise; the whole table spans −0.041 to +0.016,
against a DOCKSTRING effect of +0.14 to +0.45 at the same budgets and the same
`H`. No budget clears 3 SEM at any `H`. The SEMs here are 0.003–0.031, so this
is not an underpowered null — an effect a quarter the size of DOCKSTRING's would
have been visible.

The single-task baselines are sound: `stl` reaches 0.476 at n = 80, consistent
with section 10's ChEMBL curve (0.542). The instrument works; there is no
transfer to find.

### The fitted task correlation is the visible difference

| dataset | `r` selected by marginal likelihood |
|---|---|
| DOCKSTRING (58 Vina targets) | **1.00**, every target × budget |
| ChEMBL (4 measured kinases) | median **0.40**, 10th 0.00, 90th 1.00 |

Docking scores across pockets are nearly one function — generic ligand size and
lipophilicity drive much of the variance, and every pocket sees them. Measured
pIC50 is not, and the wide spread of selected `r` says it is not even *uniformly*
coupled: some target–repeat combinations want full sharing and others want none,
which a single shared `r` cannot express.

### A confound this section does **not** resolve

The two datasets differ in a second way, and it is not controlled.

| | own-library NN similarity | auxiliary NN | reach ratio |
|---|---|---|---|
| ChEMBL libraries | 0.64–0.71 | 0.31–0.40 | **0.53** |
| DOCKSTRING, uniform pool | 0.38 | 0.38 | 1.00 |

A ChEMBL library is a chemotype someone assembled for one target; DOCKSTRING
hands every task the same uniform ZINC sample, so its auxiliary observations sit
right next to the compounds the primary is scored on. If the Tanimoto kernel
cannot reach from the auxiliary chemistry to the primary's test set, transfer
fails for a reason that has nothing to do with task correlation.

`--c-aux-region distant` was built to test that, drawing DOCKSTRING's
auxiliaries from a separated half of the pool. The effect barely moved (+0.377
vs +0.398 at H = 100). **That does not acquit chemical reach**: the split
measures a reach ratio of **0.85**, against ChEMBL's 0.53, so it is a weaker
separation than the thing it imitates. The instrument now prints its own ratio
next to the table for exactly this reason.

The two candidate mechanisms therefore remain:

1. **Task correlation** — real kinases are coupled at r ≈ 0.4 and heterogeneously,
   not at r = 1.
2. **Library geometry** — real per-target libraries are disjoint chemotypes, and
   the kernel cannot carry transfer between them.

They are not exclusive and both are plausible. The decisive experiment is a
cluster-based split of the DOCKSTRING pool — Butina clusters as surrogate
libraries, primary and auxiliaries in different clusters — tuned to reproduce a
reach ratio near 0.53 at a task correlation of 1.0. `src/molbo/clusters.py`
already has the clustering. That is the next thing to run.

### Status of H1, corrected

| | claim | verdict |
|---|---|---|
| H1-split | spread a fresh budget across coupled targets | **rejected** (§14), docking |
| H1-cold | cold target borrows from free history on relatives | **docking only** (§16); **does not replicate on measured pIC50** (§17) |

Section 16's +0.445 stands as a statement about AutoDock Vina and about nothing
else. **No version of H1 currently has support on measured data.** Before
building a multi-task surrogate for the paper, either the cluster experiment
above has to identify a recoverable mechanism, or H1 should be demoted from a
headline hypothesis to a negative control that H2 and H3 are reported against.

One repair is worth trying first, because it is cheap and it is the known
weakness of the model rather than of the hypothesis: the one-parameter
`B = (1-r)I + r11'` cannot represent kinases that are coupled unevenly, and the
ChEMBL fits visibly want that (r spanning 0.00 to 1.00 across repeats). A free
PSD `B`, or low-rank plus diagonal, is the first thing to relax — and on
measured data a fitted `B` is itself a result, being a target-similarity matrix
estimated from assay data rather than from sequence.

## 18. H1-cold holds on measured data — and (on JAK) the coregionalisation, not the extra data, is what works

> **Amended by §19.** The main result stands. The mechanism claim does not
> generalise: on the CYP block, naive pooling *ties* the fitted model. What
> decides it is how unevenly the family is coupled, not multi-task modelling
> per se. See section 19.

Section 17 left two explanations for the ChEMBL null and could not separate
them: real kinases are coupled at r ≈ 0.4, or real per-target libraries are
disjoint chemotypes the Tanimoto kernel cannot bridge. Transfer flows through
`k_tanimoto(m, m') × r`, and on ChEMBL those were ≈ 0.35 and ≈ 0.40 — an
effective coupling of **0.14**. The two failures multiply, so fixing either one
alone could not have worked.

A **dense block** removes one factor outright. BindingDB contains selectivity
panels — compound sets deliberately run against every member of a family — and
inside one, the auxiliary observations are on *the same molecules* as the
primary's. Reach is 1.0 by construction and cannot explain anything.

Instrument: `scripts/dense_block_h1.py`. Block: **JAK1/JAK2/JAK3/TYK2, 1,563
compounds × 4 targets, 6,252 cells, fully observed**, pIC50 (median over
replicates) from `data/cache/tdc/bindingdb_ic50.csv`.

| | JAK1 | JAK2 | JAK3 | TYK2 |
|---|---|---|---|---|
| **JAK1** | — | 0.71 | 0.59 | 0.42 |
| **JAK2** | 0.71 | — | 0.54 | 0.62 |
| **JAK3** | 0.59 | 0.54 | — | 0.24 |
| **TYK2** | 0.42 | 0.62 | 0.24 | — |

Measured, heterogeneous, median +0.57 — well below DOCKSTRING's 1.0 and exactly
the regime H1 is supposed to exploit.

### H1-cold holds, at every budget

| n | H=0 (`stl`) | H=25 | H=50 | H=100 | H=200 |
|---|---|---|---|---|---|
| 10 | 0.329 | +0.067 | +0.140 | +0.176 | **+0.204 ± 0.029** |
| 20 | 0.442 | +0.048 | +0.101 | +0.133 | +0.151 ± 0.017 |
| 40 | 0.550 | +0.020 | +0.050 | +0.065 | +0.089 ± 0.013 |
| 80 | 0.630 | +0.011 | +0.016 | +0.037 | +0.057 ± 0.008 |

Break-even at **H ≈ 50** per auxiliary (H ≈ 100 at n = 80), 88–95% of runs
positive. **This is the first support for any form of H1 outside docking
scores.** The effect is about half DOCKSTRING's and break-even is 5–10× higher —
the realistic discount — but it is there and it is resolvable.

Section 17's confound is resolved in favour of **geometry**: r here is only 0.60
(fitted; empirical median 0.57), far below DOCKSTRING's 1.0, and transfer works
anyway *because the tasks share their compounds*. Low task correlation was never
the blocker. **Reach was.**

### The control that matters: pooling makes it worse

The obvious objection is that H = 200 adds 600 observations and of course more
data helps. It does not. Discarding the task structure — fixing r = 1, which
after per-task z-scoring is exactly "pool all four targets into one function" —
gives:

| n | arm | H=50 | H=100 | H=200 |
|---|---|---|---|---|
| 80 | `stl` (H=0) | 0.630 | 0.630 | 0.630 |
| 80 | pooled | 0.538 | 0.519 | **0.512** |
| 80 | fitted ICM | 0.646 | 0.667 | **0.687** |

**Naive pooling is worse than ignoring the auxiliary data entirely**, and it gets
worse as more of it arrives — 0.630 → 0.512 at n = 80. The fitted model beats it
by +0.175 there. The same inversion holds at every budget (fitted − pooled:
+0.086, +0.146, +0.169, +0.175 at H = 200 for n = 10/20/40/80).

So the gain is **not** extra rows. It is the estimate of how much the targets
share: pool naively and the auxiliary targets drag the primary toward a
consensus that is wrong for it. That also explains why section 14's H1-split
fails — and it fails here too, on measured data, at −0.127 ± 0.014 at 80 cells,
reproducing the DOCKSTRING result without either caveat.

### H1, final status

| | claim | verdict |
|---|---|---|
| H1-split | spread a fresh budget across coupled targets | **rejected** — docking (§14) *and* measured (§18, −0.127) |
| H1-cold, disjoint libraries | separate per-target libraries | **no effect** (§17) — reach 0.53 kills it |
| H1-cold, dense block | co-measured selectivity panel | **holds** (§18), break-even H ≈ 50 |

### What this means for the project

1. **H1 is alive, with a precondition.** It is not "multi-task transfer helps."
   It is: *transfer helps when the auxiliary targets were measured on compounds
   that reach the primary's chemistry, and the sharing is estimated rather than
   assumed.* Both halves are load-bearing and each has a control here.
2. **Pick data accordingly.** Selectivity panels and family dense blocks, not
   four separate ChEMBL libraries. `h1_dense_blocks.json` already lists the
   candidates; the CYP block (1,472 × 4) is the obvious replication target, and
   it is a different protein class, which is the right robustness check.
3. **Report the pooling control.** Any multi-task number in this project without
   it is unreadable — a reviewer cannot tell modelling from extra rows, and here
   the two have *opposite signs*.
4. **Free `B` is now more interesting, not less.** The family is unevenly
   coupled (0.24 to 0.71) and one shared r still helped; a matrix that can hold
   JAK3–TYK2 at 0.24 while holding JAK1–JAK2 at 0.71 should help more, and on
   measured data a fitted `B` is a target-similarity matrix estimated from
   assays rather than from sequence.

### Verification addendum: the holdout, and what it moves

Section 18's numbers were produced with `te = pool - draw`, so the H = 200 arm
was scored on 600 fewer compounds than the H = 0 baseline it is differenced
against. The removed compounds are random, so the mean should not move — but
"should not" is a claim, and `--holdout` now reserves one test set (625
compounds, 40%) that every H is scored on. Re-run:

| n | H | gain, varying test set | gain, shared test set |
|---|---|---|---|
| 20 | 100 | +0.133 ± 0.016 | +0.099 ± 0.017 |
| 20 | 200 | +0.151 ± 0.017 | **+0.128 ± 0.016** |
| 80 | 100 | +0.037 ± 0.008 | +0.046 ± 0.010 |
| 80 | 200 | +0.057 ± 0.008 | **+0.063 ± 0.009** |

The effect survives; the artifact is worth about ±0.02 and does not have a
consistent sign. **The break-even moves, though**: under a shared test set,
H = 25 at n = 20 is +0.022 ± 0.021 (not resolvable) and H = 100 is +0.099 ±
0.017. Read break-even as **H ≈ 100**, not the H ≈ 50 stated above.

The pooling control is unchanged and slightly stronger under the shared test
set: at n = 80, H = 200, pooled reaches 0.494 against `stl`'s 0.617 and the
fitted model's 0.680 — pooling still loses to using no auxiliary data at all,
by more than a tenth of a Spearman.

The effect also replicates across seeds. At n = 80, H = 200: +0.057 (seed 0),
+0.059 (seed 1), +0.058 (seed 2). At n = 20 it is noisier, +0.151 / +0.114 /
+0.128, which is inside its own SEM and is the budget to be careful quoting.

**Prefer `--holdout 0.4` for anything quoted.** The default of 0 is kept only
because sections 14–18 were produced with it and changing it silently would
make the earlier tables unreproducible.

## 19. CYP replicates H1-cold — and corrects §18 on *why* it works

The CYP block (CYP3A4/2D6/2C9/2C19, 1,472 × 4) was the robustness check §18
named: a different protein class, same instrument. It replicates H1-cold and it
overturns §18's explanation of the mechanism.

### First: 77% of that block is not a measurement

| | value held by >2% of the column |
|---|---|
| CYP3A4 | 5.00 ×317, 4.52 ×122, 4.30 ×90, 5.30 ×68, 4.00 ×64, … |
| CYP2D6 | 5.00 ×399, 4.52 ×136, 4.30 ×88, 4.00 ×79, … |

Those are pIC50 for 10, 30, 50, 5 and 100 µM — the fixed doses of a screening
panel, stored as if they were potencies. **40.3% of CYP compounds carry the
identical value on all four targets** (JAK: 3.5%); 16% are exactly 5.00 on all
four. Every quantity here moves the same way under that: cross-target
correlation inflates, fitted `r` rails to 1, pooling looks optimal, and
"transfer" becomes the copying of a censored constant.

`--drop-censored` removes compounds touching any pile-up value. **CYP keeps 335
of 1,472 (22.8%). JAK keeps 904 of 1,563 (57.8%).** Every number below is from
the filtered blocks with a shared holdout.

### H1-cold replicates

| block | n=10 | n=20 | n=40 | break-even |
|---|---|---|---|---|
| CYP (clean, H=50) | **+0.256 ± 0.017** | +0.181 ± 0.021 | +0.117 ± 0.020 | H ≈ 10–25 |
| JAK (clean, H=50) | +0.101 | +0.086 | +0.049 ± 0.015 | H ≈ 25–50 |

Two protein classes, both measured, both co-measured panels, both positive at
every budget. H1-cold is not a kinase quirk.

### But §18's mechanism claim was over-general

Section 18 concluded "the coregionalisation, not the extra data, is what works,"
from JAK, where naive pooling was *worse than using no auxiliary data at all*.
On CYP, pooling ties the fitted model exactly:

| block | fitted − pooled, at H=50 | coupling range |
|---|---|---|
| JAK | **+0.043 / +0.066 / +0.083** (n=10/20/40) | 0.24 – 0.71 (**spread 0.47**) |
| CYP | −0.001 / −0.006 / −0.003 | 0.52 – 0.74 (**spread 0.22**) |

The discriminator is not the protein class and not the average coupling — both
families sit near +0.6 on average. It is **how unevenly the family is coupled**.

- JAK3–TYK2 at 0.24 while JAK1–JAK2 sits at 0.71. A single shared `r` is wrong
  for both, pooling is badly wrong, and estimating `r` rescues most of it.
- Every CYP pair falls in 0.52–0.74. One number describes the family, that
  number is near 1 after z-scoring, and pooling *is* the right model. The fitted
  arm selects r = 1.00 and recovers it — correctly, adding nothing.

**Corrected claim:** modelling the task structure earns its cost in proportion
to how heterogeneous the coupling is. Where a family is uniformly coupled,
pooling is not a strawman — it is the answer, and a multi-task surrogate should
be expected to tie it, not beat it.

> **Falsified by §20.** Free `B` was run and loses on both blocks. The
> heterogeneity explanation below did not survive its own prediction.

This also sharpens the free-`B` step. A full PSD `B` can only help where one `r`
is a poor summary, so the predicted ordering is **large gain on JAK, ~none on
CYP** — which makes it a falsifiable next experiment rather than an upgrade.

### Standing rules, updated

1. **Screen dense blocks for pile-up values before using them.** A block can be
   three-quarters fixed screening doses and look fine in aggregate. Report the
   kept fraction.
2. **Always run the pooling control**, and report the family's coupling *spread*
   beside it — the two together say whether the model is earning anything.
3. **`--holdout`** for anything quoted (§18 addendum).

## 20. Free B does not pay — the §19 prediction is falsified

Section 19 predicted a full task correlation matrix would help where the family
is unevenly coupled (JAK, spread 0.47) and not where it is uniform (CYP, spread
0.22). It was stated as a falsifiable prediction. It is false.

`CoregionalisedGP(structure="free")` fits every pair by coordinate ascent on the
exact marginal likelihood, two starts, PSD-checked. Clean blocks, shared
holdout, 8 repeats, free-B minus shared-r:

| | H=10 | H=25 | H=50 | fitted B spread |
|---|---|---|---|---|
| **JAK** (predicted to gain) | −0.003 | −0.005 | **−0.029** | 0.49 |
| **CYP** (predicted flat) | −0.009 | −0.011 | −0.006 | 0.26 |

Not a null on JAK — a small consistent *loss*, largest where there is most
auxiliary data to overfit to.

The fitted matrix recovers the structure it was supposed to: spread 0.49 on JAK
against an empirical 0.47, 0.26 on CYP against 0.22. So the failure is not that
it cannot see the heterogeneity. It sees it and still predicts worse, because
six correlations estimated from 10–40 primary observations cost more in variance
than the uneven coupling costs in bias. **One shared `r` is not a simplification
to be improved on here; at these budgets it is the right amount of structure.**

That also retires the §19 explanation for why fitting beat pooling on JAK and
tied on CYP. Coupling heterogeneity predicted the free-B result and the
prediction failed, so it should not be carried as the mechanism for the pooling
result either. What survives is the measurement: fitted `r` beats pooling on
JAK, ties on CYP, and why is open.

**Retired.** Shared-`r` ICM stays the surrogate. Do not spend more on the task
covariance parameterisation at these budgets.

## 21. The transfer gain survives as a campaign result — *bounded by §22*

> **Read §22 first.** The direction holds, the magnitude does not: most of the
> gain below comes from the history containing the primary top-10, and the
> free-history-vs-extra-assays exchange rate is 5:1 against. The JAK/CYP
> pooling split does not survive the control.

Sections 18–20 measure Spearman. The project's claim is about Bayesian
optimisation, and section 4 already records offline proxies inverting against
campaign outcome twice, so a regression gain is a hypothesis about campaigns,
not a campaign result. `scripts/dense_block_campaign.py` runs the loop.

Same clean dense blocks, same free history (H = 50 per auxiliary, fixed before
the campaign and never grown — history, not budget). Every arm gets the
identical initial design, budget, batch and acquisition; only the surrogate
differs. 60 assays on the primary, batch 5, n_init 10, EI, 15 repeats × 4
targets.

| block | arm | AUC top-10 | top-10 recall | ΔAUC vs stl |
|---|---|---|---|---|
| **JAK** | stl | 8.1563 | 27.2% | — |
| | **mt** | **8.3069** | **37.3%** | **+0.151 ± 0.022** (80% of campaigns) |
| | pooled | 8.2267 | 32.2% | +0.070 ± 0.028 |
| **CYP** | stl | 6.1773 | 38.0% | — |
| | **mt** | **6.4667** | **57.3%** | **+0.289 ± 0.029** (95% of campaigns) |
| | pooled | 6.4434 | 54.0% | +0.266 ± 0.031 |

**Multi-task BO with free family history recovers 1.4–1.5× as much of the top-10
at the same assay budget** — 27% → 37% on JAK, 38% → 57% on CYP. This is the
contribution, stated at the level the paper claims it.

The pooling control splits the same way it did on regression: on JAK the fitted
model more than doubles pooling's gain (+0.151 vs +0.070), on CYP they tie
(+0.289 vs +0.266). That the split reproduces at campaign level, having been
found at regression level, is worth more than either measurement alone — though
§20 killed the *explanation* for it, so it stays a finding without a mechanism.

### Two bugs this found, both invisible to the regression work

`CoregionalisedGP.predict_gram` returned the posterior on the **per-task
z-scored** scale. Spearman is invariant to affine maps, so every number in
§§18–20 is unaffected — but expected improvement compares the mean against an
incumbent in raw pIC50, so every candidate looked equally hopeless, EI collapsed
to zero, and the loop degenerated to its tie-break jitter. The symptom was `mt`
and `pooled` agreeing to four decimals across twelve campaigns, which is not
something two different models do.

The unit test that should have caught it asserted the single-task equivalence
*up to an affine map* — precisely the slack the bug lived in. It now asserts
equality, plus scale equivariance.

**A surrogate that is only ever scored by rank correlation can carry a bug that
makes it useless in a loop.** Anything added to `molbo.surrogate` should be
driven through a campaign before it is trusted, not just through a ρ.

## 22. Two pre-paper controls that move the headline

Section 21 reported 1.4–1.5× top-10 recovery at fixed budget. Both controls a
reviewer runs first were run. The result survives; the headline does not.

### Control 1: what is the free history actually worth in assays?

Single-task BO at increasing budget, JAK, 15 repeats:

| stl budget | 60 | **90** | 120 | 180 | 240 |
|---|---|---|---|---|---|
| top-10 recall | 27.2% | **37.7%** | 49.5% | 67.0% | 71.5% |

`mt` at budget 60 scores 37.3%. **A single-task campaign reaches the same place
at budget 90.** So 150 free auxiliary assays (50 × 3 targets) are worth 30
primary assays: an exchange rate of **5:1 against auxiliary data**.

That is not a contradiction — it is §14 and §18 restated in campaign units, and
the arithmetic is consistent across all three. The gain is real *because the
history is free*, not because auxiliary cells are efficient. A paper that quotes
1.4× without the exchange rate invites a reviewer to derive it and conclude the
number was framed.

### Control 2: does the history point at the answer?

The history is drawn uniformly from the pool the campaign then searches, so it
can contain the primary's top-10 outright. `--history-excludes-top` removes
them.

| block | arm | recall, uniform history | recall, top-10 withheld |
|---|---|---|---|
| **JAK** | stl | 27.2% | 27.2% |
| | mt | **37.3%** | **30.8%** |
| | pooled | 32.2% | 31.2% |
| **CYP** | stl | 38.0% | 38.0% |
| | mt | **57.3%** | **40.7%** |
| | pooled | 54.0% | 43.3% |

ΔAUC falls from +0.151 to +0.084 (JAK) and +0.289 to +0.102 (CYP). **Between
half and two-thirds of the headline gain comes from the historical set happening
to contain the compounds the campaign is looking for.**

Neither condition is the honest one on its own. Uniform history is optimistic:
real historical data is not a uniform sample of the library you are screening
now — §17 measured real per-target libraries reaching only 0.53 of the primary's
own nearest-neighbour similarity. Withholding the top-10 is adversarial: it
guarantees zero overlap with the answer, which no real history does.

> **§23 correction.** The overlap explanation below was tested and is
> unsupported: gain is flat from reach 0.96 to 0.44. What is sensitive is
> whether the history contains the top-k, not chemical overlap generally.

**The true value lies between +2.7 and +19.3 recall points and is governed by
how much the historical set overlaps the chemotypes now being searched.** That
is the paper's actual claim, and it is a better one than a single ratio: it says
*when* transfer pays, and it connects directly to §17, where transfer failed on
libraries with no overlap.

### The JAK/CYP split does not survive either

Under withheld top-10, `mt` (30.8%) sits *below* `pooled` (31.2%) on JAK. The
§19/§21 observation that fitting `r` beats pooling on JAK and ties on CYP holds
only in the optimistic condition. §20 already removed its explanation; §22
removes the observation. **Treat fitted-`r`-beats-pooling as unsupported.**

### The pattern, three times in one session

DOCKSTRING's H1-cold (+0.45) → ChEMBL (nothing) → dense block (+0.20). §19's
mechanism → falsified by §20. §21's 1.4× → §22's 1.07–1.13× under the control.
Every headline in this line of work has shrunk when the control was run, and
none has been overturned in sign. **Run the control before writing the number,
not after.**

## 23. The reach hypothesis is dead; the baselines hold

Two additions to the screening claim: a curve that was supposed to be the main
figure, and the baselines the contribution has to clear. The curve does not
exist. The baselines are cleared decisively.

### Gain does not track chemical overlap

Section 22 framed the campaign gain as "governed by how much the historical set
overlaps the chemotypes being searched," and proposed measuring it as a curve
against the reach ratio. `--history-cluster` draws a controllable fraction of
the history from one tight neighbourhood instead of uniformly; the achieved
reach is measured per draw. Top-10 always withheld, so this isolates overlap
from *contains-the-answer*. 15 repeats, budget 60.

| | reach | stl | **mt** | mt − stl AUC |
|---|---|---|---|---|
| **JAK** | 0.86 | 27.2% | 30.8% | +0.084 ± 0.023 |
| | 0.84 | 27.2% | 32.8% | +0.110 ± 0.023 |
| | 0.80 | 27.2% | 29.8% | +0.083 ± 0.025 |
| | 0.74 | 27.2% | 27.0% | +0.060 ± 0.025 |
| | **0.44** | 27.2% | 28.3% | **+0.077 ± 0.020** |
| **CYP** | 0.96 | 38.0% | 40.7% | +0.102 ± 0.028 |
| | 0.63 | 38.0% | 36.2% | +0.042 ± 0.026 |

**Flat.** Across a reach range of 0.86 → 0.44 the JAK gain moves from +0.084 to
+0.077, wandering ±0.03 with SEMs of 0.023. CYP drifts downward but every point
overlaps its neighbours.

This matters beyond a missing figure. Real ChEMBL libraries measure **0.53**, and
JAK at reach **0.44** — below that — still gains +0.077. So chemical reach
**cannot** be why §17 found nothing on separate ChEMBL libraries. §22's
explanation is withdrawn, and with it the one unifying story this line of work
had.

What remains different between §17 and the dense blocks: the ChEMBL set was four
*different kinase families* (JAK2, CDK2, VEGFR2, EGFR), the blocks are one
family each. Target relatedness is the surviving candidate and it is not tested.
**Why transfer fails across separate libraries is open.**

### What *is* sensitive is whether the history contains the answer

§22 conflated two things this separates cleanly:

| effect | size |
|---|---|
| history contains the primary's top-10 | **large** — JAK 37.3% → 30.8% when withheld |
| chemical reach, 0.96 → 0.44 | **none measurable** |

Report the first as the sensitivity. Do not report the second.

### The baselines are cleared, and one of them is not close

| block | random | **similarity search** | stl (single-task BO) | **mt** |
|---|---|---|---|---|
| JAK | 6.0% | 11.7–13.5% | 27.2% | **27.0–32.8%** |
| CYP | 17.3% | 25.3–28.7% | 38.0% | **36.2–40.7%** |

Similarity search — rank the library by Tanimoto to the best actives known so
far, from history and measurements alike, which is what a medicinal chemist does
without a model — is **worse than single-task BO on both blocks**, by 14 points
on JAK. It uses exactly the information the multi-task arm uses and costs
nothing, so this is the comparison that says the modelling earns its place. It
does.

Random sits where it should, and the spread between random and `mt` (6% → 30%
on JAK) is the honest measure of how much all the machinery is worth.

### The claim, as it now stands

> Given free historical assay data on a co-measured target family, a
> coregionalised GP improves top-10 recovery over single-task BO by 3–6 points
> at a 60-assay budget, robustly across how the history is distributed in
> chemical space, and beats a model-free similarity baseline by 14–17 points.
> The gain is larger when the history happens to include the compounds sought
> (up to +19 points) and the auxiliary assays are worth about one fifth of a
> primary assay each, so the result depends on the history being already paid
> for.

Flat-in-reach is a *better* claim than the conditional one — it is robust rather
than contingent. It is simply not the claim §22 predicted.

## 24. Toward practical guidance: budget matters, and the per-target lottery blocks a pre-flight check

Two questions a practitioner has before spending money on molecules they have
not yet assayed: *is my budget in the range where this helps?* and *can I tell
in advance whether it will help for my target?* Budget sweep with per-target
breakdown, top-10 withheld from history, 15 repeats.

| budget | JAK stl | JAK mt | Δrecall | targets improving | per-target spread |
|---|---|---|---|---|---|
| 20 | 4.7% | 4.0% | **−0.7%** | 2/4 | −3.3% .. +1.3% |
| 40 | 15.2% | 16.8% | +1.7% | 2/4 | −12.0% .. +14.7% |
| 60 | 27.2% | 30.8% | +3.7% | 3/4 | −15.3% .. +15.3% |
| 100 | 41.7% | 52.7% | **+11.0%** | 2/4 | −2.0% .. +30.7% |
| 150 | 60.8% | 68.7% | +7.8% | 3/4 | −6.7% .. +18.7% |

| budget | CYP stl | CYP mt | Δrecall | targets improving | per-target spread |
|---|---|---|---|---|---|
| 20 | 11.3% | 7.0% | **−4.3%** | 0/4 | −6.0% .. −2.0% |
| 40 | 22.0% | 22.2% | +0.2% | 2/4 | −5.3% .. +8.0% |
| 60 | 38.0% | 40.7% | +2.7% | 2/4 | −8.0% .. +14.0% |
| 100 | 58.5% | 67.5% | +9.0% | **4/4** | +2.7% .. +16.0% |
| 150 | 73.8% | 85.0% | **+11.2%** | **4/4** | +5.3% .. +14.0% |

### The gain grows with budget — the opposite of what the regression work implied

Sections 18–19 showed the *regression* gain shrinking as the primary target
accumulated data (+0.20 at n=10 down to +0.06 at n=80), and the natural
inference was that transfer is an early-campaign trick. **It is not.** At budget
20 the campaign gain is *negative* on both blocks; it turns positive around 40
and reaches +9 to +11 recall points at 100–150.

The two are not in conflict. A better posterior early does not convert into
better picks when there are only four rounds to exploit it; the advantage
compounds over rounds. Which is a reminder of section 21's lesson in a second
form: **a regression curve does not tell you the shape of the campaign curve,
even in sign.**

Practical consequence: **there is a minimum budget, around 40 assays, below
which this hurts.**

### The per-target lottery is the real obstacle to deploying this

A practitioner screening one cold target gets **one** draw, not an average over
four. The per-target spread says what that draw looks like: at budget 60 on JAK,
one target gained 15 recall points and another lost 15. Pooled, that is
"+3.7%". Individually, it is a coin flip.

This kills the pre-flight check that would have made the method deployable.
The natural self-test — hold out each target you already have, see whether
multi-task beats single-task, and extrapolate to the new one — requires the
answer to be consistent across held-out targets. It is 2/4 or 3/4 on JAK at
every budget. **Hold out a different target and you get a different
recommendation**, so the check cannot license a decision about a target you have
not measured.

Only CYP at 100+ assays reaches 4/4. That is the one regime in this work where a
practitioner could be told "this will probably help you" rather than "this helps
on average across targets like yours".

### What can honestly be recommended today

1. **Budget ≥ 40 assays, ideally ≥ 100.** Below 40 it costs you.
2. **Use history you already own.** Auxiliary assays are worth ~1/5 of a primary
   one (§22), so never buy them for this.
3. **Co-measured family panels only.** Separate per-target libraries showed
   nothing (§17), and the reason is still unknown (§23).
4. **Do not use similarity search** — it is worse than plain single-task BO by
   14 points (§23).
5. **Expect variance, not a guarantee.** Across targets the effect at 60 assays
   ranges −15 to +15 points.

**What cannot be recommended: a way to know in advance which side of that range
you will land on.** Until that exists, this is a method with a positive expected
value and an unpredictable per-campaign outcome — honest to publish, premature
to deploy on a single target where one draw is all you get.

## 25. The operating window: the one regime where the per-target lottery stops

Section 24 measured the gain against budget in **assays** and ended on the
obstacle: at 60 assays on JAK one held-out target gained 15 recall points and
another lost 15, pooling to a meaningless "+3.7%". A practitioner screening one
cold target gets one draw, not an average over four, so a method that is
positive in expectation and unpredictable per campaign cannot be recommended.

This section changes the x-axis and the obstacle goes away over part of the
range. That window is the paper's central practical claim.

> **Read §31 with this.** The window below is real and unchanged. What §31
> removes is the attribution: the `pooled` control was never run on CYP, and
> when it is, naive pooling matches `mt` on JAK and *beats* it on CYP by up to
> 9.1 points (p = 0.0004). The gain belongs to the free family history, not to
> the fitted task correlation. The `pooled` column below is from §31's runs.
>
> **Read §34 with this too, and before quoting the unanimity.** §34 runs this
> protocol on four more families and re-runs these cells at 32 repeats. Three
> things change:
>
> 1. **The "targets up" column does not reproduce.** Over six families
>    unanimity holds in 7 of 12 window cells, and the verdict **flips in 5 of 12
>    when the repeats go 8 → 32 — including JAK at both window budgets below**.
>    It is a minimum over four noisy per-target estimates and does not converge,
>    while ΔAUC does (ρ = +0.916 between the two runs). **The leave-one-target-out
>    self-check argued for at the end of this section is retired.**
> 2. **The JAK-vs-CYP effect-size difference is a history-coverage artefact.**
>    `H=50` is 17% of JAK's library and 46% of CYP's. At matched 25% coverage
>    the two become +7.8 and +6.3 — the same number.
> 3. **§31's attribution above does not hold outside these two blocks.** `mt`
>    beats `pooled` on AUC in 5 of 12 cells across six families and loses in
>    none, and `pooled` drops *below* single-task on PI3K while `mt` never does.
>
> What survives unchanged: the mean gain, positive in 12/12 window cells on
> 6/6 families, and the three-regime shape of the budget axis.

### Why fraction of library, not number of assays

The two clean blocks are different sizes — JAK 904 compounds, CYP 335. A
100-assay budget is 11% of one library and 30% of the other, so the assay axis
compares two different experiments: §24 has JAK peaking at 100 assays and CYP at
150, which looks like a disagreement between the blocks and is mostly an
artefact of dividing by different denominators.

On the fraction axis they agree. That is not a presentational choice. A
recommendation stated in assays does not transfer to a library of another size,
and the quantity a practitioner actually controls is what share of their
catalogue they can afford to screen.

### Setup

`scripts/dense_block_campaign.py`, clean blocks (censoring filter applied: JAK
1,563 → 904, CYP 1,472 → 335). `H = 50` free historical observations per
auxiliary, fixed before the campaign and never grown. Batch 10, `n_init` 10, EI,
8 repeats × 4 held-out targets. **The primary's top-10 is withheld from the
history** (`--history-excludes-top`) — §22's conservative condition, not the
optimistic one, so every number below is a floor rather than a headline.

| block | % lib | budget | stl | mt | pooled | Δ (mt−stl) | targets up | per-target spread |
|---|---|---|---|---|---|---|---|---|
| **JAK** (N=904) | 11% | 100 | 40.0% | 46.6% | 45.0% | +6.6 | 2/4 | −5.0 .. +18.8 |
| | **22%** | 200 | 68.8% | **75.3%** | 72.8% | **+6.6** | **4/4** | +2.5 .. +11.2 |
| | **33%** | 300 | 76.2% | **83.4%** | 82.5% | **+7.2** | **4/4** | +1.2 .. +15.0 |
| | 50% | 450 | 92.8% | 94.1% | 94.4% | +1.2 | 2/4 | −0.0 .. +3.8 |
| | 66% | 600 | 97.8% | 98.1% | 97.8% | +0.3 | 1/4 | +0.0 .. +1.2 |
| **CYP** (N=335) | 11% | 37 | 17.5% | 18.1% | 18.4% | +0.6 | 2/4 | −5.0 .. +6.2 |
| | **22%** | 74 | 37.8% | 52.2% | **58.1%** | **+14.4** | **4/4** | +8.8 .. +17.5 |
| | **33%** | 111 | 58.1% | 70.0% | **79.1%** | **+11.9** | **4/4** | +5.0 .. +18.7 |
| | 50% | 168 | 78.1% | 87.5% | **94.7%** | +9.4 | 4/4 | +2.5 .. +12.5 |
| | 66% | 221 | 89.1% | 94.1% | **97.8%** | +5.0 | 3/4 | −0.0 .. +13.7 |

Top-10 recall, points. "Targets up" and "spread" are for `mt`; §31 has them for
`pooled`. "Targets up" counts how many of the four held-out targets individually
improved — the quantity §24 showed matters more than the mean, and the spread is
the per-target range behind it, which is what a practitioner with one draw
actually faces. **Bold in the `pooled` column marks the cells where the simpler
model beats `mt`**; on CYP that is every cell from 22% up (§31).

### Three regimes

1. **Below ~11% the method costs you or does nothing.** At 11% the mean is
   positive on both blocks but only 2/4 targets improve on either, and §24's
   assay-axis sweep is outright negative lower down (−0.7 JAK at 20 assays,
   −4.3 CYP). There is not enough budget for a better posterior to be converted
   into better picks.
2. **At 22–33% the gain is both large and unanimous**: +6.6 to +14.4 points,
   **4/4 targets on both blocks, all four cells**. This is the window.
3. **Beyond 50% it decays on recall** as the single-task baseline saturates —
   JAK `stl` is already at 92.8% — and unanimity breaks with it (2/4, then 1/4).

### The AUC gain does not decay, and saying so matters

Reading only recall would overstate regime 3. ΔAUC of the top-10 trace stays
positive and above 3 SEM at *every* JAK budget:

| % lib | 11% | 22% | 33% | 50% | 66% |
|---|---|---|---|---|---|
| ΔAUC | +0.114 ± 0.040 | +0.085 ± 0.029 | +0.066 ± 0.021 | +0.049 ± 0.015 | +0.037 ± 0.011 |

The two metrics are answering different questions. AUC credits finding the
top-10 *earlier*; final recall credits finding them *at all*. Past 50% both arms
end up finding nearly everything, so recall converges while `mt` is still
getting there sooner. **The decay claim is about recall and should be stated as
such** — "there is nothing left to win at the end", not "the surrogate stops
working."

### Unanimity is the result, not the effect size

+7.2 points is a modest gain. 4/4 is what changes the method's status — and it
is also the only axis on which the fitted model currently has any claim over
naive pooling (§31), though at 8 repeats that claim is not established.

The self-check available to a practitioner is leave-one-target-out: hold out each
target you have already measured, check whether multi-task beats single-task, and
extrapolate to the cold one. That check licenses a decision **only if its answer
does not depend on which target you held out**. At 2/4 or 3/4 it does — hold out
a different target, get a different recommendation — so outside the window the
check is uninformative and the method cannot be recommended for a single target.
Inside the window, on two unrelated protein families, it is unanimous eight times
out of eight.

That is the difference between "this helps on average across targets like yours"
and "this will probably help you".

### What this claim does not cover

* **Batch size is a confound against §24.** That sweep is batch 5, this one is
  batch 10, and the two are not comparable cell by cell. The window is stated in
  this section's units; do not read a §24 cell against a §25 cell.
* **8 repeats × 4 targets per cell.** Enough for a 4/4 count to mean something,
  not enough to separate +6.6 from +7.2.
* **Unanimity is demonstrated at N ≈ 900 and N ≈ 335 only.** §26 cuts JAK to 566
  at random and unanimity drops to 1/4. The window is a claim about these blocks
  at these sizes, not about arbitrary library size.
* **Co-measured panels only.** §17 found nothing on separate per-target
  libraries and §23 leaves the reason open. Nothing here extends to a target
  whose relatives were assayed on different compounds.
* **The history must be free.** §22's exchange rate is 5:1 against auxiliary
  assays. Buying the history to obtain this window loses.

### Audit

Run under `--audit`, which was added in §29 and — until §30 went looking — had
never been run on the configurations that produced this table. All ten cells
were re-run audited and all ten pass: equal initial design across arms, shared
history within a repeat, exact budget, no repeat picks, no top-10 leaking into
the history, no auxiliary column equal to the primary's values. The audited runs
reproduce the stored numbers exactly.

Two defects surfaced in the re-run and are fixed:

* **The `pooled` arm was missing from every stored CYP run.** Only `stl` and
  `mt` were recorded, against standing rule 2. See §31 for what it changed.
* **The pool-size guard rejected a published cell.** It required
  `budget + H(T−1) + k` distinct compounds, and CYP at 66% needs
  `221 + 150 + 10 = 381` against a 335-compound block. But the history and the
  campaign draw *independently* from the same pool and are allowed to overlap —
  that is the defining property of a dense block (§18), not a leak — so the
  requirement is the largest single draw, not the sum. Fixed; the cell then ran
  and reproduced the stored numbers exactly.

Artefacts `results/win_{jak,cyp}_*.json`, each carrying its own `audit` block
and the per-campaign `auc`/`recall` traces, so any two arms can be compared
pairwise from the artefact without re-running.

## 26. Drug-likeness as an admission filter: the effect survives

Sections 18--25 optimise raw potency, so nothing stopped the method selecting a
large greasy compound that happens to bind. Adding a drug-likeness *term* to the
objective is not the repair -- that is exactly what ruins the published
DOCKSTRING objectives (§14). Applying it as a library **admission filter** keeps
the score entirely assayed, so a free descriptor can earn no recall and the
floor stays at chance.

`--qed-min 0.5` on the clean blocks: JAK 904 → 566 (62.6%), CYP 335 → 192
(57.3%).

### The filter is unbiased with respect to potency

| target | top-10 kept | QED of top-10 | QED of the rest |
|---|---|---|---|
| JAK1 | 6/10 | 0.59 | 0.54 |
| JAK2 | 4/10 | 0.51 | 0.54 |
| JAK3 | 4/10 | 0.53 | 0.54 |
| TYK2 | 7/10 | 0.64 | 0.54 |

Retention of each target's top-10 (4–8 of 10) tracks the overall 63% rate, and
the potent compounds are no less drug-like than the library mean. The filter is
not quietly deleting the actives and turning the benchmark into a search among
mediocrities.

### Effect size survives; the unanimity is a library-size effect, not the filter

JAK at 33% of the library screened, three variants of the same block:

| variant | N | Δrecall | targets improving |
|---|---|---|---|
| full clean block | 904 | +7.2% | **4/4** |
| **QED ≥ 0.5** | 566 | **+6.9%** | 3/4 |
| random cut, size control | 566 | +2.5% | 1/4 |

The drug-like subset holds the effect ($+6.9$ vs $+7.2$). A *random* cut to the
same size does not ($+2.5$, 1/4) — so cutting the library costs consistency and
effect, and the drug-likeness criterion costs **less** than an arbitrary cut of
equal size. The 4/4 unanimity of §25 needs the larger pool, not the unfiltered
one.

This is a benchmark-size artifact rather than a limitation of the method: in
deployment the filter is applied to a large purchasable catalogue and leaves
thousands of compounds, not 566. But it does mean **the unanimity result is
demonstrated only at $N \approx 900$**, and a paper should say so rather than
imply it holds at any library size.

CYP was also run filtered, but its history had to be cut from $H=50$ to $H=20$
(150 history compounds out of a 192-compound pool is not a campaign), so that
comparison is confounded and is not reported.

**Recommendation.** Keep the filter. It costs ~0.3 recall points on JAK, removes
the objection that the method optimises binding with no regard for
developability, and preserves the zero-assay floor at chance because it never
enters the score.

## 27. Drug-likeness as a second *objective* fails; as a filter it works

Section 26 added QED as a library admission filter and the effect survived. The
natural next question is whether to go further and optimise potency and
drug-likeness jointly. We implemented it --- ParEGO selection over
(potency, QED) with a redrawn Tchebycheff weight each round, Pareto-front recall
as the metric, and a `qed_only` arm --- specifically to avoid section 14's
failure, where summing QED into the score left it carrying 93% of the ranking.
**A Pareto front does not avoid it.**

### The front is tiny and sits at the QED extreme

| target | $|F|$ | QED-only recall, 0 assays | potency-only recall | front median QED | library median |
|---|---|---|---|---|---|
| JAK1 | 6 | 50\% | 33\% | 0.91 | 0.58 |
| JAK2 | 6 | 33\% | 33\% | 0.88 | 0.58 |
| JAK3 | 5 | 40\% | 40\% | 0.90 | 0.58 |
| TYK2 | 10 | 50\% | 40\% | 0.90 | 0.58 |
| CYP3A4 | 8 | 25\% | 50\% | 0.84 | 0.54 |
| CYP2C19 | 2 | 50\% | 50\% | 0.82 | 0.54 |

Three things go wrong at once:

1. **A free continuous objective puts part of the front on sale.** The
   maximum-QED compound is non-dominated by construction, and so are its
   neighbours. Ranking by QED alone recovers $20$--$50\%$ of the front having
   measured nothing.
2. **The front is 2--10 compounds** out of 335--904. A single compound is
   $10$--$50\%$ of the metric, so the measurement has almost no resolution and
   the per-target variance is enormous.
3. **The front is unrepresentative.** Median QED on the front is $0.72$--$0.91$
   against a library median of $0.54$--$0.58$: the success set is a
   QED-extremity artifact, not a set of compounds a chemist would want.

In the campaign this shows up directly: at budget 100 on JAK, `qed_only` reaches
$50.0\%$ front recall, `mt` reaches $50.0\%$, and `stl` $51.7\%$. The
zero-assay arm ties the modelled ones.

### Why the filter is the right construction and the objective is not

| | what it changes | success set | zero-assay floor |
|---|---|---|---|
| QED as score term (§14) | the score | contaminated | 48\% of top-50 |
| QED as Pareto objective (§27) | the success set | 2--10 compounds, QED-extreme | 20--50\% of front |
| **QED as admission filter (§26)** | **the library** | **top-10 potency, all assayed** | **chance** |

The filter changes *which compounds are eligible* and leaves the objective and
the metric alone. Potency still has to be measured, the success set is still ten
compounds that require assays, and a free descriptor earns nothing. That is why
it costs only $0.3$ recall points while removing the developability objection.

**Retired.** Do not add drug-likeness as an objective on blocks of this size.
The implementation is kept behind `--multiobjective` so the result is
reproducible and so a future block large enough for a non-degenerate front
(hundreds of non-dominated compounds) can revisit it.

### The general lesson

**A cheap objective in a multi-objective benchmark is as dangerous as a cheap
term in a scalarised one.** Scalarisation hands the free quantity a weight;
Pareto hands it a whole region of the front. Either way an optimiser is credited
for rediscovering something it was given. Any multi-objective molecular BO
benchmark should report what a zero-assay ranking on each free objective
recovers, and treat that as the floor.

## 28. Where to apply a free descriptor: filter the library, or only the spending?

Section 26 applied QED as a library filter and section 27 rejected it as an
objective. A third option remained, and it is the intuitive one: let BO optimise
potency over everything and apply drug-likeness *afterwards*, since QED is free
for any SMILES. It is not obviously worse --- a potent but undevelopable
compound still teaches the surrogate about the SAR, which a pre-filter discards.

Three designs, **identical success set** (top-10 potency among the 566 drug-like
compounds) and **identical absolute assay budget**; only where the budget may be
spent and what the free history covers differ. JAK, batch 10, 8 repeats.

| budget | design | `stl` | `mt` | Δrecall | targets up |
|---|---|---|---|---|---|
| 62 | post-hoc: search all 904 | 19.4% | 14.7% | −4.7% | 2/4 |
| | pre-filter: library cut to 566 | 27.5% | 31.9% | +4.4% | 3/4 |
| | **restricted: spend 566, history 904** | 27.5% | **37.5%** | **+10.0%** | 3/4 |
| 125 | post-hoc | 42.2% | 45.3% | +3.1% | 3/4 |
| | pre-filter | 52.5% | 58.4% | +5.9% | 2/4 |
| | **restricted** | 52.5% | **59.7%** | +7.2% | 2/4 |
| 187 | post-hoc | 51.9% | 57.5% | +5.6% | 3/4 |
| | **pre-filter** | 66.6% | **73.4%** | +6.9% | 3/4 |
| | restricted | 66.6% | 70.0% | +3.4% | 2/4 |

### Post-hoc scoring loses badly

Searching the whole library and filtering at the end costs $13$--$17$ recall
points against either alternative. Roughly $37\%$ of the library cannot count as
a success, so roughly $37\%$ of the budget is spent on compounds that will be
discarded, and the SAR information they provide does not repay it. **Applying a
free criterion late is not free** --- it is free to *evaluate* and expensive to
*ignore while spending*.

### But the pre-filter was discarding free information

Cutting the library to 566 before anything else also cut the historical
auxiliary data to 566. History is not charged, so there was no reason to. The
`restricted` design confines the *budget* to compounds that can count while
letting the surrogate's free history span all 904.

It is the best design at the budgets where the method matters most: $+10.0$
against the pre-filter's $+4.4$ at 62 assays, $+7.2$ against $+5.9$ at 125. At
187 it reverses ($+3.4$ vs $+6.9$), which at 8 repeats and 4 targets is inside
the noise this measurement carries; the honest reading is that the two converge
once the budget is large relative to the eligible set.

An implementation check falls out of the design: `stl` uses no history, so
`restricted` must reduce exactly to `pre-filter` for it. It does --- $27.5\%$,
$52.5\%$, $66.6\%$ to the decimal in both --- which confirms the eligibility
machinery changes only what it should.

### The principle

A free descriptor has three possible roles, and they are not interchangeable:

| role | what it changes | verdict |
|---|---|---|
| term in the score | the objective | **fails** (§14): carries 93% of the ranking |
| Pareto objective | the success set | **fails** (§27): owns 20--50% of a tiny front |
| filter on *scoring* only | nothing until the end | **fails** (§28): wastes 37% of the budget |
| **constraint on *spending*** | **which compounds the budget may buy** | **works**, and is best at low budget |

The distinction that holds across all four rows: a free quantity should
constrain **what you pay for**, and nothing else. Let it into the score and it
dominates; let it into the success set and it gives away part of the answer;
apply it only at the end and it wastes the budget it should have been steering.

## 29. Runtime invariant audit

Every fairness claim in §§21--28 was asserted in prose and checked nowhere. A
violation of any of them would not crash --- it would quietly change the
numbers. `--audit` checks them where the code runs.

`dense_block_campaign.py --audit`:

* arms start from the **same initial design** (the claim that a difference
  between arms is the surrogate and not a luckier draw);
* arms see the **same history** within a repeat;
* each arm spends **exactly** the budget, with **no compound picked twice**;
* with `--restrict-selection`, **no pick lands outside the eligible set**;
* with `--history-excludes-top`, **no top-k compound is in the history**;
* **no auxiliary column equals the primary target's values** --- the answer
  relabelled.

`dense_block_h1.py --audit`: evaluation rows disjoint from training *and*
history rows, no duplicate draws, no target used as its own auxiliary.

`transfer_headroom.py --audit` and `cold_target_chembl.py --audit` were added in
section 30 and close the gap this section left open; the checks now live in
`src/molbo/audit.py` and are unit-tested in `tests/test_audit.py`.

### Status

All configurations that produced published numbers pass: JAK and CYP, batch 5
and 10, `--qed-min`, `--report-druglike --restrict-selection`,
`--multiobjective`, and `h1_cold` with and without `--holdout`.

**Both audits were validated by deliberate injection**, because a check that has
never fired is not evidence. Giving the `mt` arm a different RNG produced
*"arms started from different initial designs"* on all 8 cells; adding three
training rows to the evaluation set produced *"3 evaluation rows are also
training/history rows"* on all 16. Both injections were reverted and the files
verified clean.

Two further checks, from outside the audit:

* **Determinism.** The same command twice gives byte-identical JSON.
* **No regression from this session's refactors.** Re-running the stored
  headline configuration (JAK, budget 200) after adding the eligibility
  parameter and audit hooks reproduces `stl` $0.6875$ and `mt` $0.7531$ exactly
  --- so `rng.choice(np.arange(n))` and `rng.choice(n)` do draw identically, a
  thing worth confirming rather than assuming.

**Run `--audit` on any configuration before quoting it.** It costs nothing and
it is the only thing standing between a fairness bug and a published number.

## 30. Closing the audit gap on §§14–17 — the unpaired split, and what it was worth

Section 29 audited the campaign instruments and left the two behind §§14–17
unaudited, on the reasoning that those sections are mostly *negative* results
and a leak would have flattered transfer rather than hidden it. That reasoning
was right about leaks, and silent about the defect that was actually there.

Instruments: `scripts/transfer_headroom.py --audit` (checks A, B, C) and
`scripts/cold_target_chembl.py --audit`. The checks moved into
`src/molbo/audit.py` so both scripts run the same ones and so they could be
unit-tested — `tests/test_audit.py`, 24 tests.

### What the audit found

Checks A and B are **clean**: 17,538 checks over 2,520 cells and 601 over 120,
no violation. Evaluation rows are never trained on, no arm draws a row twice, no
target is its own auxiliary or appears in its own ceiling features, every
multi-task arm spends no more cells than `stl`, and in check B all four arms are
scored on the same rows with the free and assayed parts of the objective summing
back to the objective.

Check C and `cold_target_chembl.py` both failed one invariant, in **every cell**:

```
- evaluation rows: differs across arms in 120 of 120 cells -- not paired
- primary training rows: differs across arms in 120 of 120 cells -- not paired
```

`H` was in the RNG seed. The `H = 0` control and every `H > 0` arm it is
differenced against were therefore fitted on **different primary observations**
and scored on **different held-out rows** — and without a reserved holdout the
evaluation set also *shrinks* as `H` grows, by `H × |aux|` rows, because the
history is carved out of it. `dense_block_h1.py` had already noticed the second
half of this and fixed it with `--holdout`; the fix was never carried across.

This is not a leak and not a bias: the gain stays unbiased in expectation. What
it costs is **pairing**. Two new flags on each script fix it, both default off
so the stored tables stay reproducible:

| flag | what it binds |
|---|---|
| `--c-holdout` / `--holdout` | one evaluation set per repeat, scored by every `H` |
| `--c-paired` / `--paired` | one set of primary training rows, fitted by every `H` |

> **The three implementations of this split had diverged. The evaluation
> stream is now fixed; the feasibility rule still differs.**
>
> The reserved evaluation set was drawn from a different stream in
> `dense_block_h1.py` (`seed + 977*r`) than in the other two
> (`seed + 977*r + 7919*ti`), which made `--holdout` runs incomparable across
> instruments. All three now use the target-dependent stream, so each held-out
> target gets its own evaluation rows rather than every target in a repeat
> sharing one set --- which is also what rule 4 wants, since a shared holdout
> correlates the per-target results it exists to keep separate. **No stored
> number changed**: `dense_block_h1.py` had no `holdout > 0` artefact, and the
> `holdout = 0` path does not touch this stream.
>
> Still divergent: the feasibility guard is `n_pool//2` (h1, `holdout = 0`),
> `len(avail)` (h1, `holdout > 0`), `len(home_pool)//2` (transfer_headroom) and
> `n > len(avail)` (cold_target_chembl). These decide which grid cells come back
> NaN, so unifying them would move published numbers and is deferred. The deeper
> fix remains one shared split helper owning the stream derivation and the
> feasibility rule, rather than three parallel copies; that change should be made
> deliberately, with every affected table regenerated.

**What pairing can and cannot fix here.** The ± in both tables is a SEM across
the *four held-out targets*, not across repeats — rule 4, report per target.
Pairing removes split-to-split variance *within* a target; it does nothing to
genuine between-target heterogeneity, which is the larger term. So the
prediction is that pairing moves the point estimates and leaves the error bars
roughly where they are. That is what happens in §16. It is not what happens in
§17, and the difference is informative.

### §16 re-run paired — conclusion holds, effect sizes shrink

DOCKSTRING, 4 targets, 10 repeats, pool 2,000, 18 cells. Gain over the
single-task GP:

| n | H | published (unpaired) | audited (paired) | change |
|---|---|---|---|---|
| 10 | 5 | +0.093 ± 0.014 | +0.055 ± 0.018 | −0.038 |
| 10 | 25 | +0.256 ± 0.010 | +0.208 ± 0.019 | −0.049 |
| 10 | 200 | **+0.453 ± 0.015** | **+0.427 ± 0.028** | −0.025 |
| 20 | 25 | +0.157 ± 0.023 | +0.175 ± 0.016 | +0.018 |
| 20 | 200 | +0.346 ± 0.022 | +0.353 ± 0.024 | +0.007 |
| 80 | 25 | +0.070 ± 0.005 | +0.043 ± 0.009 | −0.026 |
| 80 | 200 | +0.172 ± 0.011 | +0.154 ± 0.011 | −0.018 |

**Break-even stays at H = 5–10 per auxiliary and 4/4 targets still improve at
every `H` past it.** The gain shrinks in **15 of 18 cells**, mean −0.014, worst
−0.049; the three that grow are all at n = 20 and all within their own SEM.
Mean SEM is unchanged (ratio 1.05), exactly as predicted above. The n = 10 row
is hit hardest, which is what a pairing defect predicts, since that is where
split variance is largest relative to the signal.

This is the seventh headline this project has re-measured under a control and
the seventh that shrank without reversing sign. Standing rule 6 holds.

### §17 re-run paired — the null gets stronger, which is the point

ChEMBL, four kinase libraries, 1,000 compounds each, 18 cells.

| n | H | published (unpaired) | audited (paired) |
|---|---|---|---|
| 10 | 50 | +0.016 ± 0.026 | +0.001 ± 0.022 |
| 10 | 200 | +0.011 ± 0.026 | +0.006 ± 0.008 |
| 20 | 25 | −0.041 ± 0.020 | −0.036 ± 0.013 |
| 20 | 200 | −0.008 ± 0.023 | +0.008 ± 0.007 |
| 80 | 25 | −0.016 ± 0.006 | −0.011 ± 0.005 |
| 80 | 200 | −0.006 ± 0.006 | −0.001 ± 0.004 |

The gain itself does not systematically move — 9 cells up, 9 down, mean −0.001,
which is what an unbiased-but-unpaired estimator should do. **The error bars
do**: mean SEM 0.0177 → 0.0099, median ratio 0.61, a ~40% reduction. The largest
gain anywhere in the grid halves, +0.0165 → +0.0085.

Here pairing *did* shrink the SEM, against the prediction above, and the reason
is the finding: on ChEMBL there is no between-target signal for the SEM to be
made of, so the error bar was almost entirely split noise — the thing pairing
removes. On DOCKSTRING the SEM survives pairing because it is measuring real
differences between targets. **The error bar responding to pairing is itself
evidence about which dataset has a transfer effect.**

§17 asserted "this is not an underpowered null — an effect a quarter the size of
DOCKSTRING's would have been visible." That was a claim about power written
without the paired comparison that establishes it. It is now measured and it
survives: **no cell clears 3 SEM at any budget or any `H`, on an instrument ~40%
tighter than the one that first said so.**

### A number §17 argued for in prose and never reported

`cold_target_chembl.py` keeps compounds appearing in two libraries, on the
argument that a VEGFR2 number on a molecule the JAK2 campaign is scored on is
prior information a deployed model would genuinely have. The argument is sound
and the exposure was never quantified. It now is, per run:

> **0.09% of the primary's evaluation compounds carry an auxiliary measurement
> in the drawn history.**

That is the ceiling on how much of any ChEMBL gain could come from cross-library
overlap rather than from task structure. At 0.09% the design choice rests on
measurement rather than on argument alone — and it independently supports §23's
conclusion that chemical overlap does not explain the ChEMBL null, since there
is almost none of it to explain anything with.

### Validation, by deliberate injection

A check that has never fired is not evidence, so each was fired.

* **Unit** — `tests/test_audit.py`, 24 tests. Every check gets a violation it
  must catch *and* a clean case it must not flag: wrong draw size, duplicate
  rows, leaked evaluation rows, an eligibility escape, an auxiliary column equal
  to the primary's values, a target used as its own auxiliary, an unpaired
  split, and an order-permuted split that must **not** count as unpaired. The
  clean twins are the half that would otherwise rot: a false positive is worse
  than no check, because it teaches the reader to skip the line.
* **End-to-end** — the unit tests prove the checks work; they do not prove the
  call sites pass the right arrays. So three training rows were appended to `te`
  in check C, and the history draw in `cold_target_chembl.py` was changed to
  `h + 1`. Both fired in every cell (*"3 rows are in both [C] evaluation and
  training/history"*, *"26 rows, expected 25"*). Both were reverted and the
  files verified clean.

### Two further checks, from outside the audit

* **`--audit` changes no number.** The same command with and without it produces
  identical JSON apart from the audit block itself.
* **Determinism.** The same command twice gives identical JSON. The stored
  `transfer_headroom.json` differs from a fresh run by ~6 × 10⁻⁹, in the `mt_gp`
  arm at n = 10 only — `CoregionalisedGP` was edited after that file was
  written, and `mt_gp` is the only arm that uses it. No conclusion depends on
  the ninth decimal, but the stored artefact is stale and
  `results/transfer_headroom_audited.json` supersedes it.

### Status

| instrument | sections | audit |
|---|---|---|
| `dense_block_campaign.py` | §§21–28 | clean (§29) |
| `dense_block_h1.py` | §§18–20 | clean (§29) |
| `transfer_headroom.py` check A | §14 | **clean** |
| `transfer_headroom.py` check B | §14 | **clean** |
| `transfer_headroom.py` check C | §16 | clean **when run paired**; published table unpaired |
| `cold_target_chembl.py` | §17 | clean **when run paired**; published table unpaired |

Every instrument behind a quoted number now has a runtime audit, and the audit
result travels inside the run's JSON under `audit`, so a table can be checked
for provenance without the terminal it was produced in. **A result whose JSON
says `"clean": false` is not quotable.**

Artefacts: `results/transfer_headroom_audited.json`,
`results/cold_target_audit_paired.json`,
`results/cold_target_audit_unpaired.json`,
`results/cold_target_chembl_paired.json`.

## 31. The coregionalisation does not earn its place — pooling ties on JAK and beats it on CYP

> ## RETRACTED by §32.
>
> **The conclusion below is wrong and the error is instructive.** Every number
> in this section is an EI comparison between surrogates whose *predictive
> uncertainties are on different scales* (mean sd: stl 0.671, fitted 0.644,
> pooled 0.561). EI trades mu against sd, so part of every arm difference here
> is acquisition calibration rather than the model.
>
> §32 ran the control this section needed and did not have: a **pure-noise
> auxiliary**, which carries no information and must therefore produce no gain.
> It produced **+14.7 points for `pooled` on CYP** — 70% of the +20.9 this
> section credits to pooling's better use of the auxiliary data.
>
> Under `--acquisition greedy`, which uses the posterior mean alone and removes
> the sd channel, the finding **inverts**: on CYP `pooled` is *worse* than
> single-task (−7.5 at rho = 1, −20.6 with noise), and `mt` is the arm that
> gains. The fitted coregionalisation is not beaten by pooling; it was never
> being compared to it fairly.
>
> The framing "run the pooling control" was right. The execution was not: a
> control arm has to be comparable on the axis the acquisition actually reads,
> and sd was never checked. Read §32 and disregard the verdict below.

Standing rule 2 says *always run the pooling control*. Half the operating-window
table was produced without it: the stored `hb_cyp_*` runs carry only `stl` and
`mt`, no `pooled` arm at all. Running it changes what the paper can claim about
the model.

`pooled` is the same coregionalised GP with `r` **fixed at 1**: one function
wearing four labels, everything that distinguishes the targets pushed into the
per-task noise term. It still z-scores per task — so it is "standardise each
target, then pool", not a naive concatenation of raw pIC50 — but it fits *no*
task correlation. If the coregionalisation is the contribution, `mt` has to beat
it.

### Paired, per campaign

The arms share the initial design and the history within a repeat (verified by
`--audit`), so `pooled − mt` is a paired difference over 32 campaigns per cell,
not two independent comparisons against `stl`.

| block | % lib | stl | mt | pooled | mt − stl | **pooled − mt** | p |
|---|---|---|---|---|---|---|---|
| JAK | 11% | 40.0% | 46.6% | 45.0% | +6.6 | −1.6 ± 3.4 | 0.65 |
| JAK | 22% | 68.8% | 75.3% | 72.8% | +6.6 | −2.5 ± 2.5 | 0.33 |
| JAK | 33% | 76.2% | 83.4% | 82.5% | +7.2 | −0.9 ± 1.6 | 0.57 |
| JAK | 50% | 92.8% | 94.1% | 94.4% | +1.2 | +0.3 ± 1.0 | 0.74 |
| JAK | 66% | 97.8% | 98.1% | 97.8% | +0.3 | −0.3 ± 0.8 | 0.71 |
| CYP | 11% | 17.5% | 18.1% | 18.4% | +0.6 | +0.3 ± 1.6 | 0.85 |
| CYP | 22% | 37.8% | 52.2% | **58.1%** | +14.4 | **+5.9 ± 3.0** | 0.057 |
| CYP | **33%** | 58.1% | 70.0% | **79.1%** | +11.9 | **+9.1 ± 2.3** | **0.0004** |
| CYP | 50% | 78.1% | 87.5% | **94.7%** | +9.4 | **+7.2 ± 1.8** | **0.0004** |
| CYP | 66% | 89.1% | 94.1% | **97.8%** | +5.0 | **+3.7 ± 1.2** | 0.005 |

**On JAK the two are indistinguishable** — `mt` is nominally ahead at four of
five budgets and not once at p < 0.3. **On CYP `pooled` wins, significantly,
everywhere from 33% up**, by up to 9.1 recall points at the window's own
headline cell. *(Both readings are confounded by sd calibration; see the
retraction above.)*

**The fitted task correlation is never significantly better than not fitting
one, anywhere in this work.** §22 already reported it as "unsupported"; at window
budgets with the top-10 withheld it is not merely unsupported, on CYP it is
reversed.

### What this does and does not overturn

| claim | status |
|---|---|
| `mt` beats `stl` in the 22–33% window | **holds** — unchanged, both blocks |
| free family history is worth having | **holds** |
| *a coregionalised GP* is what extracts it | ~~not supported~~ → **supported after all (§32)** |

The headline is unaffected in direction and improves in magnitude: on CYP the
best available use of the auxiliary data gains **+20.9** points at 33%, not
+11.9. What fails is the modelling claim. The right recommendation from this
data is the *simpler* one — per-task standardise, then pool the family into one
function — and that is a better deployable result than a fitted ICM, not a worse
one.

### The one thing that might still favour the fitted model

Per-target unanimity, which §25 argues matters more than effect size:

| cell | mt | pooled |
|---|---|---|
| JAK 22% | **4/4** | 2/4 |
| JAK 33% | 4/4 | 4/4 |
| CYP 22% | 4/4 | 4/4 |
| CYP 33% | 4/4 | 4/4 |

`mt` is 4/4 in all four window cells; `pooled` is 4/4 in three. If real, that
would say the coregionalisation buys *consistency across targets* rather than
mean effect — which is exactly the property the window claim rests on, so it
would matter more than the effect size it loses.

**It is not established.** One cell, 8 repeats per target, and the JAK 22%
per-target range for `pooled` is [−5.0, +13.8] — a single target slipping five
points negative is well inside noise at this sample size. Do not report the
consistency advantage without more repeats. It is, however, the one measurement
that could rescue the fitted model, so it is the one to run.

### What to do

1. **Re-run every arm comparison with `pooled` present.** It was missing from
   half the window table and it changes the model recommendation.
2. **Restate the contribution as being about the data, not the surrogate**:
   free co-measured family history is worth +7 to +21 recall points in the
   22–33% window. Which multi-task model extracts it is open, and the simplest
   one is currently ahead.
3. **Settle the consistency question with more repeats** before either
   retiring the fitted ICM or claiming it.

Artefacts: `results/win_{jak,cyp}_*.json`, all ten cells audited clean, each
carrying the per-campaign `auc` and `recall` traces so a paired comparison
between any two arms can be made from the artefact without re-running.

## 32. The null-information control: a pure-noise auxiliary, and what it invalidates

Every arm comparison in §§18–31 asks whether a surrogate given auxiliary data
beats one that is not. None of them ran the control that makes the question
answerable: **give the multi-task arm an auxiliary that carries no information
at all.** It must then tie the single-task arm. Anything it gains is not
transfer, and is being credited to transfer by every table in this package.

`--aux-quality RHO` corrupts each auxiliary column to approximately Spearman
`rho` against its own true values, using the construction from
`scripts/prior_quality_threshold.py` (§8). The primary target, the history
indices and the success set are untouched; only what the multi-task arm is
*told* about its relatives degrades. Noise is redrawn per (target, repeat).
At `rho = 1.0` it reproduces the stored numbers exactly, so the path is a
verified no-op.

### The control fails

Budget at 33% of library, top-10 withheld, 32 campaigns per cell, EI:

| block | arm | **rho = 0 (no information)** | rho = 1 (measured panel) | share of the gain that is not transfer |
|---|---|---|---|---|
| JAK | `mt` − `stl` | **+2.8 (p = 0.037)** | +7.2 | ~39% |
| CYP | `mt` − `stl` | +1.6 (n.s.) | +11.9 | ~13% |
| CYP | `pooled` − `stl` | **+14.7 (p < 0.001)** | +20.9 | **~70%** |

Pure noise buying 14.7 recall points is not a small anomaly. It is most of what
§31 credited to pooling.

### It is not the surrogate

Checked directly on the JAK block, 12 draws, 100 primary observations and 150
auxiliary:

| | fitted `r` | Spearman vs truth | mean predictive sd |
|---|---|---|---|
| measured auxiliaries | 0.621 | `stl` 0.733 → **fitted 0.747** | 0.671 → 0.685 |
| **noise auxiliaries** | **0.083** | `stl` 0.733 → **fitted 0.715** | 0.671 → **0.644** |

The marginal likelihood correctly rejects noise (`r` = 0.083) and the noise
posterior is genuinely *worse* than single-task. `CoregionalisedGP` does the
right thing. The posterior gets worse and the campaign gets better, so the
campaign gain is not coming through the posterior mean.

### It is the acquisition

The three arms report predictive uncertainty on different scales — `stl` 0.671,
fitted 0.644, `pooled` 0.561 — and EI trades mu against sd. Adding rows to the
kernel compresses sd whether or not those rows say anything, which makes the arm
more exploitative. On these blocks that helps, and it is indistinguishable from
transfer in the output.

`--acquisition greedy` ranks on the posterior mean alone and removes the channel:

| block | acq | rho | `mt` − `stl` | `pooled` − `stl` |
|---|---|---|---|---|
| JAK | EI | 0 | +2.8 (p = 0.037) | +1.9 (n.s.) |
| JAK | EI | 1 | +7.2 (p < 0.001) | +6.2 (p < 0.001) |
| JAK | **greedy** | 0 | +3.1 (**n.s.**, p = 0.51) | +7.5 (n.s.) |
| JAK | **greedy** | 1 | **+16.2 (p = 0.008)** | +4.7 (n.s.) |
| CYP | EI | 0 | +1.6 (n.s.) | **+14.7 (p < 0.001)** |
| CYP | EI | 1 | +11.9 (p < 0.001) | +20.9 (p < 0.001) |
| CYP | **greedy** | 0 | −2.5 (n.s.) | **−20.6 (p = 0.001)** |
| CYP | **greedy** | 1 | +4.4 (n.s.) | **−7.5 (p = 0.081)** |

Three things follow, and they point in different directions.

**1. The noise gain is an EI artifact.** Under greedy it is not significant on
either block (p = 0.51, 0.57). The control passes once sd is removed.

**2. §31 inverts.** Under greedy, `pooled` on CYP is *worse* than single-task —
−20.6 with noise, −7.5 with the real panel. Pooling never used the auxiliary
data better; it had the most compressed sd and therefore the most exploitative
EI. **The fitted coregionalisation is not beaten by pooling.** §31 is retracted.

**3. Transfer is real, and larger than reported.** On JAK under greedy, `mt`
gains **+16.2 points (p = 0.008)** with the measured panel against +3.1 with
noise. That contrast — same surrogate, same budget, same history *indices*, only
the auxiliary *values* replaced by noise — is the cleanest evidence in this
package that the effect is information transfer and not bookkeeping. §§18–31
never produced it.

### The auxiliary-quality threshold — provisional

The original question: could a *computed* auxiliary (a docking score, a Boltz-2
affinity) replace a measured panel, which is the gate keeping this off a
purchasable catalogue? Sweeping `rho` under EI:

| achieved rho | JAK `mt` − `stl` | CYP `mt` − `stl` |
|---|---|---|
| 0.00 | +2.8 | +1.6 |
| 0.19 | +2.2 | +1.6 |
| 0.28 | +2.8 | +2.8 |
| 0.37 | +2.8 | +4.7 |
| 0.46 | +3.8 | +7.8 (p = 0.046) |
| 0.56 | +4.4 | +8.4 (p = 0.033) |
| 0.75 | +5.9 | +12.5 |
| 1.00 | +7.2 | +11.9 |

Monotone on both blocks, and separating from the `rho = 0` baseline at **rho ≈
0.45–0.5** (CYP) and **0.5–0.6** (JAK). Docking against measured pIC50 typically
reaches 0.2–0.4, **below** both — so on this evidence a docking-grade computed
auxiliary does not pay, consistent with §8's prior-mean threshold of 0.37 rather
than beating it.

**Provisional, because the baseline it is measured against is the contaminated
one.** The sweep ran under EI, where `rho = 0` already buys +2.8. The threshold
has to be re-measured under greedy, or with sd calibrated across arms, before it
is quotable. The *ordering* — threshold above docking quality — is the part that
survives, because the artifact inflates the low-`rho` end and therefore makes the
threshold look *lower* than it is.

### What to do

1. **Run `--aux-quality 0` beside every arm comparison**, as routinely as the
   pooling control. It is the only check that separates transfer from
   acquisition calibration, and it costs one extra run.
2. **Do not compare arms under EI without checking their sd scales.** Either
   calibrate, or report greedy alongside. Standing rule 2 asked for a pooling
   control and got one; it did not ask whether the control was comparable on the
   axis the acquisition reads, and that is where §31 went wrong.
3. **Re-measure the auxiliary-quality threshold under greedy** before deciding
   on the docking cache.
4. **Do not build the docking cache yet.** Both the provisional threshold and §8
   put it below the line.

Artefacts: `results/auxq_{jak,cyp}_*.json` (EI sweep),
`results/auxqg_{jak,cyp}_*.json` (greedy control), all audited clean.

## 33. Selectivity: the objective choice is worth +40 points, the surrogate is worth nothing

Every campaign in §§18--32 optimises a single potency column. That is not the
objective a kinase programme fails on. Compounds rarely die from insufficient
potency; they die from hitting something else, and off-target inhibition is a
clinical failure mode rather than a benchmark metric. This section optimises

    y_sel(m) = pIC50(m, on-target) - max_{j} pIC50(m, anti-target j)

on the same clean blocks. Instrument: `scripts/selectivity_campaign.py`.

**No budget splitting.** Every arm observes the *full panel* on any compound it
selects, which is what selectivity profiling is. Budget is compounds, no arm
sees more measurements than another, and §14's rejected H1-split is not
re-litigated -- at budget 200 across four targets it would leave 50 compounds
per target, which tests nothing.

### Selectivity is a different problem from potency

| block | overlap between the selectivity top-10 and the potency top-10 |
|---|---|
| JAK1, JAK2, TYK2 | **0%** |
| JAK3 | 40% |
| CYP (four targets) | 20--40% |

On three of four JAK targets the most selective compounds and the most potent
compounds are **disjoint sets**. Optimising potency and hoping selectivity
follows is not a conservative approximation; on this block it is a different
search.

### The result

Budget 200 compounds on JAK (22% of the library) and 74 on CYP, 8 repeats x 4
on-targets = 32 campaigns per cell, top-10 selectivity recall, `--audit` clean.

| arm | JAK | CYP |
|---|---|---|
| `random` | 21.9% | 21.6% |
| `potency` -- optimise on-target potency, score on selectivity | 55.9% | 41.6% |
| **`contrast`** -- single-task GP on `y_sel` | **95.6%** | 36.9% |
| `compose` -- coregionalised GP per task, compose the contrast | 93.4% | 38.4% |

| comparison | JAK | CYP |
|---|---|---|
| model selectivity vs optimise potency | **+39.7 (p < 0.0001)** | −4.7 (n.s.) |
| compose vs optimise potency | +37.5 (p < 0.0001) | −3.1 (n.s.) |
| **compose vs regress the contrast** | **−2.2 (n.s.)** | +1.6 (n.s.) |

**1. Optimise the objective you actually want.** On JAK, regressing the
selectivity contrast recovers 95.6% of the selective top-10 against 55.9% for
optimising potency -- **+40 recall points for changing what you regress, at
identical cost and identical data.** This is the largest effect anywhere in this
package, it needs no new machinery, and it is the one finding here that is
directly actionable in a programme.

**2. The multi-task surrogate does not earn its place -- again.** Selectivity
was the objective where a coregionalised model *should* have a structural
advantage: it models every task, so it can express a between-task contrast,
where a single-task model can only regress the composite scalar. It does not
win. `compose` ties `contrast` on both blocks (−2.2 and +1.6, both n.s.).

That null is stronger than it looks. `compose` carries four times the rows and
therefore the compressed posterior variance that §32 showed *flatters* an arm
under EI. It is flattered and still does not win. A plain Tanimoto GP regressing
the right scalar is sufficient, which is the cheaper and more deployable answer.

**3. On CYP nothing works.** No arm separates from `potency`, and `contrast`
is nominally *below* it. CYP selectivity is not learnable at this budget by any
method here. The same family split as everywhere else in this work: what pays
on one panel does not pay on the other, and nothing predicts which in advance.

### Data integrity: is the target set real, or the assay's reporting floor?

A *difference* objective is far more exposed to censoring than a potency column.
If an anti-target value is a censored floor -- "IC50 > 10 uM" stored as pIC50
5.00 -- then `y(on) − max(off)` is large for a reason unrelated to selectivity,
and the campaign chases the limit of detection. `drop_censored` removes values
held by >2% of a column, but a floor just under that threshold survives.

`integrity()` inspects the winners directly: for the top-10 it reports the
*binding* anti-target value and how many compounds in the block share it.

| target | binding anti-target value | shared by (median / max) | on a repeated value |
|---|---|---|---|
| JAK1 | 5.08--7.28 | 1 / 4 | 0% |
| JAK2 | 5.39--7.22 | 2 / 6 | 0% |
| JAK3 | 4.30--7.28 | 1 / 15 | 10% |
| TYK2 | 5.77--7.82 | 2 / 5 | 0% |
| CYP3A4 | 3.70--5.94 | 2 / 10 | **20%** |
| CYP2C19 | 4.69--7.02 | 1 / 2 | 0% |

Binding values span 3.70--7.82 -- the real dynamic range, near-unique, not piled
on a floor. **The optimisation target is real chemistry.**

One contaminated compound was found and is reported rather than removed: panel
`[4.7, 4.7, 9.7, 4.7]`, three identical 4.70s being the signature of "inactive
at the screening dose". It is genuinely JAK3-selective and sits in JAK3's
top-10 -- **1 compound in 1 of 4 JAK top-10 sets**. CYP3A4's 20% deserves more
scepticism, and CYP is the more censored block (22.8% of compounds survive the
filter against 57.8% for JAK), which is a second reason to read its null
cautiously.

### What a practitioner should take from this

1. **Regress the contrast, not the potency.** +40 points on JAK for a change of
   target variable. Nothing else in this package is worth as much.
2. **Do not buy a multi-task surrogate for it.** A single-task GP on `y_sel` is
   as good, on both blocks, and is far simpler to deploy.
3. **Check your family first.** It works on JAK and not on CYP, and the
   leave-one-target-out self-check is the only way to find out.
4. **Audit the contrast for censoring before trusting it.** The objective
   amplifies reporting floors, and `integrity()` costs nothing.

Artefacts: `results/selectivity_{jak_200,cyp_74}.json`, both audited clean.

## 34. Six families, not two: the effect replicates, the unanimity claim does not

§25 states the paper's central practical claim on two blocks, JAK and CYP, and
both were chosen by hand. Two hand-picked families cannot separate *this holds
on co-measured panels* from *this holds on these two panels*, and the claim that
rests on them is specifically about generality. This section removes the
hand-picking, runs the identical protocol on four more families, and re-runs
every window cell at four times the repeats.

**Two things survive and one does not.** The mean transfer gain replicates
across six protein families spanning six mechanistic classes. The *unanimity*
that §25 calls "the result, not the effect size" does not — it is not a
reproducible statistic at any repeat count tried, and it fails on JAK itself.
Two further controls move the attribution: §25's cross-block comparison was
partly reading how much of each library the free history covered, and §31's
verdict on the coregionalisation reverses outside JAK and CYP.

### Choosing the families without choosing them

`scripts/find_dense_blocks.py` searches the whole BindingDB IC50 dump for
4-target sets with the largest fully-observed sub-matrix — 775,583 unique
(compound, target) pairs, 460 targets carrying 400 or more compounds, greedy
growth from every seed, deduplicated. **JAK (1,563) and CYP (1,472) come out
first and second without being told to**, which is the check that the search
agrees with what §§18–25 already used. The next eight that clear 300 compounds
and are not a near-duplicate of one already listed are now `FAMILIES` in
`dense_block_campaign.py`.

The censoring filter decides which of those can actually be run, and it is
brutal and uneven — it costs 42% of JAK and 77% of CYP:

| family | class | raw | clean | median rho | history coverage at H=50 |
|---|---|---|---|---|---|
| jak | protein kinase | 1,563 | 904 | +0.54 | 17% |
| pi3k | lipid kinase | 1,039 | 590 | +0.55 | 26% |
| hdac | zinc deacetylase | 560 | 378 | +0.60 | 41% |
| cyp | heme monooxygenase | 1,472 | 335 | +0.64 | 46% |
| dpp | serine protease | 698 | 300 | +0.26 | 52% |
| mmp | zinc protease | 464 | 248 | +0.60 | 63% |
| cathepsin | cysteine protease | 345 | 167 | +0.37 | **96%** — not run |
| fgfr | receptor tyr. kinase | 486 | 158 | **+0.80** | — too small |
| sstr | GPCR | 413 | 92 | +0.40 | — too small |
| abc | ABC transporter | 578 | 11 | −0.41 | — too small |

rho is the median off-diagonal Spearman between the targets' columns on the
clean block. Three families are too small for the protocol: the history is
`H x (T-1)` distinct compounds drawn clear of the top-k, which at `H=50` over
three auxiliaries is 160, and below that there is no library left to search.
**Cathepsin clears that floor at 167 and is still not run**, because its history
would then be 150 of the 157 eligible compounds — "free history on relatives"
would mean labels on 96% of the library, which is the degenerate end of the
coverage axis the rest of this section is about.

**The runnable set is biased against the strongest case, not toward it.** FGFR
has the highest task correlation in the whole scan (+0.80, well above anything
that can be run) and is excluded by two compounds. Nothing below is picking
favourable families.

### The grid

`scripts/dense_block_campaign.py`, settings identical to §25 in every respect —
`H=50`, batch 10, `n_init` 10, EI, 4 held-out targets, `--history-excludes-top`,
`--audit`, arms `stl`/`mt`/`pooled` — so the only thing that varies is the
block. Budgets are 11/22/33/50% of each block's post-censoring size.
`scripts/window_across_families.py` reads the artefacts. **All 26 cells at 8
repeats and all 12 at 32 repeats audit clean.**

| family | %lib | stl | mt | pooled | d recall | d AUC | up |
|---|---|---|---|---|---|---|---|
| **jak** | 22% | 68.8% | 75.3% | 72.8% | +6.6 | +0.085 ± 0.029 | **4/4** |
| | 33% | 76.2% | 83.4% | 82.5% | +7.2 | +0.066 ± 0.021 | **4/4** |
| **cyp** | 22% | 37.8% | 52.2% | 58.1% | +14.4 | +0.140 ± 0.039 | **4/4** |
| | 33% | 58.1% | 70.0% | 79.1% | +11.9 | +0.136 ± 0.032 | **4/4** |
| **hdac** | 22% | 46.2% | 66.3% | 66.9% | **+20.0** | +0.169 ± 0.040 | **4/4** |
| | 33% | 63.1% | 78.4% | 83.4% | +15.3 | +0.146 ± 0.031 | **4/4** |
| **mmp** | 22% | 47.2% | 60.9% | 65.6% | +13.8 | +0.081 ± 0.039 | 3/4 |
| | 33% | 69.1% | 85.6% | 85.6% | +16.6 | +0.094 ± 0.032 | **4/4** |
| **dpp** | 22% | 70.0% | 75.3% | 70.3% | +5.3 | +0.065 ± 0.038 | 3/4 |
| | 33% | 85.3% | 86.2% | 86.2% | +0.9 | +0.047 ± 0.025 | 2/4 |
| **pi3k** | 22% | 69.1% | 71.9% | 63.1% | +2.8 | +0.044 ± 0.036 | 3/4 |
| | 33% | 87.8% | 88.4% | 81.2% | +0.6 | +0.020 ± 0.026 | 1/4 |

Top-10 recall, 8 repeats x 4 held-out targets. **The mean gain is positive in
12/12 window cells and on 6/6 families, and d AUC is positive in all 26 cells of
the full grid.** That part of §25 generalises, and now rests on six mechanistic
classes rather than two.

### Unanimity is not a reproducible statistic

7 of 12 window cells are unanimous, against 8 of 8 on the two hand-picked
families. That alone would be a weakened claim. Re-running every window cell at
**32 repeats** shows it is worse than weakened:

| cell | 8 repeats | 32 repeats | |
|---|---|---|---|
| jak 22% | +6.6 **4/4** | +5.8 3/4 | **flips** |
| jak 33% | +7.2 **4/4** | +7.7 3/4 | **flips** |
| cyp 22% | +14.4 **4/4** | +8.4 3/4 | **flips** |
| cyp 33% | +11.9 **4/4** | +9.7 **4/4** | same |
| hdac 22% | +20.0 **4/4** | +20.5 **4/4** | same |
| hdac 33% | +15.3 **4/4** | +21.7 **4/4** | same |
| mmp 22% | +13.8 3/4 | +12.3 **4/4** | **flips** |
| mmp 33% | +16.6 **4/4** | +16.1 **4/4** | same |
| dpp 22% | +5.3 3/4 | +6.0 **4/4** | **flips** |
| dpp 33% | +0.9 2/4 | +2.7 3/4 | same |
| pi3k 22% | +2.8 3/4 | −0.2 3/4 | same |
| pi3k 33% | +0.6 1/4 | +1.7 2/4 | same |

**The unanimity verdict flips in 5 of 12 cells, including on JAK at both window
budgets** — one of the two families the original claim was built on. The losing
targets are at −0.9 and −0.3 points, which is to say they are ties that 8
repeats happened to place on the positive side.

Contrast what happens to the two measures when power goes up 4x:

| | 8 repeats | 32 repeats |
|---|---|---|
| d AUC positive | 12/12 | 12/12 |
| d AUC above 3 SEM | 5/12 | **10/12** |
| rank agreement of d AUC between the two runs | — | **rho = +0.916** |
| unanimous (4/4) | 7/12 | 6/12 |
| unanimity verdict agrees between runs | — | **7/12** |

The continuous effect measure behaves exactly as sampling theory says it should:
the estimate does not move, the error bar shrinks, and cells cross the
significance bar. The binary count does not converge at all, because it is a
minimum over four noisy per-target estimates and a target whose true delta is
near zero will keep changing sign.

> **This retires §25's practitioner self-check.** That check — hold out each
> target you have already measured, see whether multi-task wins, extrapolate to
> the cold one — was licensed by "its answer does not depend on which target you
> held out". Its answer depends on which target you held out *and on how many
> seeds you ran*. §25's "unanimous eight times out of eight" should be read as
> a property of that run, not of those blocks.

### What does predict the per-target gain

Two quantities, measured per held-out target over the 48 window rows, and
near-independent of each other (rho = −0.33):

| predictor | Spearman | partial (controlling the other) |
|---|---|---|
| auxiliary task correlation | +0.605 (p = 5e−6) | **+0.549** (p = 5e−5) |
| single-task baseline recall | −0.647 (p = 7e−7) | **−0.599** (p = 7e−6) |

Both hold at 32 repeats (+0.433 and −0.502, p <= 2e−3). The two failures have
different causes and the predictors separate them: **DPP4 is essentially
uncorrelated with its own relatives** (aux rho = +0.04 — selectivity over
DPP8/9 is what DPP4 inhibitors are designed for, so the panel is a family by
sequence and not by SAR), while **PI3K simply has no headroom** (`stl` is
already at 87.8% by the 33% budget). Neither is a failure of the surrogate.

This also says the fraction-of-library axis §25 moved to is a *proxy*. The
direct quantity is how much the single-task baseline has left to win:
"single-task recall in 40–85%" picks out the unanimous cells slightly better
than "22–33% of library" does (8/12 vs 7/12 inside, 1/14 vs 2/14 outside).

### The history-coverage confound — only visible with six families

§25 normalises the **budget** by library size, which is the entire point of the
fraction-of-library axis, and leaves the **history** absolute at `H=50` per
auxiliary. With two blocks of similar order that is invisible. Across six it is
not: 150 free observations are 17% of JAK's 904 compounds and 63% of MMP's 248,
so the families were never compared at equal free information, and "small block"
and "history covers a lot of the library" were the same variable.

`run_window_experiments.py coverage` re-runs every window cell with `H` set per family to
hold coverage at 25%. The prediction is directional and it is met:

| family | cov at H=50 | gain at H=50 | gain at matched coverage |
|---|---|---|---|
| jak 22% | 17% | +6.6 (4/4) | **+7.8** (4/4) |
| jak 33% | 17% | +7.2 (4/4) | **+10.6** (4/4) |
| pi3k 22% | 26% | +2.8 (3/4) | +4.1 (3/4) |
| hdac 22% | 41% | +20.0 (4/4) | +12.8 (4/4) |
| cyp 22% | 46% | +14.4 (4/4) | **+6.3** (4/4) |
| dpp 22% | 52% | +5.3 (3/4) | +2.8 (3/4) |
| mmp 22% | 63% | +13.8 (3/4) | **−2.8** (2/4) |

The 33% cells move the same way; the full twelve are in
`results/window_families.json`.

**Families that gain history gain effect; families that lose it lose effect.**
Within-family, the change in gain tracks the change in coverage at rho = +0.772
(p = 0.0033). The spread across families compresses from **+0.6…+20.0 to
−2.8…+12.8**, and the two families §25 actually contrasted — CYP at +14.4
against JAK at +6.6 — become **+6.3 against +7.8**, which is no difference at
all.

> **So "CYP shows a bigger effect than JAK" was never a statement about CYP.**
> It was a statement about 150 free observations being 46% of one library and
> 17% of the other.

Both per-target predictors survive the control at reduced strength (partial
+0.348, p = 0.015 and −0.328, p = 0.023), so task correlation and headroom are
real effects whose apparent size was inflated by the confound. The practical
reading is the useful one: **the gain scales with what share of your library has
already been run against relatives**, which is a quantity a practitioner can
measure before committing.

### §31 reverses outside JAK and CYP

§31 concluded that the coregionalisation does not earn its place — naive pooling
ties on JAK and beats it on CYP, so the gain belongs to the free history rather
than to the fitted task correlation. Both arms share an initial design and are
stored per campaign, so this is a paired test and needs no new runs:

| family | rho | %lib | d recall (mt − pooled) | d AUC (mt − pooled) | p |
|---|---|---|---|---|---|
| pi3k | +0.55 | 22% | **+8.8** | **+0.040 ± 0.015** | 0.003 |
| pi3k | +0.55 | 33% | **+7.2** | **+0.057 ± 0.017** | 0.0004 |
| hdac | +0.60 | 22% | −0.6 | **+0.055 ± 0.018** | 0.001 |
| hdac | +0.60 | 33% | −5.0 | **+0.041 ± 0.014** | 0.002 |
| dpp | +0.26 | 33% | −0.0 | **+0.044 ± 0.022** | 0.029 |
| dpp | +0.26 | 22% | +5.0 | +0.055 ± 0.031 | 0.060 |
| jak | +0.54 | 22% | +2.5 | +0.012 ± 0.014 | 0.239 |
| jak | +0.54 | 33% | +0.9 | +0.008 ± 0.009 | 0.400 |
| mmp | +0.60 | 22% | −4.7 | −0.009 ± 0.021 | 0.625 |
| mmp | +0.60 | 33% | −0.0 | −0.018 ± 0.017 | 0.536 |
| cyp | +0.64 | 22% | −5.9 | −0.019 ± 0.015 | 0.217 |
| cyp | +0.64 | 33% | −9.1 | −0.022 ± 0.014 | 0.154 |

**`mt` beats `pooled` on AUC in 5 of 12 cells at p < 0.05 and loses in none.**
The sharper point is the downside rather than the mean: **`pooled` is worse than
plain single-task BO on PI3K at both budgets** (−6.0 and −6.6 recall points),
while **`mt` is never worse than single-task in any of the 12 cells**. Pooling's
risk is asymmetric, and the fitted `r` is what removes it. §31 was drawn from
the two families where pooling happens to be safe.

HDAC is the AUC/recall split §25 flags, in a new place: `mt` beats `pooled`
decisively on AUC and loses to it on final recall, which is `mt` getting there
sooner and `pooled` ending higher.

### What this section does and does not establish

* **Established.** The mean gain is positive on six families and six mechanistic
  classes, d AUC positive in 26/26 cells at 8 repeats and 12/12 at 32, with
  10/12 above 3 SEM at 32. The gain tracks task correlation and single-task
  headroom, both of which survive the coverage control.
* **Retired.** Unanimity over held-out targets, and the leave-one-target-out
  self-check that depended on it.
* **Re-attributed.** Cross-block differences in effect size, which were largely
  history coverage; and §31's verdict on pooling, which does not hold outside
  JAK and CYP.
* **Not addressed.** Still four targets per family, still 4-target blocks, still
  co-measured panels only — §17's null on separate per-target libraries is
  untouched. The three excluded families mean the runnable set spans 248–904
  compounds, and FGFR, the highest-correlation panel in the scan, is not among
  them. Matched coverage was tested at one value (25%), not swept.

### Reproducing

```bash
python scripts/find_dense_blocks.py                  # the unbiased search
python scripts/find_dense_blocks.py --labelled       # post-censoring sizes
python scripts/run_window_experiments.py --list      # every sweep, one table
python scripts/run_window_experiments.py window power coverage --dry-run
python scripts/run_window_experiments.py window power coverage
python scripts/window_across_families.py             # the tables above
python scripts/window_across_families.py \
    --pattern "results/win32_*.json" --repeats 32 --cov-pattern ""
```

Artefacts `results/win_{pi3k,hdac,dpp,mmp}_b*.json`, `results/win32_*.json`,
`results/wincov_*.json`, `results/dense_blocks_scan.json`,
`results/window_families{,_32}.json`. Every campaign artefact carries its own
`audit` block; all pass.

### One defect fixed in passing

`dense_block_campaign.py` printed "no campaign gain" for any d AUC under 3 SEM,
so a cell at +0.081 ± 0.039 — 2.1 SEM *positive* — read as a null. At 8 repeats
that branch is the common case, not the exception, and it is how an underpowered
cell gets quoted as a negative result. It now distinguishes "not established at
this power" from "no gain" and reports the SEM multiple.

## 35. The acquisition hook, re-gated on the multi-task blocks

§1 retired the acquisition hook on a hindsight ceiling of **+0.0018 (p = 0.90)**
and that number has been doing a lot of work since: it is the reason no LLM
controller was ever wired to the acquisition seam. It was measured on a
**single-task ChEMBL library** and never on a dense block, and §9 says that gap
is not free — *a prior and an acquisition function cannot be chosen separately*,
and a perfect prior scores −0.019…−0.144 under EI against +0.057…+0.113 under
UCB. The coregionalised GP changes the posterior sd, which is the only thing an
acquisition function reads. Every cell of §34 is EI.

So the gate was re-run where the contribution actually lives. **It does not
open**, but it fails differently than §1 did, and the difference is the useful
part.

### The portfolio is not five actions

Measured inside real `mt` campaigns (`hdac` 96 rounds, `mmp` 60 rounds), each
acquisition surface against EI's own:

| vs EI | ρ (hdac) | top-10 overlap | ρ (mmp) | top-10 overlap |
|---|---|---|---|---|
| ucb | +0.970 | 67.3% | +0.973 | 77.0% |
| pi | +0.996 | 66.8% | +0.996 | 79.0% |
| greedy | +0.743 | 27.9% | +0.800 | 38.1% |
| ts | +0.392 | 8.0% | +0.450 | 15.4% |

**EI, UCB and PI are one action.** Their AUC spread *within* a cell is 0.0035 to
0.0247 — against 0.058 to 0.216 for the full five, a spread produced entirely by
`greedy` and `ts` being consistently *worse*. So the action space is three
interchangeable arms plus two bad ones, and a near-zero ceiling over it is a
fact about the action space rather than evidence that adaptivity cannot pay.

### The gate

`run_window_experiments.py acquisition oracle controls` — the fixed sweep, the
per-round hindsight ceiling and the free switching schedules. EI is not re-run —
§34's artefacts are the same cells, settings and seeds, so everything is paired.
`scripts/acquisition_across_families.py` reads them. **All 84 runs audit clean.**

| cell | ei | ucb | pi | greedy | ts | best | oracle | headroom | p |
|---|---|---|---|---|---|---|---|---|---|
| cyp b74 | **6.415** | 6.405 | 6.409 | 6.381 | 6.258 | ei | 6.479 | **+0.0637** | 0.007 |
| cyp b111 | **6.668** | 6.650 | 6.663 | 6.611 | 6.523 | ei | 6.718 | **+0.0507** | 0.032 |
| mmp b55 | **8.277** | 8.276 | 8.274 | 8.219 | 8.232 | ei | 8.326 | **+0.0488** | 0.007 |
| mmp b82 | **8.603** | 8.599 | 8.598 | 8.543 | 8.553 | ei | 8.633 | +0.0302 | 0.115 |
| hdac b83 | **8.522** | 8.513 | 8.517 | 8.483 | 8.364 | ei | 8.534 | +0.0120 | 0.239 |
| pi3k b130 | **8.617** | 8.611 | 8.616 | 8.551 | 8.499 | ei | 8.621 | +0.0042 | 0.678 |
| hdac b125 | **8.718** | 8.709 | 8.714 | 8.674 | 8.596 | ei | 8.720 | +0.0018 | 0.733 |
| pi3k b195 | 8.816 | 8.807 | **8.817** | 8.735 | 8.738 | pi | 8.809 | −0.0081 | 0.172 |
| dpp b99 | **7.648** | 7.639 | 7.644 | 7.598 | 7.480 | ei | 7.636 | −0.0115 | 1.000 |
| dpp b66 | **7.352** | 7.339 | 7.349 | 7.293 | 7.136 | ei | 7.338 | −0.0135 | 0.818 |
| jak b200 | **8.823** | 8.798 | 8.821 | 8.689 | 8.732 | ei | 8.797 | −0.0262 | 0.500 |
| jak b300 | **8.972** | 8.953 | 8.970 | 8.824 | 8.896 | ei | 8.939 | −0.0324 | 0.536 |

AUC of the top-10 trace, `mt` arm, 8 repeats × 4 held-out targets.

### Three results

**1. §34's EI-only choice cost nothing.** EI is the best fixed acquisition in
**11 of 12 cells**; best-fixed minus EI is **+0.0000 mean, +0.0006 max**. This
is worth stating because it does *not* follow from §1, where EI was never best
(PI won at budget 50, UCB at 100). The single-task ranking inverts on the dense
blocks, and §34 happened to pick the right one.

**2. Switching headroom is larger than single-task but not established.**

| | value |
|---|---|
| mean headroom over 12 cells | **+0.0100 ± 0.0091** (1.1 SEM) |
| t-test vs 0 / Wilcoxon | p = 0.299 / p = 0.470 |
| positive / negative cells | 7 / 5 |
| clipped at zero (most generous bound) | +0.0176 |
| §1, single-task | +0.0018 (p = 0.90) |

The negatives are not evidence against adaptivity — the oracle is **myopic** and
can land below the best fixed arm, which §1's own artefact does at budget 100
(−0.0158). Clipping them gives +0.0176 as the most generous honest ceiling.

But the headroom is **family-structured, not uniform**: significant on `cyp`
(both cells) and `mmp` b55, zero on `hdac`/`pi3k`, negative on `jak` and `dpp`.
Three of twelve cells at p < 0.05 is more than the 0.6 chance would give, so
something real is there — it is just not a property of the method, it is a
property of some blocks.

**3. The free controls lose, so the headroom is not free — but there is little
of it to take.**

| control | vs best fixed | beats it in |
|---|---|---|
| round-robin | −0.0225 | **0/12** |
| random switch | −0.0269 | **0/12** |

Both lose, as they did single-task. A perfect *static* chooser (split-half, so
the winner's curse is removed) gains **−0.0070** — nothing. So all the headroom
that exists is in switching, none of it in picking a better fixed arm.

### Verdict: the gate stays shut, and the reason is the action space

A controller here would have to capture a ceiling of at most **+0.0176** that no
free schedule reaches, on a quantity whose 12-cell mean is 1.1 SEM from zero.
Set against the thing it would sit next to — the multi-task surrogate itself,
worth **+0.03 to +0.17 AUC** (§34) — acquisition switching is a second-order
effect even at its ceiling.

**So an LLM acquisition switcher is still not justified, but the diagnosis has
moved.** §1's reading was "adaptivity does not pay". The measurement here says
something narrower and more actionable: **the portfolio is degenerate.** Three
arms are interchangeable, two are strictly worse, and the oracle's gains come
from jumping to `greedy`/`ts` at isolated moments when their worse-on-average
surface happens to be right. That is a thin thing to build a controller on.

The honest next move is therefore *not* a controller, and not more repeats on
this ceiling. It is to make the action space non-degenerate first — a portfolio
whose members disagree for principled reasons rather than by numerical accident:

* **LogEI** (Ament et al., NeurIPS 2023). The gradient-optimisation argument
  does not apply to an enumerated library, but the underflow one does, and it is
  exactly the pathology §9 names: *"once the incumbent is the library maximum
  every remaining candidate has μ < best, so EI is zero for all of them. The arm
  ties and picks at random."* Log-space preserves the ordering through
  underflow. Cheapest change, and it targets a failure this repo has already
  measured.
* **qPO** (Gorantla et al., JCIM 2025). Batched BO over discrete chemical
  libraries — the exact setting here. Maximises P(batch contains the optimum)
  and decomposes additively, so it would replace the Tanimoto-penalty batch
  heuristic in `select_batch` with something principled rather than add another
  μ/σ transform.
* **MES / JES**. Information-theoretic, ranks by information about the optimum
  rather than improvement over an incumbent — the only candidates here that are
  not a monotone-ish function of the same two numbers.

Re-gate after that. If the ceiling over a genuinely diverse portfolio is still
inside the noise, the hook is dead for good rather than dead-pending-a-better-
action-space.

### Reproducing

```bash
python scripts/run_window_experiments.py acquisition oracle controls
python scripts/acquisition_across_families.py
```

Artefacts `results/acq_<family>_b<budget>_<acquisition>.json` (84, each with its
own `audit` block, all clean) and `results/acquisition_families.json`. EI comes
from `results/win_*.json` rather than being re-run.

`--acquisition oracle` **reads the true values** and is labelled in the code the
way `OraclePolicy` and `oracle_constraint` are: a reference line, not an arm.

## 40. §32 separated: one block was uncertainty, the other was the experimental design

*Numbering note: §§36–39 are reserved for work that has artefacts but no written
section (the co-measurement gate, the KIBA panels, the cross-family threshold,
the bad start) — see `TODO.md`. This section is independent of all four.*

§32 found that a multi-task arm given a **pure-noise** auxiliary still gains
recall under EI, and attributed it to uncertainty: *"Adding rows to the kernel
compresses sd whether or not those rows say anything, which makes the arm more
exploitative."* It supported that with `--acquisition greedy`, which removes sd
from the decision and made the noise gain vanish.

Removing sd is not the same as **matching** it. The control a reader asks for is
a single-task arm whose uncertainty is set to the multi-task arm's scale, with no
auxiliary data at all. If that arm reproduces the noise gain, the explanation is
closed. This section runs it. On one block it closes exactly. On the other it
fails, and what is underneath turns out not to be a modelling effect at all.

### The instrument

`--sd-scale X` multiplies the posterior standard deviation by `X` before the
acquisition reads it, leaving the posterior mean, the fit and the scoring
untouched — one guarded multiplication. At `X = 1.0` the multiply is skipped, and
`results/sdscale_jak_1.00.json` reproduces the stored `results/auxq_jak_0.0.json`
single-task arm to the last digit (AUC 8.9061909630, recall 76.25%).

`--aux-shuffle` is added alongside it: a null that **permutes** each auxiliary
column rather than corrupting it, preserving the marginal distribution and the
reporting pile-ups exactly, where `--aux-quality 0` preserves neither.

### The scale was predicted before it was swept

The grid `1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50` was fixed in advance, and the
value to look at came from `null_mechanism.py`, which measures posterior sd per
rung and knows nothing about campaign recall:

| primary observations | sd, single-task | sd, noise multi-task | ratio |
|---|---|---|---|
| 50 | 0.6836 | 0.5476 | **0.801** |
| 100 | 0.6428 | 0.5549 | 0.863 |
| 150 | 0.5958 | 0.5703 | 0.957 |
| 300 | 0.5190 | 0.5669 | 1.092 |

Two things worth keeping. The compression is **not constant over a campaign** —
it is strongest early and *reverses* past roughly 190 observations. And the
early-campaign value is 0.80.

### JAK: the explanation closes

EI, budget 300, 32 repeats × 4 targets, paired on (repeat, target) so every
comparison shares an initial design and a history draw.

| arm | auxiliary | Δ recall vs single-task | p |
|---|---|---|---|
| `mt` | 150 rows of pure noise | +2.03 ± 0.86 | 0.035 |
| `pooled` | 150 rows of pure noise | +0.55 ± 0.94 | 0.49 |
| **`stl`, sd × 0.80** | **none at all** | **+2.34 ± 0.73** | **0.0022** |

A single-task arm with no auxiliary data, told only to be 20% less uncertain,
**matches and slightly exceeds** everything the noise auxiliary bought. The 0.80
was not read off this table — it is the ratio a separate script measured on the
posterior at the campaign's early observation count.

The full sweep is an inverted U, which is what an over-exploratory EI predicts:

| sd × | 0.95 | 0.90 | 0.85 | **0.80** | 0.70 | 0.50 |
|---|---|---|---|---|---|---|
| Δ recall | +0.86 | +1.56 | +1.95 | **+2.34** | +2.27 | +1.09 |
| p | 0.36 | 0.082 | 0.0098 | **0.0022** | 0.0098 | 0.27 |

**On JAK, §32 is right, and is now closed.** The noise gain is an
exploration-level artefact, reproducible with no auxiliary data, at an
independently predicted scale.

#### What that costs the headline

If part of what `mt` gains is exploration level rather than information, the
comparison it should be read against is not plain single-task BO but single-task
BO at the **same** exploration level. JAK, real panel, 32 repeats:

| `mt` with the measured panel, against | Δ recall | p |
|---|---|---|
| plain `stl` — as reported since §18 | +7.66 ± 0.99 | <0.0001 |
| **`stl` at the matched sd × 0.80** | **+5.31 ± 0.91** | **<0.0001** |

Transfer on JAK is real and about **31% smaller** than every table in this
package states. That is the correction, and it does not threaten the claim.

### CYP: the explanation fails, in both directions

Same protocol, budget 111. At 32 repeats the stored 8-repeat numbers move (§34's
standing lesson): `mt` with noise falls from +1.6 to **−0.47 ± 1.55 (n.s.)**, and
`pooled` with noise from +14.7 to **+10.55 ± 1.85 (p < 0.0001)**. Still large.

The sweep was then run in **both** directions, because a flattened posterior mean
makes an arm more *exploratory*, not less:

| sd × | 0.35 | 0.50 | 0.70 | 0.80 | 1.00 | 1.50 | 2.00 | 4.00 | 8.00 | 20.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Δ recall | −15.70 | −4.69 | −1.25 | −0.23 | 0.00 | **+0.08** | −0.47 | −3.36 | −14.53 | −21.02 |

The best value anywhere in a 57-fold range of exploration is **+0.08 points**,
against a target of +10.55. Uniform uncertainty scaling explains **none** of
pooling's null gain on CYP. (Floors on the same cell: `random` −30.94,
`similarity` −0.47.)

### What it scales with is the number of noise rows

| noise rows | 30 | 75 | 150 | 300 |
|---|---|---|---|---|
| `pooled` Δ recall | −1.25 | +2.34 | **+10.55** | **+22.97** |
| p | 0.48 | 0.20 | <0.0001 | <0.0001 |

The single-task arm is identical (61.48%) in all four, so this is a clean
dose-response in row count alone. Three hundred rows of pure noise buy
**23 recall points**.

### The posterior is being destroyed while the campaign improves

`pooled` (r = 1) with noise, CYP block, 30 primary observations, 12 draws:

| noise rows | η stl | η pooled | sd ratio | sd CV, stl → pooled | mean shrink | ρ vs truth, stl → pooled |
|---|---|---|---|---|---|---|
| 30 | 0.283 | 0.867 | 0.807 | 0.094 → 0.123 | −0.01 | 0.279 → 0.214 |
| 75 | 0.283 | **1.000** | 0.734 | 0.094 → 0.140 | −0.12 | 0.279 → 0.128 |
| 150 | 0.283 | **1.000** | 0.674 | 0.094 → 0.162 | −0.35 | 0.279 → 0.108 |
| 300 | 0.283 | **1.000** | 0.582 | 0.094 → 0.090 | **−0.56** | 0.279 → **0.024** |

The fitted noise-to-signal ratio saturates at the grid ceiling by 75 rows. The
posterior mean collapses to 44% of its single-task spread. Its rank correlation
with truth falls to **0.024 — essentially uncorrelated**. And the campaign using
this posterior gains 23 points.

A posterior that has lost its ranking is not selecting compounds on the
chemistry. It is selecting them on something else.

### It is reading the experimental design

Under `pooled`, `r = 1` forces the auxiliary rows to be the *same task* as the
primary. Every history compound is therefore treated as already observed: its
posterior mean sits at its (noise) label, its sd is small, its EI is near zero.
**The arm systematically avoids the history.**

`--history-excludes-top` guarantees the primary's top-k is **not** in the history.
So avoiding the history means walking toward the answer — and the larger the
history, the smaller and more active-enriched the complement it walks into. At
H = 100 the history is 300 of 335 compounds, and the 35 left over contain all ten
actives.

The flag is the whole effect, and removing it flips the sign almost exactly:

| CYP, H = 100, `pooled` with pure noise | Δ recall |
|---|---|
| history **excludes** the top-k | **+22.97** (p < 0.0001) |
| history **may contain** the top-k | **−23.05** (p < 0.0001) |

This is not transfer, not uncertainty calibration, and not chemistry. It is an
arm reading a property of how the experiment was set up. `--history-excludes-top`
was added to remove a confound — history pointing *at* the answer — and
introduced its mirror image, which an arm that treats history as observed can
exploit.

### The single axis is history coverage of the library

Both the row-count ladder and the block-to-block difference collapse onto one
quantity: what fraction of the library the history occupies.

The leak is measured directly by the **swing**: how far `pooled`'s null gain
moves when `--history-excludes-top` is removed. That subtracts off whatever the
arm would do anyway and leaves the part that is reading the design.

| block | library | H | noise rows | coverage | excludes top-k | may contain | **swing** |
|---|---|---|---|---|---|---|---|
| jak | 904 | 50 | 150 | **17%** | +0.55 | −7.50 | **8.05** |
| cyp | 335 | 50 | 150 | **45%** | +10.55 | −18.52 | **29.06** |
| cyp | 335 | 100 | 300 | **90%** | +22.97 | −23.05 | **46.02** |

Monotone in coverage across two blocks with different libraries and budgets. The
single-flag number alone is misleading: JAK's `pooled` null gain looks clean at
+0.55, and the swing shows 8 points of design sensitivity sitting underneath it,
with the two sides happening to straddle zero.

The row-count ladder on CYP measures the same axis from the other side
(9% → −1.25, 22% → +2.34, 45% → +10.55, 90% → +22.97), with the single-task arm
identical throughout.

This connects to §34, which found that controlling for *"what share of the
library the free history covers"* collapsed the per-family gain spread from
+0.6…+20.0 to −2.8…+12.8 recall points. §34 read that as a real moderator of
transfer. It is at least partly this artefact, for any arm that pools. §34's
numbers are `mt`, which is far less exposed — but the coincidence is close enough
that the coverage variable should not be interpreted until §34 is re-run with a
null arm at matched coverage.

### Both nulls agree, so it is not the shape of the column

`--aux-quality 0` and `--aux-shuffle` destroy the information in different ways:
corruption preserves neither the marginal distribution nor the reporting
pile-ups, permutation preserves both exactly. CYP, H = 50, 32 repeats:

| design | null | `mt` | `pooled` |
|---|---|---|---|
| excludes top-k | corrupt to ρ=0 | −0.47 ± 1.55 | +10.55 ± 1.85 |
| excludes top-k | **permute** | −1.41 ± 1.46 | **+11.56 ± 1.81** |
| may contain top-k | corrupt to ρ=0 | −2.73 ± 1.44 | −18.52 ± 1.94 |
| may contain top-k | **permute** | −3.12 ± 1.47 | **−19.69 ± 2.14** |

The two nulls agree in every cell, to well inside one standard error. None of
this depends on the auxiliary column's distribution — which is the useful
negative, because it removes the cheapest remaining alternative explanation.

### What survives, and it is the part that matters

The real panel is **not** affected. CYP, H = 50, 32 repeats:

| design | arm | pure noise | real panel |
|---|---|---|---|
| history excludes top-k | `mt` | −0.47 ± 1.55 | **+9.69 ± 1.50** (p < 0.0001) |
| history excludes top-k | `pooled` | **+10.55 ± 1.85** | +17.50 ± 1.70 |
| history may contain top-k | `mt` | −2.73 ± 1.44 | **+11.25 ± 1.66** (p < 0.0001) |
| history may contain top-k | `pooled` | **−18.52 ± 1.94** | +10.00 ± 1.73 |

Read the `mt` rows first. **`mt` passes its null control in both designs** — its
noise gain is zero or negative either way — while its real-panel gain is large
and significant either way. On CYP the fitted coregionalisation's advantage is
**entirely information transfer**. §32 attributed 13% of it to the artefact; at
32 repeats that 13% is not there.

`pooled` is the opposite. Its null gain is +10.55 one way and −18.52 the other:
its behaviour under a null is dominated by a flag in the protocol.

JAK says the same thing, and says one more:

| JAK, H = 50, 32 repeats | `mt` | `pooled` |
|---|---|---|
| pure noise, history excludes top-k | +2.03 ± 0.86 | +0.55 ± 0.94 |
| pure noise, history may contain top-k | +2.19 ± 0.84 | **−7.50 ± 1.36** |
| **real panel**, history excludes top-k | **+7.66 ± 0.99** | +5.70 ± 1.01 |
| **real panel**, history may contain top-k | **+8.28 ± 1.08** | **+0.78 ± 1.47 (n.s.)** |

`mt` is stable on all four rows: its null gain is the sd artefact (+2.0, +2.2,
both reproduced by `stl` at sd × 0.80) and its real gain is large either way.

**`pooled`'s real-panel gain on JAK does not survive removing the flag** — +5.70
becomes +0.78, not significant. So the contamination is not confined to CYP or to
the null arm. On both blocks tested, `pooled` reports a transfer gain that is
substantially a function of whether the protocol put the answer outside the
history. Nothing `pooled` reports anywhere in this package should be read as a
modelling result without the flag crossed.

### Corrections to §32

1. **"It is the acquisition" is right on JAK and wrong on CYP.** On JAK the noise
   gain is uniform sd compression, fully reproduced by `stl` at sd × 0.80 with no
   auxiliary data, at a scale predicted in advance. On CYP no uniform scale in a
   57-fold range reproduces any of it.
2. **§32's CYP `mt` figure was an 8-repeat artefact.** "+1.6, ~13% not transfer"
   is −0.47 ± 1.55 at 32 repeats. That result is *stronger* than reported.
3. **"~70% of pooling's gain is not transfer" is the wrong frame.** It is not a
   share; it is a sign that a protocol flag flips.
4. **The greedy ablation was suggestive, not sufficient.** It showed that removing
   sd kills the noise gain; it could not show that *matching* sd reproduces it,
   and on CYP matching sd does not.
5. **The provisional auxiliary-quality threshold (0.45–0.6) rests on a baseline
   that is part design leak.** It still must not be quoted, and re-measuring it
   now needs the leak controlled, not just greedy.
6. **Transfer on JAK is ~31% smaller than reported** once the baseline is
   uncertainty-matched: +5.31 rather than +7.66 recall points. Still p < 0.0001.
7. **`pooled` is contaminated on both blocks, not just CYP, and in the real-panel
   arm and not only under a null.** Its JAK real-panel gain falls from +5.70 to
   +0.78 (n.s.) when `--history-excludes-top` is removed. §31's retraction was
   not broad enough.

### What to do

1. **Report every arm comparison at both settings of `--history-excludes-top`,
   or drop `pooled`.** A result that changes sign with a protocol flag is not a
   result. `mt` is clean at both; `pooled` is not.
2. **Run `--sd-scale` at the measured ratio beside `--aux-quality 0`.** Together
   they say whether a null gain is exploration level (JAK) or something else
   (CYP). Either costs one run.
3. **Quote transfer against an uncertainty-matched baseline**, not against plain
   single-task BO, wherever EI is the acquisition.
4. **Re-read §34's coverage moderator** with a null arm at matched coverage
   before it is used as guidance.
5. **Treat `pooled` as retired for null work.** §31 was already retracted on
   other grounds; this is a second, independent reason.

Artefacts: `results/sdscale_{jak,cyp}_*.json` (sweeps, 19 runs),
`results/sdref_{jak,cyp}_{0.0,1.0}.json` (matched 32-repeat references),
`results/sdrows_cyp_h{10,25,100}.json` (row-count ladder),
`results/sdleak_cyp_*.json` (the design flag, both ways),
`results/shuf_cyp_{excl,keep}.json` (the permutation null),
`results/null_mechanism_{jak,cyp}.json` (the predicted scale). All audited clean.

### Reproducing

```bash
# the control, at the independently predicted scale
python scripts/dense_block_campaign.py --family jak --arms stl --acquisition ei \
  --budget 300 --batch 10 --k 10 --history 50 --history-excludes-top \
  --repeats 32 --seed 0 --sd-scale 0.80 --out results/sdscale_jak_0.80.json

# the thing it controls
python scripts/dense_block_campaign.py --family jak --arms stl mt pooled \
  --acquisition ei --budget 300 --batch 10 --k 10 --history 50 \
  --history-excludes-top --repeats 32 --seed 0 --aux-quality 0.0 \
  --out results/sdref_jak_0.0.json

# the design flag, both ways (drop --history-excludes-top for the second)
python scripts/dense_block_campaign.py --family cyp --arms stl pooled \
  --acquisition ei --budget 111 --batch 10 --k 10 --history 100 \
  --repeats 32 --seed 0 --aux-quality 0.0 \
  --out results/sdleak_cyp_h100_keeptop.json

# the permutation null
python scripts/dense_block_campaign.py --family cyp --arms stl mt pooled \
  --acquisition ei --budget 111 --batch 10 --k 10 --history 50 \
  --history-excludes-top --repeats 32 --seed 0 --aux-shuffle \
  --out results/shuf_cyp_excl.json
```

## 41. The mechanism run down: two explanations refuted, and what the gain actually is

§40 established *that* a single-task arm at sd × 0.80 reproduces everything a
pure-noise auxiliary buys on JAK (+2.34 ± 0.73 against +2.03 ± 0.86). It did not
establish *why* 0.80, or why the same knob does nothing on CYP. The advisor's
note asks for exactly this — *"examine posterior means, variances, acquisition
rankings, and selected compounds"* — so both candidate explanations were run.
Both fail. What is left is smaller and more honest than either.

### Hypothesis 1: the GP is over-dispersed. Refuted.

The natural story is that EI over-explores because the surrogate's error bars
are wider than its errors, and sd × 0.80 is the factor that fixes it. That
predicts `std(z) ≈ 0.80` on JAK, where `z = (y − μ)/σ` on held-out candidates,
and `≈ 1.0` on CYP.

Measured, 16 draws per cell, no reference to any campaign:

| n_obs | JAK std(z) | JAK \|z\|>2 | CYP std(z) | CYP \|z\|>2 |
|---|---|---|---|---|
| 20 | 1.55 | 0.098 | 1.53 | 0.122 |
| 50 | 2.11 | 0.089 | 1.41 | 0.110 |
| 100 | 2.74 | 0.062 | 1.45 | 0.090 |
| 200 | 3.96 | 0.058 | 4.78 | 0.065 |
| 300 | 3.00 | 0.067 | 1.88 | 0.086 |

The prediction is wrong in **direction, magnitude and discrimination**. The GP
is **under**-dispersed everywhere — `std(z)` between 1.4 and 4.8, never below 1 —
so by this measure EI should be exploring *more*, not less. And the number does
not separate JAK (optimum 0.80) from CYP (optimum 1.0): both sit around 2–3.

The `|z| > 2` column says why `std(z)` is the wrong summary. Under a Gaussian it
would be 0.046; it is 0.058–0.122, so the bulk of the residuals is not far off
but the tails are heavy. Those are activity cliffs — a compound one substitution
from an observed inactive that is three log units more potent. A single such
miss with a small σ contributes a `z` of 20 and dominates the variance. **A
global variance ratio cannot describe a cliff-driven residual distribution**,
and any calibration argument built on one is unsafe here.

### Hypothesis 2: it stops buying isolated compounds. Refuted on JAK.

The replacement story is about *where* uncertainty sits rather than how large it
is. On a Tanimoto GP σ is largest for compounds far from everything observed.
Actives arrive in series, so isolated compounds are mostly not actives, and EI
at full σ should be spending picks on them.

That predicts the picks sd × 0.80 *drops* are more isolated and lower valued
than the ones it *gains*. Measured by running the two arms on identical seeds
and diffing the selected sets; isolation is the maximum Tanimoto similarity to
any other library member.

| JAK, 64 campaigns | isolation | value | in top-k | n |
|---|---|---|---|---|
| bought only at full σ (dropped) | 0.786 | 7.251 | 0.005 | 3335 |
| bought only at tight σ (gained) | 0.790 | 7.279 | **0.010** | 3335 |
| | p = 0.29 | p = 0.51 | | |

Isolation and value are **indistinguishable**. The arm is not searching
differently in any way that shows up in what it buys.

On CYP the isolation difference exists and is significant (0.527 vs 0.558,
p = 0.028), but the top-k rate moves the *wrong* way (0.022 dropped, 0.013
gained) — consistent with σ compression not helping there at all.

### What is left is a tail effect, and the arithmetic closes

One column in that table does move: the rate at which a swapped pick lands in
the top-10 **doubles**, 0.005 → 0.010. It is a tiny rate, and that is the point.

- 3,335 picks differ over 64 campaigns — **52 per campaign**, out of a budget
  of 300.
- Of those, the full-σ arm hits the top-10 on 0.5% and the tight-σ arm on 1.0%.
- On a top-10 success set that predicts **+2.50 recall points**.
- Measured: **+2.34 points**.

The same identity run on CYP predicts **−0.94** points, against a measured
−0.23 at σ × 0.80. Only ten picks per campaign change there, because the budget
is 111 rather than 300, and the top-k rate moves the *wrong* way
(0.022 → 0.013).

The gain is accounted for compound by compound. It is not a change in search
behaviour, not a calibration correction, and not a mechanism with a story. It is
a **small-probability tail shift in a pick set that is otherwise statistically
identical.**

### What that means for how this is reported

The honest statement is narrower than §32's and more useful:

> **EI's exploration level is an untuned hyperparameter, and on JAK the slack in
> it is worth about as much as a pure-noise auxiliary appears to be worth.** The
> noise auxiliary does not teach the model anything; it perturbs σ, and a σ
> perturbation of the right size is worth +2.3 recall points on this block for
> reasons that do not survive inspection beyond the arithmetic above.

That reframes the §32 question. It is not *"why does noise help?"* — it is *"why
is the acquisition's exploration level left untuned while the slack in it gets
attributed to transfer?"* Any transfer comparison under EI inherits this, which
is why §40 recommends an uncertainty-matched baseline rather than plain
single-task BO.

It also bounds the worry. The effect is **+2.3 points on one block and zero on
the other**, and it is fragile — a tail of 52 picks. It is not large enough to
threaten a +7 to +11 point transfer gain, and §40 measured the corrected JAK
figure directly: +5.31 ± 0.91, p < 0.0001.

### And the unit the claim is about

The advisor also flagged *"uncertainty estimates that reflect the experimental
units"*. Every artefact here reports a paired Wilcoxon over campaigns. Within a
block that is right — 32 repeats × 4 held-out targets are paired draws of one
experiment. For a claim ranging over protein families it is not, because those
campaigns share a block, a library, an assay history and a censoring filter.

`scripts/block_level.py` restates any sweep at three levels. §34's headline grid:

| unit | n | mean Δ recall | sem | positive | p | floor |
|---|---|---|---|---|---|---|
| campaign | 1536 | +9.37 | 0.50 | 786/1536 | **6×10⁻⁷⁷** | — |
| cell (block × budget) | 12 | +9.37 | 2.05 | 11/12 | 0.0010 | 0.0005 |
| **block** | **6** | **+9.37** | **2.99** | **6/6** | **0.031** | **0.031** |

Three things follow.

**The campaign-level p is indefensible.** 6×10⁻⁷⁷ over 1,536 non-independent
campaigns is a statement about how many times six blocks were resampled.

**The block-level p sits exactly at its floor.** With six blocks a two-sided
signed-rank test cannot return below 0.031 however large the effect, because
that is `2/2⁶`. So "p = 0.031" here means *every block agreed in sign* and
carries no information about magnitude. It must be quoted as
**"6/6 blocks, p = 0.031, the minimum attainable at n = 6"** — anything else
invites a reader to treat a unanimity statement as a strength-of-effect one.

**Unanimity is doing a lot of work.** Per block: hdac +21.1, mmp +14.2, cyp
+9.1, jak +6.7, dpp +4.3, **pi3k +0.7**. The sixth block is +0.7 points, and
"6/6" rests on it. This is §34's own lesson about unanimity, arriving again by a
different route.

Artefacts: `results/mechanism_probes.json` (both probes, JAK and CYP),
`results/block_level_win32.json`, `results/leak_audit.json`.

### Reproducing

```bash
python scripts/mechanism_probes.py --out results/mechanism_probes.json
python scripts/block_level.py "results/win32_*.json" --arm mt \
  --json results/block_level_win32.json
python scripts/leak_audit.py --json results/leak_audit.json
```

## 42. The headline under the honest design: `mt` strengthens, `pooled` dies

§40 showed `--history-excludes-top` is not a neutral control. It removes the §22
confound — a history that happens to contain the primary's top-k — and
introduces its mirror image: an arm that treats history as observed avoids it,
and the flag guarantees the answer is outside it. Both are large and they point
in opposite directions.

`scripts/leak_audit.py` scored the back catalogue. **122 stored artefacts carry
a `pooled` arm under the flag**, at history coverages from 14% to 90% — and that
includes §34's entire headline grid, at 17% (jak), 25% (pi3k), 40% (hdac), 45%
(cyp), 50% (dpp) and 60% (mmp). The number cannot be left as it stands.

So all twelve cells were re-run with a **uniform history** — the whole grid,
32 repeats, same seed, the only change being that the primary's top-k is no
longer withheld from the pool the history is drawn from. `run_window_experiments`
gained one entry, `leak`; the six pre-existing experiments replan byte for byte
(verified: 12/12 "already present").

### The result

Δ top-10 recall against single-task BO, 32 repeats × 4 held-out targets per cell:

| cell | coverage | `mt` excl | `mt` **uniform** | move | `pooled` excl | `pooled` **uniform** | move |
|---|---|---|---|---|---|---|---|
| cyp b74 | 45% | +8.44 | **+19.77** | +11.33 | +14.22 | +17.42 | +3.20 |
| cyp b111 | 45% | +9.69 | **+11.25** | +1.56 | +17.50 | +10.00 | −7.50 |
| dpp b66 | 50% | +6.02 | +4.69 | −1.33 | +1.95 | **−11.17** | −13.12 |
| dpp b99 | 50% | +2.66 | +3.05 | +0.39 | +2.11 | **−14.30** | −16.41 |
| hdac b83 | 40% | +20.55 | **+23.52** | +2.97 | +18.91 | +16.25 | −2.66 |
| hdac b125 | 40% | +21.72 | **+22.89** | +1.17 | +25.39 | +18.67 | −6.72 |
| jak b200 | 17% | +5.78 | +4.45 | −1.33 | +2.97 | −0.31 | −3.28 |
| jak b300 | 17% | +7.66 | +8.28 | +0.62 | +5.70 | +0.78 | −4.92 |
| mmp b55 | 60% | +12.34 | **+17.03** | +4.69 | +16.33 | +10.47 | −5.86 |
| mmp b82 | 60% | +16.09 | +15.16 | −0.94 | +18.75 | +9.69 | −9.06 |
| pi3k b130 | 25% | **−0.23** | **+4.30** | +4.53 | −8.83 | −9.38 | −0.55 |
| pi3k b195 | 25% | +1.72 | +3.52 | +1.80 | −8.36 | −10.23 | −1.88 |

**`mt` is not hurt by removing the flag — it improves.** Mean +9.37 → **+11.49**,
positive on 11/12 → **12/12**, paired move +2.12 (p = 0.050). The one cell that
was negative, pi3k b130 at −0.23, becomes +4.30.

**`pooled` collapses.** Mean +8.89 → **+3.16**, positive on 10/12 → 7/12, paired
move **−5.73 (p = 0.0034)**. On DPP it inverts outright, +2.0 → −11.2 and
+2.1 → −14.3.

### Restated at the unit the claim is about

Using `scripts/block_level.py`, the six families as the experimental unit:

| arm | level | n | mean | sem | positive | p | floor |
|---|---|---|---|---|---|---|---|
| **`mt`** | campaign | 1536 | +11.49 | 0.52 | 858/1536 | 2×10⁻⁹¹ | — |
| | cell | 12 | +11.49 | 2.27 | **12/12** | 0.0005 | 0.0005 |
| | **block** | **6** | **+11.49** | **3.25** | **6/6** | **0.031** | **0.031** |
| `pooled` | cell | 12 | +3.16 | 3.51 | 7/12 | 0.38 | 0.0005 |
| | **block** | **6** | **+3.16** | **5.14** | **4/6** | **0.44** | 0.031 |

Per block, `mt`: hdac +23.2, mmp +16.1, cyp +15.5, jak +6.4, pi3k +3.9,
dpp +3.9. The weakest block is now +3.9 rather than +0.7, so the unanimity claim
no longer rests on a cell that is indistinguishable from zero.

Both the cell- and block-level p-values sit at their floors, which is a statement
about having 12 cells and 6 blocks, not about effect size. Quote them as
**"12/12 cells and 6/6 blocks, p at the minimum attainable"**.

### Reconciling with §22, which pointed the other way

§22 ran this same crossing and concluded the opposite: *"Between half and
two-thirds of the headline gain comes from the historical set happening to
contain the compounds the campaign is looking for."* That is why
`--history-excludes-top` became the default. Both measurements are correct; they
were taken at different budgets, and the effect is strongly budget-dependent.

Effect of withholding on `mt`, same blocks, by budget:

| block | budget | repeats | uniform | withheld | effect of withholding |
|---|---|---|---|---|---|
| CYP | 60 (§22) | 15 | +19.33 | +2.67 | **−16.67** |
| CYP | 74 | 32 | +19.77 | +8.44 | −11.33 |
| CYP | 111 | 32 | +11.25 | +9.69 | **−1.56** |
| JAK | 60 (§22) | 15 | +10.17 | +3.67 | **−6.50** |
| JAK | 200 | 32 | +4.45 | +5.78 | +1.33 |
| JAK | 300 | 32 | +8.28 | +7.66 | **−0.63** |

Monotone in budget on both blocks, and mechanically unsurprising: with a larger
budget the campaign finds the top-$k$ on its own, so a history that already
contains them is worth less. §22 measured the confound at the budget where it is
largest.

That settles which design to prefer. At the budgets this grid uses, the confound
§22 was guarding against costs `mt` between $-1.6$ and $+1.3$ recall points,
while the leak the guard introduced costs `pooled` between $8$ and $46$ points of
swing. **The cure was larger than the disease.** §22's observation stands at
small budgets and is now reported as a budget-dependent sensitivity rather than
as grounds for the design.

### What this settles

**The primary history design is now uniform.** This reverses the project's
default. The reasoning: the §40 leak is a *correctness* problem — an arm reading
the protocol rather than the chemistry — while the §22 objection to a uniform
history is a *generalisation* problem, that real historical data is not a uniform
sample of the library now being screened. The second is real but it is already
measured separately, by `--history-cluster` and the reach ratio (§17, §23), and
it does not require a design that an arm can exploit. Every headline number in
this package is now stated under a uniform history, with the withheld-top-k
figure reported alongside as a sensitivity.

**`pooled` is out of the headline tables.** Under the honest design it has no
result: +3.16 ± 5.14, 4/6 blocks, p = 0.44. §31 retracted its §19/§21 advantage;
§40 showed its null behaviour is protocol-driven; this shows it has nothing left
to report. It stays in the methods section as a cautionary control — an arm whose
apparent gain was an artefact that three separate analyses were needed to find —
and nowhere else.

**The headline claim, restated.** Given free historical assay data on a
co-measured target family, a coregionalised GP improves top-10 recovery over
single-task BO by **+11.5 ± 3.3 recall points**, positive on **6/6 protein
families and 12/12 cells**, measured under a uniform history at 22–33% of the
library. On JAK, where an uncertainty-matched baseline was also run, the
information-attributable part is **+5.31 ± 0.91** rather than the +7.66 a plain
single-task baseline reports (§40).

### Reproducing

```bash
python scripts/run_window_experiments.py power   # the withheld-top-k side
python scripts/run_window_experiments.py leak    # the uniform side
python scripts/block_level.py "results/winleak_*.json" --arm mt \
  --json results/block_level_winleak.json
python scripts/leak_audit.py --json results/leak_audit.json
```

Artefacts: `results/winleak_*.json` (12 cells, audited clean),
`results/block_level_winleak.json`, `results/leak_audit.json`.


### Correction to §42: the between-design change was tested at the wrong unit

§42 reported the paired change between history designs over the **twelve budget
cells**. Two budgets on one family are not independent. At the family level
(§45) `pooled`'s decline holds unanimously (-5.73 +- 2.01, 0/6, p = 0.031) but
**`mt`'s improvement does not** (+2.12 +- 1.04, 4/6, p = 0.16). The uniform
design is not shown to help `mt`; it is shown not to cost it anything. §42's
headline is a level claim at the family level and is unaffected.

## 43. Controlled simulation: the exploration knob is real, quality-dependent, and small

§41 refuted both explanations the real blocks could offer. Two blocks cannot
settle it: library size, budget, assay noise and how well a Tanimoto kernel
describes the target all move at once between JAK and CYP. `scripts/sd_scale_simulation.py`
separates them on synthetic data where the answer is known.

### Design

A library of $n = 500$ points in a 5-dimensional latent space, with the
campaign's gram matrix built from those coordinates. The target is

$$y = \alpha f + \sqrt{1 - \alpha^2}\, g \;(+\ \text{cliffs})$$

where $f$ is a GP draw on that same kernel — everything the surrogate can
possibly learn — and $g$ is an independent draw it cannot see. **$\alpha$ is a
direct, monotone dial on surrogate quality**, replacing a property of a protein
with a controlled variable. Budget 75 (15% of the library), $k = 10$, 32 repeats
× 8 independent libraries = **256 campaigns per (quality level, scale)**.

A second condition adds **activity cliffs**: a sparse set of large single-point
spikes. §41 found the real blocks' residuals are cliff-driven (|z| > 2 at 6–12%
against 4.6% for a Gaussian), and a pure GP draw has none, so without this the
simulation sits in a regime the real data does not.

Everything downstream — the fit, the acquisition, the batch selection, the
tie-breaking — is `run_campaign`, unchanged.

### The hypothesis, and what happened to it

> EI trades $\mu$ against $\sigma$ at a fixed rate. That rate is right for some
> surrogates and wrong for others: one that ranks well does not need to explore,
> and EI makes it explore anyway. So the recall-optimal $\sigma$ scale should
> **fall** as the surrogate's held-out rank correlation rises.

A pilot (seed 0, 8 levels, 4 replicates) gave $\rho = -0.38$ and $-0.32$, not
significant. A **confirmatory run on fresh libraries** (seed 500, 14 levels,
8 replicates) was then run:

| landscape | $\rho$(quality, optimal $\sigma$ scale) | $p$ | $n$ |
|---|---|---|---|
| smooth (a GP draw) | **−0.546** | **0.043** | 14 |
| with activity cliffs | **−0.546** | **0.043** | 14 |

(The two coincide to three decimals by accident — the underlying sequences and
recall tables differ; they are $-0.5462$ and $-0.5459$.)

**The direction held, twice, independently.** The better the surrogate ranks,
the more compressing $\sigma$ is worth — which is the JAK/CYP contrast
(Spearman 0.73 against 0.59, optimum 0.80 against 1.00) reproduced as a
controlled trend rather than two points.

### And a pre-registered statistic that did not replicate

The pilot also suggested a cleaner response than the argmax of a flat curve: the
gain at a **fixed** compression, which is what a practitioner would actually
apply. On the pilot, gain at $\sigma \times 0.6$ correlated with quality at
$\rho = +0.738$, $p = 0.037$ (smooth). That value was selected on the pilot, so
the confirmatory run is its test. It failed:

| landscape | gain at ×0.6 vs quality | $p$ | mean gain at ×0.6 |
|---|---|---|---|
| smooth | +0.398 | 0.16 | **−0.51** |
| cliffs | +0.088 | 0.76 | **−1.15** |

Direction preserved, significance gone, and the *average* gain at a fixed ×0.6
is **negative on both landscapes**. The pilot's $p = 0.037$ was one of eight
tests (four compressions × two landscapes) and does not survive its own
replication. Reported here because pre-registering a statistic is only worth
doing if the failure is reported too.

### What the simulation actually establishes

**The effect is real and directional, and it is small.** Across 28 quality
levels the best achievable gain from tuning $\sigma$ is 0.0–2.5 recall points,
and at any fixed compression the mean gain across quality levels is near zero or
negative. There is no setting of $\sigma$ that is good everywhere, which is why
"tune $\sigma$" is not the recommendation.

**Cliffs damp it.** On the fixed-compression statistic the smooth landscape
gives $\rho = +0.398$ and the cliff landscape $+0.088$; mean gains are $-0.51$
and $-1.15$. Real chemical landscapes have cliffs, so the knob should be
expected to do less on real blocks than on a GP draw — which is what JAK
($+2.34$) and CYP ($0.00$) show.

**The usable conclusion is a control, not a method.** Because the optimal
exploration level depends on surrogate quality, and because a method given extra
kernel rows has a different $\sigma$ scale whether or not those rows are
informative (§40), **a transfer comparison under EI confounds information with
exploration level by an amount that depends on how good the surrogate is.** The
fix is the matched-uncertainty baseline of §40, which costs one extra run. On
JAK it moves the quoted transfer gain from $+7.66$ to $+5.31$ recall points.

### Limits of this simulation

* The kernel generating the data is the kernel the surrogate uses, up to
  $\alpha$ and the cliffs. Real misspecification is structural, not a mixing
  weight.
* **The synthetic surrogate is further from calibrated than the real one.**
  Measured the same way as §41, the synthetic GP has `|z| > 2` at **0.29** under
  both conditions, against 0.058–0.122 on JAK and CYP. That statistic is
  saturated here and cannot tell the two landscapes apart; excess kurtosis can —
  **0.23 without cliffs, 3.89 with**, a 17× increase, with max |z| going 7.0 to
  16.0. The cliff condition does what it was built to do, but it sits in a
  regime further from calibrated than the real blocks, so this is a study of the
  effect's *direction* rather than of its size. It is also the likeliest reason
  the two landscapes return a nearly identical Spearman: on the argmax
  statistic they are less different than the recall tables suggest.
* 14 quality levels gives a Spearman floor of $\approx 10^{-4}$, so $p = 0.043$
  is a genuine result rather than a floor artefact — but it is one test on one
  synthetic family.
* The argmax of a flat curve is a noisy statistic, which is why the fixed-
  compression response was tried; that one did not replicate, so the headline
  rests on the noisier of the two.

Artefacts: `results/sd_scale_sim.json` (pilot, seed 0),
`results/sd_scale_sim_confirm.json` (confirmatory, seed 500).

### Reproducing

```bash
python scripts/sd_scale_simulation.py --out results/sd_scale_sim.json
python scripts/sd_scale_simulation.py --seed 500 --replicates 8 --repeats 32 \
  --alphas "0.20,0.30,0.40,0.50,0.58,0.65,0.72,0.78,0.84,0.89,0.93,0.96,0.98,1.00" \
  --out results/sd_scale_sim_confirm.json
```

## 44. The decision rule, validated out of sample: ρ does not earn its place

Every threshold in this package has been **fitted**. §38 fits a zero crossing to
18 blocks and reports where it lands; that establishes the relationship exists.
It does not establish that a practitioner holding a new block and its coupling
ρ would do better by following it, because the crossing was fitted on the same
blocks it is quoted against.

This is the honest version. For each block in turn: fit on the other 17, apply
to the held-out one, record what was actually earned. And because §40 showed the
history design is not neutral, the whole thing is run **both ways**.

### The experiment

Three things to do with a free history, measured under **one protocol** so they
are comparable — same block, same budget (95 of ~430 compounds, 22%), same
history draw, 32 repeats × 4 held-out targets:

| option | what it is |
|---|---|
| **abstain** | single-task BO, random initial design. Ignore the history. |
| **seed** | single-task BO, initial design from the history's best-scoring compounds (`aux_top`). |
| **transfer** | coregionalised GP on the history, random initial design. |

crossed with the history design: **uniform** (the primary, §42) and
**withheld** (`--history-excludes-top`, how §38 measured its threshold). Seventy-two
cells, all audited clean (`scripts/run_xfam_options.py`).

The rule is the simplest thing that can use ρ: a linear fit of each option's
gain on the block's coupling, fitted on the training blocks, with the held-out
block assigned the option whose predicted gain is highest. Two parameters per
option on 17 blocks. Anything richer would be fitting the noise.

### Both uses pay, and the design decides which is better

AUC gain over abstaining, 18 blocks:

| design | option | mean | sem | positive | p |
|---|---|---|---|---|---|
| **uniform** | seed | **+0.1293** | 0.0225 | **17/18** | 0.0001 |
| | transfer | +0.1089 | 0.0276 | 12/18 | 0.0077 |
| | *seed − transfer* | *+0.0204* | *0.0124* | *13/18* | *0.054* |
| **withheld** | seed | +0.0350 | 0.0200 | 10/18 | 0.20 |
| | transfer | **+0.0494** | 0.0193 | 12/18 | 0.074 |
| | *seed − transfer* | *−0.0144* | — | *6/18* | *0.11* |

**The ordering flips with the design.** Under a uniform history, seeding wins;
under a withheld one, transfer wins. This is the third appearance of §40's
lesson, now inside the recommendation itself.

The reason is measurable and not mysterious. Under a uniform history the
30-compound `aux_top` pool contains **1.73 of the true top-10** on average,
against **0.70** expected at random — the auxiliary columns point at compounds
that are also potent on the primary. Under the withheld design that number is
**0.00 by construction**. Seeding loses most of its value when it cannot be
handed any of the answer (+0.129 → +0.035); transfer loses less
(+0.109 → +0.049). Neither number is wrong: the first is what a practitioner
with real history would see, the second is what survives when the design refuses
to hand over any of it.

### The rule does not beat the best fixed policy, in either design

| design | best fixed | rule (leave-one-block-out) | difference | better on | p | ceiling |
|---|---|---|---|---|---|---|
| **uniform** | always seed, +0.1293 | +0.1245 | **−0.0048 ± 0.0027** | **0/18** | 0.11 | +0.1421 |
| **withheld** | always transfer, +0.0494 | +0.0549 | **+0.0055 ± 0.0040** | 4/18 | 0.17 | +0.0660 |

Under the primary design the rule is **slightly worse** than simply always
seeding, and better on **no block at all**. Under the sensitivity design it is
slightly better, on four blocks of eighteen. Neither difference is significant.

**On these 18 blocks, choosing per block with ρ is not worth doing.** The honest
recommendation is the fixed policy: under a uniform history, seed; under a
withheld one, transfer.

### Why ρ cannot work here, and what could

This is not a failure of the fit. It is visible in the correlations.

| design | ρ(coupling, seed gain) | ρ(coupling, transfer gain) | ρ(coupling, **difference**) |
|---|---|---|---|
| uniform | **+0.849** (p < 10⁻⁴) | **+0.880** (p < 10⁻⁴) | −0.278 (p = 0.26) |
| withheld | **+0.897** (p < 10⁻⁴) | **+0.843** (p < 10⁻⁴) | −0.015 (p = 0.95) |

Coupling predicts how much a free history is worth — strongly, on both options,
under both designs. It does **not** predict *which option* to spend it on,
because it predicts both of them almost identically and their difference
therefore carries essentially no ρ signal. A threshold on ρ has nothing to
separate.

But the decision problem is real. The per-block ceiling — picking the best
option per block, knowing the answer — is **+0.1421 against the best fixed
policy's +0.1293** under the uniform design, and **+0.0660 against +0.0494**
under the withheld one. There is **10%** and **34%** of headroom respectively.
Something predicts the right choice; ρ is not it.

That is the opening for a richer feature, and it is the first place in this
package where one is justified by a measured gap rather than by assumption. The
candidates are the ones available before any paired measurement exists —
sequence identity, family membership, library tightness, where the history's
actives sit relative to the primary's chemistry. §36 established that paired
assay data, which is what ρ needs, does not exist outside a paralogue group;
a feature that works without it is what would make any of this deployable.

### What this adds up to

§38 said *where* the benefit of a free history crosses zero. §44 says the
threshold is **not actionable as a chooser**: knowing ρ tells you how much the
history is worth, not what to do with it. The two fixed policies are simple,
they are strong (17/18 and 12/18 blocks positive), and nothing fitted on ρ
improved on them out of sample.

This is also the answer to a question we had been treating as settled. The
project's standing line after §34 was *"build the rule where the relatives are
close; build the chooser where they are not."* Under the uniform design the
seeding rule is better **everywhere** — +0.0805 against +0.0538 below ρ = 0.28,
and +0.2269 against +0.2189 above it. The split does not survive.

Artefacts: `results/xfam_{transfer,seed}[_excl]/k01..k18.json` (72 cells, all
audited clean), `results/transfer_rule_{uniform,withheld}.json`.

### Reproducing

```bash
python scripts/run_xfam_options.py
python scripts/transfer_rule.py --out results/transfer_rule_uniform.json
python scripts/transfer_rule.py \
  --results results/xfam_transfer_excl --seed-results results/xfam_seed_excl \
  --out results/transfer_rule_withheld.json
```

### Correction to §44: the rule-vs-fixed p came from three blocks, not eighteen

The rule **picks the best fixed policy on 15 of 18 blocks** (12 of 18 under the
withheld design), so the paired difference is exactly zero there. `scipy`
discards zero pairs and, because zeros are present, also abandons its exact
null. The quoted `p = 0.1088` and `p = 0.1730` are **normal approximations over
3 and 6 pairs**. Correct, exact, on the discordant blocks only: **p = 0.25** and
**p = 0.2188**. The negative conclusion is unchanged and better supported --
the test was weaker than the block count suggested. See §46.

## 45. Advisor review: eight claims cut back, and one that changed when the unit changed

The advisor read the UAI draft and returned a list of places where the claims
outrun the evidence. Seven are wording — the claim was larger than what was
measured. One was not: re-running a test at the right unit **changed a result**.

### The one that changed

The paper quoted the paired change between history designs as *"+2.12 for `mt`
(p = 0.050) and −5.73 for `pooled` (p = 0.0034), Wilcoxon over the twelve
cells."* Twelve cells is six families at two budgets. Two budgets on the same
family share a block, a library, a censoring filter and a history draw — they
are not two independent units, exactly as §41 argued for campaigns inside a
cell. The argument was applied one level down and not here.

`scripts/block_level.py --against` re-states both contrasts at the family level:

| contrast | cell level (12) | **family level (6)** |
|---|---|---|
| `mt`: uniform − withheld | +2.12 ± 1.02, 9/12, p = 0.052 | **+2.12 ± 1.04, 4/6, p = 0.16** |
| `pooled`: uniform − withheld | −5.73 ± 1.56, 1/12, p = 0.0034 | **−5.73 ± 2.01, 0/6, p = 0.031** (the floor) |

**The `mt` improvement does not survive.** At the family level the uniform
design is not shown to help `mt` — four of six families, p = 0.16. The honest
statement is that it does not *cost* `mt` anything, which is all §42 needs: the
headline (`mt` +11.49 ± 3.25, 6/6 families, p = 0.031) is a level claim and is
untouched.

**The `pooled` decline survives, unanimously.** All six families move down; p =
0.031 is the floor at n = 6, so this says the direction was unanimous and
nothing about the size.

Per family, uniform − withheld:

| family | `mt` | `pooled` |
|---|---|---|
| cyp | +6.45 | −2.15 |
| pi3k | +3.16 | −1.21 |
| hdac | +2.07 | −4.69 |
| mmp | +1.87 | −7.46 |
| jak | −0.35 | −4.10 |
| dpp | −0.47 | −14.77 |

`--against` is the only code added: it pairs two sweeps cell by cell on
(block, budget) and runs the existing three-level aggregation on the
difference. Default behaviour is unchanged.

### The seven that were wording

| what the paper said | why it was too much | what it says now |
|---|---|---|
| matched uncertainty *"reproduces the whole effect"* on JAK (+2.34 vs +2.03) | two estimates of similar size, with SEMs of 0.7–0.9 on a 2-point effect, are consistent with a common explanation. They are not an equivalence, and no mediation test was run. | *"gives a gain of the same size"*, with an explicit paragraph saying what that does and does not license |
| a **"Not transfer: 27% / 60%"** column in the null table | the ratio of a randomized-data gain to a real-data gain is not an identified causal fraction — the two arms differ in more than the presence of information | column removed; the caption says why it is not quoted |
| `pooled` *"loses its advantage entirely"* | +3.16 ± 5.14 is a wide interval containing zero *and* most of the withheld-design value. Non-significance is not zero. | the estimate and its interval, plus the thing that *is* consistent across families: the decline |
| the simulation's ρ = −0.55, p = 0.04 in the abstract | the pre-registered statistic failed to replicate (§43); this one was selected after the pilot. Foregrounding it reads as confirmatory evidence. | out of the abstract; in the discussion as explicitly exploratory and directional; limitations item rewritten to name which statistic was pre-registered and which was not |
| `\mathcal D_q` left undefined, with a comment in the source admitting it | not reproducible | written out from `src/molbo/metrics.py:degrade_to_rho` — standardised Gaussian noise, `q·z + sqrt(1−q²)·ε` on the z-scored column, `q` exact in Pearson and *measured* in Spearman, redrawn per (target, repeat), applied before the history is indexed out |
| *"at r = 0 the model reduces to separate single-task GPs"* | §40's own diagnostic shows the shared hyperparameters move the primary posterior with `r` pinned to 0 (sd 0.643 → 0.558 on JAK). The paper contradicted itself two pages apart. | qualified: the covariance contribution vanishes, equivalence needs hyperparameters, preprocessing and noise matched too |
| *"Something predicts the right choice"* from the per-block ceiling | the ceiling is a hindsight maximum over three noisy estimates — biased upward, and not an upper bound on any learned chooser. Using a non-significant result to argue a learned method has headroom is the same error in the other direction. | *"best per block in hindsight"*, described as a reference over the options evaluated; the claim is narrowed to "ρ is not a usable chooser, and here is why", with the question left open |

### The structural change

The advisor's narrowing: *"How should transfer Bayesian optimization be
evaluated when auxiliary data alter both posterior fitting and the
candidate-selection environment?"* — controls, mechanism diagnostics and the
corrected cross-family evaluation are the main paper; the seeding-threshold
study becomes secondary.

Done: §§ *Initial-design seeding* and *Does the threshold help on a block it was
not fitted to?* moved into *Secondary and supporting analyses*, labels intact so
every cross-reference still resolves. A pointer paragraph replaces them in the
Results, and the one finding from that study that belongs in the main text — the
seeding pool holds 1.73 of the true top-10 under a uniform history and 0.00 when
the top-k is withheld, so seeding and transfer swap places between designs — was
lifted into §*The main comparison under both history designs* as the third
appearance of the design-leak lesson.

New title: **When Auxiliary Data Help Molecular Bayesian Optimization:
Transfer, Exploration, and History-Selection Bias.**

Abstract rewritten around the evaluation question, with the seeding study
reduced to one sentence and a pointer.

Also settled, since a reviewer would ask: every test in the paper is two-sided,
and every signed-rank p is exact (n ≤ 18 everywhere, inside scipy's enumeration
range). The stray source comment asking this has been replaced by the answer.

### What this does not change

The headline. `mt` +11.49 ± 3.25 recall points over `stl` under a uniform
history, positive on 6/6 families, p = 0.031 — a level claim at the family
level, which is the unit it was always tested at. §40's three mechanisms,
§41's two refutations, §43's exploratory simulation and §44's negative all
stand; what changed is how loudly each is stated.

### Reproducing

```bash
python scripts/block_level.py "results/winleak_*.json" --arm mt \
  --against "results/win32_*.json"
python scripts/block_level.py "results/winleak_*.json" --arm pooled \
  --against "results/win32_*.json"
```
### Correction to §45: two claims in it were wrong

§45's rewritten degradation paragraph claimed the mixture **preserves the
standard deviation** and gives **exact sample Pearson correlation q**. It does
neither -- the noise is standardised but not orthogonalised, so
`a = Corr(z, e)` is O(1/sqrt(n)). §45 also claimed every signed-rank p in the
paper is exact "since n <= 18"; sample size does not establish exactness, and
two tests in the paper are not exact. Both corrected in §46.


## 46. Advisor round two: a wrong equation I wrote, and a test that saw three blocks

§45 fixed the claims. In doing so it introduced a **mathematical error**, and
the second review caught it. Two of the six remaining items were also real
defects rather than wording. Recording both, because the pattern matters more
than either: **replacing a missing detail with confident prose makes a
manuscript look finished while making it less correct.**

### The error: `q` is a mixing parameter, not an achieved correlation

§45 wrote the degradation as

    y~ = s_y ( q·z + sqrt(1−q²)·e ) + ȳ,     z, e standardised

and claimed it **preserves the standard deviation** and gives **exact sample
Pearson correlation q**. Both false. The noise is standardised but **not
orthogonalised** against `z`, so `a = Corr(z, e)` is not zero. With
`q̄ = sqrt(1−q²)`:

    Var(y~)/s_y²  = 1 + 2·q·q̄·a
    Corr(y~, y)   = (q + q̄·a) / sqrt(1 + 2·q·q̄·a)

`scripts/degradation_audit.py` checks these against the implementation over 400
draws at each q and both block sizes: they agree to **8e-14** in correlation and
**2e-12** in variance ratio. The formulas are right; my description was wrong.

What it costs, **measured** (n = 335 / 433):

| q | Corr(z,e) = a | achieved Pearson | achieved Spearman | Var ratio |
|---|---|---|---|---|
| 0.0 | +0.002 ± 0.055 | +0.002 ± 0.055 | +0.002 ± 0.055 | 1.0000 |
| 0.5 | +0.002 ± 0.055 | +0.500 ± 0.036 | +0.482 ± 0.039 | 1.0014 |
| 0.8 | +0.002 ± 0.055 | +0.800 ± 0.012 | +0.784 ± 0.016 | 1.0016 |
| 1.0 | +0.002 ± 0.055 | +1.000 ± 0.000 | +1.000 ± 0.000 | 1.0000 |

So: **unbiased to ~0.002, but with draw-to-draw sd up to 0.055**; variance off by
at most 0.0018. Both correction terms carry the factor `q·q̄`, so **they vanish
at q = 0 and q = 1 — exactly where every control in the paper sits.** The
controls are not compromised. The *description* was. And the permutation null,
which preserves the marginal exactly, agrees with q = 0 — which is the
practical check that the slack does not matter here.

Stored `aux_quality_achieved` at q = 0: **+0.004 (JAK), −0.006 (CYP)**. The code
has always measured this. The manuscript quoted the request as if it were the
achieved value.

### The second defect: a test that saw three blocks, not eighteen

§44 reported the rule against the best fixed policy as *"−0.0048 ± 0.0027,
better on 0/18, p = 0.11"*. The rule **picks the fixed policy on 15 of the 18
blocks**, so the paired difference there is **exactly zero**. `scipy.wilcoxon`
discards zero pairs, and — because zeros are present — `method="auto"` also
drops its exact path and returns a **normal approximation over 3 pairs**.

| design | zeros | discordant | quoted | **correct (exact, discordant only)** |
|---|---|---|---|---|
| uniform | 15/18 | 3 (0 better, 3 worse) | p = 0.1088 | **p = 0.25** — the floor at n = 3 |
| withheld | 12/18 | 6 (4 better, 2 worse) | p = 0.1730 | **p = 0.2188** |

The conclusion is unchanged and in fact **safer**: the rule does not beat the
fixed policy, and the test had far less power than the block count suggested.
`transfer_rule.py` now computes and stores `rule_vs_best_fixed` with the zero
count, the discordant count, the exact p and scipy's default side by side.

**My §45 claim that "every signed-rank p is exact, since n ≤ 18" was wrong for
the same reason** — sample size does not establish exactness. `block_level.py`
`report()` now records zeros, ties, and the method. Checking every test in the
paper:

- **family-level tests (n = 6)**: 0 zeros, 0 ties, `auto == exact` verified
  against `method="exact"` directly. Exact, as claimed.
- **campaign level (n = 1536)**: **481 zeros, 1037 ties** — recall is a multiple
  of 1/k. Always was a normal approximation. Already not quoted for block
  claims; now it says why.
- **rule vs fixed**: above.

### The other four, which were wording

| item | fix |
|---|---|
| *"what they establish is that it does not cost `mt` anything"* | non-significance is not absence of harm. Now: **no clear difference detected**, 95% CI [−0.56, +4.80] |
| abstract called ±5.1 an "interval" | labelled **mean ± SEM over six families**, with a separate 95% CI [−10.1, +16.4] |
| parameter table said history always excludes top-k | now **uniform (primary)** for the grids and cross-family options; **withheld** as sensitivity and for the §res-null controls — read from the artefacts, not the launch scripts |
| appendix gate: *"a learned chooser has nothing to win"* | now a statement about the **resolution of the comparison**, not a demonstration |

### Counts resolved from the run records

The Methods said 8 seeds × 4 targets = 32; captions said 32 repeats × 4 targets.
**Both are true, of different experiments**, and the null table mixes them:

| artefacts | repeats | campaigns/cell | which |
|---|---|---|---|
| `auxq_*` | 8 | **32** | the q-ladder — tab:null rows 1–4 |
| `sdscale_*` | 32 | **128** | matched uncertainty — tab:null rows 5–6 |
| `win32_*`, `winleak_*` | 32 | **128** | the window grid |
| `xfam_*` | 32 | **128** | the cross-family options |

That is why the matched-uncertainty rows have visibly smaller standard errors
than the q = 0 rows they are compared against — worth knowing before reading
"+2.34 ± 0.73 against +2.03 ± 0.86" as two comparable estimates. Also confirmed
from the artefacts: `auxq_*` and `sdscale_*` both ran with
`history_excludes_top: true`, consistent with the table's caption.

### Reproducing

```bash
python scripts/degradation_audit.py --json results/degradation_audit.json
python scripts/transfer_rule.py --out results/transfer_rule_uniform.json
python scripts/transfer_rule.py \
  --results results/xfam_transfer_excl --seed-results results/xfam_seed_excl \
  --out results/transfer_rule_withheld.json
python scripts/block_level.py "results/winleak_*.json" --arm mt
```

### Standing instruction this leaves behind

**Do not write a formula into the paper that has not been run against the code
it describes.** Both errors in this round came from describing an operator or a
test from memory instead of measuring it. `degradation_audit.py` and the
zeros/ties fields exist so the next description can be checked rather than
believed. The advisor has still not read the experiment code; every number above
was verified against it here.
### Correction to §46: the campaign counts were corrected in the wrong direction

§46 said the null table's first four rows are 8 seeds x 4 targets, from
`auxq_*`. They are not. The table is built from `sdref_*` -- **32 seeds x 4
targets, 128 campaigns, every row**. The original caption was right; the
Methods' "eight seeds" never applied to this table. §46's achieved
correlations (+0.004 / -0.006) were `auxq`'s too; the right ones are
**-0.0008 (JAK)** and **-0.0022 (CYP)**. §46 also quoted the variance
ratio's *mean* deviation (0.0018) as though it bounded a draw; the per-draw sd
is **0.055** and the max deviation **0.17**. See §47.


## 47. Advisor round three: the evidence package, and the group I was citing from memory

Two defects again, and the second one is the more instructive: **my §46
"correction" to the campaign counts was itself wrong**, in the opposite
direction. Three rounds of prose produced three different statements of the
same count. The fix is not a fourth; it is that the counts are now **generated**
from the run records.

### Defect 1: I quoted a mean as if it were a maximum

§46 said the variance ratio "departs from 1 by at most 0.0018". The advisor's
algebra: the deviation is `2·q·√(1−q²)·a`, which at **q = 1/√2 equals `a`
exactly** — and I had reported sd(a) ≈ 0.055 two sentences earlier. The two
statements cannot describe the same draws.

They don't. **0.0018 is the deviation of the *mean* ratio** (a bias, driven by
mean(a) ≈ 0.002). The per-draw spread is a different statistic entirely:

| q | var ratio mean | **sd** | range | **max abs dev** |
|---|---|---|---|---|
| 0.00 | 1.0000 | 0.0000 | [1.0000, 1.0000] | **0** |
| 0.50 | 1.0014 | 0.0476 | [0.853, 1.133] | 0.147 |
| **0.70** | 1.0016 | **0.0549** | [0.830, 1.153] | **0.170** |
| 1.00 | 1.0000 | 0.0000 | [1.0000, 1.0000] | **0** |

At q = 0.70, **sd(var ratio) = 0.0549 = sd(a)**, exactly as the algebra says.
Quoting the mean as a bound understated the spread by **~105×**.
`degradation_audit.py` now prints mean, sd, range and max deviation, and says
in the header that only the spread describes what one campaign received.

Endpoint statement also corrected: at **q = 0 variance is preserved exactly but
the achieved correlation is `a`, not zero**; at **q = 1 both are exact**. The
paper previously ran the two endpoints together.

### Defect 2: the null table is built from `sdref_*`, which my inventory missed

§46 "resolved" the Methods-vs-caption conflict by declaring tab:null rows 1–4
to be 8 seeds × 4 targets, from `auxq_*`. **Wrong.** Regenerating the table from
artefacts produced +2.81/+7.19 — not the published +2.03/+7.66. Searching every
artefact for the published values found them in **`sdref_*`**: 32 seeds × 4
targets, **128 campaigns, all six rows**.

So the *original* caption was right, the Methods' "eight seeds" never applied
to this table, and my correction moved a right number to a wrong one.

**Why the inventory missed it:** I wrote `experiment_inventory.py` with a
**hand-written list of groups**, and `sdref` was not on it. An inventory that
can omit an experiment is not an inventory. It now **discovers groups by
scanning `results/`**, names each one's role (or `(not cited)`), and keeps an
explicit `SKIP` list so that excluding a group is a decision on the record.
Nine tests pin this, including the exact failure: a new prefix must be picked
up with nobody editing the script.

Corrected achieved correlations at q = 0, from the right artefacts: **−0.0008
(JAK), −0.0022 (CYP)** — not the +0.004/−0.006 of §46, which were `auxq`'s.

### The audits that were never enabled

The inventory, once it could see everything, found that **`sdref_*` and
`sdscale_*` ran without `--audit`** — `enabled: false`, which prints as
`clean: false`. These are the artefacts behind the null table *and* the
matched-uncertainty claim. Third time this mistake has appeared.

`rerun_audited.py` rebuilds each command line **from the artefact's own
recorded parameters** (not from a launch script, not from memory), re-runs with
the audit on into a separate directory, and compares every summary statistic.

**`sdscale_*`: 20/20 audited clean and reproduced the stored numbers exactly**
(184s).

**`sdref_*` did not reproduce -- and the fault was in my rebuilder.** The two
`q = 1` cells matched exactly; the two `q = 0` cells did not (CYP `mt` recall
0.610 stored against 0.712 re-run). Cause: `argv_for` never translated
`aux_quality_requested` back into `--aux-quality`, so the randomized cells
were re-run with **real auxiliary data**. `q = 1` is equivalent to omitting the
flag, which is why exactly half the cells passed -- and a half-failure like that
reads as a data problem rather than a rebuild problem.

The stored artefacts were never in question. Fixed, and the general guard added:
the rebuilder now **refuses** an artefact carrying any configuration key it
cannot translate, instead of letting the re-run take a default for it. Nine
tests pin this, including the `q = 1` masking case. After the fix,
`sdref_cyp_0.0` reproduces exactly (91s); the remaining cells are re-running.

This is the fourth time in this package that a silent omission -- a missing
`--audit`, a hand-written group list, a missing flag -- produced output that
looked fine. The pattern is always the same: **a default quietly standing in
for something that was never stated.** Every one of them was caught by a
comparison against stored results, not by reading the code.

Stored artefacts are not overwritten — the audited copies live in
`results/audited/`.

### The evidence package

What the advisor asked for, as three runnable scripts plus an appendix
(`app:repro`) that points at them:

| script | what it establishes |
|---|---|
| `degradation_audit.py` | the closed forms describe the implementation (8e-14 / 2e-12), and what the operator actually delivers, as mean **and** spread |
| `experiment_inventory.py` | every group's seeds/targets/campaigns/history design, discovered by scanning; fails on a protocol conflict; checks seeds × targets == stored array length |
| `regenerate_tables.py` | recomputes all three main tables from artefacts, naming the source group; prints zeros/ties/exactness for every signed-rank test; `--check` verifies the numbers appear in the manuscript |

`regenerate_tables.py --check report/paper.tex` now passes, and all eight
published null-control values reproduce to the stored precision.

### What the paper says now

- `q` is a mixing parameter; both closed forms given; bias and spread quoted
  separately; endpoints stated individually
- `e_j` called a **sample-standardized Gaussian draw** — after centering and
  scaling the entries are dependent, not iid standard normal
- the "an earlier version of this paper asserted…" confession is **out of the
  Methods**; the numerical checks are in the reproducibility appendix
- all main-text experiments: **32 seeds × 4 targets = 128 campaigns**; the
  8-seed ladder is the one appendix exception
- "which is why their standard errors are smaller" **removed** — both groups are
  128 campaigns; the SEMs differ because they are different contrasts

### Standing instruction, strengthened

§46 said: do not write a formula into the paper that has not been run against
the code. **Add: do not write a number into the paper that was not generated
from the run records.** Both defects this round were me citing from memory —
once an operator's properties, once an experiment group. The three scripts
above exist so that neither is possible again.

### Reproducing

```bash
python scripts/experiment_inventory.py --cited-only --tex report/inventory.tex
python scripts/degradation_audit.py --n 335 --json results/degradation_audit.json
python scripts/regenerate_tables.py --check report/paper.tex
python scripts/rerun_audited.py "results/sdref_*.json"
python scripts/rerun_audited.py "results/sdscale_*.json"
```

## 48. Preprocessing audited against the data, and two claims narrowed

The last item on the advisor's list — check preprocessing against the code —
plus a sweep of the paper before it goes to an external reviewer.

### Preprocessing: 19 of 20 claims verified exactly

`scripts/preprocessing_audit.py` recomputes every quantitative statement in
§Data from `data/bindingdb/*.csv`, prints it beside the published figure, and
exits nonzero on a mismatch.

| claim | paper | computed |
|---|---|---|
| CYP compounds before / after the filter | 1472 → 335 | **1472 → 335** |
| CYP survival | 22.8% | **22.76%** |
| CYP modal share per column | 21–27% | **20.9–27.1%** |
| CYP identical on all four targets | 40.3% | **40.29%** |
| JAK compounds before / after | 1563 → 904 | **1563 → 904** |
| JAK survival | 57.8% | **57.84%** |
| JAK identical on all four targets | 3.5% | **3.52%** |
| `kiba_prkc` n, ρ raw, ρ filtered | 904, 0.866, 0.991 | **904, 0.8663, 0.9908** |
| `kiba_cdk5` n, ρ raw, ρ filtered | 899, 0.505, 0.781 | **899, 0.5054, 0.7813** |
| `kiba_lck` n, ρ raw, ρ filtered | 843, 0.277, 0.482 | **843, 0.2774, 0.4816** |

**The potency transform and the censoring story check out completely.** The five
pIC50 values the paper names as screening-dose artefacts are exactly the fixed
doses it claims:

| dose | pIC50 = 9 − log₁₀(nM) |
|---|---|
| 5 µM | **5.30** |
| 10 µM | **5.00** |
| 30 µM | **4.52** |
| 50 µM | **4.30** |
| 100 µM | **4.00** |

All five appear in the paper's list of CYP spikes. That is a strong
confirmation that the filter is removing censored readouts and not real
potencies.

**The one mismatch:** the paper said the filter leaves "25–99 of ~900 rows" on
KIBA. Actual, on the three **pinned** blocks the sentence is about:

| block | raw | if filtered |
|---|---|---|
| `kiba_prkc` | 904 | **263** |
| `kiba_cdk5` | 899 | 59 |
| `kiba_lck` | 843 | 32 |

So **32–263**, not 25–99 — the quoted range excluded `kiba_prkc`, one of the
three blocks named in the same sentence, and understated the maximum by 2.7×.
(25–263 across all twelve labelled blocks; 25 and 99 were the min and max with
k01 dropped.) Note this **weakens** the paper's own argument slightly: 263 of
904 is less extreme than 25 of 900. Corrected.

### Two claims narrowed during the sweep

**1. The permutation null was run on one block, not "every cell".** The paper
said *"The permutation null agrees with q = 0 in every cell."* The only
permutation artefacts are `shuf_cyp_{excl,keep}` — **CYP only**, both designs,
three arms. Two numbers from one block were carrying a sentence about four.

Rewritten to say what was run, and it gains something: under the **uniform**
design the same permutation gives `pooled` **−19.69**, reproducing §40's sign
reversal with a null that preserves the marginal distribution *exactly*. That
is a better piece of evidence for the design-leak mechanism than the withheld
number, and it was sitting unused in the artefact.

Stated explicitly now: JAK was not run, so the agreement between the two nulls
rests on one block.

**2. Audits, fourth and fifth group.** `shuf_cyp_{excl,keep}` had also run
without `--audit`. Re-run from their own recorded parameters: **2/2 clean,
reproducing exactly.** Running totals: `sdscale` 20/20, `shuf` 2/2, `sdref`
3/4 so far, all clean and exact.

### The inventory now splits on history design

`shuf` holds both designs under one prefix, which the inventory flagged as a
protocol conflict. That is the right instinct but the wrong output — it is a
deliberate pairing, not a mistake. The inventory now **splits** a group that
holds both designs into two rows (`shuf [uniform]`, `shuf [withheld]`), each
internally consistent, and still **fails** on a real conflict such as two
different repeat counts sharing a name. A design split is not a failure; two
experiments sharing a name is.

### What was checked and found correct

Worth recording, since the last three rounds found something wrong every time:

- every headline number appears consistently across abstract, results,
  discussion and conclusion (10 quantities, 5 occurrences each at most, no
  disagreements)
- the DOCKSTRING/QED claims all trace to §14: sd ratio 2.02–2.65 → "2.0–2.7×";
  ρ(objective, QED) +0.929 → "+0.93"; ρ(objective, −JAK2) +0.007; QED-only
  top-50 recall 48.1%; best arm at 80 assays 50.6%; free share 95% → 33%
- the permutation-null figures −1.41 / +11.56 trace to `shuf_cyp_excl`
- JAK and CYP block provenance (`n_block`, `n_pool`, censoring, potency)
  matches the artefacts

### Reproducing

```bash
python scripts/preprocessing_audit.py --json results/preprocessing_audit.json
python scripts/experiment_inventory.py --cited-only --tex report/inventory.tex
python scripts/regenerate_tables.py --check report/paper.tex
python scripts/rerun_audited.py "results/shuf_cyp_*.json"
```
### Correction to §47: the two corrections do not share a factor

§47 said both the variance and the correlation deviation carry `q*qbar`.
Only the variance one does. The correlation deviation is `qbar^3 * a` to first
order: **`a` at q = 0**, zero only at q = 1. §47's next paragraph said this
correctly, so the text contradicted itself. Corrected and generated by
`degradation_audit.py`; see §49.


## 49. Advisor round four: the third maths error, and the package that replaces assertions

Three items were wording. The fourth was the advisor's real point, and it is
the one that changes how this gets reviewed: **"the reproducibility claims
still require evidence. I cannot independently confirm those checks from their
descriptions."**

### The error: the two corrections do not share a factor

§47 wrote *"Both deviations carry the factor q·q̄, which is zero at q = 0 and
q = 1."* True of the variance ratio, **false of the correlation**. Verified
against the exact form, at a = 0.03:

| q | var dev = 2q·q̄·a | corr dev ≈ q̄³·a | corr dev, exact |
|---|---|---|---|
| **0.00** | **+0.00000** | **+0.03000** | **+0.03000** |
| 0.50 | +0.02598 | +0.01949 | +0.01928 |
| 0.70 | +0.02999 | +0.01093 | +0.01084 |
| **1.00** | **+0.00000** | **+0.00000** | **+0.00000** |

The variance correction vanishes at **both** endpoints; the correlation
correction only at **q = 1**. At q = 0 it equals **a** — which §47's *next
paragraph* stated correctly, so the paper contradicted itself across two
sentences. First-order form within 2.3e-4 of exact.

Now generated by `degradation_audit.py` rather than written, with five
parametrised tests, per the §47 standing instruction. **Third maths error in
four rounds, and the third found by someone else reading the prose rather than
by me running the code.**

### The two other wording fixes

- **Noise independence contradicted itself.** Said entries are dependent after
  standardisation, then "the noise is independent across compounds". Now: *the
  underlying Gaussian draws are independent across compounds and targets;
  sample standardisation induces dependence within each column.*
- **Permutation null oversold.** "has no such slack" → *preserves the empirical
  marginal distribution exactly; its achieved correlation remains random.*
  Permutation fixes the marginal, not the correlation.

### The evidence package

`scripts/build_evidence.py` builds a directory and a 1.7 MB zip:

```
scripts/        67 files   the audit scripts and the campaign driver
src/molbo/      28 files   the library they import
results/       170 files   every artefact the manuscript cites, plus
                           results/audited/ (the re-runs) and their 26 logs
data/bindingdb/ 53 files   the block CSVs the preprocessing claims use
outputs/         7 files   captured stdout + exit status of each check
MANIFEST.sha256 335 files, 6.6 MB
```

**The outputs are produced by running the checks inside the package at build
time**, not transcribed from a session. The manifest is written last so it
covers them. **A failing check fails the build** — six tests pin that,
including a deliberately failing check.

Six checks run in-package, all exit 0: `preprocessing`, `inventory`,
`degradation`, `tables`, `block_level_mt`, `block_level_pooled`.

Omitted, and said so in the README: the 839 MB TDC BindingDB dump (public
download; the block CSVs cut from it are included), and the upstream DOCKSTRING
and KIBA libraries.

### Inventory terminology, and an identity that did not hold

"Campaigns/cell" collided with the Methods' use of *cell* for one
compound–target assay. Renamed **campaigns per configuration**, where a
configuration is one stored run.

While renaming, the first attempt printed `blocks × budgets × campaigns` as the
group total — and that **identity is false**: `win32` has 6 blocks and 12
distinct budget values but 12 configurations, not 72. One artefact is one
configuration, so the identity is now
**total = configurations × campaigns per configuration**, exact for every
group including the q-ladder, where a third axis (requested q) also multiplies
out. Total across all cited groups: **16,128 campaigns**.

### The audit column now counts re-runs

It read 0/4 for `sdref` because the originals are unaudited, contradicting the
appendix's "all pass". The inventory now looks for the audited twin in
`results/audited/` and counts an artefact as accounted for if it is clean in
place **or** on re-run. All eleven groups now fully accounted.

Final re-run tally: **26/26 clean and reproducing exactly** — `sdref` 4/4,
`sdscale` 20/20, `shuf` 2/2.

### The rebuild caveat is now in the paper

The appendix states the failure mode found in §47, because it is the kind a
reviewer should know happened: the first rebuild omitted `--aux-quality`, the
measured-panel cells still matched exactly (q = 1 ≡ omitting the flag), and
only the randomized half failed — which reads as a data problem, not a rebuild
problem.

### Reproducing

```bash
python scripts/build_evidence.py --zip
```

## 50. The validation project: eligibility, the scale protocol, and generated tables

Five items, worked in the advisor's priority order. The headline: **there is no
exclusion bug**, and the mechanism is now measured per compound rather than
argued.

### Item 3 — where the history actually affects selection

**The candidate set was never in question once read.** In `run_campaign`:

```python
cand = np.setdiff1d(np.arange(n_pool), obs)   # obs = PRIMARY-observed only
```

A compound measured only on an auxiliary target stays eligible. `eligible=` is
a separate opt-in used by the selectivity and QED experiments, not by any
campaign in the paper.

`scripts/selection_diagnosis.py` instruments the loop (new optional `trace`
callback on `run_campaign`, default `None`, byte-identical when unused) and
records per round: eligible IDs, primary-observed IDs, history IDs, μ/σ/acq
split by history membership, and selected IDs.

**Confirmed, both blocks:** at round 0, **148 of 894 eligible compounds (JAK)**
and **146 of 325 (CYP)** are history members, in every arm and both designs.

**The mechanism is acquisition ranking, and it is graded by arm:**

| arm | JAK base rate | picked | **ratio** | CYP base rate | picked | **ratio** |
|---|---|---|---|---|---|---|
| `stl` | 16.6% | 16.6% | **1.00** | 45.2% | 44.0% | **0.97** |
| `mt` | 17.8% | 9.9% | **0.56** | 47.3% | 25.6% | **0.54** |
| `pooled` | 19.1% | 3.0% | **0.16** | 49.8% | 14.4% | **0.29** |
| `random` | 16.6% | 16.6% | **1.00** | 45.0% | 45.6% | **1.01** |

Nothing tells any arm to avoid the history. `stl` and `random` are indifferent;
`pooled` picks from it at **a sixth to a third** of the base rate.

**The mean acquisition value is the wrong statistic and nearly misled me.** On
CYP, `pooled`'s *mean* acq over the history is **higher** than over the rest
(ratio 1.188) while it picks from the history at 0.29× the base rate. Selection
reads the **top of the ranking**, not the mean. The selection ratio is the
statistic; the mean is not.

### Item 3, second half — the advisor's random-outside-history diagnostic

Crossing the selection rule with the **eligibility** rule, which are separate
axes:

| condition | JAK: uniform → withheld | CYP: uniform → withheld |
|---|---|---|
| `eligible_all` / stl | 77.8% → 77.8% (**+0.0**) | 66.6% → 66.6% (**+0.0**) |
| `eligible_all` / mt | 86.2% → 84.7% (−1.6) | 74.1% → 70.9% (−3.1) |
| `eligible_all` / pooled | 77.2% → 81.2% (**+4.1**) | 70.9% → 79.7% (**+8.8**) |
| `eligible_all` / random | 32.8% → 32.8% (+0.0) | 36.6% → 36.6% (+0.0) |
| `exclude_hist` / stl | 70.6% → 83.4% (+12.8) | 45.0% → 84.4% (+39.4) |
| `exclude_hist` / **random** | 30.6% → 39.1% (**+8.4**) | 30.9% → 62.2% (**+31.2**) |

**Random selection from outside the history gains 8 to 31 points from the
design, with no model at all.** So the design advantage is a *mathematical
property of the evaluation*, available to anything that avoids the history —
the eligible-in-top-k share rises 2.9% → 5.4% on CYP when the top-k is
withheld and the history removed.

What the paper's campaigns actually do (`eligible_all`) shows the effect **only
for `pooled`**, and the ratio table says why: it is the only arm that avoids
the history hard enough to approximate `exclude_hist` without being told to.

**Conclusion for the paper:** the benefit comes from **acquisition
suppression**, not eligibility. There is no bug to fix. But the search problem
`pooled` ends up solving is close to "sample outside the history", and under a
design that puts the answer there, that is worth up to 31 points for free.

### Item 2 — the scale-matched baseline, specified and validated

`scripts/scale_protocol.py` fixes every degree of freedom the advisor listed:

| degree of freedom | choice |
|---|---|
| numerator model | **fitted** multi-task GP (`r` by exact LML), not pooled |
| auxiliary values | **randomized** (q = 0) — the scale the randomized arm has |
| observation count | **n_obs = 50**, below the smallest grid budget |
| candidate set C | not primary-observed and not in the history |
| scale | standardized, both models |
| fixed or per-campaign | **fixed**, one per block |
| seeds | offset 900,000 — **disjoint from every campaign** |

Measured on that independent stream: **c = 0.807 ± 0.020 (JAK)**, **0.797 ±
0.020 (CYP)**. Reproduces the 0.80 used, without any campaign outcome.

**The ratio is not constant** — the curve the manuscript should show:

| n_obs | 25 | 50 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|
| JAK | 0.815 | **0.807** | 0.858 | 0.985 | 1.059 | 1.099 |
| CYP | 0.827 | **0.797** | 0.828 | 0.884 | — | — |

It crosses 1 between 100 and 150 observations on JAK. Hence **"early posterior-
scale-matched baseline"**.

**The direct comparison the paper never made** (it only tested each against
`stl`):

| block | scale-matched | randomized-auxiliary | difference | 95% CI | p |
|---|---|---|---|---|---|
| JAK | +2.34 | +2.03 | **+0.31 ± 0.81** | [−1.28, +1.91] | 0.78 |
| CYP | −0.23 | −0.47 | **+0.23 ± 1.62** | [−2.96, +3.43] | 0.70 |

Not distinguishable. **Consistency, not equivalence, and no mediation claim.**

### Item 1, partial — the four conditions

`null_mechanism.py` already ran the exact four conditions. JAK, n_obs = 100,
mean posterior sd and EI-ranking agreement with `stl`:

| condition | mean sd | EI order agreement |
|---|---|---|
| `stl` | 0.6428 | 1.000 |
| `r=0`, hyperparameters **frozen** | **0.6428** | **1.000** |
| `r=0`, signal variance released | 1.2045 | 0.181 |
| `r=0`, **refitted** | 0.5582 | 0.081 |
| `r` fitted | 0.5549 | 0.078 |

**The mathematical control holds exactly** (frozen is identical to `stl`), and
**almost all of the compression arrives with `r = 0`** — i.e. from shared
hyperparameter refitting, not from cross-target covariance. Fitting `r` adds
0.003.

Discovery performance for the same decomposition, new `mt_r0` arm (CYP,
randomized auxiliary, 128 campaigns):

| arm | recall |
|---|---|
| `stl` | 61.5% |
| `mt` | 61.0% |
| **`mt_r0`** | **61.6%** |
| `pooled` | **72.0%** |

**JAK is the decisive one.** Same arm, same randomized auxiliary, 128
campaigns:

| arm | JAK | CYP |
|---|---|---|
| `mt_r0` - `stl` | **+2.03 +- 0.83** (p = 0.016) | +0.08 +- 1.25 (p = 0.95) |
| `mt` - `stl` | **+2.03 +- 0.86** (p = 0.020) | -0.47 +- 1.55 (p = 0.76) |
| `pooled` - `stl` | +0.55 +- 0.94 (p = 0.56) | **+10.55 +- 1.85** (p = 8e-8) |

**On JAK, forcing the task covariance to zero leaves the gain unchanged:
+2.03 against +2.03.** The entire randomized-auxiliary effect on that block is
refitting shared hyperparameters on more rows. Cross-target covariance
contributes nothing measurable.

**On CYP nothing survives at r = 0** (+0.08), and the gain is `pooled`'s alone.

So one arm separates the two blocks into two different mechanisms, and
**neither of them is transfer**. This is the campaign-outcome confirmation of
what the posterior diagnostic showed: the compression arrives at r = 0.

As a cross-check the `mt` and `pooled` columns reproduce the stored
null-control artefacts exactly (+2.031, +0.547, -0.469, +10.547), from a
separate run with a different arm list.

### Items 4 and 5 — making it reviewable

**Item 5.** `scripts/summaries.py` implements
`saved outcomes → structured summaries → LaTeX`. 18 records, each carrying
experiment, contrast, history design, budget, metric, aggregation unit, n,
estimate, SEM, CI, test, p, exactness, and source glob. It emits
`report/generated/macros.tex` (108 macros) and `table_design.tex`. The abstract
and the tables now render from the **same record**; rounding happens only at
render time.

The old text-search check is replaced: `--check` verifies every generated macro
used is defined, **and** that no value a macro renders also appears typed by
hand. Regions where a literal legitimately coincides — the per-configuration
grid contains the same CYP b111 and JAK b300 runs the null table summarises —
are marked `% summaries-exempt:` with the reason in the source. Three such
regions, each named.

**Item 4.** The audit appendix is now a four-column table (check, input,
verification, reported output), plus environment versions, commands, and the
level-of-reproducibility statement: **the package regenerates tables from
stored outcomes; it does not re-run campaigns**, except the audited-rerun check
which re-executes 26 of them. All drafting history removed.

### Reproducing

```bash
python scripts/selection_diagnosis.py --family cyp --budget 111 --repeats 8
python scripts/selection_diagnosis.py --family jak --budget 300 --repeats 8
python scripts/scale_protocol.py --family jak
python scripts/summaries.py --tex-dir report/generated --check report/paper.tex
```
