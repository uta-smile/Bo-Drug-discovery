# Questions for you — ANSWERED 2026-10-08

> **Your answers:** target **UAI**, **no LLM experiment**, and take my
> recommended default on everything else. Each heading below now carries the
> decision and what I did with it overnight. The original reasoning is kept
> underneath so you can overturn any of them on a second read.

## Decisions, at a glance

| # | decision | status |
|---|---|---|
| A1 | **UAI.** Write so the ICML material is a clean extension. | spine fixed |
| A2 | Edit both `.tex` and `FINDINGS.md`, additive only. | standing |
| B1 | Re-run **(a)** the §34 headline grid both ways. | **done, §42 — `mt` improves, `pooled` loses its result** |
| B2 | `pooled` demoted to a cautionary control; out of headline tables. | §42 |
| B3 | **Uniform history becomes the primary design.** | §42 |
| C1 | Build the controlled simulation of when σ compression pays. | **done, §43** |
| C2 | Decision rule, form (i), validated out-of-sample. | **done, §44 — it is a negative** |
| C3 | Restate at block level everywhere. | **done for both grids, §41-§42** |
| D1 | **No LLM in this paper.** Sequence-identity baseline still built — it is not an LLM and C2(ii) needs it. | deferred |
| E1 | Blinded time-split if the night allows. | **deferred — the dumps carry no date column; see TODO** |
| E2 | Nothing frozen beyond the standing constraint. | standing |

**Two of these change published numbers and you should check them first:**
B3 (every headline gain is now measured under a uniform history) and C3 (every
cross-block p-value is now a block-level one, and several sit at their floor).

---

## The original questions, for reference

Written 2026-10-08, before the answers above. Kept so the reasoning behind each
default is on the record and any of them can be overturned.

---

## A. Venue and deadline

### A1. Which venue are we actually aiming at, and when is the deadline?

> **ANSWERED: UAI.** No deadline given beyond "by tomorrow morning" for this round of work, so I scoped the night to the UAI spine and left the 18-block sweep (B1b) unstarted.


The advisor recommends UAI first, ICML only with a broader contribution. That
changes the shape of the paper substantially.

| | UAI shape | ICML shape |
|---|---|---|
| spine | "when does auxiliary data help, and when is it uncertainty?" | "three mechanisms masquerade as transfer; here is a control for each" |
| evidence needed | the two blocks + the ladder + the 18-block scan | the above **plus** the controls run across all 18 blocks and a validated decision rule |
| what is missing | an analytical/simulation account of the sd mechanism | that, plus out-of-sample validation of the rule |
| extra compute | ~1 day | ~1 week |

**My default:** UAI shape, written so the ICML material is a clean extension
rather than a rewrite. But I need the deadline to know whether the 18-block
control sweep is affordable.

**Your answer:**

---

### A2. Is there an author list / affiliation constraint that affects scope?

> **ANSWERED: default.** Editing both, additive only.


Specifically: is anyone else running experiments on this repo in parallel? I
found `report/paper.tex` had been rewritten outside this session once already,
and I reconciled it. If that is going to keep happening I should stop editing
the `.tex` and write into `results/FINDINGS.md` only.

**My default:** keep editing both, keep `check_tex.py` clean, never rewrite a
section someone else wrote — only add.

**Your answer:**

---

## B. The thing §40 just broke

§40 found that `--history-excludes-top` is **not a neutral control**. It removes
the §22 confound (a history that happens to contain the answer) and introduces
its mirror image (an arm that avoids the history walks toward an answer the
protocol guarantees is outside it). Both effects are large; they point in
opposite directions; neither setting is honest on its own.

122 stored artefacts carry a `pooled` arm under this flag, at coverages from 14%
to 90%. `scripts/leak_audit.py` lists them.

### B1. How much of the back catalogue do we re-run?

> **ANSWERED: (a).** The 12-cell §34 grid, both ways. Done -- see §42.


| option | cost | what it buys |
|---|---|---|
| **(a)** only the §34 headline grid (12 cells), both ways | running now, ~2–4 h | the headline becomes defensible |
| (b) (a) plus the 18 cross-family blocks | ~1–2 days | the threshold result becomes defensible too |
| (c) everything with a `pooled` arm | ~1 week | completeness we probably do not need |

**My default:** (a), already launched. Then (b) only if the §34 numbers move
enough to matter.

**Your answer:**

---

### B2. Do we drop `pooled` from the paper entirely?

> **ANSWERED: default.** Cautionary control in the methods section, both flag settings shown, out of every headline table.


On both blocks tested, `pooled`'s gain is substantially a function of this flag
— including its **real-panel** gain on JAK (+5.70 → +0.78, n.s.). §31 was already
retracted. Keeping `pooled` means a paragraph explaining why a retracted,
protocol-sensitive arm is still in the tables.

**My default:** keep it **only** as a cautionary control in the methods section,
with both flag settings shown, and remove it from every headline table.

**Your answer:**

---

### B3. Which history design becomes the primary one?

> **ANSWERED: uniform.** This is the one with the widest blast radius -- every headline number in the paper is now measured under a uniform history. §42 reports what moved.


Three candidates, and the paper has to pick one and report the others as
sensitivity:

1. **Uniform history** — draw from the whole library. P(top-k in history) equals
   coverage, same as any compound, so membership is uninformative. Clean against
   the §40 leak. §22 called it "optimistic" because real history is not a uniform
   sample of the library you are screening now.
2. **Top-k withheld** (`--history-excludes-top`, the current default) — clean
   against the §22 confound, carries the §40 leak.
3. **Clustered history** (`--history-cluster`) — closer to how real history is
   generated (one team, one chemotype), already implemented, never used as the
   primary.

**My default:** **uniform** as primary, because the §40 leak is a *correctness*
problem and the §22 optimism is a *generalisation* problem that
`--history-cluster` and the reach ratio already address separately. This
reverses the project's current default, so I want your sign-off.

**Your answer:**

---

## C. What the advisor asked for that we have not done

### C1. The analytical / controlled-simulation account — worth building?

> **ANSWERED: build it.** §43.


The advisor: *"A small analytical example or controlled simulation could make the
explanation much stronger."* We now have the empirical version (sd × 0.80
reproduces the noise gain on JAK). What is missing is a derivation: why an
over-exploratory EI on a Tanimoto GP over a finite library gains from mild sd
compression, and why the optimum sits near 0.8 rather than 0.5 or 1.0.

Cost: 1–2 days, mostly writing, plus a small synthetic study where the ground
truth is known.

**My default:** build it. It is the difference between "we measured an artefact"
and "we explain a failure mode of transfer-BO evaluation", and it is the cheapest
remaining step toward the ICML shape.

**Your answer:**

---

### C2. The decision rule, validated out-of-sample — which form?

> **ANSWERED: (i) first.** §44. Form (ii) needs the sequence-identity baseline, which is built but not yet used as a predictor.


The advisor wants: *"a rule for when to transfer, when to use history only for
initialization, and when to abstain."* We have the ingredients (the ρ threshold
conditional on H, the headroom gate, the bad-start result) but have never fit the
rule on some blocks and tested it on held-out ones.

The 18 cross-family blocks make this a finite experiment: leave-k-blocks-out,
measure regret against always-transfer / never-transfer / an oracle that knows
the right answer per block.

Two forms the rule could take:

- **(i) Threshold on measured ρ.** Simple, already have the number, but ρ is
  only measurable when paired assay data already exists — which §36 found is
  rare outside a paralogue group.
- **(ii) Threshold on a *predicted* ρ** from features available before any
  paired data: sequence identity, family membership, library tightness. Harder,
  but it is the only version that is deployable, and it is exactly the slot the
  advisor says an LLM would have to earn its way into.

**My default:** build (i) first because it is two days and validates the
machinery, then (ii) with sequence identity as the baseline predictor.

**Your answer:**

---

### C3. Statistical units — do we restate everything at block level?

> **ANSWERED: do it.** Started in §41 for the §34 grid, extended in §42.


The advisor flagged *"uncertainty estimates that reflect the experimental
units"*. Our per-block p-values are paired Wilcoxon over campaigns (32 repeats ×
4 targets = 128), which is right *within* a block but wrong for any claim about
transfer in general. The cross-family scan already uses blocks as units (18
blocks, correct). §§18–35 do not.

Restating means: every cross-block claim gets a block-level test, and the
within-block numbers stay but are labelled as within-block.

Cost: ~1 day, no new runs — it is a re-analysis of stored artefacts.

**My default:** do it. It is cheap, it is correct, and a reviewer will ask.

**Your answer:**

---

## D. The LLM question

### D1. Do we attempt the LLM experiment for this paper, or defer it?

> **ANSWERED: defer, no LLM.** The sequence-identity baseline is still worth having and is not an LLM.


The advisor's framing is the one that fits our data: *"Can protein descriptions
or literature-derived information help choose an auxiliary target when paired
assay measurements are scarce?"* That is §36's gap exactly — no co-measured
auxiliary exists outside a paralogue group, so ρ is unmeasurable precisely when
you need it.

Three honest problems:

1. **The baseline may win.** Kinase sequence identity predicts relatedness well.
   If it matches the LLM, the LLM contribution has not earned its place. I would
   publish that result, but it is not an LLM paper.
2. **Leakage is severe.** KIBA, Davis and BindingDB are in every modern LLM's
   training data. Preventing held-out assay outcomes from entering prompts is
   not a formality here; it may not be achievable with a public panel at all.
3. **It is a separate paper.** The advisor says so, and I agree.

**My default:** defer. Build the sequence-identity baseline *now* (it is cheap
and it is needed for C2(ii) anyway), and keep the LLM out of this paper.

**Your answer:**

---

## E. Scope boundaries I want confirmed

### E1. Is the wet-lab / prospective direction in scope at all for this paper?

> **ANSWERED: default.** Blinded time-split only, and only if the night allowed it after the UAI spine.


TODO §6 lists a blinded time-split (free, attacks the retrospective-optimism
critique directly), a Boltz-2 compute oracle, and a wet-lab validation. The
advisor's note — *"retrospective assay improvements support better retrospective
selection"* — makes the time-split look like the single highest-value cheap
addition.

**My default:** do the **blinded time-split** for this paper (it is free and it
directly answers the proportionality critique), and leave the compute oracle and
wet lab out.

**Your answer:**

---

### E2. Anything in the repo I should treat as frozen?

> **ANSWERED: nothing beyond the standing constraint.**


The standing constraint is that defaults reproduce stored results byte-identically
and stored artefacts are never rewritten. I have held to that: every flag added
this session (`--sd-scale`, `--aux-shuffle`, `excludes_top` in the driver) is a
no-op by default and verified against a stored artefact.

Is there anything beyond that — a file, a result, a figure — that should not be
touched?

**Your answer:**

---

## Advisor round four (S49) -- the package is the deliverable

The advisor stopped accepting assertions that checks pass and asked for
something runnable. `python scripts/build_evidence.py --zip` now produces a
**1.7 MB package**: the scripts, the library, all 170 cited artefacts, the 53
block CSVs, the captured output and exit status of six checks, and a sha256
manifest over 335 files. The outputs are generated by running the checks
*inside* the package; a failing check fails the build.

**Send `evidence.zip` with `paper.tex`.** The paper also needs
`report/inventory.tex` and `report/refs.bib`.

Also: a third maths error (the correlation and variance corrections do not
share a factor -- at q = 0 the achieved correlation is `a`, not zero), two
wording contradictions, and an inventory identity that did not hold. Nothing in
the headline moved.

## Advisor round three (S47)

Two defects, and the embarrassing one is that **S46's own correction to the
campaign counts was wrong**. The null table is built from `sdref_*`: 32 seeds x
4 targets, **128 campaigns, every row**. The original caption was right. I had
"fixed" it by reading a different experiment group, because the inventory I
wrote to settle exactly this question was built from a hand-written list that
omitted the group in question.

Also: I quoted the variance ratio's *mean* deviation (0.0018) as if it bounded
a single draw. The per-draw sd is **0.055**, max **0.17**.

Both are now **generated, not written**: three scripts regenerate the tables,
the inventory, and the operator's measured properties, and the manuscript is
checked against them. `app:repro` in the paper points at all three. Nothing in
the headline moved.

Also found: `sdref_*` and `sdscale_*` had run **without `--audit`** -- the
artefacts behind the null table and the matched-uncertainty claim. Re-run with
the audit on, from their own recorded parameters: clean, and reproducing the
stored numbers exactly.

## Advisor round two (S46)

The second review found **two defects, not wording**, both introduced or
repeated by my own revision:

1. The noise-mixture equation I wrote into the Methods was **wrong** -- `q` is a
   mixing parameter, not an exact sample correlation, and the standard deviation
   is not preserved. Corrected, verified against the code to 1e-13, and the cost
   measured. The controls sit at q = 0 and q = 1 where the construction *is*
   exact, so no result moves.
2. The rule-vs-fixed significance test was **reading 3 blocks while reporting
   18**, because the rule picks the fixed policy on the other 15 and scipy drops
   zero pairs. Correct exact p: **0.25** and **0.22**, not 0.11 and 0.17. The
   negative conclusion holds and is better supported.

Nothing in the headline changed. What changed is that two statements I had
written confidently were not true, and both came from describing code from
memory rather than running it. See FINDINGS S46.

## The advisor review is applied (S45)

Eight corrections, all in `report/paper.tex`. Seven were wording. **One changed
a number**: the between-design change was tested over twelve budget cells, and
at the family level `mt`'s improvement does not survive (+2.12 +- 1.04, 4/6,
p = 0.16) while `pooled`'s decline does (-5.73 +- 2.01, 0/6, p = 0.031). The
headline is a level claim and is unaffected.

The paper is narrowed to the evaluation question and retitled; the
seeding-threshold study is now a secondary section. See FINDINGS S45.

## What ran overnight

`results/FINDINGS.md` §§40-44, and `report/paper.tex` carries all of it.

| § | what |
|---|---|
| 40 | the three mechanisms separated; the design leak found |
| 41 | both mechanism explanations refuted; the gain accounted for compound by compound |
| 42 | the headline grid under both history designs; uniform becomes primary |
| 43 | controlled simulation of the exploration knob |
| 44 | the decision rule, validated out of sample — a negative |

**The one thing to decide when you read this:** §44 shows the per-block choice
between seeding and transfer is worth 10–34% of the remaining headroom, and that
the task correlation cannot make it. That is the concrete information gap the
advisor's LLM question was asking for, and it now has a number attached. It is
still not an LLM question first — the baseline to build is sequence identity.
