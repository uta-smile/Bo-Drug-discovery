# Crossed auxiliary-response and fitting diagnostic, version 1

This experiment extends C2 and supplies a historical-response-preserving C1 null.
It is a diagnostic evaluation, not a deployable policy or a causal partition of
biological mechanisms. Existing results outside this directory are unchanged.

## Protocol fixed before full results

Six BindingDB panels: JAK (budgets 200, 300), CYP (74, 111), DPP (66, 99),
HDAC (83, 125), MMP (55, 82), and PI3K (130, 195). Each panel has four held-out
primary targets and 32 repeats per target. All policies receive the same ten
initial primary observations and 50 historical observations per auxiliary
target. Histories are uniform and do not exclude primary hits. EI uses the
existing 0.01 offset and batches of ten, truncating at the budget.

The larger-budget trajectory supplies its smaller-budget prefix; this is
exactly equivalent for the deterministic batch policy, and both budgets share
the history and initial design. The original ST and measured MT recall vectors
are checked independently against the existing stored campaigns at both budgets.

M means measured responses; P means permutation of only the 50 actually observed
responses within each auxiliary target. Permutations are fixed throughout each
campaign. The first letter identifies the parameter donor; the second identifies
the conditioning responses. MM and PP are the usual refitted measured and
permuted models; MP and PM are locked-parameter reconditionings. Every donor
is re-estimated using only the primary observations available on that policy's
current trajectory. Coupling, signal variance, noise, all normalization
transforms, and jitter remain fixed within a reconditioning operation.

Predictive variance is mathematically independent of responses at fixed
locations, parameters, and transforms. The implementation reuses the donor's
Cholesky factor and recomputes only the response coefficients. Measured-response
parameters still carry measured-response information: MP is not a value-free
policy. Sequential contrasts also incorporate trajectory feedback.

## Endpoints and aggregation

Primary endpoint: final Recall@10. Primary comparison: MM minus PP, in percentage
points. Average repeats and targets within each budget, then the two budget
effects within each family. Use six family effects for the two-sided Wilcoxon
comparison; the theoretical minimum two-sided p-value is 0.03125. Report the
family vector, positive-family count, mean, SEM, and Student-t 95% interval.

The four conditional response/fitting contrasts and their interaction are
predefined descriptive comparisons. They are not selected for significance
and do not yield causal percentages. Matched-history replays score proposed
batches on shared ST and MM trajectories; average rounds within each
campaign, budgets within each campaign, and then campaigns within each family.
These one-step outcomes must not be presented as a full sequential campaign's
attribution. Candidate outcomes enter only retrospective scoring.

## Reproduction

From the repository root, using the project's Python environment:

    python scripts/crossed_auxiliary.py --stage smoke --repeats 2 --workers 4
    python scripts/summarize_crossed_auxiliary.py --stage smoke
    python scripts/run_crossed_parallel.py --workers 4
    python scripts/summarize_crossed_auxiliary.py --stage full --emit-tex

The parallel runner distributes independent checkpoints; the scientific
calculation is shared with the original runner. Native RDKit access is required.
Smoke runs have extra donor-equality and variance-invariance checks at every
visited state. Full checkpoints preserve selected indices, per-budget recall,
fixed history indices and permutation source indices, donor parameter traces,
replay metrics, input/code hashes, and audit verdicts. Resume refuses mismatched
provenance. The independent summarizer reconstructs outcomes from source panels
and compound selections before writing any manuscript macros or tables.
