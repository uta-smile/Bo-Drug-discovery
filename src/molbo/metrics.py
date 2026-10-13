"""Scoring curves and the PMO AUC top-k metric.

The headline number is the area under the "average of the best k molecules
found so far" curve plotted against *cumulative unique oracle calls*, which
rewards finding good molecules early rather than eventually.  :func:`top_auc`
reproduces the reference implementation from the PMO benchmark so numbers are
comparable to its published baselines; :func:`auc_top_k` is the same quantity
computed exactly (every call, not every hundredth).

One caveat that matters for the ChEMBL tier: PMO scores live in ``[0, 1]`` and
its AUC integrates from an implicit zero at call 0.  Raw pIC50 values do not,
so feeding them in unscaled produces a number dominated by the offset.  Use
:func:`normalize_values` first -- see its docstring.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr

Buffer = Mapping[str, tuple[float, int]]
"""``{smiles: (score, call_index)}`` -- what ``Oracle.buffer()`` returns."""


def top_auc(
    buffer: Buffer,
    top_n: int = 10,
    finish: bool = True,
    freq_log: int = 100,
    max_oracle_calls: int = 10_000,
) -> float:
    """AUC of the top-``top_n`` mean versus oracle calls, PMO-style.

    A faithful port of PMO's ``top_auc``, kept bug-for-bug so results line up
    with its 25 published baselines:

    * the trapezoid starts from ``prev = 0`` at call 0, so scores are assumed
      non-negative and roughly ``[0, 1]``-scaled;
    * the curve is sampled every ``freq_log`` calls, not every call;
    * with ``finish=True`` a run that stopped early is extended flat to
      ``max_oracle_calls``, which is what makes runs of different length
      comparable;
    * the result is divided by ``max_oracle_calls``, so a perfect method that
      instantly finds ``top_n`` molecules scoring 1.0 gets 1.0.

    Parameters
    ----------
    buffer:
        ``{smiles: (score, call_index)}`` with 1-based call indices.
    max_oracle_calls:
        The budget the curve is normalised against -- 1000 for the restricted
        setting this project uses, 10000 for PMO's original tables. Report
        which one you used; the numbers are not interchangeable.
    """
    if not buffer:
        return 0.0
    total = 0.0
    prev = 0.0
    called = 0
    ordered = sorted(buffer.items(), key=lambda kv: kv[1][1])

    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        window = sorted(ordered[:idx], key=lambda kv: kv[1][0], reverse=True)[:top_n]
        top_now = float(np.mean([v[0] for _, v in window]))
        total += freq_log * (top_now + prev) / 2
        prev = top_now
        called = idx

    window = sorted(ordered, key=lambda kv: kv[1][0], reverse=True)[:top_n]
    top_now = float(np.mean([v[0] for _, v in window]))
    total += (len(buffer) - called) * (top_now + prev) / 2

    if finish and len(buffer) < max_oracle_calls:
        total += (max_oracle_calls - len(buffer)) * top_now

    return total / max_oracle_calls


def top_k_curve(
    scores: Sequence[float],
    k: int = 10,
    budget: int | None = None,
    pad: bool = True,
) -> np.ndarray:
    """Running mean of the best ``k`` scores after each call.

    ``scores`` must be in call order. Before ``k`` molecules have been seen the
    mean is taken over however many exist, matching PMO. With ``pad`` the curve
    is extended flat to ``budget`` so runs that stopped early stay comparable.
    """
    scores = np.asarray([s for s in scores], dtype=float)
    if scores.size == 0:
        return np.zeros(budget or 0, dtype=float)

    curve = np.empty(scores.size, dtype=float)
    best: list[float] = []
    for i, s in enumerate(scores):
        # k is small (1, 10, 100); an insort into a bounded list beats re-sorting.
        best.append(s)
        if len(best) > k:
            best.sort(reverse=True)
            best = best[:k]
        curve[i] = float(np.mean(best))

    if budget is not None and pad and curve.size < budget:
        curve = np.concatenate([curve, np.full(budget - curve.size, curve[-1])])
    if budget is not None and curve.size > budget:
        curve = curve[:budget]
    return curve


def auc_top_k(
    scores: Sequence[float],
    k: int = 10,
    budget: int | None = None,
) -> float:
    """Exact AUC of :func:`top_k_curve`, normalised to ``[0, 1]`` of the budget.

    Same quantity as :func:`top_auc` without the ``freq_log`` sampling or the
    zero start. Prefer this for internal comparisons between arms; use
    :func:`top_auc` when quoting numbers against PMO's tables.
    """
    budget = budget or len(scores)
    if budget == 0:
        return 0.0
    curve = top_k_curve(scores, k=k, budget=budget, pad=True)
    return float(np.mean(curve))


def normalize_values(
    values: Iterable[float],
    lo: float,
    hi: float,
    clip: bool = True,
) -> np.ndarray:
    """Min-max scale raw assay values into ``[0, 1]`` for AUC reporting.

    ``lo``/``hi`` must come from the *library*, not from the observed subset --
    otherwise the scaling depends on what the optimiser happened to find and
    arms are no longer comparable.  For the ChEMBL tier that means
    ``lo, hi = min(oracle.values), max(oracle.values)``.
    """
    arr = np.asarray(list(values), dtype=float)
    if hi <= lo:
        raise ValueError(f"degenerate range for normalisation: lo={lo}, hi={hi}")
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0) if clip else out


@dataclass
class ArmSummary:
    """Aggregate of one method across seeds."""

    name: str
    n_seeds: int
    auc_mean: float
    auc_sem: float
    auc_std: float
    best_mean: float
    best_sem: float
    aucs: np.ndarray
    bests: np.ndarray

    def __str__(self) -> str:
        return (
            f"{self.name:24s} n={self.n_seeds:3d}  "
            f"AUC {self.auc_mean:.4f} +/- {self.auc_sem:.4f}  "
            f"best {self.best_mean:.4f} +/- {self.best_sem:.4f}"
        )


def summarize_arm(name: str, aucs: Sequence[float], bests: Sequence[float]) -> ArmSummary:
    a = np.asarray(aucs, float)
    b = np.asarray(bests, float)
    n = a.size
    sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return ArmSummary(
        name=name,
        n_seeds=n,
        auc_mean=float(a.mean()),
        auc_sem=sem(a),
        auc_std=float(a.std(ddof=1)) if n > 1 else 0.0,
        best_mean=float(b.mean()),
        best_sem=sem(b),
        aucs=a,
        bests=b,
    )


def compare_arms(a: Sequence[float], b: Sequence[float], paired: bool = True) -> dict:
    """Compare two arms' per-seed AUCs.

    With ``paired=True`` (seeds matched across arms, which is how these
    experiments should be run) this uses a Wilcoxon signed-rank test and
    reports Cohen's dz; otherwise Mann-Whitney U with Cohen's d.  The
    non-parametric tests are the right default: BO AUCs across seeds are
    bounded and routinely skewed.
    """
    from scipy import stats

    x = np.asarray(a, float)
    y = np.asarray(b, float)
    if paired and x.size != y.size:
        raise ValueError("paired comparison needs the same number of seeds per arm")

    if paired:
        diff = x - y
        if np.allclose(diff, 0):
            p, stat = 1.0, 0.0
        else:
            stat, p = stats.wilcoxon(x, y)
        sd = diff.std(ddof=1)
        effect = float(diff.mean() / sd) if sd > 0 else 0.0
        test = "wilcoxon-signed-rank"
        effect_name = "cohens_dz"
    else:
        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        pooled = np.sqrt(((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1))
                         / max(1, x.size + y.size - 2))
        effect = float((x.mean() - y.mean()) / pooled) if pooled > 0 else 0.0
        test = "mann-whitney-u"
        effect_name = "cohens_d"

    return {
        "test": test,
        "statistic": float(stat),
        "p_value": float(p),
        effect_name: effect,
        "mean_diff": float(x.mean() - y.mean()),
        "n_a": int(x.size),
        "n_b": int(y.size),
    }


def spearman0(a, b) -> float:
    """Spearman correlation, with a constant prediction scored as 0 not NaN.

    A heavily penalised ridge, or a GP whose posterior has collapsed to its
    mean, ranks nothing. That is a real outcome of the arm and 0 is what it
    earned, so it belongs in the average instead of dropping out of it -- a
    NaN would silently reduce the denominator and flatter the arm.

    Four instruments had a private copy of this (``transfer_headroom``,
    ``dense_block_h1``, ``cold_target_chembl``, ``objective_floor``). They were
    byte-identical, which is the usual reason to share: the next edit would
    have landed in one of them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def degrade_to_rho(truth: np.ndarray, rho: float,
                   rng: np.random.Generator) -> np.ndarray:
    """``truth`` corrupted to approximately Spearman ``rho`` against itself.

    Mixes the z-scored signal with standardised noise. Pearson is set exactly;
    Spearman comes out close for near-normal inputs and should be *measured*
    by the caller rather than assumed.

    Two instruments use this on different seams and their results are read
    against each other: ``prior_quality_threshold.py`` corrupts a proxy used as
    the GP's prior mean (FINDINGS 8, threshold rho ~ 0.37), and
    ``dense_block_campaign.py --aux-quality`` corrupts an auxiliary *task*
    (FINDINGS 32). That comparison is only meaningful if both are corrupted the
    same way, so the construction lives here rather than being copied into
    each -- a copy would let one drift and quietly invalidate the comparison.
    """
    truth = np.asarray(truth, dtype=float)
    mean, sd = truth.mean(), truth.std()
    z = (truth - mean) / (sd + 1e-12)
    n = rng.normal(size=len(z))
    n = (n - n.mean()) / (n.std() + 1e-12)
    mixed = rho * z + np.sqrt(max(0.0, 1.0 - rho ** 2)) * n
    return mixed * sd + mean
