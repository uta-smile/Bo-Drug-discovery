"""Acquisition functions over a discrete, enumerated library.

Every function here takes posterior ``mu``/``sd`` over the *candidate* rows and
returns a score per candidate, higher-is-better, to be argmax'd.  Minimisation
is handled once at the campaign level by negating the objective, so everything
below assumes maximisation.

Batch selection is greedy over the acquisition surface with a similarity
penalty rather than a true q-EI.  For a discrete library the q-optimal batch is
combinatorial, and the standard cheap fix -- take the top q -- fails badly on
congeneric series, where the top q by EI are routinely the same compound with
the halogen walked around the ring.  Penalising by Tanimoto to what is already
in the batch costs one Gram row per pick and recovers most of the diversity a
proper q-EI would give.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .surrogate import tanimoto_gram

EI_XI: float = 0.01
"""Default incumbent offset for EI and PI, in units of the objective.

Named rather than inlined so a campaign can write the offset it actually ran
with into its stored configuration. A run that records a literal ``0.01`` stops
being reproducible the moment this default moves; one that records ``EI_XI``
cannot disagree with the code that produced it.
"""

BETA: float = 2.0
"""Default UCB width, in posterior standard deviations. See :data:`EI_XI`."""


def expected_improvement(
    mu: np.ndarray, sd: np.ndarray, best: float, xi: float = EI_XI
) -> np.ndarray:
    """EI against the best *observed* value, with a small exploration offset.

    ``xi`` shifts the incumbent up so EI does not collapse to zero once the GP
    is confident; 0.01 is in units of the objective, which is why the campaign
    z-scores nothing -- pIC50 and PMO scores want different offsets and the
    default is tuned for the [0, 1]-ish case.
    """
    sd = np.maximum(sd, 1e-12)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def upper_confidence_bound(mu: np.ndarray, sd: np.ndarray, beta: float = BETA) -> np.ndarray:
    return mu + beta * sd


def probability_of_improvement(
    mu: np.ndarray, sd: np.ndarray, best: float, xi: float = EI_XI
) -> np.ndarray:
    sd = np.maximum(sd, 1e-12)
    return norm.cdf((mu - best - xi) / sd)


def greedy(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Pure exploitation. The ablation that shows whether uncertainty earns its keep."""
    return mu


def thompson(mu: np.ndarray, sd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return mu + sd * rng.standard_normal(mu.shape)


ACQUISITIONS = ("ei", "ucb", "pi", "greedy", "ts")


def acquisition_scores(
    kind: str,
    mu: np.ndarray,
    sd: np.ndarray,
    *,
    best: float,
    rng: np.random.Generator,
    xi: float = EI_XI,
    beta: float = BETA,
) -> np.ndarray:
    if kind == "ei":
        return expected_improvement(mu, sd, best, xi=xi)
    if kind == "ucb":
        return upper_confidence_bound(mu, sd, beta=beta)
    if kind == "pi":
        return probability_of_improvement(mu, sd, best, xi=xi)
    if kind == "greedy":
        return greedy(mu, sd)
    if kind == "ts":
        return thompson(mu, sd, rng)
    raise ValueError(f"unknown acquisition {kind!r}; expected one of {ACQUISITIONS}")


def rank_candidates(scores: np.ndarray, jitter: np.ndarray) -> np.ndarray:
    """Candidate indices, best first, with ties broken by ``jitter``.

    Shared by :func:`select_batch` and by the campaign's shortlist so the two
    order candidates identically. Without that, a selector arm at
    ``shortlist_k=1`` would stop reproducing plain BO precisely when the
    acquisition surface is flat -- every EI numerically zero, which is the
    normal early-campaign state -- because plain BO breaks that tie at random
    and a bare argsort breaks it at index 0.
    """
    return np.lexsort((jitter, np.asarray(scores, dtype=np.float64)))[::-1]


def select_batch(
    scores: np.ndarray,
    X_cand: np.ndarray,
    q: int,
    *,
    rng: np.random.Generator,
    diversity: float = 0.0,
) -> list[int]:
    """Pick ``q`` candidate indices, greedily, optionally penalising near-duplicates.

    ``diversity=0`` is plain top-q. Above zero, each already-picked molecule
    multiplies a candidate's score by ``(1 - diversity * tanimoto)``, so at
    ``diversity=1`` an exact duplicate is excluded outright and a 0.9-similar
    analogue keeps a tenth of its score.

    Ties are broken by the RNG rather than by index, otherwise a flat
    acquisition surface (every EI numerically zero, which happens early on an
    over-confident GP) would make every seed pick the same low-index molecule
    and collapse the seed-to-seed variance the whole design depends on.
    """
    n = scores.shape[0]
    q = min(q, n)
    if q <= 0:
        return []

    scores = np.asarray(scores, dtype=np.float64).copy()
    jitter = rng.random(n) * 1e-12
    if diversity <= 0:
        return list(rank_candidates(scores, jitter)[:q])

    picked: list[int] = []
    # Shift to non-negative so the multiplicative penalty cannot flip a sign:
    # penalising a negative score would otherwise *promote* the duplicate.
    work = scores - scores.min() + 1e-12 + jitter
    for _ in range(q):
        j = int(np.argmax(work))
        picked.append(j)
        work[j] = -np.inf
        sim = tanimoto_gram(X_cand, X_cand[j : j + 1]).ravel()
        live = np.isfinite(work)
        work[live] *= np.clip(1.0 - diversity * sim[live], 0.0, 1.0)
    return picked
