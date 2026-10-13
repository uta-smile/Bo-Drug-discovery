"""The simulation's dials have to be dials.

`sd_scale_simulation.py` exists to separate things the two real blocks confound.
That only works if its two controlled variables really are controlled:

* ``alpha`` must move surrogate quality monotonically and nothing else,
* ``cliffs`` must produce the heavy-tailed residuals section 41 measured on the
  real blocks, not just a shifted target,

and the campaign it drives must be the package's own, not a lookalike. A
simulation whose dial does not turn would produce a flat Spearman and be read as
"no effect".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sd_scale_simulation as sim  # noqa: E402


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(0)
    Z = rng.random((160, 4))
    return Z, sim.rbf(Z, 0.45)


def test_the_gram_has_a_unit_diagonal(small):
    """The campaign passes ``k_diag=np.ones(...)`` because Tanimoto self-
    similarity is 1. A kernel without that would make every posterior variance
    wrong by a constant and the sd sweep uninterpretable."""
    _, K = small
    assert np.allclose(np.diag(K), 1.0)
    assert np.all(K <= 1.0 + 1e-12) and np.all(K >= 0.0)


def test_alpha_moves_surrogate_quality_monotonically(small):
    _, K = small
    rhos = []
    for a in (0.2, 0.5, 0.8, 1.0):
        y = sim.target(np.random.default_rng(1), K, alpha=a, cliffs=False)
        rhos.append(sim.ranking_quality(
            K, y, n_obs=40, rng=np.random.default_rng(2), draws=6))
    assert rhos == sorted(rhos), f"alpha is not a monotone quality dial: {rhos}"
    assert rhos[0] < 0.4 < rhos[-1], f"the dial spans too little: {rhos}"


def test_alpha_one_is_pure_signal(small):
    """At alpha = 1 the unlearnable component must be gone entirely, otherwise
    the top of the quality range is capped by a leak rather than by the kernel."""
    _, K = small
    rng = np.random.default_rng(3)
    y = sim.target(rng, K, alpha=1.0, cliffs=False)
    rho = sim.ranking_quality(K, y, n_obs=60, rng=np.random.default_rng(4),
                              draws=6)
    assert rho > 0.75


def test_cliffs_produce_heavy_tails_in_the_model_residuals():
    """Section 41 measured |z| > 2 at 6-12% on the real blocks against 4.6% for
    a Gaussian, where z is the GP's standardised residual.

    Two traps on the way to the right statistic. The raw values are the wrong
    place to look, because cliffs inflate the sample standard deviation and
    dividing by it hides the tail they create -- what makes a cliff a cliff is
    that the *model* does not see it coming. And `|z| > 2` is the wrong summary
    here, because the synthetic surrogate is already so under-dispersed that it
    saturates at 0.29 under both conditions. Excess kurtosis separates them by
    more than an order of magnitude.
    """
    from scipy.stats import kurtosis

    from molbo.surrogate import TanimotoGP
    rng = np.random.default_rng(7)
    Z = rng.random((400, 4))
    K = sim.rbf(Z, 0.45)
    tails = {}
    for cliffs in (False, True):
        # Pooled over draws: a cliff set is 2% of the library, so a single
        # draw's kurtosis is itself a tail statistic and swings widely.
        zs = []
        for d in range(6):
            y = sim.target(np.random.default_rng(100 + d), K, alpha=0.9,
                           cliffs=cliffs)
            obs = np.random.default_rng(200 + d).choice(400, size=120,
                                                        replace=False)
            cand = np.setdiff1d(np.arange(400), obs)
            gp = TanimotoGP().fit_gram(K[np.ix_(obs, obs)], y[obs])
            mu, sd = gp.predict_gram(K[np.ix_(cand, obs)],
                                     k_diag=np.ones(len(cand)),
                                     return_std=True)
            zs.append((y[cand] - mu) / np.maximum(sd, 1e-12))
        tails[cliffs] = float(kurtosis(np.concatenate(zs)))
    assert tails[True] > tails[False] + 1.0, (
        f"cliffs did not make the residuals heavier-tailed: {tails}")


def test_alpha_is_a_correlation_not_a_weight(small):
    """The two components are standardised before mixing, so alpha is the
    correlation between the learnable part and the target. If it were a raw
    weight, 0.5 would not mean half the variance."""
    _, K = small
    rng = np.random.default_rng(6)
    n = K.shape[0]
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    g0 = np.random.default_rng(6)
    f = L @ g0.standard_normal(n)
    y = sim.target(np.random.default_rng(6), K, alpha=0.6, cliffs=False)
    f = (f - f.mean()) / f.std()
    assert abs(float(np.corrcoef(f, y)[0, 1]) - 0.6) < 0.12


def test_it_drives_the_package_campaign_not_a_copy():
    """If this ever stops being `run_campaign`, the simulation stops being
    about this package."""
    import dense_block_campaign as dbc
    assert sim.run_campaign is dbc.run_campaign


def test_the_scale_grid_brackets_one_on_both_sides():
    """CYP needed the sweep run upward as well: a flattened posterior mean
    makes a method more exploratory, and a one-sided grid would have called
    that negative without looking."""
    assert max(sim.SCALES) > 1.0 and min(sim.SCALES) < 1.0
    assert 1.0 in sim.SCALES
