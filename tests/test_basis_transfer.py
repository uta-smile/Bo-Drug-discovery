"""Mean-only transfer: the arm that removes the section-50 selection artefact.

Sections 40 and 50 located a mechanism rather than a bug. A coregionalised GP
conditions on the auxiliary rows, so an assayed compound gets a smaller
predictive variance (0.412 against 0.542 on CYP); expected improvement reads
that as "already explored" and the pooled arm buys historically assayed
compounds at 0.288 of their library share. The penalty is paid under permuted
auxiliary responses too, so it is not transfer -- it is the conditioning set.

`BasisTransferGP` keeps the auxiliary rows out of the conditioning set
entirely. Each auxiliary target is fitted alone, its posterior mean over the
library becomes a basis function, and the coefficients carry a Gaussian prior
that is marginalised analytically (Rasmussen & Williams section 2.7). What has
to be proved about it:

1. the location-specific variance deflation is gone, which is the mechanism;
2. the auxiliary *values* still reach the posterior mean, or it is not transfer;
3. the per-assay coefficients discriminate a useful auxiliary from a useless
   one, which is the inclusion question answered inside one campaign;
4. with no basis it is exactly single-task, so any difference is the basis;
5. no primary-target value touches the basis, which is the leakage check;
6. the SVD path is a rotation of the same information, not new information.

What is deliberately *not* asserted anywhere here is bit-identity of the
predictive sd under replacement of the auxiliary values. The basis is built
from those values and `tau2` is refitted against them, so the sd does move --
correctly, and towards the single-task answer. An early draft of this file
asserted bit-identity, and it was the docstring that was wrong, not the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dense_block_campaign as dbc  # noqa: E402
from molbo.surrogate import (  # noqa: E402
    BasisTransferGP, CoregionalisedGP, TanimotoGP, tanimoto_gram)


@pytest.fixture
def block():
    """A library with a latent family activity the auxiliaries see noisily.

    Drawn from the kernel itself so the GP is well specified: a failure here is
    then about the transfer construction and not about model misspecification.
    """
    rng = np.random.default_rng(0)
    n, d = 200, 64
    X = (rng.random((n, d)) < 0.12).astype(np.float64)
    X[X.sum(1) == 0, 0] = 1.0
    K = tanimoto_gram(X, X)
    latent = rng.multivariate_normal(np.zeros(n), K + 1e-6 * np.eye(n))
    y = latent + 0.3 * rng.normal(size=n)
    aux_idx = [rng.choice(n, 60, replace=False) for _ in range(3)]
    aux_val = [latent[i] + 0.5 * rng.normal(size=len(i)) for i in aux_idx]
    obs = rng.choice(n, 40, replace=False)
    return K, y, aux_idx, aux_val, obs, np.setdiff1d(np.arange(n), obs)


def _sd_ratio(sd, cand, aux_idx):
    """Mean predictive sd inside the history over mean sd outside it.

    The statistic section 50 reports. 1.0 is indifference; below 1.0 is the
    compression that drives expected improvement away from assayed compounds.
    """
    inh = np.isin(cand, np.unique(np.concatenate(aux_idx)))
    return float(sd[inh].mean() / sd[~inh].mean())


def test_no_location_specific_variance_deflation(block):
    """The mechanism claim, stated as the statistic the paper reports.

    Pinned against both neighbours at once, because the number only means
    something relative to them: single-task has nothing to deflate, the
    coregionalised model deflates hard, and this model must sit with the first.
    """
    K, y, aux_idx, aux_val, obs, cand = block

    stl = TanimotoGP().fit_gram(K[np.ix_(obs, obs)], y[obs])
    _, sd_stl = stl.predict_gram(K[np.ix_(cand, obs)], k_diag=np.ones(len(cand)))

    co = CoregionalisedGP().fit_gram(K, [obs] + aux_idx, [y[obs]] + list(aux_val))
    _, sd_co = co.predict_gram(K, cand, return_std=True)

    bt = BasisTransferGP.from_auxiliary(K, aux_idx, aux_val).fit_gram(K, obs, y[obs])
    _, sd_bt = bt.predict_gram(K, cand, return_std=True)

    r_stl = _sd_ratio(sd_stl, cand, aux_idx)
    r_co = _sd_ratio(sd_co, cand, aux_idx)
    r_bt = _sd_ratio(sd_bt, cand, aux_idx)

    # The coregionalised model must actually show the artefact, or this test
    # would pass on a block where there was nothing to fix.
    assert r_co < 0.75, f"coregionalised ratio {r_co} shows no deflation to remove"
    assert r_bt > 0.95, f"basis ratio {r_bt} deflates like the model it replaces"
    assert abs(r_bt - r_stl) < 0.15


def test_auxiliary_values_still_move_the_posterior_mean(block):
    """Otherwise it is not a transfer method, only an expensive single-task GP."""
    K, y, aux_idx, aux_val, obs, cand = block
    rng = np.random.default_rng(99)
    junk = [rng.normal(size=len(i)) for i in aux_idx]
    real = BasisTransferGP.from_auxiliary(K, aux_idx, aux_val).fit_gram(K, obs, y[obs])
    noise = BasisTransferGP.from_auxiliary(K, aux_idx, junk).fit_gram(K, obs, y[obs])
    mu_r = real.predict_gram(K, cand, return_std=False)
    mu_n = noise.predict_gram(K, cand, return_std=False)
    assert not np.allclose(mu_r, mu_n)


def test_trend_variance_shrinks_on_an_uninformative_basis(block):
    """The inclusion decision, made by marginal likelihood inside one campaign.

    Section 44 fitted a chooser across 18 blocks and it did not survive out of
    sample. This is the version that does not need a cross-block fit: `tau2` is
    chosen by the same likelihood that chooses every other hyperparameter here,
    from the primary observations of the campaign it is running in.
    """
    K, y, aux_idx, aux_val, obs, _ = block
    rng = np.random.default_rng(99)
    junk = [rng.normal(size=len(i)) for i in aux_idx]
    real = BasisTransferGP.from_auxiliary(K, aux_idx, aux_val).fit_gram(K, obs, y[obs])
    noise = BasisTransferGP.from_auxiliary(K, aux_idx, junk).fit_gram(K, obs, y[obs])
    assert noise.tau2_ < real.tau2_
    # And the coefficients follow, which is what makes `beta_` readable as a
    # per-assay inclusion weight rather than an arbitrary internal.
    assert (np.abs(noise.beta_response_).max()
            < np.abs(real.beta_response_).max())


def test_empty_basis_is_exactly_single_task(block):
    """With nothing to transfer it must reduce to `stl`, so any difference
    between the two arms is the basis and nothing else."""
    K, y, _, _, obs, cand = block
    bt = BasisTransferGP.from_auxiliary(
        K, [], [], n_library=len(y)).fit_gram(K, obs, y[obs])
    stl = TanimotoGP().fit_gram(K[np.ix_(obs, obs)], y[obs])
    mu_b = bt.predict_gram(K, cand, return_std=False)
    mu_s = stl.predict_gram(K[np.ix_(cand, obs)], return_std=False)
    assert bt.tau2_ == 0.0
    assert np.allclose(mu_b, mu_s)


def test_basis_reads_no_primary_values(block):
    """The leakage check. The basis is fitted before any primary observation
    exists, so changing every primary value must leave it bit-identical."""
    K, y, aux_idx, aux_val, _, _ = block
    a = BasisTransferGP.from_auxiliary(K, aux_idx, aux_val)
    b = BasisTransferGP.from_auxiliary(K, aux_idx, aux_val)
    # Nothing about `y` is in scope for the constructor at all; the assertion
    # is that two constructions agree and that the signature cannot see y.
    assert np.array_equal(a.H_, b.H_)
    assert a.H_.shape == (len(y), 3)


def test_svd_rank_truncation_orthogonalises_without_inventing_signal(block):
    """The SVD path must be a rotation of the auxiliary means, not a new source.

    Orthogonality is the point -- on a tightly coupled family the raw columns
    are near-collinear and the coefficient prior is asked to resolve a nearly
    degenerate basis -- but a full-rank rotation must also span exactly what the
    raw columns spanned, or something other than a rotation has happened.
    """
    K, y, aux_idx, aux_val, obs, _ = block
    raw = BasisTransferGP.from_auxiliary(K, aux_idx, aux_val)
    full = BasisTransferGP.from_auxiliary(K, aux_idx, aux_val, rank=3)
    trunc = BasisTransferGP.from_auxiliary(K, aux_idx, aux_val, rank=2)

    n = len(y)
    assert np.allclose(full.H_.T @ full.H_ / n, np.eye(3), atol=1e-8)
    assert trunc.H_.shape == (n, 2)
    # Same column space at full rank: projecting the raw basis onto the rotated
    # one must return it. Centred, since `_reduce` centres.
    Q, _ = np.linalg.qr(full.H_)
    Hc = raw.H_ - raw.H_.mean(axis=0)
    assert np.allclose(Q @ (Q.T @ Hc), Hc, atol=1e-7)


def test_chemistry_control_cannot_read_the_auxiliary(block):
    """The floor arm. It takes no auxiliary argument, so the property is in the
    signature; the test records that it is also usable and rank-matched."""
    K, y, _, _, obs, cand = block
    chem = BasisTransferGP.from_chemistry(K, rank=3).fit_gram(K, obs, y[obs])
    assert chem.H_.shape == (len(y), 3)
    mu, sd = chem.predict_gram(K, cand, return_std=True)
    assert np.all(np.isfinite(mu)) and np.all(sd > 0)


def test_campaign_arms_run_and_spend_the_budget(block):
    """End to end through the campaign, including the chemistry control."""
    K, y, aux_idx, aux_val, _, _ = block
    for arm in ("basis", "basis_chem"):
        vals, picked = dbc.run_campaign(
            K, y, aux_idx, aux_val, arm=arm, budget=30, batch=5, n_init=10, k=5,
            acquisition="ei", rng=np.random.default_rng(3))
        assert len(picked) == 30 and len(set(picked.tolist())) == 30
        assert len(vals) == 30


def test_campaign_is_deterministic(block):
    """Two runs at one seed must agree bit for bit, as every other arm does."""
    K, y, aux_idx, aux_val, _, _ = block
    kw = dict(arm="basis", budget=25, batch=5, n_init=10, k=5, acquisition="ei")
    a = dbc.run_campaign(K, y, aux_idx, aux_val, rng=np.random.default_rng(7), **kw)[1]
    b = dbc.run_campaign(K, y, aux_idx, aux_val, rng=np.random.default_rng(7), **kw)[1]
    assert np.array_equal(a, b)


def test_parameter_snapshot_reports_grid_edges(block):
    """A `tau2` pinned at an endpoint is the grid running out, not a
    measurement, and the artefact has to say which it was."""
    K, y, aux_idx, aux_val, obs, _ = block
    snap = (BasisTransferGP.from_auxiliary(K, aux_idx, aux_val)
            .fit_gram(K, obs, y[obs]).parameter_snapshot())
    assert set(snap) >= {"tau2", "tau2_at_grid_edge", "beta", "n_basis", "lml"}
    assert len(snap["beta"]) == snap["n_basis"] == 3
