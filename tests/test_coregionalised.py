"""The multi-task surrogate behind FINDINGS section 14's negative result.

A negative result is only worth the instrument that produced it, so the
properties these tests pin are the ones whose failure would have manufactured
that null: that ``rho = 0`` really is the independent-target model, that the
marginal likelihood picks up coupling when coupling exists, and that tasks
measured on different molecules are handled as such rather than silently
aligned by position.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbo.surrogate import CoregionalisedGP, TanimotoGP, fingerprint_matrix, tanimoto_gram

SMILES = [
    "CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)O",
    "CC(=O)N", "CCOCC", "CCCCO", "CCCCN", "c1ccncc1", "c1ccsc1", "CC(C)O",
    "CC(C)N", "CCCCCC", "OCCO", "NCCN", "CC(=O)OC",
]


@pytest.fixture(scope="module")
def gram():
    X = fingerprint_matrix(SMILES)
    return tanimoto_gram(X, X)


def _two_tasks(gram, rng, *, coupling):
    """Task 0 and a second task built to share ``coupling`` of its signal."""
    base = rng.normal(size=len(SMILES))
    other = coupling * base + np.sqrt(1 - coupling**2) * rng.normal(size=len(SMILES))
    return base, other


def test_rho_zero_reproduces_the_single_task_gp(gram):
    """With one task the coregionalised model must *be* the Tanimoto GP."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=len(SMILES))
    tr = np.arange(12)
    te = np.arange(12, len(SMILES))

    single = TanimotoGP().fit_gram(gram[np.ix_(tr, tr)], y[tr])
    mu_single = single.predict_gram(gram[np.ix_(te, tr)], return_std=False)

    multi = CoregionalisedGP(rho_grid=(0.0,)).fit_gram(gram, [tr], [y[tr]])
    mu_multi = multi.predict_gram(gram, te)

    # Both predict on the scale the caller supplied, so with one task at r = 0
    # the coregionalised model must reproduce the single-task GP exactly -- not
    # up to an affine map. An earlier version of this assertion allowed the
    # affine slack, which is what let a z-scored posterior mean reach the
    # acquisition function unnoticed.
    np.testing.assert_allclose(mu_multi, mu_single, rtol=1e-8)


def test_independent_tasks_do_not_help_the_primary(gram):
    """An uncorrelated second task must not move the primary's predictions."""
    rng = np.random.default_rng(1)
    y0, y1 = _two_tasks(gram, rng, coupling=0.0)
    tr = np.arange(12)
    te = np.arange(12, len(SMILES))

    alone = CoregionalisedGP(rho_grid=(0.0,)).fit_gram(gram, [tr], [y0[tr]])
    with_other = CoregionalisedGP(rho_grid=(0.0,)).fit_gram(
        gram, [tr, tr], [y0[tr], y1[tr]])

    np.testing.assert_allclose(alone.predict_gram(gram, te),
                               with_other.predict_gram(gram, te), atol=1e-8)


def test_marginal_likelihood_finds_the_coupling(gram):
    """Duplicate a task and the scan should prefer a high task correlation."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=len(SMILES))
    tr = np.arange(14)

    gp = CoregionalisedGP().fit_gram(gram, [tr, tr], [y[tr], y[tr]])
    assert gp.rho_ >= 0.85, f"identical tasks selected rho={gp.rho_}"

    y0, y1 = _two_tasks(gram, rng, coupling=0.0)
    indep = CoregionalisedGP().fit_gram(gram, [tr, tr], [y0[tr], y1[tr]])
    assert indep.rho_ < gp.rho_


def test_tasks_may_use_different_molecules(gram):
    """The spread design: disjoint indices, different sizes, still fits."""
    rng = np.random.default_rng(3)
    y0, y1 = _two_tasks(gram, rng, coupling=0.8)
    a, b = np.arange(0, 8), np.arange(8, 15)
    te = np.arange(15, len(SMILES))

    gp = CoregionalisedGP().fit_gram(gram, [a, b], [y0[a], y1[b]])
    mu = gp.predict_gram(gram, te)
    assert mu.shape == te.shape
    assert np.isfinite(mu).all()


def test_a_flat_task_does_not_divide_by_zero(gram):
    """Every molecule scored the same is a real outcome, not a crash."""
    tr = np.arange(10)
    flat = np.full(len(tr), 3.0)
    rng = np.random.default_rng(4)
    y0 = rng.normal(size=len(SMILES))

    gp = CoregionalisedGP().fit_gram(gram, [tr, tr], [y0[tr], flat])
    assert np.isfinite(gp.predict_gram(gram, np.arange(10, 20))).all()


def test_mismatched_shapes_are_rejected(gram):
    tr = np.arange(10)
    with pytest.raises(ValueError, match="values"):
        CoregionalisedGP().fit_gram(gram, [tr], [np.zeros(3)])
    with pytest.raises(ValueError, match="value arrays"):
        CoregionalisedGP().fit_gram(gram, [tr, tr], [np.zeros(10)])
    with pytest.raises(ValueError, match="at least one task"):
        CoregionalisedGP().fit_gram(gram, [], [])


def test_predict_before_fit_is_an_error(gram):
    with pytest.raises(RuntimeError, match="before fit_gram"):
        CoregionalisedGP().predict_gram(gram, np.arange(3))


def test_free_structure_recovers_an_uneven_family(gram):
    """A family where one pair is coupled and another is not.

    The shared-r form has to compromise between them; the free form should
    separate them, and say so in the fitted matrix. This is the property
    FINDINGS section 19 predicts free B pays for.
    """
    rng = np.random.default_rng(11)
    tr = np.arange(16)
    # Draw the tasks from the *kernel's own* prior. Independent noise would
    # leave the likelihood flat in B -- the GP explains it all as noise and the
    # ascent then wanders into whatever direction is marginally higher, which
    # is what an earlier version of this test actually measured.
    L = np.linalg.cholesky(gram + 1e-6 * np.eye(len(gram)))
    f1, f2, f3 = (L @ rng.standard_normal(len(gram)) for _ in range(3))
    base, close, far = f1, 0.95 * f1 + 0.31 * f2, f3

    free = CoregionalisedGP(structure="free").fit_gram(
        gram, [tr, tr, tr], [base[tr], close[tr], far[tr]])
    B = free._B
    assert B[0, 1] > B[0, 2], f"coupled pair {B[0,1]} not above unrelated {B[0,2]}"
    np.testing.assert_allclose(np.diag(B), 1.0)
    np.testing.assert_allclose(B, B.T)
    assert np.linalg.eigvalsh(B).min() > 0, "fitted B left the PSD cone"


def test_free_never_loses_likelihood_to_shared(gram):
    """Coordinate ascent starts at the shared optimum, so it cannot do worse."""
    rng = np.random.default_rng(12)
    tr = np.arange(16)
    ys = [rng.normal(size=len(SMILES)) for _ in range(3)]
    shared = CoregionalisedGP().fit_gram(gram, [tr] * 3, [y[tr] for y in ys])
    free = CoregionalisedGP(structure="free").fit_gram(
        gram, [tr] * 3, [y[tr] for y in ys])
    assert free.lml_ >= shared.lml_ - 1e-9


def test_free_is_deterministic(gram):
    rng = np.random.default_rng(13)
    tr = np.arange(14)
    ys = [rng.normal(size=len(SMILES)) for _ in range(3)]
    fits = [CoregionalisedGP(structure="free").fit_gram(
        gram, [tr] * 3, [y[tr] for y in ys]) for _ in range(2)]
    np.testing.assert_array_equal(fits[0]._B, fits[1]._B)
    np.testing.assert_allclose(fits[0].predict_gram(gram, np.arange(14, 20)),
                               fits[1].predict_gram(gram, np.arange(14, 20)))


def test_predictions_come_back_on_the_callers_scale(gram):
    """An acquisition function compares the mean against an incumbent.

    Returning a z-scored mean makes every candidate look equally far below a
    raw-scale best, expected improvement collapses to zero, and the BO loop
    silently degenerates to its tie-break jitter -- which is exactly how this
    was found.
    """
    rng = np.random.default_rng(21)
    tr = np.arange(12)
    y = 7.0 + 1.5 * rng.normal(size=len(SMILES))  # pIC50-like, far from zero
    other = rng.normal(size=len(SMILES))

    gp = CoregionalisedGP().fit_gram(gram, [tr, tr], [y[tr], other[tr]])
    mu, sd = gp.predict_gram(gram, np.arange(12, 20), return_std=True)
    assert abs(mu.mean() - y[tr].mean()) < 2.0 * y[tr].std(), (
        f"posterior mean {mu.mean():.2f} is nowhere near the data's {y[tr].mean():.2f}")
    assert (sd > 0).all()
    # Scale equivariance: shifting and stretching the primary task must move the
    # predictions the same way, not change the model.
    scaled = CoregionalisedGP().fit_gram(
        gram, [tr, tr], [3.0 * y[tr] + 11.0, other[tr]])
    mu2 = scaled.predict_gram(gram, np.arange(12, 20))
    np.testing.assert_allclose(mu2, 3.0 * mu + 11.0, rtol=1e-6)
