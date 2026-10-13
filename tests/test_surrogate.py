"""The Tanimoto GP: kernel validity, calibration, and the precomputed-Gram path.

The Gram path exists purely for speed, so the tests that matter most here are
the ones asserting it is *numerically identical* to the direct path -- a fast
surrogate that quietly gives different answers would silently change every
result in the project.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbo.surrogate import (
    ETA_GRID,
    LibraryKernel,
    TanimotoGP,
    fingerprint_matrix,
    tanimoto_gram,
)

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CCO",
    "c1ccccc1",
    "c1ccc2ncnc(Nc3ccccc3)c2c1",
    "COc1cc2ncnc(Nc3ccccc3)c2cc1OC",
    "CCCCCCCC",
    "c1ccc(Cl)cc1",
]


@pytest.fixture(scope="module")
def X():
    return fingerprint_matrix(SMILES)


# ---- kernel ---------------------------------------------------------------


def test_tanimoto_of_identical_molecules_is_one(X):
    K = tanimoto_gram(X, X)
    assert np.allclose(np.diag(K), 1.0)


def test_tanimoto_is_symmetric_and_bounded(X):
    K = tanimoto_gram(X, X)
    assert np.allclose(K, K.T)
    assert K.min() >= 0.0 and K.max() <= 1.0


def test_tanimoto_gram_is_positive_semidefinite(X):
    """A GP on an indefinite kernel is not a GP; Cholesky would fail downstream."""
    eig = np.linalg.eigvalsh(tanimoto_gram(X, X))
    assert eig.min() > -1e-9


def test_unparseable_molecule_is_similar_to_nothing():
    """An all-zero fingerprint carries no information, including about itself.

    Returning 1.0 on its diagonal would tell the GP that two unrelated
    unparseable strings are the same molecule.
    """
    X = fingerprint_matrix(["CCO", "not_a_molecule"])
    assert X[1].sum() == 0
    K = tanimoto_gram(X, X)
    assert K[1, 1] == 0.0 and K[0, 1] == 0.0


def test_similar_molecules_score_higher_than_dissimilar_ones(X):
    K = tanimoto_gram(X, X)
    quinazolines = K[4, 5]  # two anilinoquinazolines
    assert quinazolines > K[4, 2]  # vs ethanol
    assert quinazolines > K[4, 6]  # vs octane


# ---- fitting --------------------------------------------------------------


def test_fit_recovers_training_values_when_noise_is_tiny(X):
    y = np.array([7.5, 5.1, 4.0, 6.2, 8.1, 9.3, 3.2, 5.5])
    gp = TanimotoGP(eta_grid=(1e-6,)).fit(X, y)
    mu = gp.predict(X, return_std=False)
    assert np.allclose(mu, y, atol=1e-3)


def test_posterior_sd_is_smaller_at_observed_points(X):
    y = np.arange(len(SMILES), dtype=float)
    gp = TanimotoGP().fit(X[:4], y[:4])
    _, sd_train = gp.predict(X[:4])
    _, sd_test = gp.predict(X[4:])
    assert sd_train.mean() < sd_test.mean()


def test_constant_targets_do_not_divide_by_zero(X):
    """Every molecule scoring the same is a real early-campaign state."""
    gp = TanimotoGP().fit(X, np.full(len(SMILES), 4.2))
    mu, sd = gp.predict(X)
    assert np.allclose(mu, 4.2)
    assert np.all(np.isfinite(sd))


def test_fit_selects_a_noise_level_from_the_grid(X):
    y = np.array([7.5, 5.1, 4.0, 6.2, 8.1, 9.3, 3.2, 5.5])
    gp = TanimotoGP().fit(X, y)
    assert gp.eta_ in ETA_GRID
    assert np.isfinite(gp.lml_)


def test_noisier_data_selects_a_larger_eta(X):
    """The marginal likelihood should notice when the targets are junk."""
    rng = np.random.default_rng(0)
    signal = tanimoto_gram(X, X)[0] * 10.0
    clean = TanimotoGP().fit(X, signal)
    noisy = TanimotoGP().fit(X, rng.standard_normal(len(SMILES)) * 3.0)
    assert noisy.eta_ >= clean.eta_


def test_fit_is_bitwise_reproducible(X):
    y = np.array([7.5, 5.1, 4.0, 6.2, 8.1, 9.3, 3.2, 5.5])
    a = TanimotoGP().fit(X, y).predict(X, return_std=False)
    b = TanimotoGP().fit(X, y).predict(X, return_std=False)
    assert np.array_equal(a, b)


def test_fit_rejects_mismatched_shapes(X):
    with pytest.raises(ValueError, match="rows"):
        TanimotoGP().fit(X, np.zeros(3))


def test_fit_rejects_empty_observations():
    with pytest.raises(ValueError, match="no observations"):
        TanimotoGP().fit(np.zeros((0, 2048), dtype=np.float32), np.zeros(0))


def test_predict_before_fit_raises(X):
    with pytest.raises(RuntimeError, match="before fit"):
        TanimotoGP().predict(X)


# ---- the precomputed-Gram path -------------------------------------------


def test_gram_path_is_numerically_identical_to_the_direct_path(X):
    """The speed optimisation must not change a single number."""
    y = np.array([7.5, 5.1, 4.0, 6.2, 8.1, 9.3, 3.2, 5.5])
    train, cand = np.arange(5), np.arange(5, 8)
    K = LibraryKernel(X)

    direct = TanimotoGP().fit(X[train], y[train])
    mu_d, sd_d = direct.predict(X[cand])

    viagram = TanimotoGP().fit_gram(K.train_block(train), y[train])
    mu_g, sd_g = viagram.predict_gram(K.cross_block(cand, train), k_diag=K.k_diag[cand])

    assert np.array_equal(mu_d, mu_g)
    assert np.array_equal(sd_d, sd_g)


def test_gram_fit_selects_the_same_hyperparameters(X):
    y = np.array([7.5, 5.1, 4.0, 6.2, 8.1, 9.3, 3.2, 5.5])
    K = LibraryKernel(X)
    a = TanimotoGP().fit(X, y)
    b = TanimotoGP().fit_gram(K.train_block(np.arange(len(SMILES))), y)
    assert (a.eta_, a.scale_) == (b.eta_, b.scale_)


def test_predict_needs_fingerprints_when_fitted_from_a_gram(X):
    y = np.arange(len(SMILES), dtype=float)
    K = LibraryKernel(X)
    gp = TanimotoGP().fit_gram(K.train_block(np.arange(len(SMILES))), y)
    with pytest.raises(RuntimeError, match="predict_gram"):
        gp.predict(X)


def test_maybe_build_declines_a_library_that_would_not_fit(X):
    assert LibraryKernel.maybe_build(X, max_bytes=10) is None
    assert LibraryKernel.maybe_build(X, max_bytes=10**9) is not None


# ---- sampling -------------------------------------------------------------


def test_posterior_samples_are_seed_reproducible(X):
    y = np.arange(len(SMILES), dtype=float)
    gp = TanimotoGP().fit(X[:5], y[:5])
    a = gp.sample_posterior(X, np.random.default_rng(1), n_samples=3)
    b = gp.sample_posterior(X, np.random.default_rng(1), n_samples=3)
    assert np.array_equal(a, b)
    assert a.shape == (3, len(SMILES))


# ---- stereochemistry ------------------------------------------------------


def test_fingerprints_distinguish_enantiomers():
    """RDKit's Morgan generator ignores chirality by default.

    On the EGFR library that collapsed 133 pairs -- 204 molecules -- onto
    identical bit vectors while their measured activities differed by up to 3.0
    log units. A GP handed two identical inputs with different targets can only
    average them, which puts an irreducible noise floor under every prediction.
    """
    X = fingerprint_matrix(["C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O"])
    assert not np.array_equal(X[0], X[1])
    assert tanimoto_gram(X[:1], X[1:])[0, 0] < 1.0


def test_fingerprints_distinguish_cis_trans_isomers():
    X = fingerprint_matrix([r"C/C=C/c1ccccc1", r"C/C=C\c1ccccc1"])
    assert not np.array_equal(X[0], X[1])


def test_the_same_molecule_still_matches_itself_whatever_the_spelling():
    """Chirality must not make canonicalisation-equivalent inputs differ."""
    X = fingerprint_matrix(["CC(=O)Oc1ccccc1C(=O)O", "O=C(C)Oc1ccccc1C(O)=O"])
    assert np.array_equal(X[0], X[1])


def test_fingerprint_version_is_recorded():
    """Anything caching fingerprints has to key on this."""
    from molbo.chem import FINGERPRINT_VERSION, INCLUDE_CHIRALITY

    assert INCLUDE_CHIRALITY is True
    assert FINGERPRINT_VERSION == "v2-chiral"


def _reference_lml(K, z, eta, scale, jitter):
    """Textbook Gaussian log density, written out independently of ``_fit_gram``.

    ``_fit_gram`` evaluates the same quantity through a Cholesky factor and a
    profiling identity; this one builds the covariance and calls the general
    solver, so the two agreeing is a real check rather than a restatement.
    """
    n = z.size
    C = scale * (K + (eta + jitter) * np.eye(n))
    sign, logdet = np.linalg.slogdet(C)
    assert sign > 0
    return -0.5 * (z @ np.linalg.solve(C, z) + logdet + n * np.log(2 * np.pi))


def _standardised(y):
    return (y - y.mean()) / y.std()


def test_fixed_scale_still_selects_the_marginal_likelihood_argmax(X):
    """A pinned signal variance must not collapse the eta scan.

    The data-fit term ``z'(K + eta I)^-1 z / s`` equals ``n`` only when ``s`` is
    the profiled optimum. It was hardcoded as ``n``, so with ``fixed_scale`` set
    the scan reduced to minimising ``logdet`` and returned ``ETA_GRID[0]``
    whatever the targets looked like.
    """
    rng = np.random.default_rng(0)
    y = tanimoto_gram(X, X)[0] * 10.0 + rng.standard_normal(len(SMILES))
    K, z = tanimoto_gram(X, X), _standardised(y)
    for scale in (0.05, 1.0, 20.0):
        gp = TanimotoGP(fixed_scale=scale).fit(X, y)
        want = max(ETA_GRID, key=lambda e: _reference_lml(K, z, e, scale, gp.jitter))
        assert gp.eta_ == want, f"fixed_scale={scale} picked eta={gp.eta_}, want {want}"
        assert gp.scale_ == scale
        assert gp.lml_ == pytest.approx(
            _reference_lml(K, z, gp.eta_, scale, gp.jitter)
        )


def test_pinned_scale_moves_eta_instead_of_collapsing_the_scan(X):
    """The guard the collapsed scan could not pass: eta has to answer to ``scale``.

    Pinning the signal variance below what the data supports forces the fit to
    charge the residual to noise, and pinning it above does the reverse. Under
    the bug the scan never saw ``scale`` at all and returned ``ETA_GRID[0]`` for
    all three -- so the pinned-scale rung of ``scripts/null_mechanism.py`` would
    have been reading a model fitted at a noise level nothing selected.
    """
    rng = np.random.default_rng(1)
    y = tanimoto_gram(X, X)[0] * 10.0 + rng.standard_normal(len(SMILES))
    picked = [TanimotoGP(fixed_scale=s).fit(X, y).eta_ for s in (0.05, 1.0, 20.0)]
    assert picked[0] > picked[1] > picked[2], picked
    assert picked != [ETA_GRID[0]] * 3
    assert all(e in ETA_GRID for e in picked)


def test_profiled_scale_is_unaffected_by_the_fixed_scale_path(X):
    """The default path still profiles ``s`` out and still finds the argmax."""
    rng = np.random.default_rng(2)
    y = tanimoto_gram(X, X)[0] * 10.0 + rng.standard_normal(len(SMILES))
    K, z = tanimoto_gram(X, X), _standardised(y)
    gp = TanimotoGP().fit(X, y)
    want = max(
        ETA_GRID,
        key=lambda e: _reference_lml(
            K, z, e, max(_profiled_scale(K, z, e, gp.jitter), 1e-12), gp.jitter
        ),
    )
    assert gp.eta_ == want
    assert gp.scale_ == pytest.approx(_profiled_scale(K, z, gp.eta_, gp.jitter))
    # Pinning the scale at the value the free fit profiled out reproduces it.
    assert TanimotoGP(fixed_scale=gp.scale_).fit(X, y).eta_ == gp.eta_


def _profiled_scale(K, z, eta, jitter):
    n = z.size
    return float(z @ np.linalg.solve(K + (eta + jitter) * np.eye(n), z)) / n
