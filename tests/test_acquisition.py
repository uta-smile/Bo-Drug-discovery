"""Acquisition functions and batch selection."""

from __future__ import annotations

import numpy as np
import pytest

from molbo.acquisition import (
    ACQUISITIONS,
    acquisition_scores,
    expected_improvement,
    probability_of_improvement,
    select_batch,
    upper_confidence_bound,
)
from molbo.surrogate import fingerprint_matrix

RNG = lambda seed=0: np.random.default_rng(seed)


# ---- EI -------------------------------------------------------------------


def test_ei_is_nonnegative():
    mu = np.array([-5.0, 0.0, 5.0])
    sd = np.array([0.1, 1.0, 2.0])
    assert np.all(expected_improvement(mu, sd, best=0.0) >= 0)


def test_ei_prefers_a_higher_mean_at_equal_uncertainty():
    ei = expected_improvement(np.array([1.0, 2.0]), np.array([1.0, 1.0]), best=0.0)
    assert ei[1] > ei[0]


def test_ei_prefers_more_uncertainty_at_equal_mean():
    """The whole reason to use EI over greedy."""
    ei = expected_improvement(np.array([0.0, 0.0]), np.array([0.5, 2.0]), best=0.0)
    assert ei[1] > ei[0]


def test_ei_is_near_zero_for_a_confidently_bad_molecule():
    ei = expected_improvement(np.array([-10.0]), np.array([1e-6]), best=0.0)
    assert ei[0] == pytest.approx(0.0, abs=1e-9)


def test_ei_handles_zero_uncertainty_without_dividing_by_zero():
    ei = expected_improvement(np.array([5.0, -5.0]), np.array([0.0, 0.0]), best=0.0)
    assert np.all(np.isfinite(ei))


# ---- UCB / PI -------------------------------------------------------------


def test_ucb_beta_controls_exploration():
    mu, sd = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.argmax(upper_confidence_bound(mu, sd, beta=0.5)) == 0
    assert np.argmax(upper_confidence_bound(mu, sd, beta=5.0)) == 1


def test_pi_is_a_probability():
    p = probability_of_improvement(np.array([-3.0, 0.0, 3.0]), np.ones(3), best=0.0)
    assert np.all((p >= 0) & (p <= 1))


# ---- dispatch -------------------------------------------------------------


@pytest.mark.parametrize("kind", ACQUISITIONS)
def test_every_acquisition_returns_one_finite_score_per_candidate(kind):
    mu, sd = np.linspace(-1, 1, 7), np.linspace(0.1, 1.0, 7)
    out = acquisition_scores(kind, mu, sd, best=0.0, rng=RNG())
    assert out.shape == mu.shape
    assert np.all(np.isfinite(out))


def test_unknown_acquisition_is_rejected():
    with pytest.raises(ValueError, match="unknown acquisition"):
        acquisition_scores("nope", np.zeros(2), np.ones(2), best=0.0, rng=RNG())


def test_thompson_sampling_varies_with_the_rng():
    mu, sd = np.zeros(20), np.ones(20)
    a = acquisition_scores("ts", mu, sd, best=0.0, rng=RNG(1))
    b = acquisition_scores("ts", mu, sd, best=0.0, rng=RNG(2))
    assert not np.allclose(a, b)


def test_greedy_ignores_uncertainty():
    mu, sd = np.array([1.0, 0.5]), np.array([0.0, 10.0])
    assert np.argmax(acquisition_scores("greedy", mu, sd, best=0.0, rng=RNG())) == 0


# ---- batch selection ------------------------------------------------------


@pytest.fixture(scope="module")
def X():
    return fingerprint_matrix(
        [
            "c1ccc2ncnc(Nc3ccccc3)c2c1",
            "c1ccc2ncnc(Nc3ccc(F)cc3)c2c1",  # near-duplicate of the first
            "CCO",
            "CCCCCCCC",
            "c1ccccc1",
        ]
    )


def test_top_q_is_the_default(X):
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert select_batch(scores, X, 3, rng=RNG()) == [0, 1, 2]


def test_diversity_penalty_skips_a_near_duplicate(X):
    """Without it, a congeneric batch is one compound with the halogen moved.

    The two anilinoquinazolines here are 0.75 similar, so at ``diversity=1``
    the second keeps a quarter of its score. That is enough to displace it
    only when the runner-up is competitive -- the penalty re-ranks, it does not
    veto. Scores are close here for exactly that reason.
    """
    scores = np.array([5.0, 4.9, 4.8, 1.0, 0.9])
    plain = select_batch(scores, X, 2, rng=RNG(), diversity=0.0)
    diverse = select_batch(scores, X, 2, rng=RNG(), diversity=1.0)
    assert plain == [0, 1]
    assert diverse == [0, 2]  # the dissimilar third beats the 0.75-similar second


def test_a_strong_enough_score_gap_survives_the_diversity_penalty(X):
    """The penalty must not override a genuinely large acquisition difference."""
    scores = np.array([5.0, 4.9, 0.2, 0.1, 0.0])
    assert select_batch(scores, X, 2, rng=RNG(), diversity=1.0) == [0, 1]


def test_an_exact_duplicate_is_excluded_at_full_diversity():
    """Self-similarity is 1, so the multiplier is zero -- the documented extreme."""
    X = fingerprint_matrix(["CCO", "OCC", "c1ccccc1"])  # first two are the same molecule
    picked = select_batch(np.array([5.0, 4.9, 0.1]), X, 2, rng=RNG(), diversity=1.0)
    assert picked == [0, 2]


def test_diversity_does_not_promote_duplicates_when_scores_are_negative(X):
    """A multiplicative penalty on a negative score would flip its sign."""
    scores = np.array([-1.0, -1.01, -5.0, -6.0, -7.0])
    picked = select_batch(scores, X, 2, rng=RNG(), diversity=1.0)
    assert picked[0] == 0 and picked[1] != 1


def test_flat_acquisition_surface_breaks_ties_randomly(X):
    """Every EI numerically zero is a real early-campaign state.

    Deterministic index order there would make every seed pick the same
    molecule and collapse the seed-to-seed variance the design depends on.
    """
    scores = np.zeros(5)
    picks = {tuple(select_batch(scores, X, 2, rng=RNG(s))) for s in range(25)}
    assert len(picks) > 1


def test_batch_is_capped_by_the_candidate_count(X):
    assert len(select_batch(np.zeros(5), X, 99, rng=RNG())) == 5
    assert select_batch(np.zeros(5), X, 0, rng=RNG()) == []


def test_selection_never_repeats_a_candidate(X):
    picked = select_batch(np.arange(5.0), X, 5, rng=RNG(), diversity=0.7)
    assert sorted(picked) == list(range(5))
