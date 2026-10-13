"""The bad-start baseline: that it is bad, that it is local, that it is opt-in.

`--start bad` exists to build the one regime Akke et al. (2025) report an LLM
chooser winning in -- a campaign launched from a bad local minimum -- so that we
can ask whether a free history rule removes it. Three things have to hold for
that question to mean anything:

1. the start really is low-scoring (bad),
2. it really is one neighbourhood (local), not a scattered low-value sample,
3. and the default path is untouched, so every stored `--start random` number
   still reproduces.

The third is the project's standing constraint, not a nicety: the ladder and the
15-block sweep are both stored against the random baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import init_headroom  # noqa: E402


@pytest.fixture
def toy():
    """A library with real structure: value tracks the first coordinate, and
    similarity tracks it too, so "low value" and "one neighbourhood" coincide
    the way they do in a congeneric series."""
    rng = np.random.default_rng(0)
    n = 120
    z = np.sort(rng.random(n))
    # Tanimoto-like gram: similar where z is close, 1 on the diagonal.
    K = np.exp(-((z[:, None] - z[None, :]) ** 2) / 0.01)
    np.fill_diagonal(K, 1.0)
    y = z * 10 + rng.normal(0, 0.2, n)
    return K, y


def mean_offdiag(K, idx):
    sub = K[np.ix_(idx, idx)]
    m = ~np.eye(len(idx), dtype=bool)
    return float(sub[m].mean())


def test_bad_design_is_bad(toy):
    K, y = toy
    g = np.random.default_rng(1)
    picks = [init_headroom.bad_design(K, y, 10, g) for _ in range(20)]
    got = np.mean([y[p].mean() for p in picks])
    assert got < np.median(y), "a bad start must sit below the library median"


def test_bad_design_is_local(toy):
    K, y = toy
    g = np.random.default_rng(2)
    bad = [init_headroom.bad_design(K, y, 10, g) for _ in range(20)]
    rnd = [g.choice(len(K), size=10, replace=False) for _ in range(20)]
    assert (np.mean([mean_offdiag(K, p) for p in bad])
            > np.mean([mean_offdiag(K, p) for p in rnd])), (
        "a bad *local* minimum is tighter than a random draw; otherwise this is "
        "just a low-value sample and the regime is not the one being modelled")


def test_bad_design_has_the_right_shape(toy):
    K, y = toy
    p = init_headroom.bad_design(K, y, 10, np.random.default_rng(3))
    assert len(p) == 10 and len(set(p.tolist())) == 10
    assert p.dtype == np.intp


def test_bad_quantile_tightens_the_start(toy):
    K, y = toy
    a = [init_headroom.bad_design(K, y, 10, np.random.default_rng(s),
                                  quantile=0.05) for s in range(20)]
    b = [init_headroom.bad_design(K, y, 10, np.random.default_rng(s),
                                  quantile=0.5) for s in range(20)]
    assert np.mean([y[p].mean() for p in a]) < np.mean([y[p].mean() for p in b])


def test_random_start_is_unchanged(toy):
    """The default path must draw exactly what it drew before `--start` existed:
    `n_cand` uniform choices off the same rng, in the same order, before the
    rules touch it."""
    K, y = toy
    scaf = np.arange(len(K)) // 4
    aux_idx = [np.arange(0, 40), np.arange(40, 80)]
    aux_val = [y[a] + 0.1 for a in aux_idx]

    got, _ = init_headroom.candidate_designs(
        K, scaf, aux_idx, aux_val, n_init=10, n_cand=5, seed_top=10,
        seed_nn=5, rng=np.random.default_rng(7), y=y)

    ref_rng = np.random.default_rng(7)
    want = [ref_rng.choice(len(K), size=10, replace=False) for _ in range(5)]
    for a, b in zip(got, want):
        assert np.array_equal(a, b)


def test_start_bad_changes_only_the_baseline(toy):
    """The rules are not a function of the start. Only the thing they are scored
    against moves -- which is the whole design of the comparison."""
    K, y = toy
    scaf = np.arange(len(K)) // 4
    aux_idx = [np.arange(0, 40), np.arange(40, 80)]
    aux_val = [y[a] + 0.1 for a in aux_idx]
    kw = dict(n_init=10, n_cand=5, seed_top=10, seed_nn=5, y=y)

    r_rand, _ = init_headroom.candidate_designs(
        K, scaf, aux_idx, aux_val, rng=np.random.default_rng(11), **kw)
    r_bad, _ = init_headroom.candidate_designs(
        K, scaf, aux_idx, aux_val, rng=np.random.default_rng(11),
        start="bad", **kw)

    assert np.mean([y[p].mean() for p in r_bad]) < \
        np.mean([y[p].mean() for p in r_rand])
    assert np.mean([mean_offdiag(K, p) for p in r_bad]) > \
        np.mean([mean_offdiag(K, p) for p in r_rand])


def test_cli_default_is_random():
    """Opt-in, not opt-out. A flag that changed the baseline by default would
    silently invalidate every stored artefact."""
    src = Path(init_headroom.__file__).read_text(encoding="utf-8")
    assert 'choices=["random", "bad"], default="random"' in src


def test_rho_eff_guards_a_constant_primary():
    """A NaN in the effective-coupling list must not reach the summary.

    `rho_eff` is the x axis of both threshold fits. One block with a primary
    that is constant on its history rows yields a NaN Spearman, and `np.median`
    of anything containing a NaN is NaN -- which propagated silently into the
    cross-family fits before this guard existed. The guard drops the triple;
    `nanmedian` is the second line of defence.
    """
    src = Path(init_headroom.__file__).read_text(encoding="utf-8")
    assert "len(np.unique(y[a_i])) > 2" in src, (
        "the primary side of the correlation is unguarded again")
    assert "np.nanmedian(rho_eff)" in src
    # And the arithmetic the guard protects.
    vals = [0.3, float("nan"), 0.5]
    assert np.isnan(np.median(vals))
    assert np.nanmedian(vals) == 0.4
