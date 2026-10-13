"""Region-based search: partitioning, concentration, and the bandit baseline."""

import numpy as np
import pytest

from molbo.clusters import (
    ClusterConstraint,
    FixedClusterSource,
    UCBClusterSource,
    cluster_library,
    concentration,
)
from molbo.constraints import LibraryContext, apply_constraints

LIB = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1", "c1ccncc1", "c1ccoc1",
       "CCCCO", "CCCCN", "CCCCC", "CC(=O)O", "CC(=O)N", "CC(=O)C", "CCOCC", "CCNCC"]


def test_every_region_gets_its_own_mask():
    """LibraryContext.mask_for caches on Constraint.fingerprint(), which hashes
    describe(). The base describe() returns only {"kind"}, so inheriting it gave
    every region the FIRST region's mask -- and a bandit, a random region and the
    hindsight-best region all scored identically to four decimal places."""
    ctx = LibraryContext.build(LIB)
    labels = np.array([0] * 4 + [1] * 4 + [2] * 8)
    sizes = []
    for c in (0, 1, 2):
        out = apply_constraints(ClusterConstraint(labels=labels, clusters=(c,)),
                                ctx, min_candidates=1)
        sizes.append(out.n_allowed)
    assert sizes == [4, 4, 8], f"masks collapsed to one another: {sizes}"


def test_fingerprint_distinguishes_regions_and_labelings():
    a = ClusterConstraint(labels=np.array([0, 0, 1, 1]), clusters=(0,))
    b = ClusterConstraint(labels=np.array([0, 0, 1, 1]), clusters=(1,))
    # same cluster id, different partition -- must not share a cache entry
    c = ClusterConstraint(labels=np.array([1, 1, 0, 0]), clusters=(0,))
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert a.fingerprint() == ClusterConstraint(labels=np.array([0, 0, 1, 1]),
                                                clusters=(0,)).fingerprint()


def test_cluster_library_partitions_everything():
    for method in ("kmeans", "scaffold"):
        lab = cluster_library(LIB, 4, method=method)
        assert len(lab) == len(LIB)
        assert lab.min() >= 0


def test_concentration_detects_a_planted_hotspot():
    labels = np.array([0] * 50 + [1] * 50)
    values = np.concatenate([np.zeros(50), np.ones(50)])   # all the good ones in cluster 1
    c = concentration(labels, values, top_frac=0.1)
    assert c["best_cluster"] == 1
    assert c["coverage_best"] == 1.0
    assert c["best_lift"] == pytest.approx(2.0)            # 100% of top in 50% of library
    assert c["clusters_for_half_the_top"] == 1


def test_concentration_reports_no_signal_when_uniform():
    labels = np.tile([0, 1], 50)
    values = np.arange(100.0)                              # good ones alternate
    c = concentration(labels, values, top_frac=0.1)
    assert 0.8 < c["best_lift"] < 1.3, "uniform spread should give lift ~1"


def test_fixed_source_returns_the_named_region():
    labels = np.array([0, 0, 1, 1])
    con = FixedClusterSource(labels, 1)()
    assert con.clusters == (1,)
    np.testing.assert_array_equal(con.mask(None), [False, False, True, True])


def test_ucb_visits_an_unseen_region_before_repeating_a_seen_one():
    ctx = LibraryContext.build(LIB)
    labels = np.array([0] * 8 + [1] * 8)
    src = UCBClusterSource(labels=labels, c=1.0)
    obs = [(LIB[0], 9.0), (LIB[1], 9.5)]        # only region 0 sampled, and it looks good
    assert src(observations=obs, ctx=ctx).clusters == (1,), \
        "an unvisited region must be tried before exploiting a good one"


def test_ucb_exploits_the_better_region_once_both_are_seen():
    ctx = LibraryContext.build(LIB)
    labels = np.array([0] * 8 + [1] * 8)
    src = UCBClusterSource(labels=labels, c=0.01)   # near-greedy
    obs = [(LIB[i], 2.0) for i in range(4)] + [(LIB[8 + i], 9.0) for i in range(4)]
    assert src(observations=obs, ctx=ctx).clusters == (1,)


def test_a_prior_steers_the_first_choice():
    """The only thing a learned chooser can offer over the bandit: an opinion
    before any sampling. With no data, the prior decides."""
    ctx = LibraryContext.build(LIB)
    labels = np.array([0] * 8 + [1] * 8)
    cold = UCBClusterSource(labels=labels)
    warm = UCBClusterSource(labels=labels, prior=np.array([0.0, 10.0]), prior_weight=5.0)
    assert warm(observations=[], ctx=ctx).clusters == (1,)
    # and it is overridable by enough contrary evidence
    obs = [(LIB[8 + i], -5.0) for i in range(8)] + [(LIB[i], 8.0) for i in range(8)]
    assert warm(observations=obs, ctx=ctx).clusters == (0,)
    assert cold(observations=[], ctx=ctx) is not None


def test_prior_mean_is_inert_when_none_and_steers_when_given(tmp_path):
    """The GP prior-mean hook: zero prior must reproduce plain BO exactly, and a
    perfect prior must beat it. If the first fails the hook is not inert; if the
    second fails it is not wired into the posterior."""
    import pandas as pd

    from molbo.campaign import ArmConfig
    from molbo.experiment import run_sweep
    from molbo.oracles import LookupOracle

    lib = [f"C{'C' * i}{'O' if i % 2 else 'N'}" for i in range(1, 41)]
    vals = np.linspace(4, 10, len(lib))
    csv = tmp_path / "lib.csv"
    pd.DataFrame({"canonical_smiles": lib, "pchembl_value": vals}).to_csv(csv, index=False)
    oracle = LookupOracle.from_dataset(csv, budget=20)
    order = {s: v for s, v in zip(lib, vals)}
    truth = np.array([order[s] for s in oracle.library])

    arms = [
        ArmConfig("plain", strategy="bo", acquisition="ei", n_init=5),
        ArmConfig("zero_prior", strategy="bo", acquisition="ei", n_init=5,
                  prior_mean=np.zeros(len(oracle.library))),
        ArmConfig("perfect_prior", strategy="bo", acquisition="ei", n_init=5,
                  prior_mean=truth),
    ]
    m, _ = run_sweep(oracle, arms, range(12), budget=20, k=5,
                     library=list(oracle.library), verbose=False)
    auc = {a: np.mean([x.auc for x in m if x.arm == a])
           for a in ("plain", "zero_prior", "perfect_prior")}
    assert auc["zero_prior"] == pytest.approx(auc["plain"], abs=1e-9), \
        "a zero prior mean must reproduce plain BO exactly"
    assert auc["perfect_prior"] > auc["plain"], \
        "a prior mean equal to the truth must beat plain BO"
