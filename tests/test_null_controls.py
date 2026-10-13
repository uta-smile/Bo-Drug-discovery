"""The two section-32 controls: `--sd-scale` and `--aux-shuffle`.

Section 32 found a pure-noise auxiliary buying recall under expected
improvement and blamed uncertainty compression rather than transfer.
`null_mechanism.py` then located the compression at a rung where the task
covariance is diagonal and `r` is exactly zero -- so the channel is the shared
hyperparameters being refitted on the joint data, not the coupling. `sd_scale`
is the control that closes that argument at campaign level: a single-task arm
whose uncertainty is set to the multi-task arm's scale, with no auxiliary data
at all.

A control is only worth running if it is the thing it claims to be, so:

1. shrinking sd makes EI more exploitative (the mechanism the control asserts),
2. the posterior mean and the scoring are untouched (it is not a second knob),
3. and 1.0 is a genuine no-op, so every stored run still reproduces.

The third is the project's standing constraint. Every arm comparison in
sections 18-35 is stored against a campaign that did not have this parameter.

`--aux-shuffle` is the second null the same section is missing. `--aux-quality
0` corrupts a column towards zero rank correlation and preserves neither its
marginal distribution nor its pile-ups; a permutation destroys the information
and preserves both exactly. Running them side by side is what separates the
information in an auxiliary column from its shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dense_block_campaign as dbc  # noqa: E402
from molbo.acquisition import expected_improvement  # noqa: E402


@pytest.fixture
def toy():
    """A library where value tracks one coordinate and similarity tracks it
    too, so a GP fitted on part of it has somewhere left to explore."""
    rng = np.random.default_rng(0)
    n = 90
    z = np.sort(rng.random(n))
    K = np.exp(-((z[:, None] - z[None, :]) ** 2) / 0.02)
    np.fill_diagonal(K, 1.0)
    y = z * 10 + rng.normal(0, 0.3, n)
    return K, y


def run(K, y, *, sd_scale, seed=3, arm="stl"):
    return dbc.run_campaign(
        K, y, [], [], arm=arm, budget=30, batch=5, n_init=10, k=5,
        acquisition="ei", sd_scale=sd_scale,
        rng=np.random.default_rng(seed))


# -- 1. the mechanism the control asserts ------------------------------------

def test_shrinking_sd_makes_ei_more_exploitative():
    """The claim `sd_scale` embodies: compress uncertainty and EI stops paying
    for exploration. Checked on the acquisition itself, where it is a fact
    about EI rather than about any particular fit."""
    mu = np.array([1.0, 0.9, 0.2, 0.1])
    sd = np.array([0.1, 0.2, 1.0, 2.0])
    best = 1.0
    wide = expected_improvement(mu, sd, best)
    tight = expected_improvement(mu, sd * 0.2, best)
    # Under a wide posterior the uncertain low-mean candidate outranks the
    # confident high-mean one; compressing sd reverses that.
    assert wide.argmax() == 3
    assert tight.argmax() == 0


def test_sd_scale_changes_the_campaign(toy):
    K, y = toy
    base = run(K, y, sd_scale=1.0)[1]
    tight = run(K, y, sd_scale=0.2)[1]
    assert not np.array_equal(base, tight), (
        "a tenfold change in exploration that selects the same compounds "
        "means the knob is not reaching the acquisition")


def test_initial_design_is_untouched(toy):
    """Only the acquisition may see this. If the initial design moved, a
    difference between two sd_scale runs would be a different starting point
    rather than a different exploration level."""
    K, y = toy
    base = run(K, y, sd_scale=1.0)[1]
    tight = run(K, y, sd_scale=0.2)[1]
    assert np.array_equal(base[:10], tight[:10])


# -- 2. it is one knob, not two ----------------------------------------------

def test_greedy_is_immune(toy):
    """`greedy` ranks on the posterior mean alone. If scaling sd moved a greedy
    campaign, the knob would be touching the fit and not just the acquisition
    -- which would make it a second degradation rather than a control."""
    K, y = toy
    a = dbc.run_campaign(K, y, [], [], arm="stl", budget=30, batch=5,
                         n_init=10, k=5, acquisition="greedy", sd_scale=1.0,
                         rng=np.random.default_rng(3))[1]
    b = dbc.run_campaign(K, y, [], [], arm="stl", budget=30, batch=5,
                         n_init=10, k=5, acquisition="greedy", sd_scale=0.05,
                         rng=np.random.default_rng(3))[1]
    assert np.array_equal(a, b)


def test_random_arm_is_immune(toy):
    """The floor fits no model, so it cannot respond to a posterior knob."""
    K, y = toy
    a = run(K, y, sd_scale=1.0, arm="random")[1]
    b = run(K, y, sd_scale=0.05, arm="random")[1]
    assert np.array_equal(a, b)


# -- 3. the standing constraint ----------------------------------------------

def test_default_is_a_no_op(toy):
    """Not 'equals 1.0' but 'skipped': the stored sections were run by a
    campaign that never multiplied, and a multiply by 1.0 is only exactly
    identity for finite floats, not for the code path."""
    K, y = toy
    import inspect
    src = inspect.getsource(dbc.run_campaign)
    assert "if sd_scale != 1.0:" in src, (
        "the multiply must be guarded, not applied unconditionally")
    assert inspect.signature(dbc.run_campaign).parameters["sd_scale"].default == 1.0


def test_cli_default_is_one():
    """Opt-in, not opt-out. A flag that changed exploration by default would
    silently invalidate every stored artefact."""
    src = Path(dbc.__file__).read_text(encoding="utf-8")
    assert '"--sd-scale", type=float, default=1.0' in src


def test_sd_scale_one_matches_omitting_it(toy):
    """The byte-for-byte check the rest of the package depends on."""
    K, y = toy
    with_param = run(K, y, sd_scale=1.0)[0]
    without = dbc.run_campaign(K, y, [], [], arm="stl", budget=30, batch=5,
                               n_init=10, k=5, acquisition="ei",
                               rng=np.random.default_rng(3))[0]
    assert np.array_equal(with_param, without)


# -- the second null: permutation --------------------------------------------

def test_shuffle_preserves_the_marginal_exactly():
    """The property that distinguishes it from `--aux-quality 0`. A corrupted
    column has a different distribution; a permuted one has the same
    multiset, pile-ups included."""
    rng = np.random.default_rng(0)
    col = np.concatenate([np.full(40, 5.0), rng.normal(7, 1, 60)])
    assert np.array_equal(np.sort(rng.permutation(col)), np.sort(col))


def test_shuffle_destroys_the_information():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    col = rng.normal(size=400)
    rs = [abs(spearmanr(rng.permutation(col), col).statistic) for _ in range(30)]
    assert np.mean(rs) < 0.1


def test_the_two_nulls_are_mutually_exclusive():
    """They are different nulls against different baselines. Combining them
    would report one number for two manipulations."""
    with pytest.raises(SystemExit, match="two different"):
        dbc.main(["--family", "cyp", "--aux-shuffle", "--aux-quality", "0"])


def test_shuffle_is_opt_in():
    src = Path(dbc.__file__).read_text(encoding="utf-8")
    assert '"--aux-shuffle", action="store_true"' in src


def test_shuffle_permutes_the_whole_column_before_indexing():
    """The default remains the original full-column null, not history scope."""
    vals = np.arange(120, dtype=float).reshape(40, 3)
    idx = [np.array([2, 7, 13]), np.array([21, 27, 33])]
    rng = np.random.default_rng(19)
    expected = [rng.permutation(vals[:, j])[i] for i, j in zip(idx, [1, 2])]
    actual, provenance = dbc.shuffle_auxiliary_responses(
        vals, idx, [1, 2], np.random.default_rng(19))
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    assert all(p["scope"] == "library" for p in provenance)


# -- 4. `stl_avoid`: history avoidance with no auxiliary values --------------
#
# The C4 reversal has two candidate causes tangled together inside the pooled
# arm: it borrows auxiliary *values*, and it shies away from auxiliary-measured
# *compounds*. `stl_avoid` isolates the second. These tests check it is that
# and only that, because a control that quietly does something else is worse
# than no control.

def _avoid_run(K, y, aux_idx, aux_val, *, arm, seed=3):
    return dbc.run_campaign(
        K, y, aux_idx, aux_val, arm=arm, budget=30, batch=5, n_init=10, k=5,
        acquisition="ei", rng=np.random.default_rng(seed))


def test_stl_avoid_never_buys_a_history_compound(toy):
    """The property the arm is named for."""
    K, y = toy
    hist = np.arange(20, 50)
    _, picked = _avoid_run(K, y, [hist], [y[hist]], arm="stl_avoid")
    bought = set(picked[10:].tolist())          # the initial design is random
    assert not (bought & set(hist.tolist()))


def test_stl_fitted_alongside_does_buy_them(toy):
    """Without which the test above would pass on a library that never wanted
    those compounds anyway, and prove nothing."""
    K, y = toy
    hist = np.arange(20, 50)
    _, picked = _avoid_run(K, y, [hist], [y[hist]], arm="stl")
    assert set(picked[10:].tolist()) & set(hist.tolist())


def test_stl_avoid_reads_no_auxiliary_values(toy):
    """Identities only. Replacing every auxiliary measurement with noise must
    not move a single pick -- otherwise the arm is a transfer method and
    cannot be used to argue that selection alone explains the reversal."""
    K, y = toy
    hist = np.arange(20, 50)
    real = _avoid_run(K, y, [hist], [y[hist]], arm="stl_avoid")[1]
    junk = _avoid_run(K, y, [hist],
                      [np.random.default_rng(99).normal(size=len(hist))],
                      arm="stl_avoid")[1]
    assert np.array_equal(real, junk)


def test_stl_avoid_collapses_to_stl_without_history(toy):
    """With nothing to avoid it must be bit-identical to `stl`, so any
    difference between the two arms is the avoidance and nothing else."""
    K, y = toy
    a = _avoid_run(K, y, [], [], arm="stl")[1]
    b = _avoid_run(K, y, [], [], arm="stl_avoid")[1]
    assert np.array_equal(a, b)


def test_stl_avoid_keeps_the_initial_design(toy):
    """Avoidance is a selection rule, not a different starting point."""
    K, y = toy
    hist = np.arange(20, 50)
    a = _avoid_run(K, y, [hist], [y[hist]], arm="stl")[1]
    b = _avoid_run(K, y, [hist], [y[hist]], arm="stl_avoid")[1]
    assert np.array_equal(a[:10], b[:10])


def test_stl_avoid_spends_the_whole_budget(toy):
    """Demoted, not made ineligible: the arm must still buy `budget` compounds
    even when the history covers most of the library."""
    K, y = toy
    hist = np.arange(0, 75)
    vals, picked = _avoid_run(K, y, [hist], [y[hist]], arm="stl_avoid")
    assert len(picked) == 30 and len(set(picked.tolist())) == 30


def test_avoid_weight_one_is_a_no_op(toy):
    """W=1.0 must reproduce `stl` exactly, or the sweep has no clean origin."""
    K, y = toy
    hist = np.arange(20, 50)
    base = dbc.run_campaign(K, y, [hist], [y[hist]], arm="stl", budget=30,
                            batch=5, n_init=10, k=5, acquisition="ei",
                            rng=np.random.default_rng(3))[1]
    w1 = dbc.run_campaign(K, y, [hist], [y[hist]], arm="stl_avoid", budget=30,
                          batch=5, n_init=10, k=5, acquisition="ei",
                          avoid_weight=1.0, rng=np.random.default_rng(3))[1]
    assert np.array_equal(base, w1)


def test_avoid_weight_is_monotone_in_history_uptake(toy):
    """Lower W must not buy *more* history. The sweep's whole premise."""
    K, y = toy
    hist = np.arange(20, 50)
    hs = set(hist.tolist())
    bought = []
    for w in (1.0, 0.75, 0.5, 0.25, 0.0):
        picked = dbc.run_campaign(K, y, [hist], [y[hist]], arm="stl_avoid",
                                  budget=30, batch=5, n_init=10, k=5,
                                  acquisition="ei", avoid_weight=w,
                                  rng=np.random.default_rng(3))[1]
        bought.append(len(set(picked[10:].tolist()) & hs))
    assert bought == sorted(bought, reverse=True), bought
    assert bought[0] > bought[-1], "the knob did nothing"


def test_avoid_weight_default_is_the_strict_arm(toy):
    """Omitting the knob must leave the published `stl_avoid` runs valid."""
    K, y = toy
    hist = np.arange(20, 50)
    a = dbc.run_campaign(K, y, [hist], [y[hist]], arm="stl_avoid", budget=30,
                         batch=5, n_init=10, k=5, acquisition="ei",
                         rng=np.random.default_rng(3))[1]
    b = dbc.run_campaign(K, y, [hist], [y[hist]], arm="stl_avoid", budget=30,
                         batch=5, n_init=10, k=5, acquisition="ei",
                         avoid_weight=None, rng=np.random.default_rng(3))[1]
    assert np.array_equal(a, b)
    assert not (set(a[10:].tolist()) & set(hist.tolist()))


def test_avoid_weight_leaves_other_arms_alone(toy):
    """The knob must not reach `stl` or `pooled` in the same run."""
    K, y = toy
    hist = np.arange(20, 50)
    kw = dict(budget=30, batch=5, n_init=10, k=5, acquisition="ei")
    plain = dbc.run_campaign(K, y, [hist], [y[hist]], arm="stl", **kw,
                             rng=np.random.default_rng(3))[1]
    with_knob = dbc.run_campaign(K, y, [hist], [y[hist]], arm="stl", **kw,
                                 avoid_weight=0.0,
                                 rng=np.random.default_rng(3))[1]
    assert np.array_equal(plain, with_knob)
