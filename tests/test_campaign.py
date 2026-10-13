"""Campaigns: budget discipline, arm pairing, and constraint handling in the loop.

Most of these protect comparability rather than correctness. An arm that
quietly buys 210 molecules on a 200 budget, or that starts from a different
initial design than its rivals, produces a curve that looks like a better
method and is not one.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbo.campaign import ArmConfig, run_campaign
from molbo.constraints import LibraryContext, MaskConstraint, SubstructureConstraint
from molbo.oracles import LookupOracle
from molbo.surrogate import LibraryKernel, fingerprint_matrix

# A toy congeneric-ish series: anilinoquinazolines score high, the rest low.
TABLE = {
    "c1ccc2ncnc(Nc3ccccc3)c2c1": 9.0,
    "COc1cc2ncnc(Nc3ccccc3)c2cc1OC": 9.5,
    "c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1": 8.5,
    "c1ccc2ncnc(Nc3ccc(F)cc3)c2c1": 8.2,
    "c1ccc2ncnc(Nc3ccc(Br)cc3)c2c1": 8.8,
    "c1ccc2ncnc(Nc3cccc(C)c3)c2c1": 7.9,
    "CCO": 4.0,
    "CCCCCCCC": 4.2,
    "c1ccccc1": 5.0,
    "CC(=O)Oc1ccccc1C(=O)O": 5.5,
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O": 4.8,
    "CCCCO": 4.1,
    "c1ccc(Cl)cc1": 5.2,
    "CCN(CC)CC": 4.3,
    "OCC(O)CO": 4.4,
    "c1ccncc1": 5.1,
}
QUINAZOLINE = "c1ccc2ncnc(N)c2c1"


@pytest.fixture(scope="module")
def shared():
    oracle = LookupOracle(TABLE, name="toy")
    lib = oracle.library
    return oracle, lib, LibraryContext.build(lib), fingerprint_matrix(lib)


def run(shared, arm, seed=0, budget=10, **kw):
    oracle, lib, ctx, X = shared
    return run_campaign(oracle, arm, seed, budget=budget, library=lib, ctx=ctx, X=X, **kw)


# ---- budget ---------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["random", "bo"])
def test_campaign_spends_exactly_its_budget(shared, strategy):
    res = run(shared, ArmConfig("a", strategy=strategy, n_init=4), budget=10)
    assert res.n_calls == 10
    assert len(res.smiles) == 10


def test_campaign_never_re_proposes_a_molecule(shared):
    """A repeat is free at the oracle, so it would be a stealth budget extension."""
    res = run(shared, ArmConfig("a", strategy="bo", n_init=4), budget=12)
    assert len(set(res.smiles)) == len(res.smiles)


def test_budget_larger_than_the_library_is_rejected(shared):
    with pytest.raises(ValueError, match="exceeds the library"):
        run(shared, ArmConfig("a"), budget=len(TABLE) + 1)


def test_initial_design_can_fill_the_whole_budget(shared):
    res = run(shared, ArmConfig("a", strategy="bo", n_init=8), budget=5)
    assert res.n_calls == 5


# ---- pairing across arms --------------------------------------------------


def test_same_seed_gives_every_arm_the_same_initial_design(shared):
    """Without this, arms cannot be compared with a paired test."""
    a = run(shared, ArmConfig("a", strategy="bo", n_init=5), seed=3)
    b = run(shared, ArmConfig("b", strategy="random", n_init=5), seed=3)
    assert a.smiles[:5] == b.smiles[:5]


def test_different_seeds_give_different_initial_designs(shared):
    a = run(shared, ArmConfig("a", n_init=5), seed=1)
    b = run(shared, ArmConfig("a", n_init=5), seed=2)
    assert a.smiles[:5] != b.smiles[:5]


def test_a_campaign_is_reproducible_from_its_seed(shared):
    arm = ArmConfig("a", strategy="bo", n_init=4)
    assert run(shared, arm, seed=7).smiles == run(shared, arm, seed=7).smiles


def test_constrained_arms_keep_the_same_rng_stream_as_plain_bo(shared):
    """The constraint source must not consume draws from the campaign's RNG.

    If it did, a constrained arm would tie-break differently everywhere and the
    arms would stop being paired for reasons unrelated to the constraint.
    """
    oracle, lib, ctx, X = shared
    allow_all = MaskConstraint(values=np.ones(len(lib), bool), label="all")
    plain = run(shared, ArmConfig("plain", strategy="bo", n_init=4), seed=5)
    constrained = run(
        shared,
        ArmConfig(
            "c",
            strategy="bo",
            n_init=4,
            constraint_source=lambda **kw: allow_all,
            min_candidates=1,
        ),
        seed=5,
    )
    assert plain.smiles == constrained.smiles


# ---- the precomputed kernel -----------------------------------------------


def test_the_kernel_shortcut_does_not_change_any_choice(shared):
    """The speed path must select identical molecules in an identical order."""
    oracle, lib, ctx, X = shared
    arm = ArmConfig("a", strategy="bo", n_init=4)
    slow = run(shared, arm, seed=11, kernel=False)
    fast = run(shared, arm, seed=11, kernel=LibraryKernel(X))
    assert slow.smiles == fast.smiles
    assert slow.values == fast.values


# ---- constraints in the loop ----------------------------------------------


def test_a_hard_constraint_confines_the_search(shared):
    res = run(
        shared,
        ArmConfig(
            "c",
            strategy="bo",
            n_init=2,
            constraint_source=lambda **kw: SubstructureConstraint(require=(QUINAZOLINE,)),
            min_candidates=2,
        ),
        seed=0,
        budget=8,
    )
    after_init = res.smiles[2:]
    assert all("ncnc" in s or "c2c1" in s for s in after_init)
    assert res.constraint_log and res.constraint_log[0]["mode"] == "hard"


def test_an_exhausted_constraint_falls_back_rather_than_ending_the_run(shared):
    """Buying every allowed molecule must not truncate the budget."""
    oracle, lib, ctx, X = shared
    quinazolines = SubstructureConstraint(require=(QUINAZOLINE,)).mask(ctx)
    assert quinazolines.sum() == 6

    res = run(
        shared,
        ArmConfig(
            "c",
            strategy="bo",
            n_init=2,
            constraint_source=lambda **kw: SubstructureConstraint(require=(QUINAZOLINE,)),
            min_candidates=1,
        ),
        seed=0,
        budget=12,
    )
    assert res.n_calls == 12  # full budget spent despite only 6 allowed
    assert any(e.get("mode") in ("exhausted_fallback", "soft") for e in res.constraint_log)


def test_a_constraint_source_returning_none_runs_unconstrained(shared):
    res = run(
        shared,
        ArmConfig("c", strategy="bo", n_init=4, constraint_source=lambda **kw: None),
        budget=10,
    )
    assert res.n_calls == 10
    assert res.constraint_log[0]["mode"] == "none"


def test_constraint_source_only_sees_observed_pairs(shared):
    """It must never be handed the library's unobserved values."""
    seen = {}

    def spy(observations, ctx, rng, round_index, seed=0, **_):
        seen[round_index] = list(observations)
        return None

    oracle = shared[0]
    res = run(shared, ArmConfig("c", strategy="bo", n_init=4, constraint_source=spy), budget=9)
    first = seen[0]
    assert len(first) <= 4
    assert {s for s, _ in first} <= set(res.smiles)
    # Keys are the oracle's canonical SMILES, which is not always how TABLE
    # spelled them -- that re-keying is the whole point of the chem layer.
    truth = dict(zip(oracle.library, oracle.values))
    for smi, val in first:
        assert val == truth[smi]


def test_constraint_refresh_cadence_is_respected(shared):
    calls = []

    def counting(observations, ctx, rng, round_index, seed=0, **_):
        calls.append(round_index)
        return None

    run(
        shared,
        ArmConfig(
            "c", strategy="bo", n_init=2, constraint_source=counting, constraint_every=2
        ),
        budget=10,
    )
    assert calls == [0, 2, 4, 6]


def test_constraint_proposed_once_by_default(shared):
    calls = []

    def counting(observations, ctx, rng, round_index, seed=0, **_):
        calls.append(round_index)
        return None

    run(shared, ArmConfig("c", strategy="bo", n_init=2, constraint_source=counting), budget=8)
    assert calls == [0]


def test_the_campaign_seed_reaches_the_constraint_source(shared):
    got = []
    run(
        shared,
        ArmConfig("c", n_init=2, constraint_source=lambda seed=0, **kw: got.append(seed)),
        seed=42,
        budget=5,
    )
    assert got == [42]


# ---- result object --------------------------------------------------------


def test_bo_beats_random_on_a_series_with_obvious_structure(shared):
    """A smoke test, not a result -- but if it fails, nothing else means anything."""
    bo = [run(shared, ArmConfig("bo", strategy="bo", n_init=3), seed=s, budget=8).best()
          for s in range(8)]
    rand = [run(shared, ArmConfig("r", strategy="random", n_init=3), seed=s, budget=8).best()
            for s in range(8)]
    assert np.mean(bo) >= np.mean(rand)


def test_result_reports_the_best_value_and_drops_unscored(shared):
    res = run(shared, ArmConfig("a", n_init=4), budget=10)
    assert res.best() == max(res.scored)
    assert res.n_failed == 0
    assert res.to_dict()["arm"] == "a"


def test_unknown_strategy_is_rejected(shared):
    with pytest.raises(ValueError, match="unknown strategy"):
        run(shared, ArmConfig("a", strategy="telekinesis"))
