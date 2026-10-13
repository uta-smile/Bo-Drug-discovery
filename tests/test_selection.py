"""BO shortlists, something else decides: the selector path.

The point of the arm is that a bad decision here is recoverable -- it costs one
evaluation, unlike a bad constraint, which fences off part of the library for
the rest of the run. These tests protect the accounting that makes that true:
the selector may only choose *within* the shortlist, it may never shorten the
budget, and a selector that fails must be visibly distinguishable from one that
agrees with BO.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbo.campaign import ArmConfig, _as_indices, run_campaign
from molbo.constraints import LibraryContext
from molbo.llm import RandomShortlistSelector, build_select_prompt
from molbo.oracles import LookupOracle
from molbo.surrogate import fingerprint_matrix

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


@pytest.fixture(scope="module")
def shared():
    oracle = LookupOracle(TABLE, name="toy")
    lib = oracle.library
    return oracle, lib, LibraryContext.build(lib), fingerprint_matrix(lib)


def run(shared, arm, seed=0, budget=12):
    oracle, lib, ctx, X = shared
    return run_campaign(oracle, arm, seed, budget=budget, library=lib, ctx=ctx, X=X)


# ---- index normalisation --------------------------------------------------


def test_a_valid_choice_is_honoured():
    assert _as_indices(3, k=10, q=1).tolist() == [3]


def test_an_out_of_range_choice_is_dropped_not_clamped():
    """Snapping rank 40 to rank 9 would credit a choice the selector never made."""
    assert _as_indices(40, k=10, q=1).tolist() == [0]


def test_a_negative_choice_is_dropped():
    assert _as_indices(-1, k=10, q=1).tolist() == [0]


def test_none_defers_to_bo_ranking():
    assert _as_indices(None, k=10, q=1).tolist() == [0]


def test_duplicates_are_collapsed_and_backfilled():
    assert _as_indices([2, 2, 2], k=10, q=3).tolist() == [2, 0, 1]


def test_the_batch_is_always_filled():
    """A selector returning too few must not shorten the budget."""
    assert len(_as_indices([5], k=10, q=4)) == 4


def test_extra_choices_are_truncated_to_the_batch():
    assert len(_as_indices([1, 2, 3, 4, 5], k=10, q=2)) == 2


def test_choices_are_unique():
    out = _as_indices([7, 7, 3], k=10, q=3).tolist()
    assert len(set(out)) == len(out)


# ---- the dial -------------------------------------------------------------


def test_shortlist_of_one_is_exactly_plain_bo(shared):
    """k=1 leaves the selector no choice, so the arm must reproduce plain BO.

    This is what makes shortlist_k interpretable as 'how much of the decision
    the selector is making'.
    """
    seen = []

    def spy(shortlist, **kw):
        seen.append(len(shortlist))
        return 0

    plain = run(shared, ArmConfig("plain", strategy="bo", n_init=4), seed=3)
    dialed = run(
        shared,
        ArmConfig("k1", strategy="bo", n_init=4, selector=spy, shortlist_k=1),
        seed=3,
    )
    assert dialed.smiles == plain.smiles
    assert set(seen) == {1}


def test_always_picking_rank_zero_reproduces_plain_bo(shared):
    plain = run(shared, ArmConfig("plain", strategy="bo", n_init=4), seed=1)
    top = run(
        shared,
        ArmConfig("top", strategy="bo", n_init=4, selector=lambda **kw: 0, shortlist_k=8),
        seed=1,
    )
    assert top.smiles == plain.smiles


def test_a_selector_changes_what_is_bought(shared):
    plain = run(shared, ArmConfig("plain", strategy="bo", n_init=4), seed=1)
    last = run(
        shared,
        ArmConfig("last", strategy="bo", n_init=4,
                  selector=lambda shortlist, **kw: len(shortlist) - 1, shortlist_k=8),
        seed=1,
    )
    assert last.smiles != plain.smiles


# ---- budget discipline ----------------------------------------------------


@pytest.mark.parametrize(
    "selector",
    [lambda **kw: 0, lambda **kw: 999, lambda **kw: None,
     lambda **kw: [0, 1], RandomShortlistSelector()],
)
def test_no_selector_can_shorten_the_budget(shared, selector):
    res = run(shared, ArmConfig("s", strategy="bo", n_init=4, selector=selector,
                                shortlist_k=6), budget=12)
    assert res.n_calls == 12


def test_the_selector_never_buys_outside_the_shortlist(shared):
    """The chosen molecule must come from what BO actually proposed."""
    offered: list[set[str]] = []

    def spy(shortlist, **kw):
        offered.append(set(shortlist))
        return 1

    res = run(shared, ArmConfig("s", strategy="bo", n_init=4, selector=spy, shortlist_k=5),
              budget=10)
    # every post-init purchase came from the shortlist offered that round
    for smi, offer in zip(res.smiles[4:], offered):
        assert smi in offer


def test_selector_choices_are_logged(shared):
    res = run(shared, ArmConfig("s", strategy="bo", n_init=4,
                                selector=lambda **kw: 2, shortlist_k=6), budget=10)
    assert res.selector_log
    entry = res.selector_log[0]
    assert entry["chosen_rank"] == [2]
    assert entry["shortlist_k"] == 6


def test_no_selector_leaves_the_log_empty(shared):
    assert run(shared, ArmConfig("plain", strategy="bo", n_init=4)).selector_log == []


def test_selector_sees_the_posterior_and_the_observations(shared):
    captured = {}

    def spy(shortlist, scores, mu, sd, observations, **kw):
        captured.update(n=len(shortlist), mu=len(mu), sd=len(sd), obs=len(observations))
        return 0

    run(shared, ArmConfig("s", strategy="bo", n_init=4, selector=spy, shortlist_k=5), budget=8)
    assert captured["n"] == captured["mu"] == captured["sd"] == 5
    assert captured["obs"] >= 4


# ---- the control ----------------------------------------------------------


def test_random_shortlist_selector_stays_in_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        out = RandomShortlistSelector()(shortlist=["a"] * 7, rng=rng, q=1)
        assert all(0 <= i < 7 for i in out)


def test_random_shortlist_selector_actually_varies():
    """Otherwise it is not a control for 'any deviation from rank 0 helps'."""
    picks = {
        tuple(RandomShortlistSelector()(shortlist=["a"] * 9,
                                        rng=np.random.default_rng(s), q=1))
        for s in range(30)
    }
    assert len(picks) > 1


# ---- prompt ---------------------------------------------------------------


def test_select_prompt_numbers_every_candidate_and_shows_uncertainty():
    text = build_select_prompt(
        ["CCO", "CCC", "CCN"], mu=[8.1, 7.4, 6.9], sd=[0.5, 1.2, 0.3],
        observations=[("c1ccccc1", 5.0)],
    )
    for i in range(3):
        assert f"[ {i}]" in text
    assert "8.10" in text and "+/- 0.50" in text
    assert "5.00" in text  # the observed compound


def test_k_of_one_matches_plain_bo_even_on_a_flat_surface(shared):
    """The degenerate case the shared ranking exists for.

    Early on, every EI is numerically zero. Plain BO breaks that tie at random;
    a bare argsort would break it at index 0, so the two would silently diverge
    exactly when the claim "k=1 is plain BO" is most load-bearing.
    """
    oracle, lib, ctx, X = shared
    flat = ArmConfig("flat", strategy="bo", acquisition="greedy", n_init=2)
    sel = ArmConfig("flat_k1", strategy="bo", acquisition="greedy", n_init=2,
                    selector=lambda **kw: 0, shortlist_k=1)
    for seed in range(4):
        assert run(shared, sel, seed=seed).smiles == run(shared, flat, seed=seed).smiles


# ---- warm start -----------------------------------------------------------


def test_warm_start_replaces_the_initial_design(shared):
    oracle, lib, ctx, X = shared
    chosen = [3, 1, 4, 1, 5][:4]

    res = run(
        shared,
        ArmConfig("ws", strategy="bo", n_init=4,
                  init_source=lambda **kw: [3, 1, 4, 5]),
        seed=0,
    )
    assert res.smiles[:4] == [lib[i] for i in [3, 1, 4, 5]]


def test_warm_start_falling_back_gives_the_random_design(shared):
    """A failed model call must not change what the arm starts from."""
    plain = run(shared, ArmConfig("plain", strategy="bo", n_init=4), seed=2)
    fell_back = run(
        shared, ArmConfig("ws", strategy="bo", n_init=4, init_source=lambda **kw: None), seed=2
    )
    assert fell_back.smiles[:4] == plain.smiles[:4]


def test_warm_start_is_truncated_to_n_init(shared):
    res = run(
        shared,
        ArmConfig("ws", strategy="bo", n_init=3, init_source=lambda **kw: [0, 1, 2, 3, 4, 5]),
        seed=0,
    )
    assert len(set(res.smiles[:3])) == 3


def test_diversity_warm_start_picks_spread_out_molecules(shared):
    """The control must actually be diverse, or it is not a control."""
    from molbo.llm import DiverseInitSelector
    from molbo.surrogate import fingerprint_matrix, tanimoto_gram

    oracle, lib, ctx, X = shared
    picked = DiverseInitSelector(sample_size=len(lib))(library=lib, seed=0, n_init=5)
    rnd = list(np.random.default_rng(0).choice(len(lib), size=5, replace=False))

    def spread(idx):
        g = tanimoto_gram(fingerprint_matrix([lib[i] for i in idx]),
                          fingerprint_matrix([lib[i] for i in idx]))
        return g[np.triu_indices(len(idx), k=1)].mean()

    assert spread(picked) < spread(rnd)  # lower mean similarity = more diverse


def test_diversity_warm_start_is_deterministic(shared):
    from molbo.llm import DiverseInitSelector

    oracle, lib, ctx, X = shared
    a = DiverseInitSelector()(library=lib, seed=1, n_init=6)
    b = DiverseInitSelector()(library=lib, seed=1, n_init=6)
    assert a == b
    assert len(set(a)) == 6


def test_diversity_warm_start_varies_with_the_seed(shared):
    """Only through the sample it draws -- MaxMin itself is deterministic."""
    from molbo.llm import DiverseInitSelector

    oracle, lib, ctx, X = shared
    sel = DiverseInitSelector(sample_size=8)  # must be smaller than the library
    assert sel(library=lib, seed=1, n_init=4) != sel(library=lib, seed=2, n_init=4)


def test_diversity_warm_start_is_seed_independent_when_it_sees_everything(shared):
    """With the sample covering the whole library there is nothing for the seed
    to vary, so every seed shares one initial design. Harmless at the real scale
    (40 drawn from 2,086), but it would understate the arm's variance if it held."""
    from molbo.llm import DiverseInitSelector

    oracle, lib, ctx, X = shared
    sel = DiverseInitSelector(sample_size=len(lib) * 2)
    assert sel(library=lib, seed=1, n_init=4) == sel(library=lib, seed=2, n_init=4)
