"""The LLM-only arm and the shared ladder.

``strategy="llm"`` is the arm the project never had: a model choosing molecules
with **no surrogate underneath it**. It exists so that a BO+LLM result can be
attributed. Without it, "bo_llm_select beat random" is compatible with the model
contributing nothing and the GP carrying the whole arm.

The properties worth pinning are the ones that would make the arm dishonest if
they broke:

* it must be refused without a selector, or it is ``random`` under a better name
* it must not see a posterior, and must not be shown a fabricated one
* it must spend its full batch, like every other arm, or its budget accounting
  drifts and the AUC is not comparable
* at ``shortlist_k = 1`` it must reproduce ``random`` exactly, the same way
  ``shortlist_k = 1`` collapses the BO+selector path to plain BO
"""

from __future__ import annotations

import numpy as np
import pytest

from molbo.campaign import ArmConfig, run_campaign
from molbo.constraints import LibraryContext
from molbo.ladder import CATEGORY, NEEDS_MODEL, LadderSpec, build_ladder, skipped_arms
from molbo.llm import build_select_prompt
from molbo.oracles import LookupOracle
from molbo.surrogate import fingerprint_matrix

TABLE = {
    "c1ccc2ncnc(Nc3ccccc3)c2c1": 9.0,
    "COc1cc2ncnc(Nc3ccccc3)c2cc1OC": 9.5,
    "c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1": 8.5,
    "c1ccc2ncnc(Nc3ccc(F)cc3)c2c1": 8.2,
    "c1ccc2ncnc(Nc3ccc(Br)cc3)c2c1": 8.8,
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


def run(shared, arm, seed=0, budget=10):
    oracle, lib, ctx, X = shared
    return run_campaign(oracle, arm, seed, budget=budget, library=lib, ctx=ctx, X=X)


class RecordingSelector:
    """Always takes the first entry, and keeps what it was shown."""

    def __init__(self, pick: int = 0):
        self.pick = pick
        self.seen: list[dict] = []

    def __call__(self, *, shortlist, scores, mu, sd, **kw):
        self.seen.append({"shortlist": list(shortlist), "mu": np.asarray(mu),
                          "sd": np.asarray(sd), "scores": np.asarray(scores)})
        return self.pick


# ---- the arm is refused when it would be a mislabelled control -------------
def test_llm_strategy_requires_a_selector(shared):
    """Without a selector the arm is a uniform draw reported as an LLM result."""
    with pytest.raises(ValueError, match="requires a selector"):
        run(shared, ArmConfig("llm_only", strategy="llm", n_init=2))


def test_unknown_strategy_still_refused(shared):
    with pytest.raises(ValueError, match="unknown strategy"):
        run(shared, ArmConfig("nope", strategy="magic", n_init=2))


# ---- what the selector may and may not see ---------------------------------
def test_llm_arm_is_shown_no_posterior(shared):
    """No GP is fitted, so there is nothing to report -- and a placeholder
    number would be indistinguishable from a real prediction in the prompt."""
    sel = RecordingSelector()
    run(shared, ArmConfig("llm_only", strategy="llm", selector=sel,
                          shortlist_k=5, n_init=2))
    assert sel.seen, "the selector was never called"
    for call in sel.seen:
        assert np.all(np.isnan(call["mu"]))
        assert np.all(np.isnan(call["sd"]))


def test_select_prompt_drops_the_prediction_column_without_a_posterior():
    blank = [float("nan"), float("nan")]
    text = build_select_prompt(["CCO", "CCN"], blank, blank, [("CC", 5.0)])
    assert "pred " not in text, "the prediction column survived"
    assert "+/-" not in text, "an uncertainty column survived"
    assert "nan" not in text
    assert "drawn at random" in text
    # and the list is still numbered, or the model cannot answer with an index
    assert "[ 0]" in text and "[ 1]" in text


def test_select_prompt_keeps_the_prediction_column_with_a_posterior():
    text = build_select_prompt(["CCO", "CCN"], [1.0, 2.0], [0.1, 0.2], [("CC", 5.0)])
    assert "pred" in text and "ranked by the optimiser" in text


def test_llm_arm_never_offers_an_already_bought_molecule(shared):
    """The shortlist is drawn from the *untested* pool. If an evaluated molecule
    could reappear, the model could spend the budget re-buying its own pick and
    the arm would look stable while doing nothing."""
    sel = RecordingSelector()
    res = run(shared, ArmConfig("llm_only", strategy="llm", selector=sel,
                                shortlist_k=4, n_init=2), budget=10)
    assert len(res.smiles) == len(set(res.smiles)), "a molecule was bought twice"

    # Round i is offered a shortlist; whatever it buys must never be offered
    # again. Reconstruct the bought-so-far set at each call and check it.
    n_init = 2
    for i, call in enumerate(sel.seen):
        bought_before = set(res.smiles[: n_init + i])
        assert not (set(call["shortlist"]) & bought_before), (
            f"round {i} was offered an already-evaluated molecule"
        )


# ---- budget and comparability ----------------------------------------------
def test_llm_arm_spends_its_whole_budget(shared):
    sel = RecordingSelector()
    res = run(shared, ArmConfig("llm_only", strategy="llm", selector=sel,
                                shortlist_k=5, n_init=2), budget=10)
    assert res.n_calls == 10


def test_llm_arm_backfills_when_the_selector_declines(shared):
    """A model that returns nothing must not shorten the run: a short run is not
    comparable to a full one, and the arm would look better than it is."""

    class Declines:
        def __call__(self, **kw):
            return None

    res = run(shared, ArmConfig("llm_only", strategy="llm", selector=Declines(),
                                shortlist_k=5, n_init=2), budget=10)
    assert res.n_calls == 10


def test_shortlist_k_one_reproduces_random(shared):
    """With one candidate there is nothing to choose, so the arm must be the
    uniform draw exactly -- the same collapse `shortlist_k=1` gives on the BO
    path. If this drifts, the arm's floor is not the control it claims."""
    sel = RecordingSelector()
    a = run(shared, ArmConfig("llm_only", strategy="llm", selector=sel,
                              shortlist_k=1, n_init=2), seed=3, budget=10)
    b = run(shared, ArmConfig("random", strategy="random", n_init=2), seed=3, budget=10)
    assert a.smiles == b.smiles


def test_llm_arm_records_that_it_had_no_surrogate(shared):
    """The selector log has to distinguish this arm's picks from BO-shortlist
    picks, or a later reader cannot tell which ranking the rank refers to."""
    sel = RecordingSelector()
    res = run(shared, ArmConfig("llm_only", strategy="llm", selector=sel,
                                shortlist_k=5, n_init=2))
    assert res.selector_log
    assert all(e["surrogate"] is False for e in res.selector_log)


# ---- the ladder itself ------------------------------------------------------
def test_ladder_without_a_backend_skips_exactly_the_model_arms():
    arms = build_ladder(LadderSpec())
    names = {a.name for a in arms}
    assert not (names & NEEDS_MODEL)
    assert {"random", "bo_ei", "bo_random_select", "bo_policy_scripted"} <= names


def test_ladder_every_arm_has_a_declared_category():
    """An arm the report cannot categorise would be silently dropped from the
    grouped tables rather than shown as uncategorised."""
    arms = build_ladder(LadderSpec(), values=np.arange(5.0))
    for a in arms:
        assert a.name in CATEGORY, f"{a.name} is not in ladder.CATEGORY"


def test_ladder_ceiling_only_when_values_are_passed():
    assert "bo_policy_oracle" not in {a.name for a in build_ladder(LadderSpec())}
    with_values = build_ladder(LadderSpec(), values=np.arange(5.0))
    assert "bo_policy_oracle" in {a.name for a in with_values}


def test_skipped_arms_reports_what_did_not_run():
    assert skipped_arms(True, ("none", "bo", "llm")) == []
    assert set(skipped_arms(False, ("none", "bo", "llm", "bo+llm", "agentic"))) == set(
        NEEDS_MODEL
    )


def test_ladder_arms_share_the_pairing_knobs():
    """Arms differing in more than the thing under test are not an ablation."""
    arms = build_ladder(LadderSpec(n_init=7, batch_size=3, candidate_subset=99))
    for a in arms:
        assert a.n_init == 7 and a.batch_size == 3
        assert a.candidate_subset == 99


def test_menu_mine_sample_reaches_the_constraint_proposer():
    """Without this the constraint arm mines substructures from every molecule.

    On the ChEMBL tier (~9k compounds) that is fine and is the default. On PMO
    the pool is 249,455 molecules and mining walks all of them, so the first
    constraint call stalls the run before it has scored anything. The PMO runner
    passes 3,000; this pins that the value actually arrives.
    """
    from molbo.llm import LlamaCppBackend

    backend = LlamaCppBackend()
    sampled = build_ladder(LadderSpec(menu_mine_sample=3000), backend=backend)
    arm = next(a for a in sampled if a.name == "bo_llm_constraint")
    assert arm.constraint_source.menu_kwargs == {"mine_sample": 3000}

    whole = build_ladder(LadderSpec(), backend=backend)
    arm = next(a for a in whole if a.name == "bo_llm_constraint")
    assert arm.constraint_source.menu_kwargs == {}
