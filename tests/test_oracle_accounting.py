"""The budget accounting is the part that silently invalidates results.

If repeats consume budget, the numbers are not comparable to PMO. If a disk
cache hit skips the budget, a 20-seed sweep quietly gets cheaper every seed and
later seeds look better than earlier ones. Both failures produce plausible
curves, so they get tested rather than eyeballed.
"""

from __future__ import annotations

import pytest

from molbo.chem import canonical_smiles
from molbo.oracles.base import Oracle, OracleBudgetExceeded

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_ALT = "O=C(C)Oc1ccccc1C(O)=O"  # same molecule, different string
ASPIRIN_SALT = "CC(=O)Oc1ccccc1C(=O)O.[Na+].[Cl-]"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


class CountingOracle(Oracle):
    """Scores by heavy-atom count and records every molecule it actually computes."""

    def __init__(self, **kwargs):
        self.computed: list[str] = []
        self.n_batches = 0
        super().__init__(kwargs.pop("name", "counting"), **kwargs)

    def _evaluate_batch(self, smiles):
        from rdkit import Chem

        self.n_batches += 1
        self.computed.extend(smiles)
        return [float(Chem.MolFromSmiles(s).GetNumHeavyAtoms()) for s in smiles]


# ---- memoisation ----------------------------------------------------------


def test_repeat_proposals_are_free():
    o = CountingOracle()
    for _ in range(50):
        o(ASPIRIN)
    assert o.n_calls == 1, "a re-proposed molecule must not consume budget"
    assert o.n_queries == 50, "total queries are still tracked, for diagnostics"
    assert len(o.computed) == 1


def test_memoisation_is_by_canonical_smiles_not_string():
    o = CountingOracle()
    a = o(ASPIRIN)
    b = o(ASPIRIN_ALT)
    assert a == b
    assert o.n_calls == 1, "same molecule written two ways is one oracle call"


def test_salt_is_stripped_before_keying():
    o = CountingOracle()
    o(ASPIRIN)
    o(ASPIRIN_SALT)
    assert o.n_calls == 1, "hydrochloride and free base key to the same compound"


def test_duplicates_within_one_batch_collapse():
    o = CountingOracle()
    vals = o([ASPIRIN, ASPIRIN_ALT, CAFFEINE, ASPIRIN])
    assert o.n_calls == 2
    assert vals[0] == vals[1] == vals[3]
    assert len(o.computed) == 2


def test_batch_returns_values_in_request_order():
    o = CountingOracle()
    order = [CAFFEINE, ASPIRIN, CAFFEINE]
    vals = o(order)
    direct = [CountingOracle()(s) for s in order]
    assert vals == direct


# ---- budget ---------------------------------------------------------------


def test_budget_counts_unique_calls_only():
    o = CountingOracle(budget=3)
    o([ASPIRIN] * 10)
    assert o.remaining == 2
    o([CAFFEINE, "CCO"])
    assert o.n_calls == 3 and o.remaining == 0


def test_budget_raises_on_overflow():
    o = CountingOracle(budget=2)
    o([ASPIRIN, CAFFEINE])
    with pytest.raises(OracleBudgetExceeded, match="budget 2 exhausted"):
        o("CCO")


def test_budget_overflow_does_not_partially_evaluate_when_raising():
    o = CountingOracle(budget=2)
    with pytest.raises(OracleBudgetExceeded):
        o([ASPIRIN, CAFFEINE, "CCO"])
    assert o.n_calls == 0, "a rejected batch must leave the ledger untouched"
    assert o.computed == []


def test_on_exhausted_stop_truncates_instead_of_raising():
    o = CountingOracle(budget=2, on_exhausted="stop")
    vals = o([ASPIRIN, CAFFEINE, "CCO", "CCC"])
    assert o.n_calls == 2
    assert o.exhausted is True
    assert vals[0] is not None and vals[1] is not None
    assert vals[2] is None and vals[3] is None, "past-budget molecules return None"


def test_exhausted_oracle_still_serves_cached_molecules():
    o = CountingOracle(budget=1, on_exhausted="stop")
    first = o(ASPIRIN)
    o("CCO")
    assert o(ASPIRIN_ALT) == first, "already-paid-for molecules stay readable"


# ---- invalid input --------------------------------------------------------


def test_invalid_smiles_scores_invalid_value_and_is_charged():
    o = CountingOracle(invalid_value=0.0)
    assert o("not_a_molecule") == 0.0
    assert o.n_calls == 1, "PMO charges for garbage; a generative arm cannot dodge budget"
    assert o.computed == [], "the subclass never sees unparseable input"


def test_repeated_invalid_smiles_charged_once():
    o = CountingOracle()
    for _ in range(10):
        o("not_a_molecule")
    assert o.n_calls == 1


def test_distinct_invalid_strings_are_distinct_calls():
    o = CountingOracle()
    o("not_a_molecule")
    o("also[not]a(molecule")
    assert o.n_calls == 2


def test_count_invalid_false_makes_garbage_free():
    o = CountingOracle(count_invalid=False, budget=2)
    o(["not_a_molecule", "bad[[", ASPIRIN])
    assert o.n_calls == 1
    assert o.remaining == 1


# ---- disk cache -----------------------------------------------------------


def test_disk_cache_saves_compute_across_runs(tmp_path):
    db = tmp_path / "cache.sqlite"
    first = CountingOracle(cache_path=db)
    first([ASPIRIN, CAFFEINE])
    assert len(first.computed) == 2

    second = CountingOracle(cache_path=db)
    second([ASPIRIN, CAFFEINE])
    assert second.computed == [], "second run reuses the stored values"
    assert second.n_disk_hits == 2


def test_disk_cache_hit_still_consumes_budget(tmp_path):
    """The crucial one: a cache hit saves GPU time, not budget.

    Seed 2 did not know the answer just because seed 1 computed it, so charging
    it keeps every seed on the same budget and stops later seeds from looking
    better purely through cache warmth.
    """
    db = tmp_path / "cache.sqlite"
    CountingOracle(cache_path=db)([ASPIRIN, CAFFEINE])

    second = CountingOracle(cache_path=db, budget=2)
    second([ASPIRIN, CAFFEINE])
    assert second.n_calls == 2 and second.remaining == 0
    with pytest.raises(OracleBudgetExceeded):
        second("CCO")


def test_cache_is_namespaced_by_standardizer(tmp_path):
    from molbo.chem import StandardizeOptions

    db = tmp_path / "cache.sqlite"
    CountingOracle(cache_path=db)(ASPIRIN)
    other = CountingOracle(cache_path=db, opts=StandardizeOptions(keep_stereo=False))
    other(ASPIRIN)
    assert other.computed == [ASPIRIN], "different keying must not reuse old values"


def test_cached_failures_are_not_recomputed(tmp_path):
    class Failing(CountingOracle):
        def _evaluate_batch(self, smiles):
            super()._evaluate_batch(smiles)
            return [None] * len(smiles)

    db = tmp_path / "cache.sqlite"
    Failing(cache_path=db)(ASPIRIN)
    again = Failing(cache_path=db)
    assert again(ASPIRIN) is None
    assert again.computed == [], "a deterministic failure re-fails; do not pay twice"


# ---- trace ----------------------------------------------------------------


def test_history_is_ordered_and_indexed_from_one():
    o = CountingOracle()
    o([CAFFEINE, ASPIRIN, "CCO"])
    assert [r.call for r in o.history] == [1, 2, 3]
    assert o.history[0].smiles == canonical_smiles(CAFFEINE)
    assert all(r.source == "computed" for r in o.history)


def test_buffer_matches_pmo_shape():
    o = CountingOracle()
    o([CAFFEINE, ASPIRIN])
    buf = o.buffer()
    assert set(buf) == {canonical_smiles(CAFFEINE), canonical_smiles(ASPIRIN)}
    assert buf[canonical_smiles(CAFFEINE)][1] == 1


def test_reset_clears_run_state_but_not_disk_cache(tmp_path):
    db = tmp_path / "cache.sqlite"
    o = CountingOracle(cache_path=db, budget=5)
    o([ASPIRIN, CAFFEINE])
    o.reset()
    assert o.n_calls == 0 and o.n_queries == 0 and o.remaining == 5
    o(ASPIRIN)
    assert o.n_calls == 1, "budget is charged again after reset"
    assert o.n_disk_hits == 1, "but the value came from disk, not recomputation"


def test_save_trace_round_trips(tmp_path):
    import json

    o = CountingOracle(budget=10)
    o([ASPIRIN, CAFFEINE])
    p = o.save_trace(tmp_path / "trace.json", extra={"seed": 7})
    payload = json.loads(p.read_text())
    assert payload["n_calls"] == 2
    assert payload["seed"] == 7
    assert payload["oracle"]["budget"] == 10
    assert len(payload["history"]) == 2


def test_evaluate_batch_length_mismatch_is_caught():
    class Broken(Oracle):
        def _evaluate_batch(self, smiles):
            return [1.0]

    o = Broken("broken")
    with pytest.raises(RuntimeError, match="returned 1 values for 2"):
        o([ASPIRIN, CAFFEINE])
