"""The primary oracle: exactness, determinism, and the out-of-library policy."""

from __future__ import annotations

import json

import pytest

from molbo.chem import canonical_smiles
from molbo.oracles import LookupOracle
from molbo.oracles.base import OracleBudgetExceeded

TABLE = {
    "CC(=O)Oc1ccccc1C(=O)O": 7.5,   # aspirin
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O": 5.1,  # caffeine
    "CCO": 4.0,
    "c1ccccc1": 6.2,
}
ASPIRIN_ALT = "O=C(C)Oc1ccccc1C(O)=O"
OFF_LIBRARY = "CCCCCCCCCC"


@pytest.fixture
def oracle():
    return LookupOracle(TABLE, name="test")


def test_lookup_is_exact(oracle):
    assert oracle("CCO") == 4.0
    assert oracle("CC(=O)Oc1ccccc1C(=O)O") == 7.5


def test_lookup_is_deterministic(oracle):
    a = [oracle(s) for s in TABLE]
    oracle.reset()
    b = [oracle(s) for s in TABLE]
    assert a == b


def test_equivalent_smiles_hits_the_same_entry(oracle):
    assert oracle(ASPIRIN_ALT) == 7.5
    assert oracle.n_calls == 1


def test_table_is_rekeyed_on_construction():
    o = LookupOracle({ASPIRIN_ALT: 7.5}, name="t")
    assert o.library == [canonical_smiles(ASPIRIN_ALT)]
    assert o("CC(=O)Oc1ccccc1C(=O)O") == 7.5


def test_library_and_values_are_aligned(oracle):
    assert len(oracle.library) == len(TABLE)
    for smi, val in zip(oracle.library, oracle.values):
        assert oracle(smi) == val


def test_contains(oracle):
    assert ASPIRIN_ALT in oracle
    assert OFF_LIBRARY not in oracle
    assert "not_a_molecule" not in oracle


def test_empty_table_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        LookupOracle({"not_a_molecule": 1.0}, name="t")


# ---- ground truth ---------------------------------------------------------


def test_best_value_and_regret(oracle):
    assert oracle.best_value == 7.5
    assert oracle.best_smiles == canonical_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert oracle.simple_regret() is None, "no regret before anything is evaluated"
    oracle("CCO")
    assert oracle.simple_regret() == pytest.approx(3.5)
    oracle("CC(=O)Oc1ccccc1C(=O)O")
    assert oracle.simple_regret() == pytest.approx(0.0)


def test_minimize_flips_the_optimum():
    o = LookupOracle(TABLE, name="t", maximize=False)
    assert o.best_value == 4.0
    o("c1ccccc1")
    assert o.simple_regret() == pytest.approx(2.2)


def test_top_fraction_found(oracle):
    assert oracle.top_fraction_found(0.25) == 0.0
    oracle("CC(=O)Oc1ccccc1C(=O)O")
    assert oracle.top_fraction_found(0.25) == 1.0


# ---- out-of-library policy ------------------------------------------------


def test_missing_returns_none_by_default_and_is_charged(oracle):
    assert oracle(OFF_LIBRARY) is None
    assert oracle.n_calls == 1, "an unmakeable proposal is not free in a real campaign"


def test_missing_can_raise():
    o = LookupOracle(TABLE, name="t", on_missing="error")
    with pytest.raises(KeyError, match="not in the enumerated library"):
        o(OFF_LIBRARY)


def test_missing_can_take_a_sentinel_value():
    o = LookupOracle(TABLE, name="t", on_missing="value", missing_value=0.0)
    assert o(OFF_LIBRARY) == 0.0


def test_bad_on_missing_is_rejected():
    with pytest.raises(ValueError, match="on_missing"):
        LookupOracle(TABLE, name="t", on_missing="shrug")


def test_missing_molecules_are_excluded_from_top_k(oracle):
    oracle([OFF_LIBRARY, "CCO"])
    assert oracle.top_k(5) == [(canonical_smiles("CCO"), 4.0)]


# ---- integration with the real series -------------------------------------


@pytest.fixture(scope="module")
def egfr():
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data/chembl/egfr_anilinoquinazoline.csv"
    if not p.exists():
        pytest.skip("run scripts/build_chembl_series.py first")
    return LookupOracle.from_dataset(p, budget=200)


def test_real_series_loads_with_provenance(egfr):
    assert len(egfr) > 1000
    assert egfr.metadata["target_chembl_id"] == "CHEMBL203"
    assert egfr.metadata["core"] is not None
    assert 4.0 <= min(egfr.values) < max(egfr.values) <= 12.0


def test_real_series_contains_the_reference_drugs(egfr):
    """If the filters drop gefitinib, they are dropping the well-measured
    compounds -- which is what a range-based replicate filter does."""
    gefitinib = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    erlotinib = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
    assert gefitinib in egfr and erlotinib in egfr


def test_real_series_budget_holds_under_repeats(egfr):
    egfr.reset()
    lib = egfr.library[:50]
    for _ in range(20):
        egfr(lib)
    assert egfr.n_calls == 50
    assert egfr.n_queries == 1000


def test_from_dataset_skips_recanonicalisation_when_provenance_matches(egfr, tmp_path):
    from molbo.chem import DEFAULT_OPTS

    assert egfr.metadata["standardizer"] == DEFAULT_OPTS.key()
    # A file claiming a different standardiser must be re-keyed, not trusted.
    import pandas as pd
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "data/chembl/egfr_anilinoquinazoline.csv"
    pd.read_csv(src).head(20).to_csv(tmp_path / "x.csv", index=False)
    (tmp_path / "x.meta.json").write_text(json.dumps({"standardizer": "v0:stale"}))
    o = LookupOracle.from_dataset(tmp_path / "x.csv")
    assert len(o) == 20


def test_budget_is_enforced_on_the_real_series(egfr):
    egfr.reset()
    with pytest.raises(OracleBudgetExceeded):
        egfr(egfr.library)
