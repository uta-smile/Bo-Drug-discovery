"""The PMO tier: the conventions that make our numbers comparable to published ones.

These tests are about *protocol compliance*, not about chemistry. PMO's numbers
mean something only because every method is scored the same way -- unique calls
counted, invalid molecules scored zero and charged for. Getting any of that
wrong produces a number that looks comparable and is not, which is worse than
producing no number at all.

Everything that needs TDC skips cleanly when it is not installed.
"""

from __future__ import annotations

import pytest

from molbo.oracles.pmo import (
    PMO_TASKS,
    RDKIT_ONLY,
    PMOOracle,
    install_rdkit_six_shim,
)

tdc = pytest.importorskip("tdc", reason="PMO tier needs PyTDC")


@pytest.fixture(scope="module")
def qed():
    return PMOOracle("qed", budget=None)


# ---- the compatibility shim ----------------------------------------------


def test_the_shim_makes_rdkit_six_importable():
    install_rdkit_six_shim()
    from rdkit.six import iteritems

    assert list(iteritems({"a": 1})) == [("a", 1)]


def test_the_shim_is_idempotent():
    install_rdkit_six_shim()
    assert install_rdkit_six_shim() is False  # already there, nothing replaced


# ---- task registry --------------------------------------------------------


def test_the_benchmark_has_its_23_tasks():
    assert len(PMO_TASKS) == 23
    assert len(set(PMO_TASKS)) == 23


def test_the_ml_backed_tasks_are_called_out():
    """Those three unpickle sklearn models and can fail on a newer sklearn."""
    assert set(PMO_TASKS) - RDKIT_ONLY == {"drd2", "gsk3b", "jnk3"}


def test_an_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown PMO task"):
        PMOOracle("make_it_bind")


def test_task_name_is_case_insensitive():
    assert PMOOracle("QED", budget=None).task == "qed"


# ---- PMO's scoring conventions -------------------------------------------


def test_scores_land_in_the_unit_interval(qed):
    for smi in ("CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"):
        assert 0.0 <= qed(smi) <= 1.0


def test_a_known_molecule_scores_its_own_task_perfectly():
    """celecoxib on celecoxib_rediscovery is the benchmark's own sanity check."""
    celecoxib = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    assert PMOOracle("celecoxib_rediscovery", budget=None)(celecoxib) == pytest.approx(1.0)


def test_an_unrelated_molecule_scores_low():
    celecoxib_oracle = PMOOracle("celecoxib_rediscovery", budget=None)
    assert celecoxib_oracle("CCO") < 0.3


def test_invalid_smiles_scores_zero_not_none(qed):
    """PMO's convention. Returning None would drop it from the curve instead."""
    assert qed("not_a_molecule") == 0.0


def test_an_invalid_proposal_still_costs_budget():
    """Otherwise a generative arm buys extra evaluations by emitting garbage."""
    o = PMOOracle("qed", budget=10)
    o("not_a_molecule")
    assert o.n_calls == 1


def test_scores_need_no_rescaling():
    """Unlike the ChEMBL tier, PMO output is already [0, 1] higher-is-better."""
    d = PMOOracle("qed", budget=None).describe()
    assert d["score_range"] == [0.0, 1.0]
    assert d["maximize"] is True


# ---- budget accounting ----------------------------------------------------


def test_repeats_are_not_charged(qed_free=None):
    o = PMOOracle("qed", budget=100)
    for _ in range(5):
        o("CCO")
    assert o.n_calls == 1
    assert o.n_queries == 5


def test_equivalent_smiles_count_once():
    o = PMOOracle("qed", budget=100)
    o("CC(=O)Oc1ccccc1C(=O)O")
    o("O=C(C)Oc1ccccc1C(O)=O")  # the same molecule, spelled differently
    assert o.n_calls == 1


def test_the_budget_is_enforced():
    from molbo.oracles.base import OracleBudgetExceeded

    o = PMOOracle("qed", budget=2)
    o(["CCO", "CCC"])
    with pytest.raises(OracleBudgetExceeded):
        o("CCCC")


def test_batch_and_single_calls_agree(qed):
    singles = [qed(s) for s in ("CCO", "CCC", "c1ccccc1")]
    qed.reset()
    assert qed(["CCO", "CCC", "c1ccccc1"]) == singles


# ---- provenance -----------------------------------------------------------


def test_the_task_is_part_of_the_cache_key():
    a = PMOOracle("qed", budget=None)
    b = PMOOracle("albuterol_similarity", budget=None)
    assert a.cache_key != b.cache_key


def test_describe_records_the_benchmark_and_task(qed):
    d = qed.describe()
    assert d["benchmark"] == "PMO"
    assert d["task"] == "qed"
