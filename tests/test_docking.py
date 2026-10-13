"""Docking as a budgeted tool. No Vina: the score cache is a fixture."""

import json

import numpy as np
import pytest

from molbo.docking import (
    BudgetedDockSelector,
    DockingBudget,
    DockingScores,
    DockSelector,
    incremental,
)

MOLS = [f"C{'C' * i}O" for i in range(12)]


def write_cache(tmp_path, scores, redock_ok=True):
    p = tmp_path / "d.jsonl"
    p.write_text("".join(json.dumps({"smiles": s, "score": v, "note": ""}) + "\n"
                         for s, v in scores.items()), encoding="utf-8")
    (tmp_path / "d.meta.json").write_text(json.dumps(
        {"pdb": "1M17", "redock": {"ok": redock_ok, "nn_rmsd": 1.1, "within_2A": 27,
                                   "n_atoms": 29}}), encoding="utf-8")
    return p


def test_vina_sign_is_flipped_so_argmax_means_best():
    """Vina reports kcal/mol, lower is better; the project maximises."""
    d = DockingScores(scores={"a": -9.5, "b": -6.0})
    assert d.value("a") == 9.5 and d.value("b") == 6.0
    assert np.argmax(d.values(["b", "a"])) == 1
    assert np.isnan(d.value("missing"))


def test_load_refuses_a_cache_whose_redocking_failed(tmp_path):
    p = write_cache(tmp_path, {"a": -9.0}, redock_ok=False)
    with pytest.raises(ValueError, match="redocking validation"):
        DockingScores.load(p)


def test_load_skips_failed_rows_and_keeps_meta(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"smiles": "a", "score": -9.0, "note": ""}) + "\n"
                 + json.dumps({"smiles": "b", "score": None, "note": "no 3D"}) + "\n",
                 encoding="utf-8")
    d = DockingScores.load(p)
    assert len(d) == 1 and "b" not in d.scores
    assert d.coverage(["a", "b"]) == 0.5


def test_budget_is_enforced_and_counted(tmp_path):
    d = DockingScores.load(write_cache(tmp_path, {m: -float(i) for i, m in enumerate(MOLS)}))
    b = DockingBudget(scores=d, budget=4)
    b.request(MOLS[:10], round_index=0)
    assert b.spent == 4 and b.remaining == 0 and b.refused == 6
    assert len(b.known) == 4
    # only what was paid for is visible
    v = b.values(MOLS[:6])
    assert np.isfinite(v[:4]).all() and np.isnan(v[4:]).all()


def test_re_requesting_a_known_molecule_is_free(tmp_path):
    d = DockingScores.load(write_cache(tmp_path, {m: -1.0 for m in MOLS}))
    b = DockingBudget(scores=d, budget=5)
    b.request(MOLS[:3])
    b.request(MOLS[:3])
    assert b.spent == 3


def test_reset_clears_everything(tmp_path):
    d = DockingScores.load(write_cache(tmp_path, {m: -1.0 for m in MOLS}))
    b = DockingBudget(scores=d, budget=5)
    b.request(MOLS[:5])
    b.reset()
    assert (b.spent, b.refused, b.known, b.log) == (0, 0, {}, [])


def test_allocator_none_spends_nothing_and_is_plain_max_mu(tmp_path):
    d = DockingScores.load(write_cache(tmp_path, {m: -float(i) for i, m in enumerate(MOLS)}))
    sel = BudgetedDockSelector(budget=DockingBudget(scores=d, budget=99),
                               allocator="none", per_round=3)
    mu = np.arange(len(MOLS), dtype=float)
    assert sel(shortlist=MOLS, scores=mu, mu=mu, sd=np.ones(len(MOLS))) == len(MOLS) - 1
    assert sel.budget.spent == 0


def test_top_allocator_docks_the_highest_acquisition_candidates(tmp_path):
    d = DockingScores.load(write_cache(tmp_path, {m: -float(i) for i, m in enumerate(MOLS)}))
    b = DockingBudget(scores=d, budget=99)
    sel = BudgetedDockSelector(budget=b, allocator="top", per_round=2)
    acq = np.zeros(len(MOLS))
    acq[5], acq[7] = 10.0, 9.0
    sel(shortlist=MOLS, scores=acq, mu=np.zeros(len(MOLS)), sd=np.ones(len(MOLS)))
    assert set(b.known) == {MOLS[5], MOLS[7]}


def test_uncertain_allocator_docks_where_the_surrogate_is_least_sure(tmp_path):
    d = DockingScores.load(write_cache(tmp_path, {m: -float(i) for i, m in enumerate(MOLS)}))
    b = DockingBudget(scores=d, budget=99)
    sel = BudgetedDockSelector(budget=b, allocator="uncertain", per_round=2)
    sd = np.zeros(len(MOLS))
    sd[1], sd[9] = 5.0, 4.0
    sel(shortlist=MOLS, scores=np.zeros(len(MOLS)), mu=np.zeros(len(MOLS)), sd=sd)
    assert set(b.known) == {MOLS[1], MOLS[9]}


def test_undocked_candidates_are_not_penalised(tmp_path):
    """Not having paid to look at a molecule is not evidence against it."""
    d = DockingScores.load(write_cache(tmp_path, {m: -1.0 for m in MOLS}))
    b = DockingBudget(scores=d, budget=1)
    sel = BudgetedDockSelector(budget=b, allocator="top", per_round=1, blend_mu=1.0)
    mu = np.zeros(len(MOLS))
    mu[-1] = 100.0          # an undocked but overwhelmingly promising candidate
    acq = np.zeros(len(MOLS))
    acq[0] = 1.0            # docking is spent on candidate 0
    assert sel(shortlist=MOLS, scores=acq, mu=mu, sd=np.ones(len(MOLS))) == len(MOLS) - 1


def test_dock_selector_falls_back_when_nothing_is_scored(tmp_path):
    d = DockingScores.load(write_cache(tmp_path, {"other": -9.0}))
    sel = DockSelector(docking=d)
    assert sel(shortlist=MOLS, mu=np.arange(len(MOLS), dtype=float)) == 0
    assert sel.misses == 1


def test_incremental_detects_a_signal_orthogonal_to_mu():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=400)
    extra = rng.normal(size=400)
    true = mu + extra + 0.1 * rng.normal(size=400)
    out = incremental(docking=extra, mu=mu, true=true)
    assert out["delta_r2"] > 0.2          # extra explains a lot beyond mu
    assert out["partial_rho"] > 0.5
    assert abs(out["rho_dock_mu"]) < 0.2   # and is near-orthogonal to it


def test_incremental_detects_a_redundant_signal():
    rng = np.random.default_rng(1)
    mu = rng.normal(size=400)
    true = mu + 0.1 * rng.normal(size=400)
    out = incremental(docking=mu * 2.0, mu=mu, true=true)
    assert out["delta_r2"] < 0.01          # a copy of mu adds nothing
    assert out["rho_dock_mu"] > 0.95


def test_campaign_resets_a_budgeted_selector_between_seeds(tmp_path):
    """Without this the first seed spends the whole allowance and every later
    seed runs with none -- an arm that silently degrades along the seed axis."""
    from molbo.campaign import ArmConfig
    from molbo.experiment import run_sweep
    from molbo.oracles import LookupOracle

    import pandas as pd

    lib = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1", "CCOC", "CCCO",
           "CCCN", "CCCC", "CCCCl", "CCCBr", "c1ccncc1", "CCOCC", "CCCCO", "CCCCN"]
    csv = tmp_path / "lib.csv"
    pd.DataFrame({"canonical_smiles": lib,
                  "pchembl_value": np.linspace(5, 9, len(lib))}).to_csv(csv, index=False)
    d = DockingScores.load(write_cache(tmp_path, {s: -float(i) for i, s in enumerate(lib)}))
    sel = BudgetedDockSelector(budget=DockingBudget(scores=d, budget=3),
                               allocator="top", per_round=2)
    oracle = LookupOracle.from_dataset(csv, budget=14)
    arm = ArmConfig("bo_dock", strategy="bo", acquisition="ei", n_init=4,
                    selector=sel, shortlist_k=5)
    run_sweep(oracle, [arm], range(3), budget=14, k=3, verbose=False)

    assert len(sel.history) >= 2, "each seed should have recorded its own spend"
    assert all(h["spent"] <= 3 for h in sel.history), "budget breached within a seed"
    assert any(h["spent"] > 0 for h in sel.history[1:]), \
        "later seeds got no budget -- the selector was not reset"
