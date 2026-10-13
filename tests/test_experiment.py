"""Sweeps end to end: metrics, pairing, arm registry, and result files.

The statistics tests here are the ones that decide whether a published number
means anything. A sweep that silently compares unpaired seeds, or normalises
each arm by its own range, will happily report a significant difference that is
an artifact of the harness.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from molbo.campaign import ArmConfig, CampaignResult
from molbo.constraints import LibraryContext, SubstructureConstraint
from molbo.experiment import default_arms, run_sweep, save_sweep, score_run, summarize_sweep
from molbo.llm import ScriptedProposer
from molbo.oracles import LookupOracle

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


@pytest.fixture
def oracle():
    return LookupOracle(TABLE, name="toy", budget=10)


# ---- scoring --------------------------------------------------------------


def _run(values, arm="a"):
    return CampaignResult(
        arm=arm, seed=0, budget=len(values), smiles=[f"s{i}" for i in range(len(values))],
        values=list(values), n_calls=len(values), wall_seconds=0.0,
        constraint_log=[], arm_config={}, oracle={},
    )


def test_a_shared_library_range_preserves_the_true_ordering_of_arms():
    """The comparability property the whole results table rests on.

    A weak arm and a strong arm scored against the *library's* range keep their
    real ordering. Scaled by each run's own range instead, both would map to
    [0, 1] and score identically -- the weak arm would look as good as the
    strong one.
    """
    weak, strong = _run([4.0, 5.0]), _run([8.0, 9.0])
    shared = dict(lo=0.0, hi=10.0, budget=2)
    assert score_run(weak, **shared).auc < score_run(strong, **shared).auc

    # What self-normalisation would have done: identical, and therefore useless.
    self_weak = score_run(weak, lo=4.0, hi=5.0, budget=2)
    self_strong = score_run(strong, lo=8.0, hi=9.0, budget=2)
    assert self_weak.auc == self_strong.auc


def test_score_run_keeps_the_raw_best_whatever_the_scaling():
    res = _run([4.0, 5.0, 6.0, 7.0])
    assert score_run(res, lo=0.0, hi=10.0, budget=4).best == 7.0
    assert score_run(res, lo=4.0, hi=7.0, budget=4).best == 7.0
    assert score_run(res, lo=4.0, hi=7.0, budget=4).best_normalized == 1.0


def test_score_run_reports_regret_and_recall():
    res = CampaignResult(
        arm="a", seed=0, budget=3, smiles=["a", "b", "c"], values=[4.0, 9.0, 5.0],
        n_calls=3, wall_seconds=0.0, constraint_log=[], arm_config={}, oracle={},
    )
    m = score_run(res, lo=0.0, hi=10.0, budget=3, best_possible=11.0, top_set={"b", "z"})
    assert m.simple_regret == pytest.approx(2.0)
    assert m.top1pct_recall == pytest.approx(0.5)


def test_score_run_counts_constraint_modes():
    res = CampaignResult(
        arm="a", seed=0, budget=2, smiles=["a", "b"], values=[4.0, 5.0], n_calls=2,
        wall_seconds=0.0,
        constraint_log=[{"mode": "hard"}, {"mode": "soft"}, {"mode": "hard"}],
        arm_config={}, oracle={},
    )
    m = score_run(res, lo=0.0, hi=10.0, budget=2)
    assert m.constraint_modes == {"hard": 2, "soft": 1}


def test_unscored_proposals_are_dropped_from_the_curve():
    res = CampaignResult(
        arm="a", seed=0, budget=3, smiles=["a", "b", "c"], values=[4.0, None, 9.0],
        n_calls=3, wall_seconds=0.0, constraint_log=[], arm_config={}, oracle={},
    )
    assert score_run(res, lo=0.0, hi=10.0, budget=3).best == 9.0


# ---- arm registry ---------------------------------------------------------


def test_default_arms_include_the_controls_and_the_ceiling():
    names = {a.name for a in default_arms(list(TABLE.values()))}
    assert {
        "random",
        "bo_ei",
        "bo_random_constraint",
        "bo_oracle_constraint",
        "bo_data_driven_constraint",
    } <= names


def test_llm_arm_is_added_when_a_source_is_given():
    arms = default_arms(list(TABLE.values()), llm_source=ScriptedProposer([{}]))
    assert "bo_llm_constraint" in {a.name for a in arms}


def test_random_control_mask_varies_with_the_seed():
    """Otherwise all 20 seeds share one mask and the arm's variance is understated."""
    arm = next(a for a in default_arms(list(TABLE.values())) if a.name == "bo_random_constraint")
    ctx = LibraryContext.build(sorted(TABLE))
    rng = np.random.default_rng(0)
    a = arm.constraint_source(observations=[], ctx=ctx, rng=rng, round_index=0, seed=1)
    b = arm.constraint_source(observations=[], ctx=ctx, rng=rng, round_index=0, seed=2)
    assert not np.array_equal(a.mask(ctx), b.mask(ctx))


def test_arm_config_describe_is_json_serialisable():
    for arm in default_arms(list(TABLE.values())):
        json.dumps(arm.describe())


# ---- sweeps ---------------------------------------------------------------


def test_sweep_runs_every_arm_on_every_seed(oracle):
    arms = [ArmConfig("a", strategy="random", n_init=4), ArmConfig("b", strategy="bo", n_init=4)]
    metrics, traces = run_sweep(oracle, arms, range(3), budget=10, verbose=False)
    assert len(metrics) == len(traces) == 6
    assert {m.arm for m in metrics} == {"a", "b"}
    assert {m.seed for m in metrics} == {0, 1, 2}


def test_sweep_pairs_arms_on_seeds(oracle):
    """Same seed, same initial design across arms -- the basis of the paired test."""
    arms = [ArmConfig("a", strategy="random", n_init=4), ArmConfig("b", strategy="bo", n_init=4)]
    _, traces = run_sweep(oracle, arms, range(2), budget=10, verbose=False)
    by = {(t.arm, t.seed): t for t in traces}
    assert by[("a", 0)].smiles[:4] == by[("b", 0)].smiles[:4]
    assert by[("a", 1)].smiles[:4] != by[("a", 0)].smiles[:4]


def test_summarize_compares_every_arm_against_the_reference(oracle):
    arms = [
        ArmConfig("bo_ei", strategy="bo", n_init=4),
        ArmConfig("random", strategy="random", n_init=4),
    ]
    metrics, _ = run_sweep(oracle, arms, range(6), budget=10, verbose=False)
    summary = summarize_sweep(metrics, reference="bo_ei")
    assert set(summary["summaries"]) == {"bo_ei", "random"}
    assert set(summary["comparisons"]) == {"random"}  # the reference is not compared to itself
    assert summary["comparisons"]["random"]["test"] == "wilcoxon-signed-rank"


def test_summarize_refuses_to_pair_mismatched_seed_sets(oracle):
    a, _ = run_sweep(oracle, [ArmConfig("bo_ei", n_init=4)], range(4), budget=10, verbose=False)
    b, _ = run_sweep(oracle, [ArmConfig("other", n_init=4)], range(3), budget=10, verbose=False)
    summary = summarize_sweep(list(a) + list(b), reference="bo_ei")
    assert "error" in summary["comparisons"]["other"]


def test_summarize_tolerates_a_missing_reference(oracle):
    metrics, _ = run_sweep(oracle, [ArmConfig("a", n_init=4)], range(2), budget=10, verbose=False)
    assert summarize_sweep(metrics, reference="not_here")["comparisons"] == {}


# ---- the LLM arm's wiring -------------------------------------------------


def test_a_scripted_llm_arm_runs_end_to_end(oracle):
    """Exercises the real arm path with a hand-written response standing in for
    the model, so the wiring is covered without an API call."""
    response = {
        "hypothesis": "The anilinoquinazoline core drives potency.",
        "require_smarts": [QUINAZOLINE],
        "forbid_smarts": [],
        "property_bounds": [],
        "confidence": 0.9,
    }
    arm = ArmConfig(
        "bo_llm_constraint",
        strategy="bo",
        n_init=3,
        constraint_source=ScriptedProposer([response]),
        min_candidates=2,
    )
    metrics, traces = run_sweep(oracle, [arm], range(3), budget=9, verbose=False)
    assert all(m.n_calls == 9 for m in metrics)
    assert all(t.constraint_log for t in traces)
    assert any("hard" in m.constraint_modes for m in metrics)


def test_a_correct_constraint_helps_on_a_library_with_obvious_structure(oracle):
    """Not a result -- a wiring check that the constraint reaches the optimiser."""
    good = ScriptedProposer([{
        "hypothesis": "", "require_smarts": [QUINAZOLINE], "forbid_smarts": [],
        "property_bounds": [], "confidence": 1.0,
    }])
    plain = ArmConfig("bo_ei", strategy="bo", n_init=3)
    guided = ArmConfig(
        "guided", strategy="bo", n_init=3, constraint_source=good, min_candidates=2
    )
    metrics, _ = run_sweep(oracle, [plain, guided], range(6), budget=8, verbose=False)
    by = {}
    for m in metrics:
        by.setdefault(m.arm, []).append(m.auc)
    assert np.mean(by["guided"]) >= np.mean(by["bo_ei"])


# ---- output files ---------------------------------------------------------


def test_save_sweep_writes_metrics_traces_and_summary(oracle, tmp_path):
    arms = [ArmConfig("bo_ei", strategy="bo", n_init=4), ArmConfig("random", strategy="random", n_init=4)]
    metrics, traces = run_sweep(oracle, arms, range(3), budget=10, verbose=False)
    summary = summarize_sweep(metrics)
    out = save_sweep(tmp_path, metrics, traces, summary, meta={"dataset": "toy"})

    assert (out / "metrics.csv").exists()
    lines = (out / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    assert json.loads(lines[0])["arm"]

    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["meta"]["dataset"] == "toy"
    assert payload["arms"]["bo_ei"]["n_seeds"] == 3
    assert len(payload["arms"]["bo_ei"]["aucs"]) == 3


def test_saved_traces_are_enough_to_recompute_the_metrics(oracle, tmp_path):
    """Everything needed to recompute any metric offline must be on disk."""
    metrics, traces = run_sweep(
        oracle, [ArmConfig("a", strategy="bo", n_init=4)], range(2), budget=10, verbose=False
    )
    save_sweep(tmp_path, metrics, traces, summarize_sweep(metrics))
    rows = [
        json.loads(line)
        for line in (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    for row, metric in zip(rows, metrics):
        assert len(row["values"]) == metric.n_calls
        assert row["best"] == metric.best
