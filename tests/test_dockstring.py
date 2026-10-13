"""The DOCKSTRING objective family, and the line between free and assayed.

FINDINGS section 14 turns on one measurement: how much of an objective a
free descriptor already supplies. That number is only meaningful if
``Objective`` splits itself where it claims to, and if a filtered objective
really removes molecules from the library rather than scoring them badly --
a reject left in at a bad score is still a term the optimiser can spend
budget rediscovering.

These tests need no dataset; they run on a hand-built frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molbo import dockstring as ds


@pytest.fixture
def frame():
    return pd.DataFrame({
        "smiles": ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O"],
        "JAK2": [-9.0, -7.0, -11.0, -6.0, -8.0],
        "LCK": [-7.0, -9.5, -8.4, -5.0, -12.0],
        "PPARA": [-8.0, -6.0, -7.0, -9.0, -5.0],
        "PPARD": [-7.5, -6.5, -9.0, -8.0, -4.0],
        "PPARG": [-6.0, -7.5, -8.0, -7.0, -6.5],
        "QED": [0.9, 0.4, 0.75, 0.2, 0.55],
    })


def test_published_objective_matches_the_notebook_formula(frame):
    """The transcription is load-bearing: it is the comparison to Mobley."""
    got = ds.OBJECTIVES["jak2"](frame)
    want = (-frame.JAK2 + np.minimum(frame.LCK, -8.1) - 10 * (1 - frame.QED))
    np.testing.assert_allclose(got, want)
    np.testing.assert_allclose(ds.score_jak2(frame), want)


def test_the_selectivity_term_saturates_on_weak_lck(frame):
    """Docking is negative-is-better, and the objective wants LCK *weak*.

    ``min(LCK, -8.1)`` caps the reward: once LCK binds worse than -8.1 there is
    no further credit for being even less potent on it, and the gradient in LCK
    vanishes. That flat region is what a scalarised surrogate cannot see, so it
    is the reason the objective is multi-objective rather than a weighted sum.
    """
    a, b = frame.copy(), frame.copy()
    a.loc[0, "LCK"] = -7.0   # weaker than the -8.1 cut
    b.loc[0, "LCK"] = -2.0   # weaker still
    assert ds.OBJECTIVES["jak2"](a)[0] == pytest.approx(ds.OBJECTIVES["jak2"](b)[0])

    # Below the cut the term is live again: stronger LCK binding is penalised.
    c = frame.copy()
    c.loc[0, "LCK"] = -12.0
    assert ds.OBJECTIVES["jak2"](c)[0] < ds.OBJECTIVES["jak2"](a)[0]


def test_free_plus_assayed_reconstructs_the_objective(frame):
    for name, obj in ds.OBJECTIVES.items():
        total = np.asarray(obj(frame), dtype=float)
        parts = obj.assayed(frame) + obj.free(frame)
        np.testing.assert_allclose(total, parts, err_msg=name)


def test_the_free_part_needs_no_assay(frame):
    """It must depend on QED alone: change a docking column, nothing moves."""
    obj = ds.OBJECTIVES["jak2"]
    before = obj.free(frame)
    moved = frame.copy()
    moved["JAK2"] = moved["JAK2"] - 5.0
    moved["LCK"] = moved["LCK"] - 5.0
    np.testing.assert_allclose(before, obj.free(moved))


def test_potency_variants_have_nothing_free(frame):
    for name, obj in ds.OBJECTIVES.items():
        if not name.endswith("_potency"):
            continue
        assert obj.qed_weight == 0.0
        np.testing.assert_allclose(obj.free(frame), 0.0)
        np.testing.assert_allclose(obj(frame), obj.assayed(frame))


def test_reweighting_scales_only_the_free_part(frame):
    base = ds.OBJECTIVES["jak2"]
    half = base.reweighted(5.0)
    np.testing.assert_allclose(half.assayed(frame), base.assayed(frame))
    np.testing.assert_allclose(half.free(frame), 0.5 * base.free(frame))
    assert half.targets == base.targets


def test_a_filter_removes_molecules_rather_than_scoring_them_low(frame):
    """The distinction section 14 rests on: rejects must leave the library."""
    published = ds.objective_table(frame, "jak2")
    filtered = ds.objective_table(frame, "jak2_potency")
    assert len(published) == len(frame)
    # QED 0.4 and 0.2 are below the 0.5 cut.
    assert set(filtered) == {"CCO", "CCC", "CC(=O)O"}
    assert "c1ccccc1" not in filtered


def test_admits_tracks_the_threshold(frame):
    assert ds.OBJECTIVES["jak2"].admits(frame).all()
    admitted = ds.OBJECTIVES["jak2_potency"].admits(frame)
    np.testing.assert_array_equal(admitted, frame.QED.to_numpy() >= ds.QED_MIN)


def test_a_nan_qed_is_never_admitted(frame):
    bad = frame.copy()
    bad.loc[1, "QED"] = np.nan
    for name in ("jak2", "jak2_potency"):
        assert not ds.OBJECTIVES[name].admits(bad)[1]
    assert "CCN" not in ds.objective_table(bad, "jak2")


def test_objective_table_skips_non_finite_scores(frame):
    bad = frame.copy()
    bad.loc[2, "JAK2"] = np.nan
    assert "CCC" not in ds.objective_table(bad, "jak2")


def test_every_objective_declares_the_columns_it_reads(frame):
    for name, obj in ds.OBJECTIVES.items():
        assert obj.targets, name
        assert set(obj.targets) <= set(ds.TARGETS), name
        # Perturbing a declared target must move the assayed part; anything it
        # reads but does not declare would be invisible to a per-target
        # surrogate and to a target-allocation policy.
        for t in obj.targets:
            moved = frame.copy()
            moved[t] = moved[t] - 3.0
            assert not np.allclose(obj.assayed(frame), obj.assayed(moved)), (name, t)


def test_the_kinase_subset_is_a_subset(frame):
    assert set(ds.KINASES) <= set(ds.TARGETS)
    assert len(set(ds.TARGETS)) == len(ds.TARGETS) == 58
