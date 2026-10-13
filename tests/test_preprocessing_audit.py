"""Preprocessing claims must match the block files, and the check must bite.

Three review rounds found that numbers written from memory are wrong often
enough that none should be trusted unchecked. The preprocessing numbers had
never been verified against the CSVs since they were written; one of twenty was
wrong (the KIBA surviving-row range, quoted as 25--99 when the pinned blocks
give 32--263, excluding one of the three blocks the sentence names).

These tests pin the two things that make the audit worth running: the censoring
rule is the one the paper describes, and a claim that stops matching is a
failure rather than a line of output nobody reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import preprocessing_audit as pa  # noqa: E402
from dense_block_h1 import drop_censored, nanomolar_potency  # noqa: E402

HAVE_BLOCKS = (ROOT / "data" / "bindingdb" /
               "block_cyp2c19-cyp2c9-cyp2d6-cyp3a4.csv").exists()


def test_the_potency_transform_maps_screening_doses_where_the_paper_says():
    """The paper names pIC50 4.00, 4.30, 4.52, 5.00, 5.30 as the fixed doses
    100, 50, 30, 10 and 5 micromolar. If the transform ever changed, those
    five sentences would quietly become wrong."""
    for um, expected in ((5.0, 5.30), (10.0, 5.00), (30.0, 4.52),
                         (50.0, 4.30), (100.0, 4.00)):
        assert nanomolar_potency(um * 1000.0) == pytest.approx(expected,
                                                               abs=0.005)


def test_higher_is_better_after_the_transform():
    """A stronger binder has a *lower* IC50 and must come out *higher*.
    Inverting this would have every campaign optimise for weak binders."""
    assert nanomolar_potency(1.0) > nanomolar_potency(1000.0)


def test_the_filter_removes_a_spike_and_keeps_a_smooth_column():
    """The rule as stated: any value held by more than 2% of a column is a
    pile-up, and compounds touching one go."""
    rng = np.random.default_rng(0)
    smooth = rng.normal(6.0, 1.0, 500)
    block = pd.DataFrame({"smiles": [f"C{i}" for i in range(500)],
                          "a": smooth.copy(), "b": smooth.copy()})
    kept, piles = drop_censored(block, ["a", "b"], pile_frac=0.02,
                                verbose=False)
    assert not piles and len(kept) == 500

    block.loc[:49, "a"] = 5.00          # 10% of the column on one value
    kept, piles = drop_censored(block, ["a", "b"], pile_frac=0.02,
                                verbose=False)
    assert 5.00 in {float(k) for k in piles["a"]}
    assert len(kept) == 450


def test_a_spike_just_under_the_threshold_survives():
    """2% is the rule, so 1.8% of a column must not be dropped -- otherwise
    the filter is stricter than the paper says."""
    rng = np.random.default_rng(1)
    block = pd.DataFrame({"smiles": [f"C{i}" for i in range(1000)],
                          "a": rng.normal(6.0, 1.0, 1000)})
    block.loc[:17, "a"] = 5.00          # 18 of 1000 = 1.8%
    kept, piles = drop_censored(block, ["a"], pile_frac=0.02, verbose=False)
    assert not piles and len(kept) == 1000


@pytest.mark.skipif(not HAVE_BLOCKS, reason="block CSVs not present")
def test_the_audit_passes_on_the_manuscript_as_committed():
    """The whole point. If a preprocessing number in the paper drifts from
    the data, this fails."""
    assert pa.main([]) == 0


@pytest.mark.skipif(not HAVE_BLOCKS, reason="block CSVs not present")
def test_the_censored_block_is_the_one_the_paper_describes():
    """CYP is heavily censored and JAK is not; the paper's argument for the
    filter rests on that contrast being this large."""
    cyp = pa.censoring(pa.BINDINGDB["cyp"])
    jak = pa.censoring(pa.BINDINGDB["jak"])
    assert cyp["identical_all_targets"] > 0.35
    assert jak["identical_all_targets"] < 0.05
    assert cyp["survival"] < jak["survival"]
