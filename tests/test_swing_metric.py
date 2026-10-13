"""The design swing is in recall points, and an artefact's delta is not.

A campaign artefact stores `arms[a]["delta_vs_stl"]` as the **AUC top-k**
difference, built from `acc[arm]["auc"]`. Recall lives in `recall_mean` and in
the per-campaign `recall` vector. The two are not proportional: on the CYP
45%-coverage pair the pooled design swing is 29.06 recall points and 0.1528 in
AUC, a ratio of 190; on the 90% pair it is 46.02 and 0.4165, a ratio of 110.

Reading `delta_vs_stl` as recall is therefore not a rounding difference -- it
produced a reported swing of 15.28 "points" against a published C4 value of
29.06, from the same runs. The checks here pin the three things that would have
caught it:

* which metric the stored field is, on a real artefact;
* that the calibration script forms its deltas from the recall vectors;
* that the swing it reports for `pooled` is the published C4 swing, so the
  calibration and the manuscript cannot drift apart silently.

These read `results/`, deliberately: the failure being fixed is an analysis
that disagreed with the artefacts it claimed to summarise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import avoidance_calibration as ac  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# The published C4 pair: CYP, budget 111, 300 historical entries, randomized
# auxiliary responses. paper.tex quotes +22.97 -> -23.05, "approximately 46".
C4 = {"withheld": ("avoid_cyp_h100_withheld.json", 22.97),
      "uniform": ("avoid_cyp_h100_uniform.json", -23.05)}


def load(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not in results/")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("design,expected",
                         [(d, v) for d, (_, v) in C4.items()])
def test_published_c4_values_are_recall_not_auc(design, expected):
    art = load(C4[design][0])
    stats = ac.arm_stats(art, "pooled")
    assert stats["delta_recall"] == pytest.approx(expected, abs=0.01)
    # The same artefact's stored field is a different number entirely.
    assert art["arms"]["pooled"]["delta_vs_stl"] != pytest.approx(expected / 100,
                                                                 abs=1e-4)


def test_stored_delta_field_is_the_auc_difference():
    art = load(C4["withheld"][0])
    stats = ac.arm_stats(art, "pooled")
    assert ac.check_delta_field(art, "pooled", stats) == "auc"
    assert art["arms"]["pooled"]["delta_vs_stl"] == pytest.approx(
        stats["delta_auc"], abs=1e-12)


def test_arm_stats_recomputes_from_the_per_campaign_vectors():
    art = load(C4["uniform"][0])
    stats = ac.arm_stats(art, "pooled")
    rec = (np.asarray(art["arms"]["pooled"]["recall"], float)
           - np.asarray(art["arms"]["stl"]["recall"], float)) * 100
    assert stats["delta_recall"] == pytest.approx(float(rec.mean()), abs=1e-12)
    assert stats["n_campaigns"] == len(rec)
    assert stats["delta_recall_sem"] > 0


def test_pooled_swing_is_the_published_c4_swing():
    """The calibration's reference swing and the manuscript's C4 reversal are
    the same quantity from the same runs, so they move together or not at
    all."""
    rows = ac.sweep_table(100)
    cal = ac.calibrate(100, rows)
    assert cal["recall"]["pooled_swing"] == pytest.approx(46.02, abs=0.01)
    assert cal["recall"]["pooled_withheld"] == pytest.approx(22.97, abs=0.01)
    assert cal["recall"]["pooled_uniform"] == pytest.approx(-23.05, abs=0.01)
    # ... and the AUC swing is the number that was mistaken for it.
    assert cal["auc"]["pooled_swing"] == pytest.approx(0.4165, abs=0.0001)
    assert cal["stored_delta_field_is"] == "auc"


def test_the_two_metrics_do_not_rank_the_recovered_fraction_alike():
    """Not a defect -- a reason the metric has to be named.

    At 90% coverage avoidance recovers 73% of the swing in recall and 84% in
    AUC. A report that does not say which it means is unreadable.
    """
    rows = ac.sweep_table(100)
    cal = ac.calibrate(100, rows)
    assert cal["recall"]["interpolated_fraction"] != pytest.approx(
        cal["auc"]["interpolated_fraction"], abs=0.02)
