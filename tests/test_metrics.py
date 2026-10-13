"""AUC top-k: the metric the whole comparison rests on."""

from __future__ import annotations

import numpy as np
import pytest

from molbo.metrics import (
    auc_top_k,
    compare_arms,
    normalize_values,
    summarize_arm,
    top_auc,
    top_k_curve,
)


def buffer_from(scores):
    """``{smiles: (score, call_index)}`` with call order matching the list."""
    return {f"C{i}": (s, i + 1) for i, s in enumerate(scores)}


# ---- top_k_curve ----------------------------------------------------------


def test_curve_is_running_mean_of_best_k():
    assert top_k_curve([0.1, 0.9, 0.5], k=2).tolist() == pytest.approx([0.1, 0.5, 0.7])


def test_curve_uses_fewer_than_k_before_k_exist():
    assert top_k_curve([0.4], k=10).tolist() == [0.4]


def test_curve_is_monotone_non_decreasing_for_k_equals_one():
    rng = np.random.default_rng(0)
    c = top_k_curve(rng.random(200), k=1)
    assert np.all(np.diff(c) >= -1e-12)


def test_curve_pads_flat_to_budget():
    c = top_k_curve([0.5, 0.7], k=1, budget=5)
    assert c.tolist() == pytest.approx([0.5, 0.7, 0.7, 0.7, 0.7])


def test_curve_truncates_past_budget():
    assert top_k_curve([0.1] * 10, k=1, budget=3).size == 3


def test_empty_scores():
    assert top_k_curve([], k=10, budget=4).tolist() == [0.0] * 4
    assert auc_top_k([], k=10, budget=4) == 0.0
    assert top_auc({}, max_oracle_calls=100) == 0.0


# ---- ordering sensitivity: the whole point of the metric ------------------


def test_finding_good_molecules_early_scores_higher():
    early = [1.0] + [0.0] * 99
    late = [0.0] * 99 + [1.0]
    assert auc_top_k(early, k=1, budget=100) > auc_top_k(late, k=1, budget=100)


def test_same_final_best_can_have_different_auc():
    """A metric that only looked at the best found would call these equal."""
    early, late = [1.0] + [0.0] * 99, [0.0] * 99 + [1.0]
    assert max(early) == max(late)
    assert auc_top_k(early, k=1, budget=100) == pytest.approx(1.0)
    assert auc_top_k(late, k=1, budget=100) == pytest.approx(0.01)


def test_auc_is_bounded_by_zero_and_one_for_unit_scores():
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.random(50)
        assert 0.0 <= auc_top_k(s, k=10, budget=50) <= 1.0


def test_perfect_run_scores_one():
    assert auc_top_k([1.0] * 10, k=10, budget=10) == pytest.approx(1.0)


# ---- PMO parity -----------------------------------------------------------


def test_top_auc_matches_hand_computed_trapezoid():
    """Reproduce PMO's own arithmetic on a tiny case.

    freq_log=1, budget=4, k=1, scores [0.2, 0.8, 0.5, 0.5]:
    running best is 0.2, 0.8, 0.8, 0.8 and the trapezoid starts from 0.
    sum = (0+0.2)/2 + (0.2+0.8)/2 + (0.8+0.8)/2 + (0.8+0.8)/2 = 0.1+0.5+0.8+0.8
    """
    got = top_auc(buffer_from([0.2, 0.8, 0.5, 0.5]), top_n=1, freq_log=1, max_oracle_calls=4)
    assert got == pytest.approx(2.2 / 4)


def test_top_auc_extends_short_runs_flat_when_finish():
    short = buffer_from([1.0, 1.0])
    with_finish = top_auc(short, top_n=1, freq_log=1, max_oracle_calls=10, finish=True)
    without = top_auc(short, top_n=1, freq_log=1, max_oracle_calls=10, finish=False)
    assert with_finish > without
    assert without == pytest.approx((0.5 + 1.0) / 10)


def test_top_auc_is_insensitive_to_dict_insertion_order():
    scores = [0.3, 0.9, 0.1, 0.7]
    forward = buffer_from(scores)
    shuffled = dict(reversed(list(forward.items())))
    assert top_auc(forward, top_n=2, freq_log=1, max_oracle_calls=4) == pytest.approx(
        top_auc(shuffled, top_n=2, freq_log=1, max_oracle_calls=4)
    )


def test_top_auc_and_auc_top_k_agree_closely_at_freq_log_one():
    """They differ only by PMO's zero start, worth one call of area."""
    rng = np.random.default_rng(3)
    s = rng.random(500).tolist()
    pmo = top_auc(buffer_from(s), top_n=10, freq_log=1, max_oracle_calls=500)
    exact = auc_top_k(s, k=10, budget=500)
    assert abs(pmo - exact) < 0.01


def test_freq_log_sampling_biases_the_auc_downward():
    """Coarse sampling *understates* AUC, and the bias grows with ``freq_log``.

    PMO samples the curve every ``freq_log`` calls and starts the trapezoid at
    zero, so the first window is integrated as a straight line from 0 up to the
    top-k mean at call ``freq_log``. On a curve that rises fast -- which is
    every curve here, since the top-10 mean is already ~0.96 by call 100 -- that
    line runs well below the truth and the lost area is never recovered.

    This is not a rounding artifact to be tolerated; at ``freq_log=100`` on a
    1000-call budget it is worth ~4 AUC points, which is larger than the
    between-arm differences this project is trying to resolve. Hence: quote
    PMO-comparable numbers at PMO's own ``freq_log`` and budget, and use
    :func:`auc_top_k` for comparisons between our own arms.
    """
    rng = np.random.default_rng(4)
    s = rng.random(1000).tolist()
    auc = lambda fl: top_auc(buffer_from(s), top_n=10, freq_log=fl, max_oracle_calls=1000)

    fine, coarse = auc(1), auc(100)
    assert coarse < fine  # systematic, not noise
    assert [auc(fl) for fl in (1, 10, 50, 100, 200)] == sorted(
        [auc(fl) for fl in (1, 10, 50, 100, 200)], reverse=True
    )

    # The whole gap is the zero-start deficit over the first window.
    curve = top_k_curve(s, k=10, budget=1000)
    first_window_deficit = (curve[:100].sum() - 100 * curve[99] / 2) / 1000
    assert fine - coarse == pytest.approx(first_window_deficit, abs=0.002)


# ---- normalisation --------------------------------------------------------


def test_normalize_uses_library_range_not_observed_range():
    vals = normalize_values([5.0, 7.0], lo=4.0, hi=11.0)
    assert vals.tolist() == pytest.approx([1 / 7, 3 / 7])


def test_normalize_clips_out_of_range():
    assert normalize_values([2.0, 99.0], lo=4.0, hi=11.0).tolist() == [0.0, 1.0]


def test_normalize_rejects_degenerate_range():
    with pytest.raises(ValueError, match="degenerate"):
        normalize_values([1.0], lo=5.0, hi=5.0)


def test_raw_pic50_auc_is_dominated_by_the_offset():
    """Why normalisation is not optional on the ChEMBL tier.

    Two runs with very different search quality look nearly identical on raw
    pIC50, because PMO's AUC integrates from zero and a pIC50 of 4 is already
    most of the way to 11.
    """
    good = [11.0] + [4.0] * 99
    bad = [4.0] * 99 + [11.0]
    raw_gap = auc_top_k(good, k=1, budget=100) / auc_top_k(bad, k=1, budget=100)
    lo, hi = 4.0, 11.0
    norm_gap = auc_top_k(normalize_values(good, lo, hi), k=1, budget=100) / max(
        1e-9, auc_top_k(normalize_values(bad, lo, hi), k=1, budget=100)
    )
    assert raw_gap < 3, "raw values compress the difference"
    assert norm_gap > 50, "normalised values expose it"


# ---- cross-seed statistics ------------------------------------------------


def test_summarize_arm_reports_sem():
    s = summarize_arm("a", [0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
    assert s.n_seeds == 3
    assert s.auc_mean == pytest.approx(0.2)
    assert s.auc_sem == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1) / np.sqrt(3))
    assert "n=  3" in str(s)


def test_summarize_single_seed_has_zero_sem():
    assert summarize_arm("a", [0.5], [1.0]).auc_sem == 0.0


def test_paired_comparison_detects_a_consistent_shift():
    rng = np.random.default_rng(5)
    base = rng.random(20)
    res = compare_arms(base + 0.05, base, paired=True)
    assert res["test"] == "wilcoxon-signed-rank"
    assert res["p_value"] < 0.001
    assert res["mean_diff"] == pytest.approx(0.05)


def test_paired_comparison_on_identical_arms_is_not_significant():
    x = [0.1, 0.2, 0.3, 0.4]
    res = compare_arms(x, x, paired=True)
    assert res["p_value"] == 1.0
    assert res["cohens_dz"] == 0.0


def test_paired_comparison_requires_matched_seeds():
    with pytest.raises(ValueError, match="same number of seeds"):
        compare_arms([1, 2, 3], [1, 2], paired=True)


def test_unpaired_comparison_falls_back_to_mann_whitney():
    rng = np.random.default_rng(6)
    res = compare_arms(rng.random(15) + 1.0, rng.random(12), paired=False)
    assert res["test"] == "mann-whitney-u"
    assert res["p_value"] < 0.001
    assert "cohens_d" in res


def test_noise_alone_does_not_look_significant():
    """The reason the goal asks for 20+ seeds: two identical methods must not
    separate by luck."""
    rng = np.random.default_rng(7)
    false_positives = 0
    for _ in range(100):
        a, b = rng.random(20), rng.random(20)
        if compare_arms(a, b, paired=False)["p_value"] < 0.05:
            false_positives += 1
    assert false_positives <= 12, f"{false_positives}/100 false positives at alpha=0.05"
