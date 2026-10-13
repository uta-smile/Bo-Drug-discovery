"""Every check in :mod:`molbo.audit`, fired on purpose.

FINDINGS section 29 states the rule these tests exist to satisfy: *a check that
has never fired is not evidence*. The audits in `dense_block_campaign.py` and
`dense_block_h1.py` were validated by editing the script to inject a violation,
running it, and reverting -- which works once and leaves nothing behind. Now
that the checks are a module, the injection can be permanent: each test below
hands `Audit` a specific violation and asserts it is caught, and each has a
clean twin asserting it is not caught when nothing is wrong.

The clean twins are the half that would otherwise rot. An audit that reports a
problem on every run is as useless as one that never does, and a false positive
is worse than no check at all, because it trains the reader to ignore the line.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbo.audit import Audit


def _a() -> Audit:
    return Audit(enabled=True)


# ---- the disabled path ---------------------------------------------------


def test_disabled_audit_records_nothing():
    """The call sites are unguarded, so `enabled=False` has to be inert."""
    a = Audit(enabled=False)
    a.cell()
    a.exactly([1, 1, 1], 99, "nonsense")
    a.disjoint([1, 2], [2, 3], "x", "y")
    a.subset([9], [1], "x", "y")
    a.not_relabelled([1.0, 2.0], [[1.0, 2.0]], "x")
    a.same_across("g", 0, "te", [1, 2])
    a.same_across("g", 1, "te", [3, 4])
    assert a.problems == []
    assert a.checks == 0 and a.cells == 0
    assert a.report() is True
    assert a.to_json() == {"enabled": False, "checks": 0, "cells": 0,
                           "clean": False, "problems": []}


def test_disabled_audit_is_not_reported_clean():
    """`clean` must mean "checked and passed", never "not checked"."""
    assert Audit(enabled=False).to_json()["clean"] is False
    assert _a().to_json()["clean"] is True


# ---- exactly -------------------------------------------------------------


def test_exactly_accepts_the_right_draw():
    a = _a()
    a.exactly(np.array([4, 1, 7]), 3, "draw")
    assert a.problems == []


@pytest.mark.parametrize("idx, size, needle", [
    ([1, 2], 3, "expected 3"),
    ([1, 2, 3, 4], 3, "expected 3"),
    ([1, 2, 2], 3, "duplicate"),
])
def test_exactly_catches_wrong_size_and_duplicates(idx, size, needle):
    a = _a()
    a.exactly(np.array(idx), size, "draw")
    assert any(needle in p for p in a.problems), a.problems


def test_exactly_reports_size_and_duplicates_separately():
    """A short draw caused by a repeat is two different bugs; say both."""
    a = _a()
    a.exactly(np.array([1, 1]), 3, "draw")
    assert len(a.problems) == 2


# ---- disjoint / subset ---------------------------------------------------


def test_disjoint_clean_and_dirty():
    a = _a()
    a.disjoint([1, 2, 3], [4, 5], "evaluation", "training")
    assert a.problems == []
    a.disjoint([1, 2, 3], [3, 4], "evaluation", "training")
    assert len(a.problems) == 1
    assert "1 rows are in both evaluation and training" in a.problems[0]


def test_disjoint_counts_every_leaked_row():
    """The count is the point: one shared row is a slip, three hundred is a bug."""
    a = _a()
    a.disjoint(np.arange(10), np.arange(5, 15), "evaluation", "training")
    assert "5 rows are in both" in a.problems[0]


def test_subset_clean_and_dirty():
    a = _a()
    a.subset([2, 3], [1, 2, 3, 4], "picks", "eligible")
    assert a.problems == []
    a.subset([2, 99], [1, 2, 3, 4], "picks", "eligible")
    assert "1 rows of picks fall outside eligible" in a.problems[0]


# ---- not_relabelled ------------------------------------------------------


def test_not_relabelled_passes_for_a_genuine_relative():
    a = _a()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    a.not_relabelled(y, [y + 0.3, y * 0.5], "JAK2")
    assert a.problems == []


def test_not_relabelled_catches_the_answer_relabelled():
    a = _a()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    a.not_relabelled(y, [y + 0.3, y.copy()], "JAK2")
    assert len(a.problems) == 1
    assert "auxiliary column 1 equals the primary" in a.problems[0]


def test_not_relabelled_ignores_columns_of_another_length():
    """History columns are shorter than the primary's; they are not comparable."""
    a = _a()
    a.not_relabelled(np.arange(5.0), [np.arange(3.0)], "JAK2")
    assert a.problems == []


def test_not_relabelled_treats_matching_nans_as_a_match():
    """A column that is NaN wherever the primary is NaN is the same column."""
    a = _a()
    y = np.array([1.0, np.nan, 3.0])
    a.not_relabelled(y, [y.copy()], "JAK2")
    assert len(a.problems) == 1


# ---- excludes ------------------------------------------------------------


def test_excludes_catches_a_target_used_as_its_own_auxiliary():
    a = _a()
    a.excludes("JAK2", ["LCK", "SRC"], "aux set")
    assert a.problems == []
    a.excludes("JAK2", ["LCK", "JAK2"], "aux set")
    assert "JAK2 is its own auxiliary" in a.problems[0]


# ---- same_across ---------------------------------------------------------


def test_same_across_is_silent_when_the_arms_shared_a_split():
    a = _a()
    for h in (0, 50, 200):
        a.same_across(("JAK2", 20, 0), h, "evaluation rows", [3, 1, 2])
    assert a.report() is True


def test_same_across_is_order_insensitive():
    """The same rows in another order are the same evaluation set."""
    a = _a()
    a.same_across("g", 0, "te", [1, 2, 3])
    a.same_across("g", 1, "te", [3, 2, 1])
    assert a.report() is True


def test_same_across_catches_an_unpaired_comparison():
    a = _a()
    a.same_across(("JAK2", 20, 0), 0, "evaluation rows", [1, 2, 3])
    a.same_across(("JAK2", 20, 0), 200, "evaluation rows", [1, 2, 4])
    assert a.report() is False
    problems = a.to_json()["problems"]
    assert len(problems) == 1
    assert "not paired" in problems[0]


def test_same_across_reports_once_per_quantity_not_once_per_cell():
    """A split that varies with the arm varies everywhere; do not print 500 copies."""
    a = _a()
    for cell in range(500):
        a.same_across(cell, 0, "evaluation rows", [1, 2])
        a.same_across(cell, 200, "evaluation rows", [1, 3])
    problems = a.to_json()["problems"]
    assert len(problems) == 1
    assert "500 of 500 cells" in problems[0]


def test_same_across_separates_the_quantities_it_tracks():
    """Training rows can be shared while evaluation rows are not, and vice versa."""
    a = _a()
    for h in (0, 200):
        a.same_across("cell", h, "primary training rows", [1, 2])
    a.same_across("cell", 0, "evaluation rows", [5])
    a.same_across("cell", 200, "evaluation rows", [6])
    problems = a.to_json()["problems"]
    assert len(problems) == 1
    assert "evaluation rows" in problems[0]


def test_a_single_arm_is_not_an_unpaired_comparison():
    """One H in the grid has nothing to be paired against."""
    a = _a()
    a.same_across("cell", 0, "evaluation rows", [1, 2])
    assert a.report() is True


# ---- reporting -----------------------------------------------------------


def test_report_counts_checks_and_cells():
    a = _a()
    for _ in range(3):
        a.cell()
    a.exactly([1, 2], 2, "draw")
    assert a.to_json()["cells"] == 3
    assert a.to_json()["checks"] == 2


def test_to_json_caps_the_problem_list_but_not_the_count():
    """The JSON travels with the result; it must not become the result."""
    a = _a()
    for i in range(80):
        a.want(False, f"problem {i}")
    js = a.to_json()
    assert js["clean"] is False
    assert len(js["problems"]) == 50
    assert js["checks"] == 80


def test_report_returns_false_once_anything_failed():
    a = _a()
    a.disjoint([1], [1], "evaluation", "training")
    assert a.report("[C]") is False
