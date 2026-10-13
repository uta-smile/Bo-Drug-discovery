"""The reporting half of `dense_block_campaign`, on arm sets that have no `stl`.

Every delta the script reports is *against* the single-task arm, and `--arms`
does not have to include it: a sweep that only wants the multi-task trace is a
legitimate run, and the budget ladder in section 25 runs `stl` alone from the
other side. Running `--arms mt` used to raise

    TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'

from the summary loop, after the whole campaign had been computed -- so the
failure cost the full runtime and produced nothing.

The fix had a choice, and the wrong half of it is what these tests pin. Filling
the deltas with 0.0 would also stop the crash, and would be worse than the
crash: `delta_vs_stl: 0.0` in an artefact reads as a measured tie, and
`acquisition_across_families` and `window_across_families` both difference that
field. Null is the only honest value when there is no reference arm, so the
tests below assert null and assert that a reader can tell the two cases apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

dbc = pytest.importorskip("dense_block_campaign")


def _run(tmp_path, *arms, family="mmp", budget=25):
    """A deliberately tiny campaign: this exercises reporting, not science."""
    out = tmp_path / "r.json"
    rc = dbc.main([
        "--family", family, "--history", "10", "--budget", str(budget),
        "--batch", "5", "--n-init", "5", "--k", "5", "--repeats", "2",
        "--arms", *arms, "--out", str(out),
    ])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_arms_without_stl_does_not_crash(tmp_path):
    """The regression itself: the summary loop must survive a missing `stl`."""
    report = _run(tmp_path, "mt")
    assert "mt" in report["arms"]
    assert "stl" not in report["arms"]


def test_missing_stl_reports_null_delta_not_zero(tmp_path):
    """Null, never 0.0 -- a zero here would read as a measured tie."""
    mt = _run(tmp_path, "mt")["arms"]["mt"]
    assert mt["delta_vs_stl"] is None
    assert mt["delta_sem"] is None
    assert mt["frac_better"] is None
    # The arm's own numbers are still real and still written.
    assert mt["auc_mean"] > 0
    assert len(mt["auc"]) == len(mt["recall"]) > 0


def test_delta_is_a_number_once_stl_is_present(tmp_path):
    """The clean twin. A guard that fires on every run is not a guard."""
    arms = _run(tmp_path, "stl", "mt")["arms"]
    assert arms["mt"]["delta_vs_stl"] is not None
    assert arms["mt"]["delta_sem"] is not None
    assert arms["mt"]["frac_better"] is not None
    # stl against itself is a true zero, and must stay one.
    assert arms["stl"]["delta_vs_stl"] == 0.0
    assert arms["stl"]["delta_sem"] == 0.0


def test_stl_alone_is_still_a_budget_ladder(tmp_path):
    """The other arm set with no comparison to make: `stl` on its own."""
    arms = _run(tmp_path, "stl")["arms"]
    assert arms["stl"]["delta_vs_stl"] == 0.0
    assert arms["stl"]["auc_mean"] > 0
