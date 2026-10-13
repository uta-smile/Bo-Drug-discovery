"""Aggregation level is a claim about what was sampled, so it has to be right.

`block_level.py` exists because every campaign artefact reports a paired
Wilcoxon over campaigns, and a claim ranging over blocks cannot use that number:
the 1,536 campaigns behind §34's grid are 6 blocks, and the campaign-level p of
6e-77 is a statement about how many times one block was resampled.

The floor matters as much as the p. With six blocks a signed-rank test cannot
return less than 0.031 whatever the effect size, so a p at the floor says "every
block agreed" and nothing about magnitude. Reporting it without the floor
invites exactly the over-reading the script was written to stop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import block_level as bl  # noqa: E402


def write(path: Path, *, family, budget, stl, mt):
    path.write_text(json.dumps({
        "family": family, "budget": budget,
        "arms": {"stl": {"recall": list(stl)}, "mt": {"recall": list(mt)}},
    }), encoding="utf-8")


@pytest.fixture
def grid(tmp_path):
    """Two blocks x two budgets, a constant +10-point gain, 20 campaigns each.

    Constant so that the three levels must agree on the mean and can only
    differ in n -- which is the whole point being tested.
    """
    for fam in ("a", "b"):
        for b in (50, 100):
            stl = np.full(20, 0.50)
            write(tmp_path / f"w_{fam}_b{b}.json", family=fam, budget=b,
                  stl=stl, mt=stl + 0.10)
    return tmp_path


def test_the_three_levels_count_different_things(grid):
    rows = [bl.read(f, "mt", "stl") for f in sorted(grid.glob("*.json"))]
    camp = np.concatenate([r["per_campaign"] for r in rows])
    cell = np.array([r["cell_mean"] for r in rows])
    assert len(camp) == 80 and len(cell) == 4
    # Two blocks, each run at two budgets.
    assert len({r["block"] for r in rows}) == 2
    for x in (camp, cell):
        assert np.isclose(x.mean(), 10.0)


def test_floor_is_what_a_signed_rank_can_actually_reach():
    """2 / 2**n, and 0.031 at the six blocks section 34 has."""
    assert bl.signed_rank_floor(6) == pytest.approx(0.03125)
    assert bl.signed_rank_floor(18) < 1e-4
    assert bl.signed_rank_floor(4) == pytest.approx(0.125)
    assert bl.signed_rank_floor(1) == 1.0


def test_a_unanimous_six_block_result_sits_at_the_floor():
    """The case that must be labelled. Six blocks all positive, an enormous
    effect, and the p is still 0.031 -- so the p is about the block count."""
    r = bl.report("block", np.array([1.0, 50.0, 2.0, 80.0, 3.0, 99.0]))
    assert r["p"] == pytest.approx(0.03125)
    assert r["at_floor"]


def test_a_large_sample_is_not_at_the_floor():
    rng = np.random.default_rng(0)
    r = bl.report("campaign", rng.normal(5.0, 1.0, 400))
    assert not r["at_floor"]
    assert r["p"] < 1e-10


def test_an_all_zero_contrast_does_not_crash():
    """`wilcoxon` raises on an all-zero difference; the script must report
    p = 1 rather than fail, because a tied arm is a real outcome."""
    r = bl.report("block", np.zeros(6))
    assert r["p"] == 1.0
    assert r["positive"] == 0


def test_missing_arm_is_skipped_not_guessed(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"family": "a", "budget": 1,
                             "arms": {"stl": {"recall": [0.1]}}}),
                 encoding="utf-8")
    assert bl.read(p, "mt", "stl") is None


def test_it_refuses_rather_than_reporting_nothing(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"arms": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no artefact"):
        bl.main([str(tmp_path / "*.json")])


def test_against_subtracts_the_matching_cell(grid, tmp_path):
    """The second sweep is the same grid with a smaller gain. The difference
    must be the gap between them, at every level, matched on (block, budget)
    and not on filename order."""
    other = tmp_path / "other"
    other.mkdir()
    for fam in ("b", "a"):              # deliberately the wrong order
        for b in (100, 50):
            stl = np.full(20, 0.50)
            # +4 points on `a`, +6 on `b`, so a mismatch would show up as a
            # per-block mean that is neither +6 nor +4.
            gain = 0.04 if fam == "a" else 0.06
            write(other / f"w_{fam}_b{b}.json", family=fam, budget=b,
                  stl=stl, mt=stl + gain)
    out = tmp_path / "d.json"
    bl.main([str(grid / "*.json"), "--against", str(other / "*.json"),
             "--json", str(out)])
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["against"] == str(other / "*.json")
    assert d["per_block"]["a"] == pytest.approx([6.0, 6.0])
    assert d["per_block"]["b"] == pytest.approx([4.0, 4.0])
    levels = {L["level"]: L for L in d["levels"]}
    assert levels["block"]["n"] == 2
    assert levels["campaign"]["n"] == 80


def test_against_refuses_an_unpaired_grid(grid, tmp_path):
    """One cell missing from the second sweep. Dropping it quietly would
    report a difference over a different set of blocks than the header says."""
    other = tmp_path / "half"
    other.mkdir()
    stl = np.full(20, 0.50)
    write(other / "w_a_b50.json", family="a", budget=50, stl=stl, mt=stl)
    with pytest.raises(SystemExit, match="no matching cell"):
        bl.main([str(grid / "*.json"), "--against", str(other / "*.json")])


def test_against_refuses_a_different_number_of_campaigns(grid, tmp_path):
    """Same grid, half the repeats. The subtraction would not be paired."""
    other = tmp_path / "short"
    other.mkdir()
    for fam in ("a", "b"):
        for b in (50, 100):
            stl = np.full(10, 0.50)
            write(other / f"w_{fam}_b{b}.json", family=fam, budget=b,
                  stl=stl, mt=stl)
    with pytest.raises(SystemExit, match="not paired"):
        bl.main([str(grid / "*.json"), "--against", str(other / "*.json")])


def test_without_against_nothing_changes(grid):
    """The flag is additive: the default path must be byte-identical to what
    it was before, because stored block-level numbers were read off it."""
    assert bl.main([str(grid / "*.json")]) == 0


def test_exactness_is_recorded_not_inferred_from_n():
    """scipy falls back to a normal approximation when zeros or ties are
    present, whatever `n` is. The paper claimed exactness from `n <= 18`
    alone; `report` has to carry the two conditions that actually decide it."""
    clean = bl.report("block", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert clean["exact"] and clean["zeros"] == 0 and clean["ties"] == 0
    assert clean["method"] == "exact"

    tied = bl.report("block", np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert tied["ties"] == 1 and not tied["exact"]
    assert tied["method"] == "normal approximation"

    zeroed = bl.report("block", np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert zeroed["zeros"] == 1 and not zeroed["exact"]


def test_a_floor_is_only_claimed_for_an_exact_test():
    """A p at or below the floor means "every pair agreed" only if the exact
    null was enumerated. With ties the approximation can dip under the floor
    and the label would be a lie."""
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])       # all tied, all positive
    r = bl.report("block", x)
    assert not r["exact"]
    assert not r["at_floor"], "a non-exact p must never be labelled the floor"


def test_campaign_level_is_flagged_as_approximate(grid):
    """Recall is a multiple of 1/k, so campaign-level differences are riddled
    with ties. That is a second reason not to quote them for block claims."""
    rows = [bl.read(f, "mt", "stl") for f in sorted(grid.glob("*.json"))]
    camp = np.concatenate([r["per_campaign"] for r in rows])
    assert not bl.report("campaign", camp)["exact"]
