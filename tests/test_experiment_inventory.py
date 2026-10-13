"""The inventory exists because counts were written from memory twice.

Round two of review "fixed" the null table's campaign count by reading the
wrong artefact group, and the first version of this script could not have
caught that: it worked from a hand-written list of groups which omitted
`sdref_*`, the group the null table is actually built from.

So the property that matters is not that the inventory is correct on today's
results tree. It is that the inventory **cannot silently omit an experiment**,
and that it **refuses** rather than averages when a group's artefacts disagree
about how they were run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_inventory as inv  # noqa: E402


def write(path: Path, *, family, budget=100, repeats=8, targets=4,
          campaigns=32, excludes_top=True, history=50, audit=True,
          acquisition="ei", k=10, batch=10, pile_frac=0.02):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "family": family, "budget": budget, "repeats": repeats,
        "history": history, "history_excludes_top": excludes_top,
        "acquisition": acquisition, "k": k, "batch": batch,
        "pile_frac": pile_frac,
        "per_target": {f"T{i}": {} for i in range(targets)},
        "audit": {"enabled": audit, "clean": audit},
        "arms": {"stl": {"recall": [0.5] * campaigns}},
        "provenance": {"source": "data/x.csv", "targets": ["a"] * targets},
    }), encoding="utf-8")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(inv, "RESULTS", tmp_path)
    monkeypatch.setattr(inv, "ROOT", tmp_path.parent)
    return tmp_path


def test_discovery_finds_a_group_nobody_listed(tree):
    """The failure that happened. A new prefix appears in results/ and the
    inventory must pick it up without anyone editing the script."""
    write(tree / "sdref_jak_0.0.json", family="jak")
    write(tree / "brandnew_jak_1.json", family="jak")
    found = {name for name, _, _ in inv.discover()}
    assert "sdref" in found
    assert "brandnew" in found, "a group not in FEEDS must still be inventoried"


def test_an_uncited_group_is_labelled_not_hidden(tree):
    write(tree / "brandnew_jak_1.json", family="jak")
    feeds = {name: f for name, _, f in inv.discover()}
    assert feeds["brandnew"] == "(not cited)"


def test_directory_groups_are_found_too(tree):
    write(tree / "xfam_seed" / "k01.json", family="k01")
    assert "xfam_seed" in {name for name, _, _ in inv.discover()}


def test_a_group_holding_both_history_designs_is_split_not_averaged(tree):
    """The history design is the protocol axis the paper argues about. A
    group holding both must come out as two rows -- averaging them would hide
    exactly the confound the design section is about."""
    write(tree / "g_a_1.json", family="a", excludes_top=True)
    write(tree / "g_b_1.json", family="b", excludes_top=False)
    g = inv.summarise("g", "g_*.json", "test")
    assert "history_excludes_top" in g["conflicts"]

    uni = inv.summarise("g", "g_*.json", "test", only_design=False)
    wit = inv.summarise("g", "g_*.json", "test", only_design=True)
    assert uni["history_design"] == ["uniform"] and uni["n_artefacts"] == 1
    assert wit["history_design"] == ["withheld"] and wit["n_artefacts"] == 1
    assert uni["conflicts"] == [] and wit["conflicts"] == []
    assert inv.main([]) == 0, "a split is not a failure"


def test_a_real_protocol_conflict_still_fails(tree):
    """Differing repeat counts inside one group is not a design split; it
    means two different experiments share a name."""
    write(tree / "g_a_1.json", family="a", repeats=8, campaigns=32)
    write(tree / "g_b_1.json", family="b", repeats=32, campaigns=128)
    g = inv.summarise("g", "g_*.json", "test")
    assert "repeats" in g["conflicts"]
    assert inv.main([]) == 1


def test_seeds_times_targets_must_equal_the_stored_campaign_count(tree):
    """The check that would have caught the original discrepancy: a stated
    repeat count that does not match the array actually saved."""
    write(tree / "g_a_1.json", family="a", repeats=8, targets=4, campaigns=32)
    assert inv.summarise("g", "g_*.json", "t")["seeds_times_targets_consistent"]

    write(tree / "h_a_1.json", family="a", repeats=8, targets=4, campaigns=128)
    assert not inv.summarise("h", "h_*.json", "t")[
        "seeds_times_targets_consistent"]


def test_differing_arms_are_not_a_conflict(tree):
    """A sweep may hold a single-arm rung beside a multi-arm reference run.
    That is not a protocol disagreement and must not be reported as one."""
    write(tree / "g_a_1.json", family="a")
    p = tree / "g_b_1.json"
    write(p, family="b")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["arms"]["mt"] = {"recall": [0.6] * 32}
    p.write_text(json.dumps(d), encoding="utf-8")
    assert inv.summarise("g", "g_*.json", "t")["conflicts"] == []


def test_unaudited_artefacts_are_counted(tree):
    write(tree / "g_a_1.json", family="a", audit=False)
    write(tree / "g_b_1.json", family="b", audit=True)
    g = inv.summarise("g", "g_*.json", "t")
    assert g["audit_clean"] == 1 and g["n_artefacts"] == 2


def test_it_refuses_an_empty_tree(tree):
    with pytest.raises(SystemExit, match="no campaign artefacts"):
        inv.main([])
