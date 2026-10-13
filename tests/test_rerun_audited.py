"""Rebuilding a command line from an artefact must not omit a setting.

The first version of `rerun_audited.py` did not translate
`aux_quality_requested` back into `--aux-quality`, so the two randomized
null-control cells were re-run with *real* auxiliary data. They failed the
numeric comparison, which is how it surfaced -- but the two `q = 1` cells
passed, because omitting `--aux-quality 1.0` happens to be equivalent to using
the measured column. A silent omission that only shows up on half the cells,
and there looks like a data problem rather than a rebuild problem, is exactly
the failure worth pinning.

So the rebuilder now refuses an artefact carrying a configuration key it does
not know about, instead of quietly letting the re-run take a default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rerun_audited as ra  # noqa: E402


def artefact(**over):
    d = {
        "family": "cyp", "budget": 111, "batch": 10, "k": 10, "history": 50,
        "repeats": 32, "acquisition": "ei", "pile_frac": 0.02,
        "sd_scale": 1.0, "history_cluster": 0.0,
        "history_excludes_top": True, "scaffold_split": False,
        "seeding": {"mode": "random"},
        "arms": {"stl": {}, "mt": {}},
        "provenance": {"source": "data/x.csv", "seed_pool": "random"},
        "audit": {}, "per_target": {}, "n_pool": 300,
    }
    d.update(over)
    return d


def test_aux_quality_is_carried_back_into_the_command_line():
    """The bug. Without this flag the re-run uses real auxiliary data."""
    argv = ra.argv_for(artefact(aux_quality_requested=0.0), Path("out.json"))
    assert "--aux-quality" in argv
    assert argv[argv.index("--aux-quality") + 1] == "0.0"


def test_q_one_also_carries_it_even_though_it_is_a_no_op():
    """q = 1 is equivalent to omitting the flag, which is why omitting it hid
    the bug on half the cells. The rebuild states it anyway."""
    argv = ra.argv_for(artefact(aux_quality_requested=1.0), Path("out.json"))
    assert "--aux-quality" in argv


def test_the_shuffled_null_becomes_its_own_flag():
    argv = ra.argv_for(artefact(aux_quality_requested="shuffled"),
                       Path("out.json"))
    assert "--aux-shuffle" in argv and "--aux-quality" not in argv


def test_an_unknown_setting_is_refused_not_defaulted():
    """The general fix. A key the rebuilder cannot translate must stop the
    re-run, not silently become whatever the default is."""
    with pytest.raises(SystemExit, match="does not know how"):
        ra.argv_for(artefact(some_new_knob=0.5), Path("out.json"))


def test_outputs_are_not_mistaken_for_settings():
    """Diagnostics an artefact records about what happened must not trip the
    refusal -- only things that configured the run."""
    d = artefact(aux_quality_achieved=-0.0022, reach=0.95,
                 targets_improving=3)
    assert "--aux-quality" not in ra.argv_for(d, Path("o.json"))


def test_the_history_design_is_rebuilt_from_the_artefact():
    uni = ra.argv_for(artefact(history_excludes_top=False), Path("o.json"))
    wit = ra.argv_for(artefact(history_excludes_top=True), Path("o.json"))
    assert "--history-excludes-top" not in uni
    assert "--history-excludes-top" in wit


def test_a_mismatch_is_reported_field_by_field():
    old = {"arms": {"stl": {"auc_mean": 1.0, "recall_mean": 0.5}}}
    new = {"arms": {"stl": {"auc_mean": 1.5, "recall_mean": 0.5}}}
    bad = ra.compare(old, new)
    assert len(bad) == 1 and "auc_mean" in bad[0]
    assert ra.compare(old, old) == []


def test_a_missing_arm_is_a_mismatch():
    old = {"arms": {"stl": {"auc_mean": 1.0}, "mt": {"auc_mean": 2.0}}}
    new = {"arms": {"stl": {"auc_mean": 1.0}}}
    assert any("mt" in m for m in ra.compare(old, new))


def test_an_artefact_without_provenance_is_refused():
    d = artefact()
    d.pop("provenance")
    with pytest.raises(SystemExit, match="provenance"):
        ra.argv_for(d, Path("o.json"))
