"""What a stored campaign result has to say about the data it came from.

Two things went into `results/` in October with no way to tell them apart: a
BindingDB block and a KIBA panel block, both written as `"family": "k10"` with
no source, no `--potency`, and no record of whether the censoring filter had
run. That is the §30 defect again -- the operating-window table reached a paper
unaudited because `--audit` printed its verdict to the terminal and stored
nothing -- and it is worse here, because `k01`/`k02`/`k10` are *positions in a
scan*. Re-running `find_dense_blocks` at a different `--min-per-target` moves
them, and the old results would then name the wrong proteins while looking
perfectly well-formed.

So the tests below pin two separate things:

* `block_provenance` writes down the flags that change what the numbers mean;
* `PANEL_FAMILIES` names the same accessions the stored KIBA artefacts were
  measured on, which is what makes pinning them an improvement over the scan
  labels rather than a second thing to keep in sync by hand.

The second is a tie to `results/kiba_block_rho.json` on purpose. A registry that
drifts from the artefacts it claims to describe is the failure being fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dense_block_campaign import (  # noqa: E402
    FAMILIES,
    PANEL_FAMILIES,
    PANELS,
    block_provenance,
)

ROOT = Path(__file__).resolve().parents[1]

# registry label -> the scan label the stored results were written under.
SCAN_LABELS = {"kiba_prkc": "k01", "kiba_cdk5": "k02", "kiba_lck": "k10"}


def _args(**over):
    base = dict(source="data/cache/tdc/kiba_score.csv", potency="identity",
                blocks_from=None, keep_censored=True, pile_frac=0.02,
                qed_min=None, subsample=None)
    base.update(over)
    return argparse.Namespace(**base)


def _provenance(**over):
    return block_provenance(_args(**over), targets=["LCK", "GSK3B"],
                            block_csv="data/bindingdb/block_x.csv",
                            n_block=843, n_pool=843)


# ---- the record itself ---------------------------------------------------

def test_records_source_and_potency():
    """The two fields that decide what the stored numbers mean.

    `--potency` is not cosmetic: pointing `nanomolar` at a KIBA score inverts
    the objective, so a result that does not say which was used cannot be read
    at all.
    """
    p = _provenance()
    assert p["source"] == "data/cache/tdc/kiba_score.csv"
    assert p["potency"] == "identity"


def test_paths_are_posix():
    """Compared between runs and quoted in the write-up, so not backslashes."""
    p = block_provenance(_args(source=Path("data") / "cache" / "kiba.csv"),
                         targets=["LCK"],
                         block_csv=Path("data") / "bindingdb" / "b.csv",
                         n_block=10, n_pool=10)
    assert "\\" not in p["source"]
    assert "\\" not in p["block_csv"]


def test_censoring_distinguishes_skipped_from_run():
    """`pile_frac` alone cannot say whether the filter ran.

    It is recorded at the top level of every report whether or not
    `--keep-censored` was passed, which is how the KIBA runs look filtered in
    the artefact and are not. The JAK block loses 42% of its compounds to that
    filter, so the two cases are not a detail.
    """
    assert "skipped" in _provenance(keep_censored=True)["censoring"]
    assert _provenance(keep_censored=False)["censoring"] == "pile_frac=0.02"


def test_block_size_before_and_after_filters():
    p = block_provenance(_args(keep_censored=False), targets=["JAK1"],
                         block_csv="b.csv", n_block=1563, n_pool=904)
    assert (p["n_block"], p["n_pool"]) == (1563, 904)


def test_usable_from_a_sibling_script_without_every_flag():
    """The helper is shared, and the other scripts have smaller argument sets.

    `dense_block_h1` and `selectivity_campaign` cut the same blocks and have no
    `--qed-min` or `--subsample`. Requiring them would mean a second copy of
    this function, which is how the two reporting paths diverged the first time.
    """
    bare = argparse.Namespace(source="x.csv", potency="nanomolar")
    p = block_provenance(bare, targets=["JAK1"], block_csv="b.csv",
                         n_block=5, n_pool=5)
    assert p["potency"] == "nanomolar"
    assert p["censoring"] == "pile_frac=None"
    assert p["qed_min"] is None


# ---- the pinned panel blocks ---------------------------------------------

def test_panel_families_do_not_shadow_the_registry():
    """The two are merged at `--family` resolution, so a collision would
    silently redirect a published family to a different dump."""
    assert not set(FAMILIES) & set(PANEL_FAMILIES)


def test_every_panel_family_resolves_to_a_panel():
    """The prefix is load-bearing -- it is what fills in `--source`."""
    for name in PANEL_FAMILIES:
        assert name.split("_")[0] in PANELS


def test_panels_are_identity_scored():
    """Each panel's `--potency` must match the direction of its own column.

    This test used to assert that *both* panels were identity-scored, on the
    reasoning that a kinome panel reports a score where higher already means
    stronger. That is true of KIBA and false of Davis, whose `Y` is a
    dissociation constant in nM -- lower is better. So the test did not catch
    the mismatch; it pinned it in place, and would have kept failing anyone who
    tried to fix it.

    The lesson is worth the comment: a test that encodes an assumption is only
    as good as the assumption. This one now checks each row against what its
    source actually measures.
    """
    assert PANELS["kiba"]["potency"] == "identity", (
        "KIBA score is an integrated higher-is-better statistic")
    assert PANELS["davis"]["potency"] == "nanomolar", (
        "Davis is Kd in nM; with identity a campaign maximises Kd, and since "
        "71% of Davis sits at the 10,000 nM ceiling the top-k becomes the "
        "non-binders")
    for name, panel in PANELS.items():
        assert panel["potency"] in {"identity", "nanomolar"}, name
        assert "source" in panel, name


@pytest.mark.parametrize("name,scan", sorted(SCAN_LABELS.items()))
def test_pinned_accessions_match_the_stored_artefact(name, scan):
    """The registry entry is the same block the stored KIBA results came from.

    Checked against the gene labels in `results/kiba_block_rho.json`, which is
    the artefact that reported those blocks' task correlations and runnability.
    A mismatch means the pinned set has drifted from the results that cite it.
    """
    art = ROOT / "results" / "kiba_block_rho.json"
    if not art.exists():
        pytest.skip(f"{art.name} not present")
    stored = json.loads(art.read_text())["labelled"][scan]["targets"]
    assert sorted(PANEL_FAMILIES[name].values()) == sorted(stored)
