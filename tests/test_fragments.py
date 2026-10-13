"""The substructure menu handed to the model.

The menu exists so the arm measures chemistry judgment rather than SMARTS
typing ability. Two properties have to hold for that to be true: every entry
must be a valid, matchable pattern (so nothing the model picks can fail), and
the menu must be built from structure alone (so no unobserved value leaks into
the arm and breaks comparability with the controls).
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from molbo.constraints import LibraryContext, SubstructureConstraint
from molbo.fragments import (
    FUNCTIONAL_GROUPS,
    build_menu,
    menu_digest,
    render_menu,
    ring_systems,
)

LIB = [
    "c1ccc2ncnc(Nc3ccccc3)c2c1",
    "COc1cc2ncnc(Nc3ccccc3)c2cc1OC",
    "c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1",
    "c1ccc2ncnc(Nc3ccc(F)cc3)c2c1",
    "c1ccc2ncnc(Nc3ccc(Br)cc3)c2c1",
    "COc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OC",
    "CCO",
    "CCCCCCCC",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C1CCNCC1",
    "C1COCCN1",
    "c1ccncc1",
    "CCN(CC)CC",
    "OCC(O)CO",
    "CC(=O)Nc1ccccc1",
    "O=S(=O)(N)c1ccccc1",
    "N#Cc1ccccc1",
    "FC(F)(F)c1ccccc1",
    "C#Cc1ccccc1",
]


@pytest.fixture(scope="module")
def menu():
    return build_menu(LIB, min_fraction=0.0, max_fraction=1.0)


@pytest.fixture(scope="module")
def ctx():
    return LibraryContext.build(LIB)


# ---- the guarantee that makes the arm measurable --------------------------


def test_every_curated_group_is_valid_smarts():
    """A broken entry here would silently poison every run that picked it."""
    for label, smarts in FUNCTIONAL_GROUPS.items():
        assert Chem.MolFromSmarts(smarts) is not None, label


def test_every_menu_entry_is_a_usable_query(menu, ctx):
    """The whole point: nothing the model can pick is able to fail."""
    assert menu
    for frag in menu:
        assert Chem.MolFromSmarts(frag.smarts) is not None, frag.label
        SubstructureConstraint(require=(frag.smarts,)).mask(ctx)  # must not raise


def test_reported_counts_match_reality(menu, ctx):
    """The model is told how much each entry matches; that has to be true."""
    for frag in menu:
        actual = int(SubstructureConstraint(require=(frag.smarts,)).mask(ctx).sum())
        assert actual == frag.count, f"{frag.label}: said {frag.count}, matches {actual}"


def test_indices_are_dense_and_ordered(menu):
    assert [f.index for f in menu] == list(range(len(menu)))


# ---- selectivity window ---------------------------------------------------


def test_entries_matching_almost_everything_are_excluded():
    """A pattern covering the whole library cannot narrow it."""
    menu = build_menu(LIB, max_fraction=0.5)
    assert all(f.count <= 0.5 * len(LIB) for f in menu)


def test_entries_matching_almost_nothing_are_excluded():
    """A pattern covering two molecules would starve the search."""
    menu = build_menu(LIB, min_fraction=0.25)
    assert all(f.count >= 0.25 * len(LIB) for f in menu)


def test_menu_length_is_capped(menu):
    assert len(build_menu(LIB, max_entries=5, min_fraction=0.0, max_fraction=1.0)) <= 5


def test_menu_is_ordered_by_coverage(menu):
    counts = [f.count for f in menu]
    assert counts == sorted(counts, reverse=True)


# ---- no leakage -----------------------------------------------------------

def test_the_menu_depends_only_on_structure():
    """Built twice from the same molecules in a different order: same content.

    If the menu could vary with anything but structure it would be a channel
    for information the optimiser should not have, and the controls would stop
    being comparable.
    """
    a = build_menu(LIB, min_fraction=0.0, max_fraction=1.0)
    b = build_menu(list(reversed(LIB)), min_fraction=0.0, max_fraction=1.0)
    assert {(f.smarts, f.count) for f in a} == {(f.smarts, f.count) for f in b}


def test_build_menu_is_deterministic(menu):
    again = build_menu(LIB, min_fraction=0.0, max_fraction=1.0)
    assert [f.smarts for f in again] == [f.smarts for f in menu]
    assert menu_digest(again) == menu_digest(menu)


def test_digest_changes_with_the_menu(menu):
    other = build_menu(LIB, max_entries=4, min_fraction=0.0, max_fraction=1.0)
    assert menu_digest(other) != menu_digest(menu)


# ---- ring systems ---------------------------------------------------------


def test_fused_rings_come_back_as_one_system():
    """A quinazoline is one ring system, not two rings."""
    systems = ring_systems("c1ccc2ncnc(Nc3ccccc3)c2c1")
    assert len(systems) == 2  # the fused bicycle, plus the pendant phenyl
    assert any(len(s) > 12 for s in systems)


def test_acyclic_molecules_have_no_ring_systems():
    assert ring_systems("CCCCCCCC") == []


def test_unparseable_input_is_handled():
    assert ring_systems("not_a_molecule") == []


# ---- rendering ------------------------------------------------------------


def test_rendered_menu_shows_index_and_coverage(menu):
    text = render_menu(menu, len(LIB))
    assert "[ 0]" in text
    assert "%" in text
    for frag in menu[:3]:
        assert frag.label[:20] in text


def test_empty_menu_renders_without_crashing():
    assert "no substructure menu" in render_menu([], 10)


def test_patterns_matching_the_same_molecules_appear_once():
    """Three spellings of piperidine are one choice, not three menu slots.

    Without this, near-duplicate SMILES spellings crowd out genuinely different
    chemistry and the model's choice between them is meaningless.
    """
    lib = ["C1CCNCC1", "C1CCNCC1CC", "CCO", "c1ccccc1", "CC(=O)Nc1ccccc1"] * 6
    menu = build_menu(lib, min_fraction=0.0, max_fraction=1.0, max_entries=40)
    ctx = LibraryContext.build(lib)
    seen = set()
    for frag in menu:
        key = SubstructureConstraint(require=(frag.smarts,)).mask(ctx).tobytes()
        assert key not in seen, f"{frag.label} duplicates an earlier entry's match set"
        seen.add(key)


def test_mining_on_a_subsample_still_counts_on_the_whole_library():
    """The cost saving must not turn the coverage figure into an estimate."""
    lib = LIB * 8
    menu = build_menu(lib, mine_sample=10, min_fraction=0.0, max_fraction=1.0)
    ctx = LibraryContext.build(lib)
    assert menu
    for frag in menu:
        actual = int(SubstructureConstraint(require=(frag.smarts,)).mask(ctx).sum())
        assert actual == frag.count
