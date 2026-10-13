"""Constraints: compilation, the graceful-degradation ladder, and the controls.

The degradation tests are the important ones. A constraint that empties the
search space must not be allowed to end a run early or silently shorten its
budget -- either would make the arm's curve incomparable to every other arm's,
which is the one thing the experiment cannot tolerate.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbo.constraints import (
    AllOf,
    AnyOf,
    ConstraintError,
    LibraryContext,
    MaskConstraint,
    PropertyConstraint,
    RandomSubsetConstraint,
    SubstructureConstraint,
    apply_constraints,
    oracle_constraint,
    parse_spec,
)

LIB = [
    "c1ccc2ncnc(Nc3ccccc3)c2c1",        # anilinoquinazoline
    "COc1cc2ncnc(Nc3ccccc3)c2cc1OC",    # anilinoquinazoline, dimethoxy
    "c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1",    # anilinoquinazoline, chloro
    "CCO",
    "CCCCCCCCCCCCCCCC",
    "c1ccccc1",
]
QUINAZOLINE = "c1ccc2ncnc(N)c2c1"


@pytest.fixture(scope="module")
def ctx():
    return LibraryContext.build(LIB)


# ---- substructure ---------------------------------------------------------


def test_require_matches_only_the_series(ctx):
    mask = SubstructureConstraint(require=(QUINAZOLINE,)).mask(ctx)
    assert mask.tolist() == [True, True, True, False, False, False]


def test_require_is_a_conjunction(ctx):
    """Two patterns means both, not either."""
    both = SubstructureConstraint(require=(QUINAZOLINE, "[Cl]")).mask(ctx)
    assert both.tolist() == [False, False, True, False, False, False]


def test_forbid_excludes(ctx):
    mask = SubstructureConstraint(require=(QUINAZOLINE,), forbid=("[Cl]",)).mask(ctx)
    assert mask.tolist() == [True, True, False, False, False, False]


def test_invalid_smarts_is_rejected_loudly():
    with pytest.raises(ConstraintError, match="unparseable SMARTS"):
        SubstructureConstraint(require=("this is not smarts [[[",))


# ---- properties -----------------------------------------------------------


def test_property_bounds_are_inclusive(ctx):
    mw = ctx.descriptors["mw"]
    lo = float(mw.min())
    mask = PropertyConstraint(bounds={"mw": (lo, lo)}).mask(ctx)
    assert mask.sum() == int((mw == lo).sum())


def test_open_ended_bounds(ctx):
    assert PropertyConstraint(bounds={"mw": (None, None)}).mask(ctx).all()
    heavy = PropertyConstraint(bounds={"mw": (200.0, None)}).mask(ctx)
    assert heavy.tolist() == (ctx.descriptors["mw"] >= 200.0).tolist()


def test_unknown_descriptor_is_rejected():
    with pytest.raises(ConstraintError, match="unknown descriptor"):
        PropertyConstraint(bounds={"vibes": (0, 1)})


def test_unevaluable_molecule_fails_rather_than_passes():
    """A NaN descriptor is not evidence that the rule is satisfied."""
    ctx = LibraryContext.build(["CCO", "not_a_molecule"])
    assert np.isnan(ctx.descriptors["mw"][1])
    assert PropertyConstraint(bounds={"mw": (0.0, 10_000.0)}).mask(ctx).tolist() == [True, False]


# ---- composition ----------------------------------------------------------


def test_all_of_intersects(ctx):
    c = AllOf(parts=(
        SubstructureConstraint(require=(QUINAZOLINE,)),
        PropertyConstraint(bounds={"mw": (0.0, 260.0)}),
    ))
    mask = c.mask(ctx)
    assert mask.sum() < SubstructureConstraint(require=(QUINAZOLINE,)).mask(ctx).sum()


def test_any_of_unions(ctx):
    c = AnyOf(parts=(
        SubstructureConstraint(require=("[Cl]",)),
        SubstructureConstraint(require=("CCO",)),
    ))
    assert c.mask(ctx).sum() >= 2


def test_empty_any_of_allows_everything(ctx):
    assert AnyOf(parts=()).mask(ctx).all()


# ---- controls -------------------------------------------------------------


def test_random_subset_keeps_the_requested_fraction(ctx):
    mask = RandomSubsetConstraint(fraction=0.5, seed=0).mask(ctx)
    assert mask.sum() == 3


def test_random_subset_varies_with_its_seed():
    ctx = LibraryContext.build(LIB * 10)
    a = RandomSubsetConstraint(fraction=0.3, seed=1).mask(ctx)
    b = RandomSubsetConstraint(fraction=0.3, seed=2).mask(ctx)
    assert not np.array_equal(a, b)


def test_random_subset_is_reproducible_for_one_seed(ctx):
    a = RandomSubsetConstraint(fraction=0.5, seed=7).mask(ctx)
    b = RandomSubsetConstraint(fraction=0.5, seed=7).mask(ctx)
    assert np.array_equal(a, b)


def test_random_subset_always_keeps_at_least_one(ctx):
    assert RandomSubsetConstraint(fraction=0.0, seed=0).mask(ctx).sum() == 1


def test_oracle_constraint_selects_the_true_best(ctx):
    values = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0]
    mask = oracle_constraint(values, fraction=0.5).mask(ctx)
    assert mask.tolist() == [False, True, False, True, False, True]


def test_mask_constraint_rejects_a_length_mismatch(ctx):
    with pytest.raises(ConstraintError, match="entries for a library"):
        MaskConstraint(values=np.ones(3, bool)).mask(ctx)


# ---- degradation ladder ---------------------------------------------------


def test_no_constraint_allows_everything(ctx):
    out = apply_constraints(None, ctx)
    assert out.mode == "none" and out.mask.all()


def test_a_roomy_constraint_is_applied_hard(ctx):
    out = apply_constraints(
        SubstructureConstraint(require=(QUINAZOLINE,)), ctx, min_candidates=2
    )
    assert out.mode == "hard"
    assert out.n_allowed == 3
    assert not out.mask[3]
    assert np.allclose(out.penalty, 0.0)


def test_a_too_tight_constraint_degrades_to_a_soft_penalty(ctx):
    """It must not starve the optimiser, but the hypothesis is still expressed."""
    out = apply_constraints(
        SubstructureConstraint(require=("[Cl]",)), ctx, min_candidates=5, soft_penalty=2.0
    )
    assert out.mode == "soft"
    assert out.mask.all()  # nothing is excluded outright
    assert out.penalty[2] == 0.0  # the chloro compound is preferred
    assert out.penalty[0] == -2.0  # everything else is penalised


def test_a_constraint_matching_nothing_falls_through_to_unconstrained(ctx):
    out = apply_constraints(SubstructureConstraint(require=("[Au]",)), ctx, min_candidates=2)
    assert out.mode == "none"
    assert out.mask.all()
    assert out.n_allowed == 0


def test_tightness_is_judged_on_unseen_molecules(ctx):
    """A mask admitting only compounds already bought is as useless as an empty one."""
    seen = np.array([True, True, True, False, False, False])
    out = apply_constraints(
        SubstructureConstraint(require=(QUINAZOLINE,)),
        ctx,
        min_candidates=2,
        already_seen=seen,
    )
    assert out.mode == "soft"


def test_fraction_allowed_is_reported(ctx):
    out = apply_constraints(SubstructureConstraint(require=(QUINAZOLINE,)), ctx, min_candidates=1)
    assert out.fraction_allowed == pytest.approx(0.5)


# ---- spec parsing ---------------------------------------------------------


def test_parse_substructure_spec(ctx):
    c = parse_spec({"require": [QUINAZOLINE], "forbid": ["[Cl]"]})
    assert c.mask(ctx).tolist() == [True, True, False, False, False, False]


def test_parse_property_spec(ctx):
    c = parse_spec({"kind": "property", "bounds": {"mw": [0, 260]}})
    assert isinstance(c, PropertyConstraint)


def test_parse_list_is_an_implicit_conjunction(ctx):
    c = parse_spec([{"require": [QUINAZOLINE]}, {"bounds": {"mw": [0, 260]}}])
    assert isinstance(c, AllOf)


def test_parse_nested_any_of(ctx):
    c = parse_spec({"kind": "any_of", "parts": [{"require": ["[Cl]"]}, {"require": ["CCO"]}]})
    assert isinstance(c, AnyOf)


def test_parse_rejects_nonsense_rather_than_ignoring_it():
    """Silently dropping a clause would score a nonsense model as if it proposed one."""
    with pytest.raises(ConstraintError, match="unrecognised"):
        parse_spec({"kind": "telepathy"})
    with pytest.raises(ConstraintError, match="must be a dict or list"):
        parse_spec("require quinazoline")


def test_parse_rejects_malformed_bounds():
    with pytest.raises(ConstraintError, match=r"\[lo, hi\]"):
        parse_spec({"kind": "property", "bounds": {"mw": [1, 2, 3]}})


# ---- identity -------------------------------------------------------------


def test_fingerprint_is_stable_and_spec_sensitive():
    a = SubstructureConstraint(require=(QUINAZOLINE,))
    b = SubstructureConstraint(require=(QUINAZOLINE,))
    c = SubstructureConstraint(require=(QUINAZOLINE,), forbid=("[Cl]",))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


# ---- the context cannot silently produce an empty mask --------------------


def test_a_directly_constructed_context_masks_identically_to_a_built_one():
    """The bug this guards against cost a whole PMO sweep.

    ``LibraryContext(smiles=...)`` used to leave ``mols`` empty, so every
    substructure mask came back all-False. That is indistinguishable from "the
    constraint matched nothing", so apply_constraints degraded to unconstrained
    and the arm silently became plain BO -- reporting dAUC = +0.0000 on every
    task, which looks like a finding rather than a failure.
    """
    c = SubstructureConstraint(require=(QUINAZOLINE,))
    direct = LibraryContext(smiles=LIB)
    built = LibraryContext.build(LIB)
    assert c.mask(direct).tolist() == c.mask(built).tolist()
    assert c.mask(direct).any()


def test_a_directly_constructed_context_computes_descriptors():
    direct = LibraryContext(smiles=LIB)
    assert not np.isnan(direct.descriptors["mw"]).all()
    assert len(direct.mols) == len(LIB)


def test_context_work_is_cached_not_repeated():
    ctx = LibraryContext(smiles=LIB)
    assert ctx.mols is ctx.mols
    assert ctx.descriptors is ctx.descriptors


def test_an_all_false_mask_is_reported_as_matching_nothing(ctx):
    """So a degraded constraint is visible in the log rather than silent."""
    out = apply_constraints(SubstructureConstraint(require=("[Au]",)), ctx, min_candidates=1)
    assert out.mode == "none"
    assert out.n_allowed == 0
