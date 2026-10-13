"""The LLM constraint arm: caching, parsing, and the non-LLM ablation.

No test here calls the API. That is deliberate and it is also the design claim
being tested: the arm's determinism comes from the response cache, because
Claude Opus 5 removed ``temperature`` and there is no "set it to zero" option.
If these tests needed a network call, the experiment would not be reproducible
either.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from molbo.constraints import AllOf, LibraryContext, PropertyConstraint, SubstructureConstraint
from molbo.fragments import Fragment, build_menu
from molbo.llm import (
    RESPONSE_SCHEMA,
    DataDrivenProposer,
    LLMConstraintProposer,
    LLMUnavailable,
    ResponseCache,
    ScriptedProposer,
    build_prompt,
    format_observations,
    parse_response,
    request_key,
)

LIB = [
    "c1ccc2ncnc(Nc3ccccc3)c2c1",
    "COc1cc2ncnc(Nc3ccccc3)c2cc1OC",
    "c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1",
    "CCO",
    "CCCCCCCC",
    "c1ccccc1",
]
QUINAZOLINE = "c1ccc2ncnc(N)c2c1"

# The v3 schema: substructures are chosen from a mined menu by index, never
# written as SMARTS. See molbo.fragments for why.
GOOD = {
    "hypothesis": "The 4-anilinoquinazoline core drives potency.",
    "require_fragments": [0],
    "forbid_fragments": [],
    "property_bounds": [{"descriptor": "mw", "min": 200.0, "max": 500.0}],
    "confidence": 0.8,
}

# A menu whose entry 0 is the quinazoline core; entry 1 is a halogen.
MENU = [
    Fragment(index=0, label="quinazoline core", smarts=QUINAZOLINE, count=3,
             source="scaffold"),
    Fragment(index=1, label="halogen", smarts="[F,Cl,Br,I]", count=1,
             source="functional_group"),
]


@pytest.fixture(scope="module")
def ctx():
    return LibraryContext.build(LIB)


# ---- parsing --------------------------------------------------------------


def test_a_well_formed_response_becomes_a_conjunction(ctx):
    parsed = parse_response(GOOD, menu=MENU)
    assert isinstance(parsed.constraint, AllOf)
    assert parsed.constraint.mask(ctx).tolist() == [True, True, True, False, False, False]
    assert parsed.dropped == []
    assert parsed.confidence == 0.8


def test_substructure_only_response_needs_no_wrapper(ctx):
    parsed = parse_response({**GOOD, "property_bounds": []}, menu=MENU)
    assert isinstance(parsed.constraint, SubstructureConstraint)


def test_property_only_response(ctx):
    parsed = parse_response({**GOOD, "require_fragments": []}, menu=MENU)
    assert isinstance(parsed.constraint, PropertyConstraint)


def test_invalid_smarts_is_dropped_and_recorded_not_raised(ctx):
    """A crash would report nothing; a recorded drop rate is itself a result."""
    parsed = parse_response({**GOOD, "require_smarts": ["[[[bad"]}, menu=MENU)
    assert parsed.constraint is not None
    assert any("bad" in d for d in parsed.dropped)
    assert parsed.constraint.mask(ctx).sum() == 3  # the good pattern still applied


def test_an_entirely_unusable_response_yields_no_constraint():
    parsed = parse_response(
        {
            "hypothesis": "",
            "require_fragments": [99],
            "forbid_fragments": [],
            "property_bounds": [],
            "confidence": 0.0,
        },
        menu=MENU,
    )
    assert parsed.constraint is None
    assert parsed.dropped


def test_unknown_descriptor_is_dropped():
    parsed = parse_response(
        {**GOOD, "property_bounds": [{"descriptor": "vibes", "min": 0, "max": 1}]},
        menu=MENU,
    )
    assert any("vibes" in d for d in parsed.dropped)


def test_inverted_bounds_are_dropped():
    parsed = parse_response(
        {**GOOD, "property_bounds": [{"descriptor": "mw", "min": 500, "max": 200}]},
        menu=MENU,
    )
    assert any("inverted" in d for d in parsed.dropped)


def test_an_all_null_bound_is_dropped_as_meaningless():
    parsed = parse_response(
        {**GOOD, "property_bounds": [{"descriptor": "mw", "min": None, "max": None}]},
        menu=MENU,
    )
    assert any("empty" in d for d in parsed.dropped)


def test_a_one_sided_bound_is_kept(ctx):
    parsed = parse_response(
        {
            **GOOD,
            "require_fragments": [],
            "property_bounds": [{"descriptor": "mw", "min": 200.0, "max": None}],
        },
        menu=MENU,
    )
    assert parsed.constraint.mask(ctx).tolist() == (ctx.descriptors["mw"] >= 200.0).tolist()


def test_missing_fields_do_not_crash_the_parser():
    assert parse_response({}, menu=MENU).constraint is None


# ---- schema ---------------------------------------------------------------


def test_schema_is_strict_so_the_model_cannot_invent_fields():
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert set(RESPONSE_SCHEMA["required"]) == set(RESPONSE_SCHEMA["properties"])


def test_schema_constrains_descriptors_to_the_known_set():
    from molbo.constraints import DESCRIPTORS

    enum = RESPONSE_SCHEMA["properties"]["property_bounds"]["items"]["properties"]["descriptor"][
        "enum"
    ]
    assert set(enum) == set(DESCRIPTORS)


# ---- prompt ---------------------------------------------------------------


def test_observations_are_ordered_most_potent_first():
    text = format_observations([("CCO", 4.0), ("CCC", 9.0), ("CCN", 6.0)])
    assert text.index("9.00") < text.index("6.00") < text.index("4.00")


def test_a_long_series_shows_both_extremes_and_says_what_it_omitted():
    obs = [(f"C{'C' * i}O", float(i)) for i in range(100)]
    text = format_observations(obs, max_show=10)
    assert "99.00" in text and "0.00" in text
    assert "omitted" in text


def test_empty_observations_are_stated_plainly():
    assert "No compounds" in format_observations([])


def test_prompt_reports_the_library_size():
    assert "2103 compounds" in build_prompt([("CCO", 4.0)], library_size=2103)


# ---- caching / determinism ------------------------------------------------


def test_the_request_key_is_stable():
    a = request_key("be", "sys", "prompt", RESPONSE_SCHEMA, "v1")
    b = request_key("be", "sys", "prompt", RESPONSE_SCHEMA, "v1")
    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend_key": "other"},
        {"system": "different"},
        {"prompt": "different"},
        {"version": "v2"},
    ],
)
def test_changing_any_part_of_the_request_changes_the_key(kwargs):
    base = dict(backend_key="be", system="sys", prompt="p", schema=RESPONSE_SCHEMA, version="v1")
    assert request_key(**base) != request_key(**{**base, **kwargs})


def test_cache_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path)
    assert cache.get("k") is None
    cache.put("k", {"response": GOOD})
    assert cache.get("k")["response"] == GOOD
    assert len(cache) == 1


def test_a_warm_cache_needs_no_api_call(tmp_path, ctx):
    """The reproducibility guarantee: re-running a sweep makes zero calls."""
    proposer = LLMConstraintProposer(cache_dir=tmp_path, offline=True)
    key, _ = proposer.request_key_for([("CCO", 4.0)], ctx)
    ResponseCache(tmp_path).put(key, {"response": GOOD})

    constraint = proposer(observations=[("CCO", 4.0)], ctx=ctx, round_index=0, seed=0)
    assert constraint is not None
    assert proposer.calls == 0
    assert proposer.cache_hits == 1


def test_offline_mode_refuses_rather_than_silently_calling(tmp_path, ctx):
    proposer = LLMConstraintProposer(cache_dir=tmp_path, offline=True)
    with pytest.raises(LLMUnavailable, match="no cached response"):
        proposer(observations=[("CCO", 4.0)], ctx=ctx, round_index=0, seed=0)


def test_the_proposer_logs_what_it_proposed(tmp_path, ctx):
    proposer = LLMConstraintProposer(cache_dir=tmp_path, offline=True)
    key, _ = proposer.request_key_for([("CCO", 4.0)], ctx)
    ResponseCache(tmp_path).put(key, {"response": GOOD})

    proposer(observations=[("CCO", 4.0)], ctx=ctx, round_index=0, seed=11)
    entry = proposer.log[0]
    assert entry["seed"] == 11
    assert entry["hypothesis"] == GOOD["hypothesis"]
    assert entry["constraint"] is not None


# ---- scripted / ablation --------------------------------------------------


def test_scripted_proposer_replays_in_order(ctx):
    p = ScriptedProposer([GOOD, {**GOOD, "require_fragments": []}], menu=MENU)
    first = p(observations=[], ctx=ctx)
    second = p(observations=[], ctx=ctx)
    assert isinstance(first, AllOf)
    assert isinstance(second, PropertyConstraint)


def test_scripted_proposer_repeats_its_last_response(ctx):
    p = ScriptedProposer([GOOD], menu=MENU)
    assert p(observations=[], ctx=ctx) is not None
    assert p(observations=[], ctx=ctx) is not None


def test_data_driven_proposer_brackets_the_best_observed(ctx):
    obs = [(LIB[0], 9.0), (LIB[1], 9.5), (LIB[2], 8.5), (LIB[3], 4.0),
           (LIB[4], 4.2), (LIB[5], 5.0), ("CCCCO", 4.1), ("c1ccncc1", 5.1)]
    c = DataDrivenProposer()(observations=obs, ctx=ctx)
    assert isinstance(c, PropertyConstraint)
    mask = c.mask(ctx)
    assert mask[:3].all()  # the potent compounds it was derived from survive
    assert not mask[3]     # ethanol does not


def test_data_driven_proposer_waits_for_enough_data(ctx):
    assert DataDrivenProposer()(observations=[("CCO", 4.0)], ctx=ctx) is None


def test_data_driven_proposer_uses_only_observed_values(ctx):
    """It must derive its box from the observations, not from the library."""
    obs = [(LIB[i], float(i)) for i in range(6)] + [("CCCCO", 1.0), ("c1ccncc1", 2.0)]
    c = DataDrivenProposer(top_fraction=0.25)(observations=obs, ctx=ctx)
    assert c.mask(ctx)[5]  # benzene scored best here, so benzene-like survives


def test_menu_kwargs_reach_the_menu_builder(ctx):
    """A 250k-molecule pool needs mine_sample, or candidate mining walks it all."""
    small = LLMConstraintProposer(cache_dir="/tmp/x", offline=True,
                                  menu_kwargs={"max_entries": 3})
    assert len(small.menu_for(ctx)) <= 3
    default = LLMConstraintProposer(cache_dir="/tmp/x", offline=True)
    assert len(default.menu_for(ctx)) >= len(small.menu_for(ctx))


def test_the_menu_is_built_once_per_library(ctx):
    """Mining is the expensive half; a 20-seed sweep must not repeat it."""
    p = LLMConstraintProposer(cache_dir="/tmp/x", offline=True)
    first = p.menu_for(ctx)
    assert p.menu_for(ctx) is first
