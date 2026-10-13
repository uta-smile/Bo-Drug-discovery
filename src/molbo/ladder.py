"""One arm ladder, shared by the ChEMBL and PMO runners.

Both runners used to build their own arm lists, and they had drifted: the ChEMBL
runner grew five arms and the PMO runner had two. A comparison is only as good
as the arms it is read against, so the ladder lives here once and both tiers run
the identical set.

The ladder answers one question -- **is it the BO, the LLM, or the two
together?** -- and every arm exists to isolate one term of that:

=====================  ========================================================
category               what a win by this category would mean
=====================  ========================================================
``none``               ``random``. The floor. Every other arm must clear it
                       before any of its machinery is worth discussing.
``bo``                 ``bo_ei``/``bo_ucb``/``bo_greedy``. A surrogate helps,
                       with no model in the loop at all.
``llm``                ``llm_only``. The model picks molecules with **no
                       surrogate underneath it**. Its control is ``random``:
                       the same uniform draw, picked from uniformly instead of
                       by the model. This is the arm the project never had, and
                       without it a BO+LLM number cannot be attributed -- a
                       reader cannot tell the model from the GP carrying it.
``bo+llm``             ``bo_llm_constraint``/``bo_llm_select``. The model
                       advising a surrogate. Both already ran on the congeneric
                       tier (-0.030 p=0.006, and neutral-with-positional-bias
                       respectively); they are here so the new arms have the
                       project's own prior results in the same table.
``agentic``            ``bo_policy_*``. The model, or a control, choosing the
                       acquisition and its hyperparameters each round. This is
                       the SARA coupling.
=====================  ========================================================

Within ``agentic`` the controls are the point. ``bo_policy_scripted`` is four
napkin rules reading the identical diagnostics the model is shown; ``random``
and ``roundrobin`` switch without reading anything. If ``bo_policy_llm`` does
not clear all three, the agency is not what is helping -- the same test that
caught a free MaxMin heuristic matching the LLM warm start at its own strategy.

Pairing
-------
Every arm here takes the same ``n_init``, ``batch_size`` and
``candidate_subset``, so arms differing in one field are an ablation rather than
two unrelated configurations. ``llm_only`` is the exception that proves it: it
takes ``shortlist_k`` from the same knob ``bo_llm_select`` uses, so "the model
chose 1 of 15" means the same thing in both rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .campaign import ArmConfig

CATEGORY = {
    "random": "none",
    "bo_ei": "bo",
    "bo_ucb": "bo",
    "bo_greedy": "bo",
    "llm_only": "llm",
    "bo_llm_constraint": "bo+llm",
    "bo_llm_select": "bo+llm",
    "bo_random_select": "bo+llm",
    # Selector controls that need no model. `bo_maxmu_select` buys the
    # shortlist's highest posterior mean and `bo_clm_select` is the trained
    # contrastive selector (scripts/clm_train.py). Categorised with the other
    # selector arms so the report groups the shortlist question in one place.
    "bo_maxmu_select": "bo+llm",
    "bo_clm_select": "bo+llm",
    "bo_oracle_select": "ceiling",
    "bo_policy_scripted": "agentic",
    "bo_policy_random": "agentic",
    "bo_policy_roundrobin": "agentic",
    "bo_policy_llm": "agentic",
    "bo_policy_oracle": "ceiling",
}
"""Which question each arm answers. Drives the grouping in the report.

``ceiling`` is deliberately not a category a method can belong to:
``bo_policy_oracle`` reads the answer key, exactly as
``molbo.constraints.oracle_constraint`` does, and is a reference line saying how
much room exists rather than a result.
"""

NEEDS_MODEL = frozenset(
    {"llm_only", "bo_llm_constraint", "bo_llm_select", "bo_policy_llm"}
)
"""Arms that cost model calls. Everything else runs with no endpoint at all,
which is what makes a partial run useful rather than a blocked one."""

LADDER_ARMS = frozenset(
    {
        "random",
        "bo_ei",
        "bo_ucb",
        "bo_greedy",
        "llm_only",
        "bo_random_select",
        "bo_llm_constraint",
        "bo_llm_select",
        "bo_policy_scripted",
        "bo_policy_random",
        "bo_policy_roundrobin",
        "bo_policy_llm",
        "bo_policy_oracle",
    }
)
"""The arms :func:`build_ladder` can actually produce.

:data:`CATEGORY` is deliberately wider: it also classifies arms built only by
individual scripts (``bo_maxmu_select``, ``bo_clm_select``,
``bo_oracle_select``) so a report can colour and group them wherever they turn
up. A report that renders "the ladder" must read *this* set, not ``CATEGORY``,
or those script-only arms appear as ladder arms that were skipped -- which
says "this arm lost the chance to run" about an arm the ladder never contained.
"""


@dataclass
class LadderSpec:
    """The knobs both tiers share. Defaults are the ChEMBL tier's."""

    n_init: int = 10
    batch_size: int = 1
    candidate_subset: int | None = None
    shortlist_k: int = 15
    acquisitions: Sequence[str] = ("ei", "ucb", "greedy")
    units: str = "pIC50 (higher is more potent)"
    constraint_min_candidates: int = 50

    menu_mine_sample: int | None = None
    """Molecules sampled when mining the substructure menu for the constraint
    arm. ``None`` mines the whole library.

    That default is right for the ChEMBL tier (~9k compounds) and wrong for PMO:
    the ZINC pool is 249,455 molecules and mining walks every one of them, so the
    first constraint call would stall the run before it scored anything. The PMO
    runner passes 3,000, matching the value its own constraint arm has always
    used. The menu is mined from the sample but its coverage fractions are
    counted on the full pool, so the numbers shown to the model stay honest."""

    policy_every: int = 1
    """Rounds between LLM policy calls. ``1`` asks every round.

    This is the cost dial for ``bo_policy_llm`` and the reason it is separate
    from the other LLM arms: ``bo_llm_constraint`` asks once per run, while this
    one can ask once per evaluation. On a 1,000-call PMO task that is the
    difference between 5 calls and 100 per task."""

    def common(self) -> dict:
        d = dict(n_init=self.n_init, batch_size=self.batch_size)
        if self.candidate_subset is not None:
            d["candidate_subset"] = self.candidate_subset
        return d


def build_ladder(
    spec: LadderSpec,
    *,
    backend=None,
    cache_root: str = "data/cache",
    values: np.ndarray | None = None,
    maximize: bool = True,
    include: Sequence[str] = ("none", "bo", "llm", "bo+llm", "agentic"),
    offline: bool = False,
) -> list[ArmConfig]:
    """Assemble the ladder.

    ``backend`` is any :class:`molbo.llm.Backend`. When it is ``None`` every arm
    in :data:`NEEDS_MODEL` is skipped and the rest still run, so the non-model
    half of the grid is never blocked on an endpoint being up. The skip is
    silent here and reported by the caller, because a missing arm must be
    visible in the results file rather than inferred from a gap in a table.

    ``values`` enables the hindsight ceiling ``bo_policy_oracle``. Pass it only
    where a ceiling is wanted; it reads the answer key.
    """
    from .agentic.policies import (
        LLMPolicy,
        OraclePolicy,
        RandomPolicy,
        RoundRobinPolicy,
        ScriptedPolicy,
    )

    common = spec.common()
    arms: list[ArmConfig] = []
    want = set(include)

    if "none" in want:
        arms.append(ArmConfig("random", strategy="random", **common))

    if "bo" in want:
        for acq in spec.acquisitions:
            arms.append(
                ArmConfig(f"bo_{acq}", strategy="bo", acquisition=acq, **common)
            )

    if "llm" in want and backend is not None:
        from .llm import LLMSelector

        arms.append(
            ArmConfig(
                "llm_only",
                strategy="llm",
                selector=LLMSelector(
                    backend=backend,
                    cache_dir=f"{cache_root}/llm_only",
                    offline=offline,
                    units=spec.units,
                ),
                shortlist_k=spec.shortlist_k,
                **common,
            )
        )

    if "bo+llm" in want:
        from .llm import LLMSelector, RandomShortlistSelector

        # The selector control runs with or without an endpoint: it is the
        # shortlist draw with a uniform pick, and it is what `bo_llm_select`
        # has to beat.
        arms.append(
            ArmConfig(
                "bo_random_select",
                strategy="bo",
                acquisition="ei",
                selector=RandomShortlistSelector(),
                shortlist_k=spec.shortlist_k,
                **common,
            )
        )
        if backend is not None:
            from .llm import LLMConstraintProposer

            arms.append(
                ArmConfig(
                    "bo_llm_constraint",
                    strategy="bo",
                    acquisition="ei",
                    constraint_source=LLMConstraintProposer(
                        backend=backend,
                        cache_dir=f"{cache_root}/llm",
                        offline=offline,
                        units=spec.units,
                        menu_kwargs=(
                            {} if spec.menu_mine_sample is None
                            else {"mine_sample": spec.menu_mine_sample}
                        ),
                    ),
                    min_candidates=spec.constraint_min_candidates,
                    **common,
                )
            )
            arms.append(
                ArmConfig(
                    "bo_llm_select",
                    strategy="bo",
                    acquisition="ei",
                    selector=LLMSelector(
                        backend=backend,
                        cache_dir=f"{cache_root}/llm_select",
                        offline=offline,
                        units=spec.units,
                    ),
                    shortlist_k=spec.shortlist_k,
                    **common,
                )
            )

    if "agentic" in want:
        arms.append(
            ArmConfig(
                "bo_policy_scripted",
                strategy="bo",
                acquisition="ei",
                policy_source=ScriptedPolicy(),
                **common,
            )
        )
        arms.append(
            ArmConfig(
                "bo_policy_random",
                strategy="bo",
                acquisition="ei",
                policy_source=RandomPolicy(),
                **common,
            )
        )
        arms.append(
            ArmConfig(
                "bo_policy_roundrobin",
                strategy="bo",
                acquisition="ei",
                policy_source=RoundRobinPolicy(),
                **common,
            )
        )
        if backend is not None:
            from .llm import ResponseCache

            arms.append(
                ArmConfig(
                    "bo_policy_llm",
                    strategy="bo",
                    acquisition="ei",
                    policy_source=LLMPolicy(
                        backend=backend,
                        cache=ResponseCache(f"{cache_root}/llm_policy"),
                        every=spec.policy_every,
                    ),
                    **common,
                )
            )

    if values is not None:
        arms.append(
            ArmConfig(
                "bo_policy_oracle",
                strategy="bo",
                acquisition="ei",
                policy_source=OraclePolicy(values=np.asarray(values), maximize=maximize),
                **common,
            )
        )

    return arms


def skipped_arms(backend_available: bool, include: Sequence[str]) -> list[str]:
    """Which model arms a run will not contain, for the results file.

    A reader must be able to tell "this arm lost" from "this arm never ran",
    and a gap in a table does not distinguish them.
    """
    if backend_available:
        return []
    want = set(include)
    return sorted(name for name in NEEDS_MODEL if CATEGORY[name] in want)
