"""Agentic BO: let a controller choose the search strategy, and control for it.

Standard BO fixes its policy before the run -- surrogate, acquisition function,
bounds, objective split -- and adapts only *where it samples*, never *how it
decides*. SARA (arXiv 2608.00316, github.com/uta-smile/AgenticBo) relaxes that
by putting an LLM in charge of the optimiser itself.

Two of SARA's headline actions map onto arms this project already ran:

===========================  ================================  ================
SARA action                  this repo                         result
===========================  ================================  ================
modify search bounds         ``bo_llm_constraint``             -0.030, p=0.006
select evaluations           ``bo_llm_select``                 neutral
switch acquisition function  *(new here)*                      untested
inspect surrogate state      *(new here)*                      untested
===========================  ================================  ================

So the useful experiment is not "run SARA on EGFR" -- it is the bottom two rows.
This package implements exactly those, with the same control discipline the rest
of the project uses: a null, two cheap controls, a hand-written policy reading
identical inputs, and a hindsight ceiling.

``molbo.campaign.ArmConfig.policy_source`` is the hook. See
:mod:`molbo.agentic.policies` for the ladder and
``scripts/acquisition_headroom.py`` for the measurement that should run first.

Imports here are kept lazy: :mod:`molbo.campaign` imports
:mod:`~molbo.agentic.diagnostics`, and pulling :mod:`molbo.llm` into every
campaign import to get it would be a needless coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .diagnostics import (
    ACQUISITIONS,
    Diagnostics,
    PolicyDecision,
    build_diagnostics,
)

if TYPE_CHECKING:  # pragma: no cover
    from .policies import (
        FixedPolicy,
        LLMPolicy,
        OraclePolicy,
        RandomPolicy,
        RoundRobinPolicy,
        ScriptedPolicy,
    )
    from .sara import SaraBackend

_LAZY = {
    "FixedPolicy": ".policies",
    "RandomPolicy": ".policies",
    "RoundRobinPolicy": ".policies",
    "ScriptedPolicy": ".policies",
    "OraclePolicy": ".policies",
    "LLMPolicy": ".policies",
    "build_policy_prompt": ".policies",
    "POLICY_SYSTEM": ".policies",
    "POLICY_SCHEMA": ".policies",
    "SaraBackend": ".sara",
}

__all__ = [
    "ACQUISITIONS",
    "Diagnostics",
    "PolicyDecision",
    "build_diagnostics",
    *sorted(_LAZY),
]


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(_LAZY[name], __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
