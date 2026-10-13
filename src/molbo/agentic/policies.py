"""Controllers: things that choose the acquisition strategy each round.

The arm under test is :class:`LLMPolicy`. Everything else in this module exists
so that a win by :class:`LLMPolicy` would mean something -- the same discipline
the rest of this project applies to its LLM arms, carried over to agentic BO.

The ladder, weakest claim to strongest:

``FixedPolicy``
    Changes nothing. Reduces to plain BO *exactly*, and a test pins that. It is
    the ``shortlist_k=1`` of this hook: proof that adding the machinery does not
    itself move the number.
``RandomPolicy``
    Switches acquisition at random each round. Settles "does any switching
    help?" -- the question :class:`~molbo.constraints.RandomSubsetConstraint`
    settles for the constraint arm, where the answer turned out to be *no, it
    significantly hurts*.
``RoundRobinPolicy``
    Cycles deterministically. Separates "switching helps" from "switching
    *unpredictably* helps".
``ScriptedPolicy``
    A hand-written policy reading the same diagnostics the model reads. **This
    is the control that matters.** It is the ``bo_diverse_warmstart`` of this
    study: the free thing that does what the agent claims to do. The warm-start
    result is the precedent -- the model had the right idea, said so explicitly
    in its rationales, and arithmetic did it better.
``OraclePolicy``
    A ceiling, not a method. Bounds what any controller could achieve.
``LLMPolicy``
    The arm under test.

Run the ceiling first. If ``OraclePolicy`` cannot beat ``FixedPolicy`` by more
than the noise between seeds, then no controller can, and the honest conclusion
is that this library is the wrong testbed for agentic BO -- not that the agent
failed. ``scripts/acquisition_headroom.py`` is that measurement, and it needs no
model, no endpoint and no GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..acquisition import acquisition_scores
from ..llm import Backend, LLMUnavailable, ResponseCache, request_key
from .diagnostics import ACQUISITIONS, Diagnostics, PolicyDecision

POLICY_VERSION = "v1"
"""Bumped when the prompt or schema changes, so a cache built under the old
wording cannot be mistaken for the new one. Mirrors ``llm.request_key``'s
``version`` argument, which exists for the same reason."""


# --------------------------------------------------------------------------
# non-LLM policies
# --------------------------------------------------------------------------


@dataclass
class FixedPolicy:
    """Never changes anything. The null arm for this hook.

    An arm carrying this policy must produce a run byte-identical to the same
    arm with ``policy_source=None``, because the decision it returns is the
    arm's own configuration. ``tests/test_agentic.py`` pins that: if the hook
    perturbs the RNG stream or the scoring path, every downstream comparison is
    contaminated and this test is the thing that catches it.
    """

    acquisition: str = "ei"
    xi: float = 0.01
    beta: float = 2.0

    def __call__(self, **_: object) -> PolicyDecision:
        return PolicyDecision(
            acquisition=self.acquisition,
            xi=self.xi,
            beta=self.beta,
            rationale="fixed",
        ).validated()


@dataclass
class RandomPolicy:
    """Uniform over the available acquisitions each round.

    Control for "any adaptivity helps". Draws from the campaign's own ``rng`` so
    the choice is reproducible from the seed, and so this arm consumes RNG in
    the same way an LLM arm does not -- which is why it is a control for
    *switching*, not a drop-in placebo for the LLM call.
    """

    choices: Sequence[str] = ACQUISITIONS

    def __call__(self, *, rng: np.random.Generator, **_: object) -> PolicyDecision:
        kind = str(rng.choice(list(self.choices)))
        return PolicyDecision(acquisition=kind, rationale="random").validated()


@dataclass
class RoundRobinPolicy:
    """Cycles through the acquisitions in a fixed order.

    Deterministic, so it isolates the effect of *varying* the acquisition from
    the effect of varying it unpredictably.
    """

    choices: Sequence[str] = ACQUISITIONS

    def __call__(self, *, diagnostics: Diagnostics, **_: object) -> PolicyDecision:
        kind = self.choices[diagnostics.round_index % len(self.choices)]
        return PolicyDecision(acquisition=kind, rationale="round-robin").validated()


@dataclass
class ScriptedPolicy:
    """Hand-written rules over the same diagnostics the model is shown.

    The rules are deliberately the obvious ones a practitioner would write on a
    napkin, in priority order:

    1. **Endgame** -- with little budget left there is no value in learning, so
       spend the remainder on the best predictions (``greedy``).
    2. **Stalled** -- the incumbent has not moved for ``stale_rounds``, so widen
       the search (``ucb`` with a larger ``beta``).
    3. **Collapsed posterior** -- mean candidate uncertainty has fallen below
       ``sd_floor`` relative to its first-round value, meaning the surrogate
       thinks it knows everything. Sample from the posterior instead
       (``ts``, Thompson sampling) rather than trusting a confident model.
    4. **Otherwise** -- ``ei``.

    Being obvious is the point. If an LLM cannot beat four napkin rules reading
    the identical inputs, the agency is not what is helping.
    """

    stale_rounds: int = 3
    endgame_fraction: float = 0.1
    """Fraction of total budget below which rule 1 fires."""

    sd_floor: float = 0.25
    """Trigger for rule 3, as a fraction of the first observed ``sd_mean``."""

    explore_beta: float = 3.0
    _sd_reference: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Forget the first-round reference. Called at the start of each run.

        Without this a policy instance reused across seeds would carry one run's
        scale into the next, which would make the arm depend on seed *order* --
        a silent violation of the paired design.
        """
        self._sd_reference = None

    def __call__(self, *, diagnostics: Diagnostics, **_: object) -> PolicyDecision:
        d = diagnostics
        if self._sd_reference is None and d.sd_mean > 0:
            self._sd_reference = d.sd_mean

        if d.budget_remaining <= max(1, int(self.endgame_fraction * d.budget)):
            return PolicyDecision(acquisition="greedy", rationale="endgame").validated()

        if d.rounds_since_improvement >= self.stale_rounds:
            return PolicyDecision(
                acquisition="ucb", beta=self.explore_beta, rationale="stalled"
            ).validated()

        if (
            self._sd_reference
            and self._sd_reference > 0
            and d.sd_mean < self.sd_floor * self._sd_reference
        ):
            return PolicyDecision(
                acquisition="ts", rationale="posterior collapsed"
            ).validated()

        return PolicyDecision(acquisition="ei", rationale="default").validated()


@dataclass
class OraclePolicy:
    """Ceiling. Picks the acquisition whose argmax has the best true value.

    **Not a method.** It reads ``values``, which no arm may do, and it is
    labelled here exactly as :func:`molbo.constraints.oracle_constraint` is
    labelled in the constraint study: a reference line saying how much room
    exists, so that a small effect can be read against the size of the prize.

    It is also a *loose* upper bound, because it optimises one round at a time.
    A controller with foresight could in principle beat it by taking a worse
    pick now for a better posterior later. That only strengthens its use as a
    screening tool: if even this greedy hindsight ceiling gains nothing, the
    action space is inert on this problem.
    """

    values: np.ndarray
    maximize: bool = True
    choices: Sequence[str] = ACQUISITIONS

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)

    def __call__(
        self,
        *,
        diagnostics: Diagnostics,
        mu: np.ndarray,
        sd: np.ndarray,
        cand: np.ndarray,
        best: float,
        rng: np.random.Generator,
        **_: object,
    ) -> PolicyDecision:
        cand = np.asarray(cand)
        if cand.size == 0:
            return PolicyDecision(rationale="no candidates").validated()

        sign = 1.0 if self.maximize else -1.0
        truth = sign * self.values[cand]

        best_kind, best_truth = "ei", -np.inf
        for kind in self.choices:
            # A fresh generator per kind: `ts` consumes randomness, and
            # scoring one kind must not shift what another kind would have seen.
            scores = acquisition_scores(
                kind,
                mu,
                sd,
                best=best,
                rng=np.random.default_rng(int(diagnostics.round_index) + 1),
                xi=0.01,
                beta=2.0,
            )
            reached = float(truth[int(np.argmax(scores))])
            if reached > best_truth:
                best_kind, best_truth = kind, reached

        return PolicyDecision(
            acquisition=best_kind,
            rationale=f"hindsight best (reaches {best_truth:.4f})",
        ).validated()


# --------------------------------------------------------------------------
# the arm under test
# --------------------------------------------------------------------------

POLICY_SYSTEM = """\
You are steering a Bayesian optimisation campaign over a fixed library of
molecules. You do not choose molecules. You choose how the next candidate is
scored, and nothing else.

Each round you are shown the state of the Gaussian Process surrogate and the
campaign's budget. Pick the acquisition function for this round.

Available:
  ei        expected improvement -- balanced; xi widens it toward exploration
  ucb       upper confidence bound -- explicit exploration; beta sets the width
  pi        probability of improvement -- conservative, exploits hard
  greedy    posterior mean only -- pure exploitation, learns nothing
  ts        Thompson sampling -- one draw from the posterior; exploration
            that respects correlations

Guidance:
- A rising `rounds since improve` means the current strategy has stopped paying.
- `posterior sd` collapsing towards zero means the surrogate believes it has
  seen everything worth seeing. It may be wrong, and it cannot tell you so.
- A large `GP noise ratio` late in a run means the representation cannot
  separate compounds the assay separates. No acquisition function repairs that.
- Exploration only pays if there is budget left to exploit what it finds.

Respond with JSON only.
"""

POLICY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "acquisition": {"type": "string", "enum": list(ACQUISITIONS)},
        "xi": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "beta": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "rationale": {"type": "string"},
    },
    "required": ["acquisition", "xi", "beta", "rationale"],
    "additionalProperties": False,
}


def build_policy_prompt(diagnostics: Diagnostics) -> str:
    """Render the state the controller reasons over.

    Numbers only. The model is not shown a single SMILES string here, which is
    deliberate: the three existing arms all asked it a chemistry question and it
    answered them by misnaming the chemotype. This arm asks whether it can read
    a surrogate, and mixing the two would leave the result uninterpretable.
    """
    return (
        f"{diagnostics.summary()}\n\n"
        "Choose the acquisition function for the next round."
    )


@dataclass
class LLMPolicy:
    """The arm under test: a model choosing the acquisition each round.

    This is the SARA idea (arXiv 2608.00316) reduced to the one action that is
    genuinely untested in this project. Two of SARA's headline capabilities map
    onto arms that were already run here and already failed -- "modify search
    bounds" is ``bo_llm_constraint`` (-0.030, p=0.006) and "select evaluations"
    is ``bo_llm_select`` (neutral, and diagnosed as positional bias rather than
    chemistry). Strategy selection is what remains.

    Every response is cached on a hash of the exact request including the
    backend's sampling settings, exactly as :class:`molbo.llm.LLMConstraintProposer`
    does, so a rerun costs nothing and a change of model cannot silently reuse
    another model's answers.

    On any failure -- unreachable endpoint, malformed JSON, an acquisition name
    outside the enum -- the policy falls back to ``fallback`` and records why.
    A failed model call must not crash a 20-seed sweep, and it must not be
    invisible either: the reason lands in the trace and
    ``scripts/run_agentic_experiment.py`` prints the failure count.
    """

    backend: Backend
    cache: ResponseCache | None = None
    fallback: str = "ei"
    every: int = 1
    """Rounds between model calls. ``1`` asks every round; higher reuses the last
    decision, which is the cost dial. Note this arm's cache hit rate is far
    lower than the single-shot arms': those prompts are a function of
    ``(seed, round, observations)``, while these vary with the GP fit."""

    _last: PolicyDecision | None = field(default=None, init=False, repr=False)
    log: list[dict] = field(default_factory=list, init=False, repr=False)

    def reset(self) -> None:
        self._last = None
        self.log = []

    def request_key_for(self, diagnostics: Diagnostics) -> str:
        return request_key(
            self.backend.key(),
            POLICY_SYSTEM,
            build_policy_prompt(diagnostics),
            POLICY_SCHEMA,
            POLICY_VERSION,
        )

    def __call__(self, *, diagnostics: Diagnostics, **_: object) -> PolicyDecision:
        if self._last is not None and diagnostics.round_index % max(1, self.every):
            return self._last

        prompt = build_policy_prompt(diagnostics)
        key = self.request_key_for(diagnostics)

        payload = self.cache.get(key) if self.cache is not None else None
        cached = payload is not None
        if payload is None:
            try:
                payload, meta = self.backend.complete(POLICY_SYSTEM, prompt, POLICY_SCHEMA)
            except (LLMUnavailable, json.JSONDecodeError, KeyError, ValueError) as exc:
                return self._fail(diagnostics, f"{type(exc).__name__}: {exc}")
            if self.cache is not None:
                self.cache.put(
                    key,
                    {
                        "backend": self.backend.key(),
                        "version": POLICY_VERSION,
                        "system": POLICY_SYSTEM,
                        "prompt": prompt,
                        "response": payload,
                        "usage": meta.get("usage", {}),
                        "diagnostics": diagnostics.to_dict(),
                    },
                )
        else:
            payload = payload.get("response", payload)

        try:
            decision = PolicyDecision(
                acquisition=str(payload["acquisition"]),
                xi=float(payload.get("xi", 0.01)),
                beta=float(payload.get("beta", 2.0)),
                rationale=str(payload.get("rationale", ""))[:500],
            ).validated()
        except (KeyError, TypeError, ValueError) as exc:
            return self._fail(diagnostics, f"unusable response: {exc}")

        self._last = decision
        self.log.append(
            {
                "round": diagnostics.round_index,
                "cached": cached,
                "decision": decision.to_dict(),
            }
        )
        return decision

    def _fail(self, diagnostics: Diagnostics, why: str) -> PolicyDecision:
        decision = PolicyDecision(
            acquisition=self.fallback, rationale=f"fallback: {why}"
        ).validated()
        self._last = decision
        self.log.append(
            {
                "round": diagnostics.round_index,
                "cached": False,
                "failed": why,
                "decision": decision.to_dict(),
            }
        )
        return decision
