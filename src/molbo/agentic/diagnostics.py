"""What a controller is allowed to see, and what it is allowed to say back.

The three LLM arms already in this project all ask the model a *chemistry*
question: narrow the library, pick from a shortlist, choose an opening batch.
All three lost to their non-LLM controls, and the recorded failure mode was
chemical -- the model misidentified the series' chemotype in every prompt
format tried.

SARA (Surrogate-Augmented Autoresearch Agent, arXiv 2608.00316) bets on a
different skill: the model as *optimisation strategist*, inspecting the
surrogate's state and adjusting search strategy. That is untested here, and
nothing in the existing results speaks to it, because the model was never shown
any surrogate state. It saw observed compounds and a substructure menu; it never
saw "the posterior has collapsed" or "you have explored 3% of the library".

:class:`Diagnostics` is that missing channel. It is assembled inside the
campaign loop from state the loop already has, so it costs nothing to produce
and cannot leak anything the arm is not entitled to.

**What is deliberately absent.** No unobserved library value appears here, in
any form -- not a true optimum, not a rank, not a count of how many good
compounds remain. A controller sees the surrogate's *beliefs* about the
candidates and the campaign's own bookkeeping, which is exactly what a real
campaign would know at that moment. The one exception is
:class:`~molbo.agentic.policies.OraclePolicy`, which takes the true values by
constructor argument and is labelled a ceiling rather than a method, following
the precedent of :func:`molbo.constraints.oracle_constraint`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

ACQUISITIONS: tuple[str, ...] = ("ei", "ucb", "pi", "greedy", "ts")
"""The strategies a controller may choose between.

These are the kinds :func:`molbo.acquisition.acquisition_scores` already
implements, spelled exactly as that function spells them -- ``ts``, not
``thompson``. The names are repeated here rather than imported because this
module is deliberately numpy-only, while :mod:`molbo.acquisition` reaches scipy
and rdkit through the surrogate. ``test_acquisition_names_match_the_scorer``
pins the two tuples equal so the duplication cannot drift again; it drifted
once, and every policy that chose Thompson sampling raised instead of running.

A controller cannot invent a name: an unrecognised one is rejected at the
boundary rather than passed through, because a typo that silently fell back to
``ei`` would make an arm quietly identical to plain BO and the comparison
meaningless.
"""


@dataclass(frozen=True)
class Diagnostics:
    """A read-only snapshot of the search, as of the start of one round.

    Everything here is derived from what the campaign has bought plus the GP's
    posterior over the current candidate pool. Nothing is derived from values
    the campaign has not paid for.
    """

    round_index: int
    calls_spent: int
    budget: int
    n_observed: int
    n_candidates: int
    library_size: int

    best_observed: float
    """Best value bought so far, in the campaign's maximise-sign convention."""

    rounds_since_improvement: int
    """How long the incumbent has stood. The cheapest signal that a strategy has
    stopped paying, and the one a scripted policy keys on."""

    mu_max: float
    mu_mean: float
    sd_max: float
    sd_mean: float
    """Posterior summary over the candidate pool. ``sd_mean`` collapsing towards
    zero means the surrogate believes it has seen everything worth seeing --
    either it is right, or it is overconfident and exploration is overdue."""

    gp_eta: float
    """Fitted noise-to-signal ratio. Large early is normal; large *late* means
    the representation cannot separate compounds the assay separates. This is
    the number that exposed the stereochemistry bug documented in the README."""

    gp_lml: float
    """Log marginal likelihood of the current fit."""

    acquisition: str
    """What is in use this round, before the controller is asked."""

    @property
    def fraction_explored(self) -> float:
        return self.n_observed / max(1, self.library_size)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.budget - self.calls_spent)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fraction_explored"] = round(self.fraction_explored, 5)
        d["budget_remaining"] = self.budget_remaining
        return d

    def summary(self) -> str:
        """A compact human/LLM-readable rendering.

        Kept deliberately terse and fully numeric. A controller that needs prose
        to understand the state of a GP is not reading the GP.
        """
        return "\n".join(
            [
                f"round {self.round_index}  "
                f"calls {self.calls_spent}/{self.budget}  "
                f"observed {self.n_observed}  candidates {self.n_candidates}",
                f"library explored      {self.fraction_explored:.2%}",
                f"best observed         {self.best_observed:.4f}",
                f"rounds since improve  {self.rounds_since_improvement}",
                f"posterior mean        max {self.mu_max:.4f}  mean {self.mu_mean:.4f}",
                f"posterior sd          max {self.sd_max:.4f}  mean {self.sd_mean:.4f}",
                f"GP noise ratio (eta)  {self.gp_eta:.4g}",
                f"GP log marg lik       {self.gp_lml:.4g}",
                f"acquisition in use    {self.acquisition}",
            ]
        )


@dataclass(frozen=True)
class PolicyDecision:
    """What a controller may change. Nothing here can spend budget.

    The action space is deliberately narrow. A controller that could also pick
    the molecule would be :class:`~molbo.llm.LLMSelector` wearing a different
    hat, and that arm was already run and came back neutral. Restricting this
    hook to *how to score* keeps the new arm a clean test of one idea:
    **can a model choose a search strategy better than a fixed one?**
    """

    acquisition: str = "ei"
    xi: float = 0.01
    """Exploration margin for ``ei`` and ``pi``."""

    beta: float = 2.0
    """Confidence width for ``ucb``."""

    rationale: str = ""
    """Free text, recorded in the trace and never acted on.

    Worth keeping: on the warm-start arm the recorded rationales are what
    revealed the model was doing MaxMin diversity in prose, which no accuracy
    number would have shown.
    """

    def validated(self) -> "PolicyDecision":
        """Reject anything the acquisition layer would not understand.

        Raises rather than clamping. A controller asking for an acquisition that
        does not exist is a bug in the controller, and silently substituting
        ``ei`` would turn that bug into an arm that looks like plain BO.
        """
        if self.acquisition not in ACQUISITIONS:
            raise ValueError(
                f"unknown acquisition {self.acquisition!r}; expected one of {ACQUISITIONS}"
            )
        if not np.isfinite(self.xi) or not np.isfinite(self.beta):
            raise ValueError(f"non-finite policy parameters: xi={self.xi}, beta={self.beta}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Incumbent:
    """Tracks how long the best-so-far has stood. Internal to the loop."""

    best: float = -np.inf
    last_improved_round: int = 0

    def update(self, best_now: float, round_index: int) -> None:
        if best_now > self.best:
            self.best = best_now
            self.last_improved_round = round_index

    def stale_for(self, round_index: int) -> int:
        return max(0, round_index - self.last_improved_round)


def build_diagnostics(
    *,
    round_index: int,
    calls_spent: int,
    budget: int,
    n_observed: int,
    library_size: int,
    mu: np.ndarray,
    sd: np.ndarray,
    best_observed: float,
    gp_eta: float,
    gp_lml: float,
    acquisition: str,
    rounds_since_improvement: int,
) -> Diagnostics:
    """Assemble a snapshot from state the campaign loop already holds.

    ``mu``/``sd`` are the posterior over the *current candidate pool*, which is
    what a controller should reason about -- not the whole library, which may be
    masked by a constraint this round.
    """
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    empty = mu.size == 0
    return Diagnostics(
        round_index=int(round_index),
        calls_spent=int(calls_spent),
        budget=int(budget),
        n_observed=int(n_observed),
        n_candidates=int(mu.size),
        library_size=int(library_size),
        best_observed=float(best_observed),
        rounds_since_improvement=int(rounds_since_improvement),
        mu_max=float(np.max(mu)) if not empty else 0.0,
        mu_mean=float(np.mean(mu)) if not empty else 0.0,
        sd_max=float(np.max(sd)) if not empty else 0.0,
        sd_mean=float(np.mean(sd)) if not empty else 0.0,
        gp_eta=float(gp_eta),
        gp_lml=float(gp_lml),
        acquisition=str(acquisition),
    )
