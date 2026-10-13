"""One optimisation run: an arm, a seed, a budget of unique oracle calls.

The unit of the whole project.  A *campaign* proposes molecules from an
enumerated library, spends budget through an :class:`~molbo.oracles.base.Oracle`
(which does the memoisation and unique-call accounting), and returns a trace
that :mod:`molbo.metrics` turns into an AUC.

Three decisions here exist to protect the statistics rather than the chemistry:

**The initial design is arm-independent.**  Seed *s* draws the same starting
molecules for every arm, so arms can be compared with a paired test.  Unpaired,
the seed-to-seed spread of BO on a congeneric series swamps the between-arm
difference and 20 seeds buys almost nothing; paired, the same 20 seeds resolve
differences several times smaller.

**Constraints are proposed from observed data only.**  A constraint source
sees the molecules the campaign has already bought and their values -- never
the library's unobserved values.  The one exception is the deliberate ceiling
arm, which is labelled as such and is not a method.

**Nothing may re-propose a molecule.**  The oracle would not charge for it
(that is the memoisation rule), so a method that re-proposes would silently get
a longer budget than its rivals.  Candidates already evaluated are removed from
the pool outright.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np

from .acquisition import acquisition_scores, rank_candidates, select_batch
from .agentic.diagnostics import _Incumbent, build_diagnostics
from .constraints import Constraint, ConstraintOutcome, LibraryContext, apply_constraints
from .oracles.base import Oracle
from .surrogate import (
    LibraryKernel,
    PackedFingerprints,
    TanimotoGP,
    fingerprint_matrix,
)

Selector = Callable[..., object]
"""``(shortlist, scores, mu, sd, observations, ctx, rng, round_index, seed, q)``
-> the chosen position(s) within ``shortlist``.

May return an int, a sequence of ints, or ``None`` to defer to BO's own
ranking. Positions, not SMILES: a selector that named a molecule could name one
that is not on the shortlist, and silently buying it would break the budget
accounting the whole comparison rests on.
"""

ConstraintSource = Callable[..., Constraint | None]
"""``(observations, ctx, rng, round_index, seed) -> Constraint | None``.

``observations`` is a list of ``(smiles, value)`` the campaign has bought so
far. Implementations must not look at anything else about the library's values.
"""

PolicySource = Callable[..., object]
"""``(diagnostics, mu, sd, cand, best, observations, ctx, rng, round_index, seed)``
-> a :class:`~molbo.agentic.diagnostics.PolicyDecision`.

Chooses *how the next candidate is scored* -- the acquisition function and its
hyperparameters -- once per round, before scoring. It cannot choose a molecule:
that is what ``selector`` does, and keeping the two hooks apart is what makes
each one a clean test of a single idea.

``diagnostics`` is the surrogate's state, assembled by
:func:`molbo.agentic.diagnostics.build_diagnostics`. Like a constraint source, a
policy must not look at the library's unobserved values; the deliberate ceiling
:class:`~molbo.agentic.policies.OraclePolicy` is the labelled exception.
"""


@dataclass
class ArmConfig:
    """A method, fully specified. Two arms differing in one field is an ablation."""

    name: str
    strategy: str = "bo"
    """``bo`` (GP + acquisition), ``random`` (uniform over the allowed pool), or
    ``llm`` (a selector picks from a uniform random shortlist, with no surrogate
    at all).

    ``llm`` exists to separate two claims that the ``bo_llm_*`` arms conflate.
    Those arms ask whether a model *improves* BO; this one asks whether the
    model can choose molecules **without** BO underneath it. Its control is
    ``random``, which is the identical draw with a uniform pick instead of the
    model's, so the difference between the two is the model's chemistry and
    nothing else. Without this arm a reader cannot tell whether a BO+LLM number
    is the LLM contributing or the GP carrying it."""

    acquisition: str = "ei"
    n_init: int = 10
    """Random molecules bought before the surrogate is first fitted."""

    batch_size: int = 1
    diversity: float = 0.0
    xi: float = 0.01
    beta: float = 2.0

    init_source: Callable | None = field(default=None, repr=False)
    """Chooses the initial design instead of drawing it at random.

    This is the one arm that deliberately breaks the arm-independent initial
    design. That is the intervention, not a bug -- but it means this arm is no
    longer paired with the others through a shared start, so its comparison
    rests on the seed alone."""

    selector: "Selector | None" = field(default=None, repr=False)
    """Chooses which of BO's shortlisted candidates actually gets evaluated.

    ``None`` keeps the usual behaviour (take the acquisition argmax). Set it to
    put a decision step between BO's ranking and the oracle."""

    candidate_subset: int | None = None
    """Score only this many randomly-chosen unevaluated candidates per round.

    A pool of a few thousand is scored in full. PMO's ZINC-250k pool is not:
    an exact GP posterior over 250,000 candidates every round is neither
    affordable nor necessary, since the acquisition argmax over a large random
    subset is nearly as good. The *search space* stays the whole pool -- only
    the per-round shortlist is sampled -- so this does not narrow what the
    method may find, and it is recorded in the arm config."""

    prior_mean: "np.ndarray | None" = field(default=None, repr=False)
    """A per-library prior on the objective, used as the GP's mean function.

    ``None`` is the standard zero-mean GP. Given an array of length
    ``len(library)``, the GP is fitted to ``y - prior`` and the prior is added
    back to the posterior mean, so it steers only where no data has arrived
    yet and is overwritten where data has. This is the *soft* way to use a
    learned prior; the hard alternative -- masking the library down to what
    the prior likes -- was measured at -0.0295 AUC against plain BO across 24
    targets, losing even where the prior was correct."""

    shortlist_k: int = 20
    """How many acquisition-ranked candidates the selector sees. ``1`` reduces
    to plain BO exactly, so this is the knob that says how much of the decision
    the selector is actually making."""

    policy_source: "PolicySource | None" = field(default=None, repr=False)
    """Chooses the acquisition function and its hyperparameters each round.

    ``None`` keeps ``acquisition``/``xi``/``beta`` fixed for the whole run, which
    is what "standard BO" means -- the policy is decided before the run and never
    adapts. Set it to test whether adapting the policy mid-run helps.

    :class:`~molbo.agentic.policies.FixedPolicy` returns the arm's own settings,
    so it must produce a run identical to ``None``. That equivalence is pinned by
    a test: if the hook perturbs the RNG stream, every paired comparison
    downstream is contaminated."""

    constraint_source: ConstraintSource | None = field(default=None, repr=False)
    constraint_every: int = 0
    """Rounds between constraint refreshes. ``0`` proposes once, after the init
    design, and keeps it; a positive value re-proposes as the SAR picture fills
    in, which is the interesting setting but costs an LLM call per refresh."""

    min_candidates: int = 20
    soft_penalty: float = 1.0
    refit_every: int = 1
    """Rounds between GP refits. ``1`` is honest; higher is a speed knob that
    makes the surrogate stale, so it is reported in the trace."""

    def describe(self) -> dict:
        d = asdict(self)
        d.pop("constraint_source", None)
        d.pop("selector", None)
        d.pop("init_source", None)
        d.pop("prior_mean", None)
        d.pop("policy_source", None)
        d["has_init_source"] = self.init_source is not None
        d["has_prior_mean"] = self.prior_mean is not None
        d["has_constraint_source"] = self.constraint_source is not None
        d["has_selector"] = self.selector is not None
        d["has_policy_source"] = self.policy_source is not None
        return d


@dataclass
class CampaignResult:
    """The trace of one run. Everything needed to recompute any metric offline."""

    arm: str
    seed: int
    budget: int
    smiles: list[str]
    values: list[float | None]
    n_calls: int
    wall_seconds: float
    constraint_log: list[dict]
    arm_config: dict
    oracle: dict
    n_failed: int = 0
    selector_log: list[dict] = field(default_factory=list)
    """Which shortlist rank the selector picked, per round. Empty when the arm
    has no selector."""

    policy_log: list[dict] = field(default_factory=list)
    """Which acquisition the policy chose, per round, with the diagnostics it
    saw and the rationale it gave. Empty when the arm has no policy source.

    The rationale is recorded and never acted on, because on the warm-start arm
    it was the rationales -- not any accuracy number -- that revealed the model
    was doing MaxMin diversity in prose."""

    @property
    def scored(self) -> list[float]:
        """Values in call order, with unscorable proposals dropped."""
        return [v for v in self.values if v is not None]

    def best(self, maximize: bool = True) -> float | None:
        s = self.scored
        if not s:
            return None
        return max(s) if maximize else min(s)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["best"] = self.best()
        return d


def run_campaign(
    oracle: Oracle,
    arm: ArmConfig,
    seed: int,
    *,
    budget: int,
    library: Sequence[str] | None = None,
    ctx: LibraryContext | None = None,
    X: np.ndarray | None = None,
    kernel: "LibraryKernel | None" = None,
    progress: Callable[[int, int], None] | None = None,
) -> CampaignResult:
    """Run ``arm`` on ``oracle`` for ``budget`` unique calls.

    ``ctx``, ``X`` and ``kernel`` (the library's descriptor context,
    fingerprint matrix and precomputed Tanimoto Gram) are accepted precomputed
    because a sweep reuses one library across every arm and seed; building them
    per run costs more than the runs themselves. Pass ``kernel=False`` to force
    the on-the-fly path.
    """
    if arm.strategy not in ("bo", "random", "llm"):
        # Checked up front, not at first use: when n_init >= budget the
        # proposal loop never runs, and a typo'd strategy would quietly return
        # a random-search result under the wrong arm's name.
        raise ValueError(
            f"unknown strategy {arm.strategy!r}; expected 'bo', 'random' or 'llm'"
        )
    if arm.strategy == "llm" and arm.selector is None:
        # An `llm` arm with no selector is just `random` wearing a better name,
        # and it would be reported as an LLM result. Refuse rather than run it.
        raise ValueError("strategy='llm' requires a selector; got None")

    t0 = time.perf_counter()
    oracle.reset()

    if library is None:
        library = getattr(oracle, "library", None)
        if library is None:
            raise ValueError("run_campaign needs a library (the oracle does not expose one)")
    library = list(library)
    if ctx is None:
        ctx = LibraryContext.build(library)
    if X is None:
        X = fingerprint_matrix(library)
    if kernel is None:
        # A packed pool is exactly the case the Gram cache cannot serve: it is
        # packed precisely because n is large, and n^2 would be worse still.
        kernel = None if isinstance(X, PackedFingerprints) else LibraryKernel.maybe_build(X)
    elif kernel is False:
        kernel = None

    n = len(library)
    if budget > n:
        raise ValueError(
            f"budget {budget} exceeds the library ({n} compounds); "
            "the run would exhaust the search space rather than measure search"
        )

    rng = np.random.default_rng(seed)
    evaluated = np.zeros(n, dtype=bool)
    obs_idx: list[int] = []
    obs_val: list[float] = []
    order_smiles: list[str] = []
    order_values: list[float | None] = []
    constraint_log: list[dict] = []
    selector_log: list[dict] = []
    policy_log: list[dict] = []
    # A stateful policy (ScriptedPolicy caches the first round's posterior scale;
    # LLMPolicy caches its last decision) must not carry one seed's state into
    # the next, or the arm would depend on the order seeds were run in and the
    # paired design would be quietly broken.
    if arm.policy_source is not None and hasattr(arm.policy_source, "reset"):
        arm.policy_source.reset()
    # Same for a stateful selector. A tool-budgeted one (BudgetedDockSelector)
    # holds a per-campaign allowance, so without this the first seed would
    # spend the lot and every later seed would run with nothing -- an arm that
    # silently degrades along the seed axis. Guarded by hasattr, so selectors
    # that deliberately accumulate across seeds (LLMSelector's rank histogram,
    # which is a diagnostic over the whole sweep) are untouched.
    if arm.selector is not None and hasattr(arm.selector, "reset"):
        arm.selector.reset()
    incumbent = _Incumbent()
    n_failed = 0

    sign = 1.0 if oracle.maximize else -1.0

    def buy(indices: Sequence[int]) -> None:
        """Spend budget on library rows, recording in proposal order."""
        nonlocal n_failed
        indices = [int(i) for i in indices if not evaluated[i]]
        if not indices:
            return
        smis = [library[i] for i in indices]
        vals = oracle(smis)
        for i, s, v in zip(indices, smis, vals):
            evaluated[i] = True
            order_smiles.append(s)
            order_values.append(None if v is None else float(v))
            if v is None:
                n_failed += 1
            else:
                obs_idx.append(i)
                obs_val.append(float(v))

    # ---- initial design (identical across arms for a given seed) ----------
    n_init = min(arm.n_init, budget)
    init = None
    if arm.init_source is not None:
        init = arm.init_source(
            library=library, ctx=ctx, rng=np.random.default_rng(seed),
            seed=seed, n_init=n_init,
        )
        if init is not None:
            init = [int(i) for i in init][:n_init]
    if not init:
        init = np.random.default_rng(seed).choice(n, size=n_init, replace=False)
    buy(init)

    constraint: Constraint | None = None
    outcome: ConstraintOutcome | None = None
    gp: TanimotoGP | None = None
    rounds = 0
    prev_mode: str | None = None

    while oracle.n_calls < budget:
        remaining = budget - oracle.n_calls
        q = int(min(arm.batch_size, remaining))

        # ---- constraint refresh ------------------------------------------
        needs_constraint = arm.constraint_source is not None and (
            outcome is None
            or (arm.constraint_every > 0 and rounds % arm.constraint_every == 0)
        )
        if needs_constraint:
            observations = [(library[i], v) for i, v in zip(obs_idx, obs_val)]
            constraint = arm.constraint_source(
                observations=observations, ctx=ctx, rng=rng, round_index=rounds, seed=seed
            )
        if arm.constraint_source is not None or outcome is None:
            outcome = apply_constraints(
                constraint,
                ctx,
                min_candidates=arm.min_candidates,
                soft_penalty=arm.soft_penalty,
                already_seen=evaluated,
            )
            # Log on refresh, and also whenever the mode changes underneath a
            # constraint that was not refreshed. A constraint starts "hard" and
            # degrades to "soft" as its allowed pool is bought out; logging only
            # at refresh time would hide that entirely, and the degradation rate
            # is exactly what needs reporting.
            if needs_constraint or outcome.mode != prev_mode:
                constraint_log.append(
                    {
                        "round": rounds,
                        "call": oracle.n_calls,
                        "mode": outcome.mode,
                        "n_allowed": outcome.n_allowed,
                        "fraction_allowed": round(outcome.fraction_allowed, 5),
                        "event": "proposed" if needs_constraint else "degraded",
                        "constraint": outcome.constraint,
                    }
                )
            prev_mode = outcome.mode

        allowed = outcome.mask & ~evaluated
        if not allowed.any():
            # The constraint plus everything already bought has exhausted the
            # allowed pool. Fall back to the unconstrained remainder rather
            # than ending the run early -- a short run is not comparable.
            allowed = ~evaluated
            constraint_log.append(
                {"round": rounds, "call": oracle.n_calls, "mode": "exhausted_fallback"}
            )
        cand = np.flatnonzero(allowed)
        if arm.candidate_subset is not None and cand.size > arm.candidate_subset:
            cand = np.sort(rng.choice(cand, size=arm.candidate_subset, replace=False))

        # ---- propose ------------------------------------------------------
        if arm.strategy == "random":
            pick = rng.choice(cand, size=min(q, cand.size), replace=False)
        elif arm.strategy == "llm":
            # No GP, no acquisition, no ranking. A uniform shortlist is drawn
            # and the selector picks from it, so this arm's only advantage over
            # `random` is whatever the model reads in the structures.
            #
            # `shortlist_k = 1` collapses to `random` exactly, by the same
            # argument that `shortlist_k = 1` collapses the BO+selector path to
            # plain BO: the draw is the same and there is nothing to choose.
            k = min(arm.shortlist_k, cand.size)
            shortlist = np.sort(rng.choice(cand, size=k, replace=False))
            # There is no posterior to report. NaN rather than a placeholder
            # number: `build_select_prompt` drops the prediction column when the
            # values are not finite, so the model is never shown a fabricated
            # prediction it might treat as information.
            blank = np.full(k, np.nan)
            chosen = arm.selector(
                shortlist=[library[i] for i in shortlist],
                scores=blank,
                mu=blank,
                sd=blank,
                observations=[(library[i], v) for i, v in zip(obs_idx, obs_val)],
                ctx=ctx,
                rng=rng,
                round_index=rounds,
                seed=seed,
                q=q,
            )
            picked_local = _as_indices(chosen, k, q)
            selector_log.append(
                {
                    "round": rounds,
                    "call": oracle.n_calls,
                    "shortlist_k": int(k),
                    "chosen_rank": [int(r) for r in picked_local],
                    "surrogate": False,
                }
            )
            pick = shortlist[picked_local]
        elif arm.strategy == "bo":
            if len(obs_val) < 2:
                pick = rng.choice(cand, size=min(q, cand.size), replace=False)
            else:
                # A learned prior enters here, as the GP's *mean function*
                # rather than as a filter over candidates. The GP is fitted to
                # the residual `y - prior` and the prior is added back to the
                # posterior mean, so the prior decides only where the data is
                # silent and is overwritten wherever it is not. Measured: the
                # same prior applied as a hard mask lost -0.0295 AUC against
                # plain BO and lost even on the targets it got right, while a
                # soft application gained +0.027 over hard. Filtering throws
                # away the ability to be wrong cheaply; a prior mean does not.
                pm = arm.prior_mean
                off_obs = (sign * pm[obs_idx]) if pm is not None else 0.0
                off_cand = (sign * pm[cand]) if pm is not None else 0.0
                y_fit = sign * np.asarray(obs_val) - off_obs
                if gp is None or rounds % max(1, arm.refit_every) == 0:
                    gp = TanimotoGP()
                    if kernel is None:
                        gp.fit(_rows(X, obs_idx), y_fit)
                    else:
                        gp.fit_gram(kernel.train_block(obs_idx), y_fit)
                if kernel is None:
                    mu, sd = gp.predict(_rows(X, cand))
                else:
                    mu, sd = gp.predict_gram(
                        kernel.cross_block(cand, obs_idx), k_diag=kernel.k_diag[cand]
                    )
                mu = mu + off_cand
                best = float(np.max(sign * np.asarray(obs_val)))

                # ---- policy: how to score, not what to pick -----------------
                # Standard BO fixes the acquisition before the run. This hook is
                # the one place that assumption is relaxed. It runs before
                # scoring and cannot touch `cand`, so an arm without a policy
                # source takes a byte-identical path.
                acq, xi, beta = arm.acquisition, arm.xi, arm.beta
                if arm.policy_source is not None:
                    incumbent.update(best, rounds)
                    diagnostics = build_diagnostics(
                        round_index=rounds,
                        calls_spent=oracle.n_calls,
                        budget=budget,
                        n_observed=len(obs_val),
                        library_size=n,
                        mu=mu,
                        sd=sd,
                        best_observed=best,
                        gp_eta=float(getattr(gp, "eta_", float("nan"))),
                        gp_lml=float(getattr(gp, "lml_", float("nan"))),
                        acquisition=arm.acquisition,
                        rounds_since_improvement=incumbent.stale_for(rounds),
                    )
                    decision = arm.policy_source(
                        diagnostics=diagnostics,
                        mu=mu,
                        sd=sd,
                        cand=cand,
                        best=best,
                        observations=[(library[i], v) for i, v in zip(obs_idx, obs_val)],
                        ctx=ctx,
                        rng=rng,
                        round_index=rounds,
                        seed=seed,
                    )
                    acq, xi, beta = decision.acquisition, decision.xi, decision.beta
                    policy_log.append(
                        {
                            "round": rounds,
                            "call": oracle.n_calls,
                            "decision": decision.to_dict(),
                            "diagnostics": diagnostics.to_dict(),
                        }
                    )

                scores = acquisition_scores(acq, mu, sd, best=best, rng=rng, xi=xi, beta=beta)
                scores = scores + outcome.penalty[cand]

                if arm.selector is not None:
                    # BO proposes a shortlist, something else decides which of
                    # them actually costs an oracle call. `shortlist_k` is the
                    # dial between the two: k=1 is exactly plain BO, and
                    # k=len(cand) is pure selection with BO only providing the
                    # posterior. Anything in between splits the credit, which
                    # is why the shortlist size is reported with the result.
                    k = min(arm.shortlist_k, cand.size)
                    # Same ranking (and the same RNG draw) plain BO would use,
                    # so k=1 reproduces it exactly even on a flat surface.
                    top = rank_candidates(scores, rng.random(scores.size) * 1e-12)[:k]
                    shortlist = cand[top]
                    chosen = arm.selector(
                        shortlist=[library[i] for i in shortlist],
                        scores=scores[top],
                        mu=mu[top],
                        sd=sd[top],
                        observations=[(library[i], v) for i, v in zip(obs_idx, obs_val)],
                        ctx=ctx,
                        rng=rng,
                        round_index=rounds,
                        seed=seed,
                        q=q,
                    )
                    picked_local = _as_indices(chosen, k, q)
                    selector_log.append(
                        {
                            "round": rounds,
                            "call": oracle.n_calls,
                            "shortlist_k": int(k),
                            "chosen_rank": [int(r) for r in picked_local],
                        }
                    )
                    pick = shortlist[picked_local]
                else:
                    local = select_batch(
                        scores, _rows(X, cand), min(q, cand.size),
                        rng=rng, diversity=arm.diversity,
                    )
                    pick = cand[local]
        else:
            raise ValueError(f"unknown strategy {arm.strategy!r}")

        before = oracle.n_calls
        buy(pick)
        rounds += 1
        if progress is not None:
            progress(oracle.n_calls, budget)
        if oracle.n_calls == before:
            # Nothing was bought: the pool is empty or every pick was a repeat.
            # Continuing would spin forever.
            break

    return CampaignResult(
        arm=arm.name,
        seed=seed,
        budget=budget,
        smiles=order_smiles,
        values=order_values,
        n_calls=oracle.n_calls,
        wall_seconds=round(time.perf_counter() - t0, 3),
        constraint_log=constraint_log,
        selector_log=selector_log,
        policy_log=policy_log,
        arm_config=arm.describe(),
        oracle=oracle.describe(),
        n_failed=n_failed,
    )


def _as_indices(chosen, k: int, q: int) -> np.ndarray:
    """Normalise whatever a selector returned into valid shortlist positions.

    Out-of-range or duplicate positions are dropped rather than clamped: a
    selector that names rank 40 of a 20-long shortlist has not made a choice,
    and snapping it to rank 19 would credit it with one. Whatever is missing is
    back-filled from the top of the shortlist, so the campaign always spends
    its full batch and stays comparable to every other arm.
    """
    if chosen is None:
        out: list[int] = []
    elif isinstance(chosen, (int, np.integer)):
        out = [int(chosen)]
    else:
        out = [int(c) for c in chosen]

    seen: set[int] = set()
    clean: list[int] = []
    for i in out:
        if 0 <= i < k and i not in seen:
            seen.add(i)
            clean.append(i)
    for i in range(k):
        if len(clean) >= q:
            break
        if i not in seen:
            seen.add(i)
            clean.append(i)
    return np.asarray(clean[:q], dtype=int)


def _rows(X, idx) -> np.ndarray:
    """Fingerprint rows for ``idx``, from a dense matrix or a packed pool."""
    return X.take(idx) if isinstance(X, PackedFingerprints) else X[idx]
