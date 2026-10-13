"""Sweeps: run every arm on every seed, then compare them properly.

The primary tier exists so this file can be cheap to run.  A sweep here is
``n_arms x n_seeds`` campaigns against a dictionary lookup, which means the
expensive part of the experiment is the GP, not the oracle, and 20 seeds is an
afternoon-free decision rather than 90 GPU-hours.

Arms are paired on seed: seed *s* hands every arm the same initial design, so
:func:`compare_arms` can use a signed-rank test over matched pairs.  The
reference arm (what everything is tested against) is plain BO, not random --
beating random is not a result, it is a smoke test.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .campaign import ArmConfig, CampaignResult, run_campaign
from .constraints import LibraryContext, MaskConstraint, RandomSubsetConstraint, oracle_constraint
from .metrics import auc_top_k, compare_arms, normalize_values, summarize_arm, top_auc
from .oracles.base import Oracle
from .surrogate import LibraryKernel, fingerprint_matrix


@dataclass
class RunMetrics:
    """Everything scalar about one campaign, ready to go into a table."""

    arm: str
    seed: int
    auc: float
    auc_pmo: float
    best: float
    best_normalized: float
    simple_regret: float | None
    top1pct_recall: float | None
    n_calls: int
    n_failed: int
    wall_seconds: float
    constraint_modes: dict[str, int] = field(default_factory=dict)

    def row(self) -> dict:
        d = dict(self.__dict__)
        d["constraint_modes"] = json.dumps(self.constraint_modes, sort_keys=True)
        return d


def score_run(
    result: CampaignResult,
    *,
    lo: float,
    hi: float,
    k: int = 10,
    budget: int,
    best_possible: float | None = None,
    top_set: set[str] | None = None,
) -> RunMetrics:
    """Turn a campaign trace into metrics.

    ``lo``/``hi`` are the *library's* value range, not the run's -- scaling by
    what a run happened to find would make each arm's AUC depend on its own
    success and destroy comparability. See :func:`molbo.metrics.normalize_values`.
    """
    scored = result.scored
    norm = normalize_values(scored, lo, hi) if scored else np.zeros(0)
    buffer = {s: (float(v), i + 1) for i, (s, v) in enumerate(zip(result.smiles, norm))}

    best = max(scored) if scored else float("nan")
    modes: dict[str, int] = {}
    for entry in result.constraint_log:
        modes[entry.get("mode", "?")] = modes.get(entry.get("mode", "?"), 0) + 1

    return RunMetrics(
        arm=result.arm,
        seed=result.seed,
        auc=auc_top_k(norm, k=k, budget=budget),
        auc_pmo=top_auc(buffer, top_n=k, freq_log=100, max_oracle_calls=budget),
        best=best,
        best_normalized=float(norm.max()) if norm.size else float("nan"),
        simple_regret=(None if best_possible is None else float(best_possible - best)),
        top1pct_recall=(
            None if top_set is None else len(top_set & set(result.smiles)) / max(1, len(top_set))
        ),
        n_calls=result.n_calls,
        n_failed=result.n_failed,
        wall_seconds=result.wall_seconds,
        constraint_modes=modes,
    )


def default_arms(
    values: Sequence[float],
    *,
    n_init: int = 10,
    constraint_fraction: float = 0.10,
    llm_source: Callable | None = None,
    include_data_driven: bool = True,
) -> list[ArmConfig]:
    """The standard suite for the primary tier.

    ``random``
        Floor. If an arm cannot beat this, nothing else about it matters.
    ``bo_ei``
        Plain BO -- the reference every other arm is tested against.
    ``bo_random_constraint``
        Plain BO restricted to a random quarter of the library. Separates
        "the LLM knows chemistry" from "any restriction helps at this budget",
        which is the confound a reviewer will reach for first.
    ``bo_data_driven_constraint``
        Descriptor ranges of the best-observed compounds, computed by
        statistics with no model involved. The sharpest control: if the LLM
        cannot beat this, its chemistry knowledge is not what is doing the
        work.
    ``bo_oracle_constraint``
        Ceiling: restricted to the true best ``constraint_fraction``. Not a
        method -- it bounds how much *any* constraint could buy, and so says
        whether the LLM arm has room to be interesting at all.

    Passing ``llm_source`` adds the arm under test alongside them.

    The default ``constraint_fraction`` is 0.10 rather than something looser
    because ``scripts/constraint_headroom.py`` measures the ceiling against
    plain BO, and headroom roughly triples between a quarter of the library
    and a twentieth. A sweep run in the loose regime cannot resolve a
    constraint effect even if one exists.
    """
    arms = [
        ArmConfig("random", strategy="random", n_init=n_init),
        ArmConfig("bo_ei", strategy="bo", acquisition="ei", n_init=n_init),
        ArmConfig(
            "bo_random_constraint",
            strategy="bo",
            acquisition="ei",
            n_init=n_init,
            constraint_source=_static(
                lambda seed: RandomSubsetConstraint(fraction=constraint_fraction, seed=seed)
            ),
        ),
        ArmConfig(
            "bo_oracle_constraint",
            strategy="bo",
            acquisition="ei",
            n_init=n_init,
            constraint_source=_static(
                lambda seed: oracle_constraint(values, fraction=constraint_fraction)
            ),
        ),
    ]
    if include_data_driven:
        from .llm import DataDrivenProposer

        arms.append(
            ArmConfig(
                "bo_data_driven_constraint",
                strategy="bo",
                acquisition="ei",
                n_init=n_init,
                constraint_source=DataDrivenProposer(),
                constraint_every=0,
            )
        )
    if llm_source is not None:
        arms.append(
            ArmConfig(
                "bo_llm_constraint",
                strategy="bo",
                acquisition="ei",
                n_init=n_init,
                constraint_source=llm_source,
                constraint_every=0,
            )
        )
    return arms


def _static(factory):
    """Wrap a seed-keyed constraint factory in the ConstraintSource signature.

    The mask is keyed on the *campaign* seed rather than on a draw from the
    campaign's RNG, for two reasons. The random-subset control must vary with
    the seed -- otherwise all 20 seeds share one mask, the arm's variance is
    understated, and it looks far more reliable than it is. And taking the seed
    directly leaves the campaign's own RNG stream untouched, so the constrained
    arms tie-break identically to plain BO and the arms stay paired.
    """

    def source(observations, ctx, rng, round_index, seed=0, **_):
        return factory(int(seed))

    return source


def run_sweep(
    oracle: Oracle,
    arms: Sequence[ArmConfig],
    seeds: Iterable[int],
    *,
    budget: int,
    k: int = 10,
    library: Sequence[str] | None = None,
    verbose: bool = True,
) -> tuple[list[RunMetrics], list[CampaignResult]]:
    """Run every ``(arm, seed)`` pair. Returns metrics and raw traces."""
    library = list(library if library is not None else oracle.library)
    ctx = LibraryContext.build(library)
    X = fingerprint_matrix(library)
    kernel = LibraryKernel.maybe_build(X)

    values = [oracle._table[s] for s in library] if hasattr(oracle, "_table") else None
    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
    best_possible = max(values) if values else None
    top_set = None
    if values:
        n_top = max(1, int(round(0.01 * len(values))))
        order = np.argsort(-np.asarray(values))
        top_set = {library[i] for i in order[:n_top]}

    metrics: list[RunMetrics] = []
    traces: list[CampaignResult] = []
    seeds = list(seeds)

    for arm in arms:
        t0 = time.perf_counter()
        for seed in seeds:
            res = run_campaign(
                oracle, arm, seed, budget=budget, library=library, ctx=ctx, X=X, kernel=kernel
            )
            traces.append(res)
            metrics.append(
                score_run(
                    res, lo=lo, hi=hi, k=k, budget=budget,
                    best_possible=best_possible, top_set=top_set,
                )
            )
        if verbose:
            arm_m = [m for m in metrics if m.arm == arm.name]
            print(
                f"  {arm.name:24s} AUC {np.mean([m.auc for m in arm_m]):.4f} "
                f"best {np.mean([m.best for m in arm_m]):.3f} "
                f"({time.perf_counter() - t0:.1f}s / {len(seeds)} seeds)",
                flush=True,
            )
    return metrics, traces


def summarize_sweep(metrics: Sequence[RunMetrics], reference: str = "bo_ei") -> dict:
    """Per-arm aggregates plus a paired test of every arm against ``reference``."""
    by_arm: dict[str, list[RunMetrics]] = {}
    for m in metrics:
        by_arm.setdefault(m.arm, []).append(m)
    for rows in by_arm.values():
        rows.sort(key=lambda m: m.seed)

    summaries = {
        name: summarize_arm(name, [m.auc for m in rows], [m.best for m in rows])
        for name, rows in by_arm.items()
    }

    comparisons: dict[str, dict] = {}
    if reference in by_arm:
        ref_seeds = [m.seed for m in by_arm[reference]]
        for name, rows in by_arm.items():
            if name == reference:
                continue
            if [m.seed for m in rows] != ref_seeds:
                comparisons[name] = {"error": "seed sets differ; cannot pair"}
                continue
            comparisons[name] = compare_arms(
                [m.auc for m in rows], [m.auc for m in by_arm[reference]], paired=True
            )
    return {"summaries": summaries, "comparisons": comparisons, "reference": reference}


def save_sweep(
    out_dir: str | Path,
    metrics: Sequence[RunMetrics],
    traces: Sequence[CampaignResult],
    summary: dict,
    meta: dict | None = None,
) -> Path:
    """Write ``metrics.csv``, ``summary.json`` and ``traces.jsonl``."""
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([m.row() for m in metrics]).to_csv(out / "metrics.csv", index=False)

    with (out / "traces.jsonl").open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t.to_dict()) + "\n")

    payload = {
        "meta": meta or {},
        "reference": summary["reference"],
        "arms": {
            name: {
                "n_seeds": s.n_seeds,
                "auc_mean": s.auc_mean,
                "auc_sem": s.auc_sem,
                "auc_std": s.auc_std,
                "best_mean": s.best_mean,
                "best_sem": s.best_sem,
                "aucs": s.aucs.tolist(),
                "bests": s.bests.tolist(),
            }
            for name, s in summary["summaries"].items()
        },
        "comparisons_vs_reference": summary["comparisons"],
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
