"""How much room is there in *choosing the initial design*, and how much of it is free.

    python scripts/init_headroom.py --family kiba_cdk5 --keep-censored

AGENTIC.md's standing rule is to run the headroom gate before wiring up a model:
if the hindsight ceiling of a hook falls inside twice the standard error of the
best free policy, the action space is inert and no controller can demonstrate
anything on it. That gate exists for the acquisition hook
(`acquisition_headroom.py`) and the region hook (`cluster_headroom.py`). This is
the same instrument for the one hook a language model has already been given --
the initial design -- now asked under the condition that arm was never given:
**with the family history in hand**.

The existing warm-start result is the reason the gate matters. `bo_llm_warmstart`
beat plain BO (+0.0239, p = 0.027 at budget 100) and then lost to
`bo_diverse_warmstart`, a free MaxMin rule picking from the same sample, at both
budgets (-0.0049 and -0.0137). The model's own recorded rationales were diversity
arguments. But its prompt says, literally, *"Nothing has been measured yet"* --
it was never shown the auxiliary history, so that result bounds the chemistry
question and not this one.

## What is measured

For each (held-out target, repeat) the history is drawn once and shared, then
several candidate initial designs are run to the same budget on the same arm:

* ``--candidates`` random draws -- the action set a chooser picks from. Their
  **mean** is plain BO and their **max** is the hindsight ceiling, which reads
  the answer key and is a bound on every possible chooser, learned or not.
* ``aux_top`` and ``aux_scaffold_nn`` -- the free history-seeding rules from
  `dense_block_campaign`, drawn from their own pools.
* ``maxmin`` -- greedy MaxMin Tanimoto, the control that beat the LLM. It reads
  no history at all, which is what makes it the floor the history has to clear.

Every candidate shares the history, the budget, the arm and the rng stream;
`run_campaign(init_idx=...)` exists so that the *only* difference between them is
where they started.

## How to read it

The headline is **ceiling minus the best free rule**, never ceiling minus random.
A free rule that beats random is a better fixed policy, not evidence for an
agent; crediting it to one would repeat exactly the mistake
`bo_diverse_warmstart` was built to catch.

One more number is printed because it decides whether this hook is worth having
at all: the **multi-task arm's** gain under each seeding rule. The coregionalised
GP already consumes the same history. If seeding only moves `stl` and closes the
`mt - stl` gap by the same amount, the history was already being spent and the
hook is a second, worse channel for it rather than new information.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from molbo.audit import Audit  # noqa: E402
from molbo.metrics import auc_top_k, degrade_to_rho  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from molbo.surrogate import fingerprint_matrix, tanimoto_gram  # noqa: E402

from dense_block_campaign import (  # noqa: E402
    FAMILIES,
    PANEL_FAMILIES,
    PANELS,
    SOURCE,
    block_provenance,
    build_block,
    draw_history,
    drop_censored,
    history_seed_pool,
    murcko_labels,
    nanomolar_potency,
    run_campaign,
)

FREE_RULES = ("maxmin", "aux_top", "aux_scaffold_nn")


def maxmin_design(K, n_init, rng):
    """Greedy MaxMin Tanimoto: the free control that beat the LLM warm start.

    Starts from a random compound and repeatedly takes whichever is furthest
    from everything taken so far. It reads no history and no label, so it is the
    floor a history-aware rule has to clear before the history is worth
    anything.
    """
    n = len(K)
    picked = [int(rng.integers(n))]
    # Similarity to the nearest already-picked compound; minimising it is
    # maximising the Tanimoto distance to the set.
    near = K[picked[0]].copy()
    while len(picked) < n_init:
        near[picked] = np.inf
        nxt = int(np.argmin(near))
        picked.append(nxt)
        near = np.minimum(near, K[nxt])
    return np.asarray(picked, dtype=np.intp)


def bad_design(K, y, n_init, rng, *, quantile=0.25):
    """A deliberately bad local start: low-scoring, and all one neighbourhood.

    Akke et al. (2025) report that an agentic workflow's clearest win over
    statistical acquisition is on a campaign *initialised in a bad local
    minimum* -- the value of a chooser is concentrated in a regime the initial
    design created. That paper has no related target to borrow from, so it
    cannot ask the next question: if a good start can be bought for free from a
    relative's measurements, does the regime that pays the chooser still exist?

    This builds the regime. A seed is drawn from the bottom `quantile` of the
    primary and the design is that seed plus its nearest Tanimoto neighbours --
    low-scoring *and* chemically tight, which is what "bad local minimum" means
    in a fixed library.

    It reads `y`, which is hindsight. That is deliberate and it is not a rule
    being scored: it is the experimental manipulation, the same status as
    choosing a bad minimum by hand. Every design that gets a dAUC reported
    *against* this baseline still reads no primary value.
    """
    lo = np.flatnonzero(y <= np.quantile(y, quantile))
    seed = int(rng.choice(lo))
    # Descending similarity to the seed; K[seed, seed] == 1 puts the seed first.
    order = np.argsort(-K[seed], kind="stable")
    return np.asarray(order[:n_init], dtype=np.intp)


def candidate_designs(K, scaf, aux_idx, aux_val, *, n_init, n_cand, seed_top,
                      seed_nn, rng, y=None, start="random", bad_quantile=0.25):
    """One batch of initial designs to score against each other.

    Returns ``(randoms, rules)``: a list of ``n_cand`` uniform draws, and a dict
    of the free rules' designs. A rule that cannot fill ``n_init`` from its own
    pool is omitted rather than topped up from outside it -- a pool too small to
    seed a campaign is a fact about the rule, and padding it would report
    something else under the rule's name.
    """
    n = len(K)
    # The baseline the rules are scored against. Under `--start bad` it stops
    # being "plain BO" and becomes "plain BO launched from a bad local
    # minimum"; the rules themselves are untouched, because a rule is free to
    # ignore where the baseline started and pick its own design.
    randoms = ([bad_design(K, y, n_init, rng, quantile=bad_quantile)
                for _ in range(n_cand)]
               if start == "bad"
               else [rng.choice(n, size=n_init, replace=False)
                     for _ in range(n_cand)])
    rules: dict[str, np.ndarray] = {"maxmin": maxmin_design(K, n_init, rng)}
    for mode in ("aux_top", "aux_scaffold_nn"):
        pool, _ = history_seed_pool(K, scaf, aux_idx, aux_val, mode=mode,
                                    n_top=seed_top, n_nn=seed_nn)
        if len(pool) >= n_init:
            rules[mode] = rng.choice(pool, size=n_init, replace=False)
    return randoms, rules


def summarise(scores, label, base, bar=2.0):
    """Mean, SEM and the verdict the gate exists to deliver."""
    v = np.asarray(scores, dtype=float) - np.asarray(base, dtype=float)
    mean, sem = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
    flag = "" if sem == 0 else f"{abs(mean) / sem:.1f} SEM"
    return {"label": label, "delta_mean": mean, "delta_sem": sem,
            "n": int(len(v)), "sem_multiple": None if sem == 0 else abs(mean) / sem,
            "clears_bar": bool(sem > 0 and abs(mean) / sem >= bar), "flag": flag}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="kiba_cdk5")
    p.add_argument("--blocks-from", metavar="SCAN.json",
                   help="take blocks from a find_dense_blocks scan (labels k01, "
                        "k02, ...) instead of the registries, as "
                        "dense_block_campaign does. The registry labels stay "
                        "available; this reaches the blocks nothing is pinned to")
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--potency", choices=["nanomolar", "identity"],
                   default="nanomolar")
    p.add_argument("--arm", default="stl", choices=["stl", "mt", "pooled"],
                   help="the surrogate every candidate runs under. Run it twice "
                        "-- stl and mt -- and compare: if seeding helps stl and "
                        "not mt, the multi-task model was already spending the "
                        "same history")
    p.add_argument("--candidates", type=int, default=12,
                   help="random initial designs per cell. Their mean is plain "
                        "BO, their max is the hindsight ceiling")
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--history", type=int, default=50)
    p.add_argument("--history-excludes-top", action="store_true",
                   help="draw the history clear of the primary's top-k. Without "
                        "it a seeding rule can be handed the answer outright, "
                        "and the overlap is reported either way")
    p.add_argument("--aux-quality", type=float, default=None,
                   help="degrade every auxiliary to this Spearman correlation "
                        "with its own true values before anything reads it, as "
                        "dense_block_campaign does. The primary's column and the "
                        "scoring are untouched. This is the only way to reach "
                        "couplings weaker than the panel contains: all 15 KIBA "
                        "blocks sit at rho >= 0.28, so the zero crossing of the "
                        "seeding gain is outside the measured range and a "
                        "synthetic ladder is what brackets it")
    p.add_argument("--start", choices=["random", "bad"], default="random",
                   help="what the baseline designs are. `random` is plain BO "
                        "and is the default, so every stored result "
                        "reproduces. `bad` launches the baseline from a bad "
                        "local minimum -- a low-scoring compound and its "
                        "nearest neighbours -- which is the regime Akke et al. "
                        "(2025) find an LLM chooser earns its keep in. The "
                        "free rules are unchanged, so the gain reported "
                        "against `bad` answers whether free history seeding "
                        "removes that regime")
    p.add_argument("--bad-quantile", type=float, default=0.25,
                   help="the quantile of the primary a --start bad seed is "
                        "drawn from. Hindsight, and deliberately so: it is the "
                        "manipulation, not a rule")
    p.add_argument("--seed-top", type=int, default=10)
    p.add_argument("--seed-nn", type=int, default=5)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pile-frac", type=float, default=0.02)
    p.add_argument("--keep-censored", action="store_true")
    p.add_argument("--audit", action="store_true", default=True)
    p.add_argument("--out", default="results/init_headroom.json")
    args = p.parse_args(argv)

    if args.blocks_from:
        from find_dense_blocks import blocks_as_families
        families = blocks_as_families(args.blocks_from, top=10_000)
    else:
        families = {**FAMILIES, **PANEL_FAMILIES}
    if args.family not in families:
        p.error(f"unknown family {args.family!r}; choose from "
                + ", ".join(sorted(families)))
    if args.family in PANEL_FAMILIES and not args.blocks_from:
        panel = PANELS[args.family.split("_")[0]]
        if args.source == SOURCE:
            args.source = panel["source"]
        if args.potency == p.get_default("potency"):
            args.potency = panel["potency"]

    t0 = time.time()
    mapping = families[args.family]
    potency = {"nanomolar": nanomolar_potency, "identity": lambda y: y}
    block, block_csv = build_block(mapping, source=args.source,
                                   potency=potency[args.potency])
    targets = [t for t in mapping.values() if t in block.columns]
    n_block = len(block)
    if not args.keep_censored:
        block, _ = drop_censored(block, targets, pile_frac=args.pile_frac)
    X = fingerprint_matrix(block.smiles.tolist())
    K = tanimoto_gram(X, X)
    scaf = murcko_labels(block)
    vals = block[targets].to_numpy(float)
    n_pool = len(block)
    print(f"{args.family}: {n_pool:,} compounds x {len(targets)} targets, "
          f"arm {args.arm}, budget {args.budget}, {args.candidates} candidate "
          f"designs x {args.repeats} repeats ({time.time() - t0:.0f}s)")

    audit = Audit(enabled=args.audit)
    rand_mean: list[float] = []
    rho_eff: list[float] = []
    ceiling: list[float] = []
    rule_scores: dict[str, list[float]] = {r: [] for r in FREE_RULES}
    rule_base: dict[str, list[float]] = {r: [] for r in FREE_RULES}
    overlap: dict[str, list[int]] = {r: [] for r in FREE_RULES}

    for ti, t in enumerate(targets):
        y = vals[:, ti]
        others = [j for j in range(len(targets)) if j != ti]
        top_k = set(np.argsort(y)[-args.k:].tolist())
        for r in range(args.repeats):
            g = np.random.default_rng(args.seed + 1009 * r + 97 * ti)
            hist = draw_history(K, y, args.history * len(others), g,
                                cluster_frac=0.0,
                                exclude_top=args.history_excludes_top, k=args.k)
            aux_idx = [hist[a * args.history:(a + 1) * args.history]
                       for a in range(len(others))]
            aux_val = [vals[aux_idx[a], others[a]] for a in range(len(others))]
            if args.aux_quality is not None:
                # Redrawn per (target, repeat), as the campaign does, so the
                # answer averages over noise draws instead of depending on one.
                gq = np.random.default_rng(args.seed + 7717 * r + 131 * ti)
                aux_val = [degrade_to_rho(vals[:, others[a]], args.aux_quality,
                                          gq)[aux_idx[a]]
                           for a in range(len(others))]
            # The coupling the seeding rule actually has to the thing being
            # optimised: how well a relative's numbers rank the primary, on the
            # rows the history exposes. This is the honest x axis -- the block's
            # own median rho is measured on all four columns over the whole
            # library, while a rule only ever sees these cells, and under
            # --aux-quality it does not even see the real ones.
            # Both sides must vary. Guarding only the auxiliary was a defect:
            # a block whose primary is constant on the history rows -- which
            # happens on a panel with heavy ties -- yields a NaN Spearman, and
            # one NaN makes the median NaN, silently poisoning every downstream
            # fit that uses this axis as its x.
            rho_eff.extend(
                float(spearmanr(a_v, y[a_i]).statistic)
                for a_i, a_v in zip(aux_idx, aux_val)
                if len(np.unique(a_v)) > 2 and len(np.unique(y[a_i])) > 2)
            gd = np.random.default_rng(args.seed + 5003 * r + 71 * ti)
            randoms, rules = candidate_designs(
                K, scaf, aux_idx, aux_val, n_init=args.n_init,
                n_cand=args.candidates, seed_top=args.seed_top,
                seed_nn=args.seed_nn, rng=gd, y=y, start=args.start,
                bad_quantile=args.bad_quantile)
            audit.cell()

            def score(init):
                # One rng stream per cell, rebuilt for every candidate, so the
                # tie-breaking jitter downstream is identical and the initial
                # design is the only thing that varies.
                seq, picked = run_campaign(
                    K, y, aux_idx, aux_val, arm=args.arm, budget=args.budget,
                    batch=args.batch, n_init=args.n_init, k=args.k,
                    acquisition="ei", init_idx=init,
                    rng=np.random.default_rng(args.seed + 1009 * r + 97 * ti))
                audit.exactly(picked, args.budget, "candidate selections")
                audit.same_across((ti, r), "history", "history",
                                  np.concatenate(aux_idx))
                return auc_top_k(seq.tolist(), k=args.k)

            rs = [score(d) for d in randoms]
            rand_mean.append(float(np.mean(rs)))
            ceiling.append(float(np.max(rs)))
            for name in FREE_RULES:
                if name not in rules:
                    continue
                rule_scores[name].append(score(rules[name]))
                rule_base[name].append(float(np.mean(rs)))
                overlap[name].append(len(top_k & set(np.asarray(
                    rules[name]).tolist())))
        print(f"    {t} done ({time.time() - t0:.0f}s)", flush=True)

    rows = [summarise(ceiling, "hindsight ceiling", rand_mean)]
    for name in FREE_RULES:
        if rule_scores[name]:
            rows.append(summarise(rule_scores[name], name, rule_base[name]))
    free = [r for r in rows[1:]]
    best = max(free, key=lambda r: r["delta_mean"]) if free else None

    base_label = "bad start" if args.start == "bad" else "random"
    print(f"\n    {'initial design':<22s}{'dAUC vs ' + base_label:>18s}{'':>12s}")
    print("    " + "-" * 52)
    for r in rows:
        print(f"    {r['label']:<22s}{r['delta_mean']:>+12.4f}"
              f" +-{r['delta_sem']:.4f}  {r['flag']}")
    for name in FREE_RULES:
        if overlap[name]:
            print(f"    {name}: {np.mean(overlap[name]):.2f} of the true "
                  f"top-{args.k} seeded outright")

    if rho_eff:
        print(f"\n    effective coupling seen by a rule: median rho "
              f"{np.nanmedian(rho_eff):+.3f} over "
              f"{int(np.sum(~np.isnan(rho_eff)))} (target, repeat, "
              f"auxiliary) triples"
              + (f", requested aux-quality {args.aux_quality:g}"
                 if args.aux_quality is not None else ""))

    gate = None
    if best is not None:
        # The gate, stated the way AGENTIC.md states it: against the best free
        # policy, not against random.
        margin = rows[0]["delta_mean"] - best["delta_mean"]
        sem = float(np.hypot(rows[0]["delta_sem"], best["delta_sem"]))
        inert = sem > 0 and margin < 2 * sem
        gate = {"ceiling_minus_best_free": margin, "sem": sem,
                "best_free": best["label"], "inert": bool(inert)}
        print(f"\n    headroom over the best free rule ({best['label']}): "
              f"{margin:+.4f} +-{sem:.4f}")
        print("    " + ("-> INERT at this power: the ceiling is inside 2 SEM of "
                        "a free rule.\n       A chooser has nothing to win here; "
                        "do not wire up a model."
                        if inert else
                        "-> room remains above every free rule. A chooser has "
                        "something to win;\n       its control is the best free "
                        "rule above, on identical inputs."))

    if args.audit:
        print(f"\n    [audit] {audit.checks} checks over {audit.cells} cells")
        print("            " + ("all invariants hold: exact budget, "
                                "shared history across candidates"
                                if not audit.problems else
                                "PROBLEMS: " + "; ".join(audit.problems)))

    report = {"family": args.family, "arm": args.arm, "budget": args.budget,
              "batch": args.batch, "n_init": args.n_init, "k": args.k,
              "history": args.history,
              "history_excludes_top": bool(args.history_excludes_top),
              "candidates": args.candidates, "repeats": args.repeats,
              "start": args.start,
              # The baseline's own level, not just the deltas taken against it.
              # Comparing a bad start to a random one needs the absolute AUC:
              # the deltas are measured against different baselines and are not
              # on the same scale.
              "baseline_auc_mean": float(np.mean(rand_mean)),
              "baseline_auc_sem": float(np.std(rand_mean, ddof=1)
                                        / np.sqrt(len(rand_mean))),
              "bad_quantile": (args.bad_quantile if args.start == "bad"
                               else None),
              "n_pool": int(n_pool), "rows": rows, "gate": gate,
              "aux_quality": args.aux_quality,
              "rho_eff": (float(np.nanmedian(rho_eff))
                          if rho_eff and not np.all(np.isnan(rho_eff))
                          else None),
              "rho_eff_mean": (float(np.nanmean(rho_eff))
                               if rho_eff and not np.all(np.isnan(rho_eff))
                               else None),
              "rho_eff_n": int(np.sum(~np.isnan(rho_eff))) if rho_eff else 0,
              "top_k_seeded": {n: float(np.mean(v))
                               for n, v in overlap.items() if v},
              "provenance": block_provenance(
                  args, targets=targets, block_csv=block_csv,
                  n_block=n_block, n_pool=n_pool),
              "audit": {"enabled": bool(args.audit), "checks": audit.checks,
                        "cells": audit.cells,
                        "clean": not audit.all_problems,
                        "problems": list(audit.all_problems)}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
