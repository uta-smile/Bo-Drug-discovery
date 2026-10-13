"""Is there anything for a multi-task surrogate to transfer? Run this before H1.

    python scripts/transfer_headroom.py --out results/transfer_headroom.json

H1 says a surrogate that sees other targets predicts a given target better than
one that sees only that target.  Building the multi-task surrogate to find out
costs days.  Measuring whether the *ceiling* on that gain is above zero costs
minutes, because DOCKSTRING is dense: for any molecule in the pool the true
value on all 58 targets is already on disk, so the auxiliary data a multi-task
model could ever hope to infer can simply be handed to a regressor for free.

This instrument mirrors ``acquisition_headroom.py``: fixed arms, then a
hindsight ceiling no deployable method can beat.

Budget is counted in **cells** -- one ``(molecule, target)`` measurement -- not
in molecules, because that is the only accounting under which the two designs
are comparable.  A single-target campaign buys ``n`` molecules on ``t``; a
multi-task one buys ``n / (1 + |aux|)`` molecules and measures every target on
each.  Compare them at equal molecules and the multi-task arm has simply been
handed more data, which makes any result it produces unfalsifiable.

Per primary target ``t``, at each budget ``n``, Spearman rho against truth on
the held-out rows of a shared pool:

* ``stl``     -- Tanimoto GP on fingerprints, all ``n`` cells spent on ``t``.
                 The independent-target surrogate H1 wants to beat, and the same
                 model the rest of this package runs.
* ``mt_gp``   -- the same ``n`` cells spread over ``t`` and its auxiliaries,
                 given to one coregionalised GP: ``k((m,a),(m',b)) =
                 k_tan(m,m') B[a,b]`` with the task correlation picked by
                 marginal likelihood.  **This is the model H1 proposes** and the
                 number the hypothesis stands or falls on.
* ``mt_eq``   -- the same cells, spent by a two-stage pipeline instead: a
                 per-target GP predicts each auxiliary column everywhere, a
                 ridge fitted on the paid-for rows maps auxiliaries to ``t``.
                 The naive way to transfer, kept as a floor.
* ``mt_free`` -- ``n`` cells on ``t`` *and* the auxiliary columns handed over
                 exactly, for nothing, on every molecule in the pool.  Not
                 deployable.  The ceiling on what knowing the other targets
                 could be worth at this budget.
* ``aux_inf`` -- ridge from the other 57 targets, fitted out-of-fold on the
                 whole pool, so budget no longer binds at all.
* ``fp_inf``  -- the Tanimoto GP fitted on 80% of the pool: what more budget
                 alone buys on the molecule-only route.

**Headroom is ``mt_gp(n) - stl(n)``.**  ``mt_free`` and ``aux_inf`` are
ceilings, and a ceiling exceeding the reachable number is the normal case, not
a contradiction.  Three outcomes, meaning three different things:

* ``aux_inf <= fp_inf`` -- the other targets carry nothing about ``t`` that the
  fingerprint does not already carry.  H1 is dead here and no multi-task
  surrogate can rescue it.
* ``mt_gp(n) > stl(n)`` -- spreading the same assay budget across coupled
  targets beats concentrating it.  H1 holds on this benchmark.
* ``aux_inf >> fp_inf`` but ``mt_gp(n) <= stl(n)`` -- the structure is real and
  the multi-task surrogate does not convert it at equal cost.  The molecules
  given up to measure the auxiliaries were worth more than the coupling
  returned, and any later multi-task result that does not hold the cell budget
  fixed is measuring that substitution rather than transfer.

Check B answers Kubincova & Mobley's own question on the same pool: given the
identical measured cells, is it better to regress the scalarised objective
directly (``joint``) or to model each docking column and compose them through
the published formula (``separated``)?  The ``min(LCK, -8.1)`` term saturates,
so a joint model has to learn a kink that a separated one gets for free.

Docking scores, not measurements: nothing here is a claim about biology.  See
``molbo/dockstring.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo import dockstring as ds  # noqa: E402
from molbo.audit import Audit  # noqa: E402
from molbo.metrics import spearman0 as _rho  # noqa: E402
from molbo.surrogate import (  # noqa: E402
    CoregionalisedGP,
    TanimotoGP,
    fingerprint_matrix,
    tanimoto_gram,
)

N_GRID = (5, 10, 20, 30, 50, 80)
"""Observation counts. The same grid as the GP accuracy curve in FINDINGS.md
section 10, so the ``stl`` column here is directly readable against it."""

AUX_DEFAULT: tuple[str, ...] = ("LCK", "SRC", "ABL1")
"""Auxiliary targets the multi-task arm spends part of its budget on.

Three, not all 57: at equal cells each extra auxiliary divides the number of
molecules the arm can afford, so a large auxiliary set does not test transfer,
it tests whether four molecules are enough.  All three are kinases, fixed here
rather than selected per primary target -- choosing auxiliaries by their
measured correlation with the target would let the hidden label matrix pick the
method's hyperparameter, which is the contamination this whole benchmark exists
to avoid."""

ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
"""Ridge penalties. Chosen per fit by leave-one-out error, which is closed-form
for ridge -- at n=5 with 57 features the penalty is doing all the work, so
letting the arm pick it is the difference between a ceiling and a shrug."""


# ---- ridge with closed-form LOO selection --------------------------------


def _ridge_fit(X: np.ndarray, y: np.ndarray, alphas=ALPHA_GRID):
    """Ridge on standardised ``X`` with the penalty chosen by LOO error.

    Returns ``(predict_fn, alpha, loo_rmse)``. LOO comes free from the hat
    matrix: ``e_loo_i = e_i / (1 - h_ii)``, so the whole grid costs one SVD.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    mu_x = X.mean(axis=0)
    sd_x = X.std(axis=0)
    sd_x[sd_x < 1e-12] = 1.0
    Z = (X - mu_x) / sd_x
    mu_y = float(y.mean())
    c = y - mu_y

    # Economy SVD: n x p with either dimension small, both cases cheap.
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    Uc = U.T @ c

    best = None
    fallback = None
    for a in alphas:
        d = s / (s**2 + a)
        coef = Vt.T @ (d * Uc)
        fitted = Z @ coef
        # h_ii = sum_k U_ik^2 s_k^2 / (s_k^2 + a)
        hii = U**2 @ (s**2 / (s**2 + a))
        # n <= p, which is every small-n arm here: a weak penalty interpolates,
        # so the residual and 1 - h_ii both go to zero and their ratio reports a
        # LOO error of ~0 for the alpha that generalises worst. Require the fit
        # to leave residual degrees of freedom rather than let it win the grid.
        edf = float(np.sum(s**2 / (s**2 + a)))
        if edf > max(1.0, len(c) - 2):
            fallback = (np.inf, a, coef)  # keep the last, i.e. most penalised
            continue
        resid = c - fitted
        loo = float(np.sqrt(np.mean((resid / (1.0 - hii)) ** 2)))
        if best is None or loo < best[0]:
            best = (loo, a, coef)

    loo, alpha, coef = best if best is not None else fallback

    def predict(Xn: np.ndarray) -> np.ndarray:
        Zn = (np.asarray(Xn, dtype=np.float64) - mu_x) / sd_x
        return Zn @ coef + mu_y

    return predict, float(alpha), loo




def _icm_fit_predict(K, te, idx_per_task, y_per_task):
    """:class:`CoregionalisedGP` on the primary task, returning ``(mu, rho)``."""
    gp = CoregionalisedGP().fit_gram(K, idx_per_task, y_per_task)
    return gp.predict_gram(K, te), gp.rho_


# ---- check A: does the other 57 targets' truth beat the fingerprint? -----


def _aux_ceiling(A: np.ndarray, y: np.ndarray, te: np.ndarray, *, folds: int, seed: int):
    """``aux_inf``: out-of-fold ridge over the whole pool, scored on ``te``.

    Fitted with every label in the pool, which no campaign has -- that is the
    point. Out-of-fold rather than in-sample so the number is a ceiling on
    generalisation and not a report of how well ridge memorises 2,000 rows.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    oof = np.empty(n)
    for f in range(folds):
        hold = order[f::folds]
        keep = np.setdiff1d(order, hold)
        pred, _, _ = _ridge_fit(A[keep], y[keep])
        oof[hold] = pred(A[hold])
    return _rho(y[te], oof[te])


def check_transfer(sub, K, report, *, targets, aux_targets, n_repeats, seed, folds,
                   audit=None):
    """Three ways to spend the same number of assayed cells, plus two ceilings.

    The unit of budget is a *cell*: one (molecule, target) measurement.  An
    independent-target campaign spends all ``n`` of them on ``t``; a multi-task
    one spreads them over ``t`` and its auxiliaries, so it learns about more
    proteins from fewer molecules each.  Comparing at equal ``n`` molecules
    instead of equal cells is the mistake that makes any multi-task result
    unfalsifiable, because the multi-task arm would simply have been given more
    data.
    """
    audit = audit or Audit()
    all_cols = sub[list(ds.TARGETS)].to_numpy(dtype=float)
    n_pool = len(sub)
    n_aux_max = len(aux_targets)
    print(f"\n[A] transfer headroom  ({len(targets)} primary targets, "
          f"aux={'+'.join(aux_targets)}, pool={n_pool}, {n_repeats} repeats)")
    print("    budget is measured in cells; the arms below all spend the same number.")
    out: dict = {}

    for ti, t in enumerate(targets):
        y = -all_cols[:, ds.TARGETS.index(t)]  # negate: every objective is maximised
        aux = [a for a in aux_targets if a != t]
        n_aux = len(aux)
        # A target whose auxiliary set is only itself has nothing to transfer
        # from, and the arms then reduce to stl vs a ridge on zero columns --
        # which still prints a full table of plausible-looking zeros.
        if n_aux == 0:
            raise SystemExit(
                f"{t} has no auxiliary targets: --aux-targets {' '.join(aux_targets)} "
                f"contains nothing but {t}. Give it at least one other target.")
        Acols = [ds.TARGETS.index(a) for a in aux]
        A = -all_cols[:, Acols]
        # The ceilings use every other target, not just the auxiliary set: they
        # are meant to bound what *any* cross-target information could give.
        A_full = -all_cols[:, [j for j, nm in enumerate(ds.TARGETS) if nm != t]]
        # The ceiling arms are only ceilings if they were denied the primary's
        # own column. `aux_inf` regressing t on t would come back at 1.000 and
        # read as "cross-target information is worth everything".
        audit.excludes(t, aux, f"[A] {t} auxiliary set")
        audit.not_relabelled(y, [A[:, j] for j in range(n_aux)], f"[A] {t}")
        audit.not_relabelled(y, [A_full[:, j] for j in range(A_full.shape[1])],
                             f"[A] {t} ceiling features")

        g = np.random.default_rng(seed + 77 * ti)
        tr_inf = g.choice(n_pool, size=int(0.8 * n_pool), replace=False)
        te_inf = np.setdiff1d(np.arange(n_pool), tr_inf)
        gp = TanimotoGP().fit_gram(K[np.ix_(tr_inf, tr_inf)], y[tr_inf])
        fp_inf = _rho(y[te_inf], gp.predict_gram(K[np.ix_(te_inf, tr_inf)],
                                                 return_std=False))
        aux_inf = _aux_ceiling(A_full, y, te_inf, folds=folds, seed=seed + 77 * ti)

        rows: dict[int, dict] = {}
        for n in N_GRID:
            # Equal cells: m molecules assayed against t and all n_aux
            # auxiliaries costs m * (1 + n_aux) cells. Below 1 + n_aux cells the
            # multi-task arm cannot afford a single molecule, which is itself
            # part of the answer, so it is reported rather than skipped.
            m = n // (1 + n_aux)
            acc = {"stl": [], "mt_eq": [], "mt_gp": [], "mt_spread": [], "mt_free": []}
            rhos_picked = []
            for r in range(n_repeats):
                gr = np.random.default_rng(seed + 1000 * r + n + 7919 * ti)
                tr = gr.choice(n_pool, size=n, replace=False)
                te = np.setdiff1d(np.arange(n_pool), tr)
                audit.cell()
                # `stl` spends all n cells on t; the arms below spend the same
                # number spread over 1 + n_aux tasks. Equal cells is the whole
                # basis of the comparison (see the module docstring), so it is
                # checked rather than asserted.
                audit.exactly(tr, n, f"[A] {t} n={n} r={r} training rows")
                audit.disjoint(te, tr, "[A] evaluation", "training")

                gp = TanimotoGP().fit_gram(K[np.ix_(tr, tr)], y[tr])
                acc["stl"].append(
                    _rho(y[te], gp.predict_gram(K[np.ix_(te, tr)], return_std=False)))

                # mt_free: the same n cells on t, and the auxiliary columns
                # handed over exactly and for nothing. Not deployable; it is the
                # ceiling on what knowing the other targets is worth at budget n.
                pf, _, _ = _ridge_fit(A[tr], y[tr])
                acc["mt_free"].append(_rho(y[te], pf(A[te])))

                # mt_eq: m molecules, every target measured on each. The map
                # from auxiliaries to t is fitted on the exact values at those m
                # rows -- cells that were paid for -- and applied to auxiliary
                # values that a per-target GP had to predict everywhere else.
                if m < 3:
                    acc["mt_eq"].append(np.nan)
                    acc["mt_gp"].append(np.nan)
                    acc["mt_spread"].append(np.nan)
                    continue
                trm = tr[:m]
                Km = K[np.ix_(trm, trm)]
                A_hat = np.empty((len(te), n_aux))
                for j in range(n_aux):
                    ga = TanimotoGP().fit_gram(Km, A[trm, j])
                    A_hat[:, j] = ga.predict_gram(K[np.ix_(te, trm)], return_std=False)
                pe, _, _ = _ridge_fit(A[trm], y[trm])
                acc["mt_eq"].append(_rho(y[te], pe(A_hat)))

                # mt_gp: the same cells, given to one coregionalised GP instead
                # of a two-stage pipeline. This is the model H1 proposes.
                cols_t = [y] + [A[:, j] for j in range(n_aux)]
                blocked = [trm] * (1 + n_aux)
                # Equal cells, checked: m molecules on each of 1 + n_aux tasks
                # must not exceed the n cells `stl` was given.
                audit.want(sum(len(b) for b in blocked) <= n,
                           f"[A] {t} n={n} r={r}: mt_gp spent "
                           f"{sum(len(b) for b in blocked)} cells, stl had {n}")
                audit.subset(trm, tr, "[A] mt_gp rows", "the paid-for draw")
                mu_hat, r_pick = _icm_fit_predict(
                    K, te, blocked, [c[i] for c, i in zip(cols_t, blocked)])
                acc["mt_gp"].append(_rho(y[te], mu_hat))
                rhos_picked.append(r_pick)

                # mt_spread: the same number of cells, but no molecule is
                # measured twice -- each task gets m molecules of its own. A
                # blocked design re-reads the same chemistry on every target,
                # which is the most redundant way to spend a multi-task budget
                # when the tasks are correlated; this is the least redundant.
                spread = [tr[a * m:(a + 1) * m] for a in range(1 + n_aux)]
                audit.want(sum(len(sp) for sp in spread) <= n,
                           f"[A] {t} n={n} r={r}: mt_spread spent "
                           f"{sum(len(sp) for sp in spread)} cells, stl had {n}")
                # "No molecule is measured twice" is what distinguishes this arm
                # from the blocked one; if it failed, the two arms are the same.
                flat = np.concatenate(spread)
                audit.exactly(flat, sum(len(sp) for sp in spread),
                              f"[A] {t} n={n} r={r} mt_spread rows")
                if min(len(s) for s in spread) >= 3:
                    mu_s, r_s = _icm_fit_predict(
                        K, te, spread, [c[i] for c, i in zip(cols_t, spread)])
                    acc["mt_spread"].append(_rho(y[te], mu_s))
                else:
                    acc["mt_spread"].append(np.nan)

            rows[n] = {"cells": n, "m_molecules_mt": m,
                       "rho_picked_median": (float(np.median(rhos_picked))
                                             if rhos_picked else float("nan"))}
            for k, v in acc.items():
                v = np.asarray(v, dtype=float)
                ok = np.isfinite(v)
                rows[n][k] = {
                    "mean": float(v[ok].mean()) if ok.any() else float("nan"),
                    "sem": float(v[ok].std() / np.sqrt(ok.sum())) if ok.any()
                    else float("nan"),
                }
            # Paired differences: the repeats share a training draw, so pairing
            # removes the split-to-split variance that dominates at small n.
            for label, arm in (("headroom", "mt_gp"), ("headroom_spread", "mt_spread"),
                               ("headroom_2stage", "mt_eq"),
                               ("ceiling_at_n", "mt_free")):
                d = np.asarray(acc[arm], float) - np.asarray(acc["stl"], float)
                ok = np.isfinite(d)
                rows[n][label] = float(d[ok].mean()) if ok.any() else float("nan")
                rows[n][f"{label}_sem"] = (
                    float(d[ok].std() / np.sqrt(ok.sum())) if ok.any() else float("nan"))

        out[t] = {"fp_inf": fp_inf, "aux_inf": aux_inf, "n_aux": n_aux, "n": rows}
        live = [r for r in rows.values() if np.isfinite(r["headroom"])]
        best = max(live, key=lambda r: r["headroom"]) if live else None
        tag = (f"best reachable {best['headroom']:+.3f}+-{best['headroom_sem']:.3f}"
               if best else "no budget affords the multi-task arm")
        print(f"    {t:<9s} fp_inf {fp_inf:.3f}  aux_inf {aux_inf:.3f}   {tag}",
              flush=True)

    audit.report("[A]", clean_lines=(
        "evaluation rows disjoint from training; exact draw sizes, no duplicates;",
        "no target as its own auxiliary or in its own ceiling features;",
        "every multi-task arm spends no more cells than stl",
    ))
    report["transfer"] = {"targets": list(targets), "aux_targets": list(aux_targets),
                          "n_repeats": n_repeats, "n_aux_max": n_aux_max,
                          "per_target": out}
    _transfer_verdict(out, targets, report)
    return out


def _nanmean(vals):
    v = np.asarray([x for x in vals], dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _transfer_verdict(out, targets, report):
    print(f"\n    {'cells':>6s}{'mols':>6s}{'stl':>9s}{'mt_gp':>9s}{'mt_spread':>11s}"
          f"{'mt_eq':>9s}{'mt_free':>9s}{'headroom(mt_gp)':>19s}{'ceiling@n':>12s}")
    print("    " + "-" * 89)
    summary = {}
    for n in N_GRID:
        r_t = [out[t]["n"][n] for t in targets]
        cells = {k: _nanmean([r[k]["mean"] for r in r_t])
                 for k in ("stl", "mt_gp", "mt_spread", "mt_eq", "mt_free")}
        m = int(np.median([r["m_molecules_mt"] for r in r_t]))
        # Across targets the per-target headrooms are the sample; their spread
        # is what a claim about "targets in general" has to clear.
        def _sem(key, _r=r_t):
            """Uncertainty on a mean headroom, from both sources it has.

            The spread *across* targets is the right scale for a claim about
            targets in general -- but with one target it is the standard
            deviation of a single number, which is 0, and a zero SEM makes any
            headroom at all clear a 3-SEM test. A probe run on one target
            therefore reported "H1 holds here" off +0.018. Fall back on the
            within-target spread across repeats, and when both are available
            take the larger, which is the conservative reading.
            """
            v = np.asarray([x[key] for x in _r], dtype=float)
            ok = np.isfinite(v)
            if not ok.any():
                return float("nan")
            across = (float(v[ok].std() / np.sqrt(ok.sum())) if ok.sum() > 1
                      else 0.0)
            w = np.asarray([x.get(f"{key}_sem", np.nan) for x in _r], dtype=float)
            okw = np.isfinite(w)
            within = (float(np.sqrt((w[okw] ** 2).sum()) / okw.sum()) if okw.any()
                      else 0.0)
            out = max(across, within)
            return out if out > 0 else float("nan")

        hs = np.asarray([r["headroom"] for r in r_t], dtype=float)
        ok = np.isfinite(hs)
        sem = _sem("headroom")
        summary[n] = {
            **cells, "m_molecules_mt": m,
            "headroom": cells["mt_gp"] - cells["stl"],
            "headroom_spread": cells["mt_spread"] - cells["stl"],
            "headroom_2stage": cells["mt_eq"] - cells["stl"],
            "headroom_sem_mean": sem,
            "headroom_spread_sem_mean": _sem("headroom_spread"),
            "ceiling_at_n": _nanmean([r["ceiling_at_n"] for r in r_t]),
            "rho_picked_median": _nanmean([r["rho_picked_median"] for r in r_t]),
            "n_targets_positive": int((hs[ok] > 0).sum()),
            "n_targets_positive_spread": int(
                (np.asarray([r["headroom_spread"] for r in r_t], float) > 0).sum()),
        }
        def cell(k, w=9, _c=cells):
            return f"{'n/a':>{w}s}" if not np.isfinite(_c[k]) else f"{_c[k]:>{w}.3f}"

        hd = (f"{'n/a':>19s}" if not np.isfinite(sem)
              else f"{summary[n]['headroom']:>+13.3f}+-{sem:<4.3f}")
        print(f"    {n:>6d}{m:>6d}{cell('stl')}{cell('mt_gp')}{cell('mt_spread', 11)}"
              f"{cell('mt_eq')}{cell('mt_free')}{hd}"
              f"{summary[n]['ceiling_at_n']:>+12.3f}")

    rp = [summary[n]["rho_picked_median"] for n in N_GRID
          if np.isfinite(summary[n]["rho_picked_median"])]
    if rp:
        print(f"\n    task correlation the ICM arm selected by marginal likelihood: "
              f"{min(rp):.2f} .. {max(rp):.2f}")

    fp_inf = float(np.mean([out[t]["fp_inf"] for t in targets]))
    aux_inf = float(np.mean([out[t]["aux_inf"] for t in targets]))
    print(f"\n    ceilings with budget removed (mean over targets):")
    print(f"      fp_inf  {fp_inf:.3f}   a Tanimoto GP on 80% of the pool")
    print(f"      aux_inf {aux_inf:.3f}   ridge from the other 57 targets, out-of-fold")
    print(f"      gap     {aux_inf - fp_inf:+.3f}   what cross-target structure is worth")
    print("                     once neither side is budget-limited")

    live = [(n, r) for n, r in summary.items() if np.isfinite(r["headroom"])]
    print("\n    verdict:")
    if aux_inf <= fp_inf:
        print("      aux_inf <= fp_inf: the other targets carry nothing the fingerprint")
        print("      does not already carry. H1 has no ceiling on this benchmark; stop.")
        verdict = "no_ceiling"
    elif not live:
        print("      no budget in the grid affords a single multi-task molecule")
        print(f"      ({1 + out[targets[0]]['n_aux']} cells per molecule). Shrink the")
        print("      auxiliary set or raise the budget before reading anything here.")
        verdict = "unaffordable"
    else:
        n_best, best = max(live, key=lambda kv: kv[1]["headroom"])
        n_worst, worst = min(live, key=lambda kv: kv[1]["headroom"])
        tol = 3 * best["headroom_sem_mean"]
        print(f"      ceiling at budget: {max(r['ceiling_at_n'] for _, r in live):+.3f}"
              "   (free exact auxiliary labels)")
        spread = max(r["headroom_spread"] for _, r in live)
        two = max(r["headroom_2stage"] for _, r in live)
        print(f"      multi-task GP, blocked  : {best['headroom']:+.3f}"
              f"+-{best['headroom_sem_mean']:.3f}  (best, at {n_best} cells,"
              f" {best['m_molecules_mt']} molecules)")
        print(f"      multi-task GP, spread   : {spread:+.3f}"
              "   (no molecule measured twice)")
        print(f"      two-stage ridge         : {two:+.3f}")
        if np.isfinite(spread) and spread > best["headroom"] + 3 * best["headroom_sem_mean"]:
            print("      -- the spread design beats the blocked one: what the auxiliary")
            print("         cells buy is coverage of new chemistry, not a second reading")
            print("         of the same molecule. H1 is judged on that row.")
        # H1 is a claim about multi-task modelling, not about one allocation of
        # the cells, so it is judged on whichever design did better.
        if np.isfinite(spread) and spread > best["headroom"]:
            n_spread, r_spread = max(live, key=lambda kv: kv[1]["headroom_spread"])
            best = {**r_spread, "headroom": r_spread["headroom_spread"],
                    "headroom_sem_mean": r_spread["headroom_spread_sem_mean"],
                    "n_targets_positive": r_spread["n_targets_positive_spread"]}
            n_best = n_spread
            tol = 3 * best["headroom_sem_mean"]
        if best["headroom"] > tol:
            print(f"      H1 holds here: {best['n_targets_positive']}/{len(targets)} "
                  "targets improve. Spreading the same assay")
            print("      budget across coupled targets beats concentrating it.")
            verdict = "live"
        elif best["headroom"] < -tol:
            print(f"      H1 fails at equal cost: the better design still loses "
                  f"{-best['headroom']:.3f} at its")
            print(f"      best budget ({n_best} cells), "
                  f"{best['n_targets_positive']}/{len(targets)} targets improving. The")
            print("      free-label ceiling above is large, so the cross-target structure")
            print("      is real; what fails is converting it once the auxiliary cells")
            print("      are paid for out of the same budget. Any later multi-task result")
            print("      that does not hold cells fixed is measuring that substitution,")
            print("      not transfer.")
            verdict = "negative_at_equal_cost"
        else:
            print("      H1 is undecided here: the multi-task arm neither beats nor")
            print("      loses to the independent one outside the spread across targets.")
            print("      The ceiling says the structure exists; converting it at equal")
            print("      cost is the open problem.")
            verdict = "inside_noise"
        print(f"      (blocked design, worst budget: {worst['headroom']:+.3f} "
              f"at {n_worst} cells)")

    report["transfer"]["summary"] = summary
    report["transfer"]["fp_inf_mean"] = fp_inf
    report["transfer"]["aux_inf_mean"] = aux_inf
    report["transfer"]["verdict"] = verdict


# ---- check C: the asymmetric case, which is the one that occurs -----------


H_GRID = (0, 25, 50, 100, 200)
"""Historical observations per auxiliary target. ``0`` is the single-task GP."""


def _distant_split(K, seed, frac=0.5):
    """Split the pool into two chemically separated halves.

    DOCKSTRING hands every task the same uniform ZINC sample, so an auxiliary
    observation lands right next to the molecules the primary is scored on.
    ChEMBL does not: each target's library is a chemotype someone assembled, and
    ``scripts/cold_target_chembl.py`` measures the auxiliaries reaching only
    0.53 of the primary's own nearest-neighbour similarity. That difference is
    confounded with task correlation in any comparison between the two datasets.

    Splitting by similarity to a random seed molecule reproduces the ChEMBL
    geometry inside DOCKSTRING, where the task correlation is held fixed, so the
    two explanations can be told apart.
    """
    g = np.random.default_rng(seed)
    anchor = int(g.integers(len(K)))
    order = np.argsort(-K[anchor])
    cut = int(frac * len(order))
    return order[:cut], order[cut:]


def check_cold_target(sub, K, report, *, targets, aux_targets, n_grid, h_grid,
                      n_repeats, seed, aux_region="same", holdout=0.0,
                      paired=False, audit=None):
    """A cold target with history on its relatives -- not a budget to split.

    Check A asked whether spreading a *fresh* budget across coupled targets
    beats concentrating it, and the answer was no at every budget: at the task
    correlation DOCKSTRING has, an auxiliary cell is nearly a duplicate of a
    primary one, so paying for it loses the molecule it displaced.

    That is not the situation a real campaign starts in.  A new target arrives
    with nothing, and its relatives arrive with years of accumulated assays that
    cost this project's budget nothing, because they were already paid for.  The
    question is then not how to split ``n`` but how much history ``H`` on the
    auxiliaries has to exist before modelling them jointly beats ignoring them.

    ``H`` is free here and the primary's ``n`` is not, which is deliberate and is
    the whole difference from check A.  Nothing in this check is a statement
    about spending a budget; it is a statement about when to *use* data you
    already have.  The historical molecules are disjoint from the primary's
    training rows and from the evaluation rows, so no arm sees a test label.

    Two flags control how tightly the ``H = 0`` control is bound to the arms it
    is differenced against.  Both default off, because the tables in FINDINGS
    sections 16 and 17 were produced without them and have to stay reproducible;
    ``--audit`` reports which of the two invariants a given run satisfies.

    ``holdout``
        Reserve one evaluation set per (target, repeat) that *every* ``H`` is
        scored on.  With ``holdout = 0`` the evaluation set is whatever the draw
        did not take, so the ``H = 200`` arm is scored on ``200 x |aux|`` fewer
        rows than the ``H = 0`` baseline it is differenced against.  Those rows
        are random and the mean should not move -- but "should not" is a claim.
        Mirrors ``--holdout`` in ``dense_block_h1.py``, where the question was
        first raised.
    ``paired``
        Draw the primary's ``n`` training rows from a stream that does not
        depend on ``H``, so the control and every arm are fitted on the *same*
        primary observations.  Without it the gain is an unbiased but unpaired
        difference carrying the split-to-split variance twice over; at ``n =
        10`` that variance dominates, and it is the variance a null has to have
        removed before it can be called a null.
    """
    audit = audit or Audit()
    all_cols = sub[list(ds.TARGETS)].to_numpy(dtype=float)
    n_pool = len(sub)
    print(f"\n[C] cold target with historical auxiliaries  "
          f"({len(targets)} targets, aux={'+'.join(aux_targets)}, "
          f"{n_repeats} repeats)")
    print(f"    H is free -- already-collected data, not budget. Contrast check A.")
    print("    pairing: "
          + (f"one evaluation set per repeat shared by every H (holdout {holdout:g})"
             if holdout > 0 else "evaluation set varies with H")
          + (", shared primary training rows" if paired
             else ", primary training rows drawn independently per H"))
    print(f"    auxiliary region: {aux_region}"
          + ("  (same chemical space as the primary, as DOCKSTRING gives it)"
             if aux_region == "same"
             else "  (a separated half of the pool)"))
    if aux_region == "distant":
        # State how hard the separation actually is. The point of this mode is
        # to imitate the geometry of real per-target libraries, and
        # cold_target_chembl.py measures those at a reach ratio of 0.53; a mode
        # that only reaches 0.8 has not reproduced them, and the table below
        # cannot then be used to rule chemical reach out as the explanation.
        rr = []
        for s in range(3):
            home, away = _distant_split(K, seed + 977 * s)
            gg = np.random.default_rng(s)
            pick = home[gg.choice(len(home), size=min(300, len(home)),
                                  replace=False)]
            own = K[np.ix_(pick, home)].copy()
            for row, j in enumerate(pick):
                own[row, np.flatnonzero(home == j)] = -1.0
            rr.append(float(K[np.ix_(pick, away)].max(axis=1).mean()
                            / own.max(axis=1).mean()))
        ratio = float(np.mean(rr))
        print(f"    measured reach ratio {ratio:.2f} (1.00 = no separation). "
              "Real ChEMBL")
        print("    libraries measure 0.53, so this is a WEAKER separation than "
              "the thing")
        print("    it imitates; it cannot on its own acquit chemical reach.")
        report.setdefault("cold_target_meta", {})["reach_ratio"] = ratio

    out: dict = {}
    for ti, t in enumerate(targets):
        y = -all_cols[:, ds.TARGETS.index(t)]
        aux = [a for a in aux_targets if a != t]
        A = -all_cols[:, [ds.TARGETS.index(a) for a in aux]]
        audit.excludes(t, aux, f"[C] {t} auxiliary set")
        audit.not_relabelled(y, [A[:, j] for j in range(len(aux))], f"[C] {t}")
        rows: dict = {}
        for n in n_grid:
            for h in h_grid:
                acc = []
                for r in range(n_repeats):
                    g = np.random.default_rng(seed + 977 * r + 31 * n + h + 7919 * ti)
                    # `h` is in `g`'s seed, so without `paired` the primary's
                    # training rows differ between the control and every arm it
                    # is differenced against. This second stream omits it.
                    g_prim = (np.random.default_rng(seed + 977 * r + 31 * n
                                                    + 7919 * ti) if paired else g)
                    if aux_region == "same":
                        home, away = np.arange(n_pool), np.arange(n_pool)
                    else:
                        home, away = _distant_split(K, seed + 977 * r)
                    if holdout > 0:
                        # Reserved before anything is drawn, from a stream that
                        # depends on neither `n` nor `h`, so every cell of the
                        # table is scored on the same rows within a repeat.
                        gh = np.random.default_rng(seed + 977 * r + 7919 * ti)
                        perm = gh.permutation(home)
                        n_te = int(holdout * len(home))
                        te_fixed, home_pool = perm[:n_te], perm[n_te:]
                        # Boolean mask rather than setdiff1d: `away` is in
                        # similarity order in the distant mode and sorting it
                        # would silently change which molecules the history
                        # draw lands on.
                        away_pool = away[~np.isin(away, te_fixed)]
                    else:
                        te_fixed, home_pool, away_pool = None, home, away
                    if n > len(home_pool) // 2 or h * len(aux) > len(away_pool) // 2:
                        acc.append(np.nan)
                        continue
                    prim = g_prim.choice(len(home_pool), size=n, replace=False)
                    tr = home_pool[prim]
                    if aux_region == "same":
                        rest = np.setdiff1d(away_pool, tr)
                        pick = g.choice(len(rest), size=h * len(aux), replace=False)
                        hist = rest[pick]
                    else:
                        # home and away are disjoint, so `tr` cannot be redrawn.
                        pick = g.choice(len(away_pool), size=h * len(aux),
                                        replace=False)
                        hist = away_pool[pick]
                    draw = np.concatenate([tr, hist])
                    te = te_fixed if te_fixed is not None else np.setdiff1d(home, draw)
                    audit.cell()
                    audit.exactly(tr, n, f"[C] {t} n={n} H={h} r={r} primary rows")
                    audit.exactly(hist, h * len(aux),
                                  f"[C] {t} n={n} H={h} r={r} history rows")
                    audit.disjoint(tr, hist, "[C] primary training", "free history")
                    audit.disjoint(te, draw, "[C] evaluation", "training/history")
                    if aux_region == "distant":
                        audit.subset(tr, home, "[C] primary rows", "the home half")
                        audit.subset(hist, away, "[C] history rows", "the away half")
                    # The two pairing invariants. These fire on the defaults --
                    # that is what they are for.
                    audit.same_across((t, n, r), h, "[C] evaluation rows", te)
                    audit.same_across((t, n, r), h, "[C] primary training rows", tr)
                    if h == 0:
                        gp = TanimotoGP().fit_gram(K[np.ix_(tr, tr)], y[tr])
                        mu = gp.predict_gram(K[np.ix_(te, tr)], return_std=False)
                    else:
                        idx = [tr] + [draw[n + j * h: n + (j + 1) * h]
                                      for j in range(len(aux))]
                        vals = [y[tr]] + [A[idx[j + 1], j] for j in range(len(aux))]
                        mu, _ = _icm_fit_predict(K, te, idx, vals)
                    acc.append(_rho(y[te], mu))
                v = np.asarray(acc, dtype=float)
                ok = np.isfinite(v)
                rows[(n, h)] = {
                    "mean": float(v[ok].mean()) if ok.any() else float("nan"),
                    "sem": (float(v[ok].std() / np.sqrt(ok.sum())) if ok.any()
                            else float("nan")),
                }
        out[t] = rows
        print(f"    {t:<9s} done", flush=True)

    print(f"\n    Spearman on held-out rows, mean over {len(targets)} targets")
    print(f"    {'n':>4s}{'H=0 (stl)':>12s}"
          + "".join(f"{'H=' + str(h):>11s}" for h in h_grid if h))
    print("    " + "-" * (16 + 11 * (len(h_grid) - 1)))
    summary = {}
    for n in n_grid:
        base = _nanmean([out[t][(n, 0)]["mean"] for t in targets])
        line, gains = f"    {n:>4d}{base:>12.3f}", {}
        for h in h_grid:
            if not h:
                continue
            v = _nanmean([out[t][(n, h)]["mean"] for t in targets])
            per = np.asarray([out[t][(n, h)]["mean"] - out[t][(n, 0)]["mean"]
                              for t in targets], dtype=float)
            ok = np.isfinite(per)
            gains[h] = {
                "value": v, "gain": v - base,
                "sem": (float(per[ok].std() / np.sqrt(ok.sum())) if ok.any()
                        else float("nan")),
                "n_targets_positive": int((per[ok] > 0).sum()),
            }
            line += f"{v:>11.3f}" if np.isfinite(v) else f"{'n/a':>11s}"
        summary[n] = {"stl": base, "by_h": gains}
        print(line)

    print(f"\n    gain over the single-task GP (+ = joint modelling helps)")
    print(f"    {'n':>4s}" + "".join(f"{'H=' + str(h):>18s}" for h in h_grid if h))
    print("    " + "-" * (4 + 18 * (len(h_grid) - 1)))
    for n in n_grid:
        cells = ""
        for h, g in summary[n]["by_h"].items():
            cells += (f"{'n/a':>18s}" if not np.isfinite(g["gain"])
                      else f"{g['gain']:>+12.3f}+-{g['sem']:<5.3f}")
        print(f"    {n:>4d}{cells}")

    # Break-even: the smallest H whose gain clears three SEM, per budget.
    print("\n    verdict:")
    # Both ways this table can be empty end in a sentence about H1 being closed,
    # which would be a conclusion drawn from nothing: a grid with no non-zero H
    # has no arm to compare against the control, and a grid whose H never fits
    # in the pool produces all-NaN gains. Refuse before the verdict prose.
    if not any(h > 0 for h in h_grid):
        print("      --c-history has no non-zero value: H=0 is the single-task")
        print("      control and there is nothing to compare it against.")
        report["cold_target"] = {"status": "no_arms"}
        return {}
    if not any(np.isfinite(g["gain"])
               for n in n_grid for g in summary[n]["by_h"].values()):
        biggest = max(n_grid) + max(h_grid) * len(aux_targets)
        print(f"      nothing in the grid could be fitted: the largest cell needs")
        print(f"      {biggest} molecules and the pool of {n_pool} left fewer after the")
        print("      draw. Not a result about H1 -- shrink --c-history or raise --pool.")
        report["cold_target"] = {"status": "no_fits"}
        return {}

    any_live = False
    for n in n_grid:
        hit = next((h for h, g in summary[n]["by_h"].items()
                    if np.isfinite(g["gain"]) and g["gain"] > 3 * g["sem"]), None)
        if hit is None:
            print(f"      n={n:<3d} no H in the grid pays.")
        else:
            g = summary[n]["by_h"][hit]
            best_h = max(summary[n]["by_h"],
                         key=lambda k: summary[n]["by_h"][k]["gain"])
            bg = summary[n]["by_h"][best_h]
            print(f"      n={n:<3d} breaks even at H={hit} per auxiliary; "
                  f"best {bg['gain']:+.3f} at H={best_h} "
                  f"({bg['n_targets_positive']}/{len(targets)} targets).")
            any_live = True
    if any_live:
        print("      Transfer pays when the auxiliary data already exists and is not")
        print("      charged to the campaign. That is the H1 worth testing, and it is")
        print("      a different claim from the one check A rejects.")
    else:
        print("      Even free history on the relatives does not beat ignoring them.")
        print("      H1 is closed on this benchmark in both of its forms.")
    audit.report("[C]", clean_lines=(
        "primary and history rows disjoint; evaluation rows disjoint from both;",
        "exact draw sizes, no duplicates, no target as its own auxiliary;",
        "control and arms share the evaluation set and the primary training rows",
    ))
    report["cold_target"] = {"targets": list(targets), "aux_region": aux_region,
                             "h_grid": list(h_grid), "holdout": holdout,
                             "paired": bool(paired),
                             "n_grid": list(n_grid), "n_repeats": n_repeats,
                             "summary": {str(k): v for k, v in summary.items()}}
    return summary


# ---- check B: joint vs separated, on the published objectives -------------


def check_joint_vs_separated(sub, K, report, *, objective, n_repeats, seed, topk,
                             audit=None):
    """Same measured cells, four ways to spend them.

    ``joint`` regresses the scalarised objective, which is what the notebook
    does.  ``separated`` regresses each docking column and composes the
    posterior means through the published formula.

    The two are not comparable without a third arm.  Every published objective
    here carries a ``-10 (1 - QED)`` term, QED is free, and check 4 of
    ``dockstring_audit.py`` measures that term at rho +0.93 against the whole
    objective.  ``separated`` is handed that term; ``joint`` has to learn it
    from fingerprints.  So:

    * ``qed_only``  -- rank by the free term alone, no assays at all.  The floor
                       any arm with access to QED starts from.
    * ``joint_qed`` -- regress only the *assayed* part of the objective and add
                       the free term back.  The fair scalarised arm.

    Read ``separated`` against ``joint_qed``, and both against ``qed_only``.
    """
    audit = audit or Audit()
    obj = ds.OBJECTIVES[objective]
    # An objective with an admission filter is not defined outside it, so the
    # arms compete over the molecules it admits -- scoring the rejects would let
    # an arm earn recall on compounds the objective has already excluded.
    keep = np.flatnonzero(obj.admits(sub))
    sub = sub.iloc[keep].reset_index(drop=True)
    K = K[np.ix_(keep, keep)]
    # A top-k wider than the evaluation set is not a hard top-k: argsort returns
    # every row, every arm "recalls" all of them, and the table fills with
    # numbers that are a function of k and the pool rather than of any method.
    # Fail rather than print that.
    if topk >= len(keep) - max(N_GRID):
        raise SystemExit(
            f"--topk {topk} leaves no room: {len(keep)} molecules admitted, "
            f"{max(N_GRID)} spent on training at the largest budget. "
            f"Use --topk below {len(keep) - max(N_GRID)} or a larger --pool.")
    print(f"\n[B] joint vs separated  (objective={objective}, "
          f"targets={'+'.join(obj.targets)}, {n_repeats} repeats, "
          f"{len(keep)} molecules admitted)")
    y = np.asarray(obj(sub), dtype=float)
    cols = {t: sub[t].to_numpy(dtype=float) for t in obj.targets}
    qed = sub["QED"].to_numpy(dtype=float)
    n_pool = len(sub)

    # Split the objective at the line between what costs an assay and what does
    # not. The objective owns that line, so a variant with a different
    # drug-likeness weight -- or none, with a filter instead -- splits correctly
    # without touching this.
    q_term = np.asarray(obj.free(sub), dtype=float)
    d_term = np.asarray(obj.assayed(sub), dtype=float)
    # `joint_qed` regresses d_term and adds q_term back, and is read against
    # `joint`, which regresses y. That reading needs the split to be exact: if
    # free + assayed did not reconstruct the objective, the two arms would be
    # optimising different things and the comparison would be empty.
    audit.want(np.allclose(q_term + d_term, y),
               "[B] the free and assayed parts do not sum to the objective")
    rho_floor = _rho(y, q_term)
    print(f"    free-descriptor floor: ranking by QED alone scores "
          f"rho {rho_floor:.3f} with zero assays")

    arms = ("qed_only", "joint", "joint_qed", "separated")
    rows = {}
    acc_rho: dict = {n: {a: [] for a in arms} for n in N_GRID}
    acc_rec: dict = {n: {a: [] for a in arms} for n in N_GRID}
    for n in N_GRID:
        for r in range(n_repeats):
            g = np.random.default_rng(seed + 31 * r + n)
            tr = g.choice(n_pool, size=n, replace=False)
            te = np.setdiff1d(np.arange(n_pool), tr)
            audit.cell()
            audit.exactly(tr, n, f"[B] n={n} r={r} training rows")
            audit.disjoint(te, tr, "[B] evaluation", "training")
            # All four arms are differenced against each other, so they have to
            # be scored on the same rows; `qed_only` in particular must be the
            # free ranking *of the evaluation set*, not of the whole pool.
            for _arm in arms:
                audit.same_across((n, r), _arm, "[B] evaluation rows", te)
            Ktr = K[np.ix_(tr, tr)]
            Kte = K[np.ix_(te, tr)]

            gp = TanimotoGP().fit_gram(Ktr, y[tr])
            gq = TanimotoGP().fit_gram(Ktr, d_term[tr])
            var = {"QED": qed[te]}
            for t in obj.targets:
                gt = TanimotoGP().fit_gram(Ktr, cols[t][tr])
                var[t] = gt.predict_gram(Kte, return_std=False)

            pred = {
                "qed_only": q_term[te],
                "joint": gp.predict_gram(Kte, return_std=False),
                "joint_qed": gq.predict_gram(Kte, return_std=False) + q_term[te],
                "separated": np.asarray(obj(var), dtype=float),
            }
            truth = y[te]
            true_top = set(np.argsort(truth)[-topk:].tolist())
            # Recall is over `te`, so the success set must live there too; a
            # top-k taken over the pool would be partly unreachable and every
            # arm's recall would be capped below 1 for a reason unrelated to it.
            audit.want(len(true_top) == topk and topk <= len(te),
                       f"[B] n={n} r={r}: top-{topk} does not fit in "
                       f"{len(te)} evaluation rows")
            # (`qed_only` is assigned `q_term[te]` directly a few lines above,
            # so checking it here would assert an identity, not an invariant.)
            for a, p in pred.items():
                acc_rho[n][a].append(_rho(truth, p))
                acc_rec[n][a].append(
                    len(true_top & set(np.argsort(p)[-topk:].tolist())) / topk)

    print(f"\n    Spearman over all held-out rows")
    print(f"    {'n':>4s}" + "".join(f"{a:>12s}" for a in arms))
    print("    " + "-" * 52)
    for n in N_GRID:
        print(f"    {n:>4d}" + "".join(
            f"{np.mean(acc_rho[n][a]):>12.3f}" for a in arms))

    print(f"\n    top-{topk} recall -- the thing a campaign is actually buying")
    print(f"    {'n':>4s}" + "".join(f"{a:>12s}" for a in arms)
          + f"{'sep - joint_qed':>19s}")
    print("    " + "-" * 71)
    for n in N_GRID:
        d = np.asarray(acc_rec[n]["separated"]) - np.asarray(acc_rec[n]["joint_qed"])
        rows[n] = {
            **{f"rho_{a}": float(np.mean(acc_rho[n][a])) for a in arms},
            **{f"recall_{a}": float(np.mean(acc_rec[n][a])) for a in arms},
            "delta": float(np.mean(d)),
            "delta_sem": float(np.std(d) / np.sqrt(n_repeats)),
            "separated_wins": int((d > 0).sum()),
        }
        print(f"    {n:>4d}" + "".join(f"{np.mean(acc_rec[n][a]):>11.1%} " for a in arms)
              + f"{np.mean(d):>+13.1%}+-{rows[n]['delta_sem']:<5.1%}")

    floor = float(np.mean(acc_rec[N_GRID[0]]["qed_only"]))
    best_rec = max(max(r[f"recall_{a}"] for a in arms if a != "qed_only")
                   for r in rows.values())
    best = max(rows.values(), key=lambda r: r["delta"])
    n_best = [n for n, r in rows.items() if r is best][0]
    has_free = float(np.ptp(q_term)) > 0
    print("\n    verdict:")
    if has_free:
        print(f"      the free descriptor alone recovers {floor:.1%} of the "
              f"top-{topk}, and")
        print(f"      rho {rho_floor:.3f} of the whole ranking. A global rank "
              "correlation")
        print("      therefore cannot tell these arms apart; top-k can.")
    else:
        print(f"      this objective has no free term, so the {floor:.1%} floor is "
              "chance")
        print(f"      ({topk} of {n_pool} admitted molecules) and `joint_qed` is "
              "`joint`.")
        print("      Everything above the floor was paid for.")
    print(f"      best assayed arm: {best_rec:.1%} recall, {best_rec - floor:+.1%} over"
          " the floor.")
    if best["delta"] > 3 * best["delta_sem"]:
        print(f"      separated beats joint_qed by {best['delta']:+.1%} at n={n_best} "
              f"({best['separated_wins']}/{n_repeats} repeats):")
        print("      modelling the docking columns and composing is worth it, and the")
        print("      saturating term is the plausible reason.")
    else:
        print(f"      separated vs joint_qed: {best['delta']:+.1%}"
              f"+-{best['delta_sem']:.1%} at n={n_best} -- no separation. Once both")
        print("      arms are given the free term, a per-target surrogate has to")
        print("      justify itself on allocation rather than on accuracy.")
    audit.report("[B]", clean_lines=(
        "evaluation rows disjoint from training; exact draw sizes, no duplicates;",
        "all four arms scored on the same rows; the top-k fits in them;",
        "free + assayed reconstruct the objective, and qed_only uses the free part only",
    ))
    report["joint_vs_separated"] = {"objective": objective, "n_repeats": n_repeats,
                                    "topk": topk, "qed_only_rho": rho_floor,
                                    "qed_only_recall": floor, "points": rows}
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(ds.DATASET))
    p.add_argument("--pool", type=int, default=2000,
                   help="shared molecule pool; one Tanimoto gram serves every arm")
    p.add_argument("--targets", nargs="+", default=None,
                   help="primary targets; default: the kinase subset "
                        "(molbo.dockstring.KINASES)")
    p.add_argument("--aux-targets", nargs="+", default=list(AUX_DEFAULT),
                   help="auxiliary targets the multi-task arm also measures. Fixed "
                        "a priori, never chosen by correlation with the primary -- "
                        "picking them from the label matrix would be the benchmark "
                        "leaking into the method")
    p.add_argument("--objective", default="jak2", choices=sorted(ds.OBJECTIVES))
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--folds", type=int, default=5, help="folds for the aux_inf ceiling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=50,
                   help="top-k recall reported by check B; the campaign-relevant "
                        "metric, since a global rank correlation on these objectives "
                        "is dominated by their free QED term")
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-b", action="store_true")
    p.add_argument("--check-c", action="store_true",
                   help="run the cold-target check: auxiliaries carry free "
                        "historical data instead of being paid for out of the "
                        "campaign budget")
    p.add_argument("--c-targets", nargs="+",
                   default=["JAK2", "CDK2", "EGFR", "MET"],
                   help="primary targets for check C; fewer than check A because "
                        "each fit is over n + 3H observations")
    p.add_argument("--c-budgets", type=int, nargs="+", default=[20, 80])
    p.add_argument("--c-history", type=int, nargs="+", default=list(H_GRID))
    p.add_argument("--c-repeats", type=int, default=10)
    p.add_argument("--c-aux-region", choices=("same", "distant"), default="same",
                   help="where the historical auxiliary compounds come from. "
                        "`distant` puts them in a chemically separated half of "
                        "the pool, reproducing the geometry of real per-target "
                        "libraries while holding the task correlation fixed")
    p.add_argument("--audit", action="store_true",
                   help="check at runtime the invariants these tables rest on: "
                        "evaluation rows never trained on, exact draw sizes, no "
                        "duplicate rows, no target used as its own auxiliary, no "
                        "auxiliary column equal to the primary's values, equal "
                        "cell budgets across arms, and -- in check C -- whether "
                        "the H=0 control and the H>0 arms actually shared a "
                        "split. Costs seconds; run it before quoting anything")
    p.add_argument("--c-holdout", type=float, default=0.0,
                   help="check C: reserve this fraction of the pool as one "
                        "evaluation set that every H is scored on. 0 -- the "
                        "default, and what sections 16-17 used -- lets the "
                        "evaluation set shrink as H grows")
    p.add_argument("--c-paired", action="store_true",
                   help="check C: fit the H=0 control and every H>0 arm on the "
                        "SAME primary training rows, so the reported gain is a "
                        "paired difference. Off by default to keep the stored "
                        "tables reproducible")
    p.add_argument("--out", default="results/transfer_headroom.json")
    args = p.parse_args(argv)

    targets = tuple(args.targets) if args.targets else ds.KINASES
    aux_targets = tuple(args.aux_targets)
    unknown = [t for t in targets + aux_targets if t not in ds.TARGETS]
    if unknown:
        raise SystemExit(f"not DOCKSTRING targets: {unknown}")
    if len(aux_targets) < 1:
        raise SystemExit("the multi-task arm needs at least one auxiliary target")

    t0 = time.time()
    print(f"loading {args.dataset} ...", flush=True)
    df = ds.load(args.dataset)
    rng = np.random.default_rng(args.seed)
    # Complete rows only. 339 of DOCKSTRING's 15M cells are missing, scattered
    # over every column; a NaN anywhere in a row would silently poison the ridge
    # arms, and dropping ~0.1% of a 260k pool costs nothing.
    cols = list(ds.TARGETS)
    full = df[np.isfinite(df[cols].to_numpy(dtype=float)).all(axis=1)]
    full = full[np.isfinite(full.QED.to_numpy(dtype=float))].reset_index(drop=True)
    sub = full.iloc[rng.choice(len(full), size=args.pool, replace=False)]
    sub = sub.reset_index(drop=True)
    print(f"loaded {len(df):,} rows, {len(full):,} complete, pool {len(sub):,}, "
          f"{time.time() - t0:.0f}s")

    t1 = time.time()
    X = fingerprint_matrix(sub.smiles.tolist())
    K = tanimoto_gram(X, X)
    print(f"fingerprints + gram in {time.time() - t1:.0f}s")

    report: dict = {"dataset": args.dataset, "pool": len(sub), "seed": args.seed}
    # One auditor per check, so a violation is attributed to the check that
    # produced it rather than to the run.
    audits = {k: Audit(enabled=args.audit) for k in ("A", "B", "C")}
    if not args.skip_a:
        check_transfer(sub, K, report, targets=targets, aux_targets=aux_targets,
                       n_repeats=args.repeats, seed=args.seed, folds=args.folds,
                       audit=audits["A"])
    if args.check_c:
        unknown_c = [t for t in args.c_targets if t not in ds.TARGETS]
        if unknown_c:
            raise SystemExit(f"not DOCKSTRING targets: {unknown_c}")
        check_cold_target(sub, K, report, targets=tuple(args.c_targets),
                          aux_targets=aux_targets, n_grid=tuple(args.c_budgets),
                          h_grid=tuple(args.c_history),
                          n_repeats=args.c_repeats, seed=args.seed,
                          aux_region=args.c_aux_region, holdout=args.c_holdout,
                          paired=args.c_paired, audit=audits["C"])
    if not args.skip_b:
        check_joint_vs_separated(sub, K, report, objective=args.objective, topk=args.topk,
                                 n_repeats=args.repeats, seed=args.seed,
                                 audit=audits["B"])
    # The audit result travels in the JSON with the numbers it vouches for. A
    # table whose provenance says `clean: false` is not quotable, and that has
    # to be readable from the artefact rather than from a lost terminal.
    report["audit"] = {k: a.to_json() for k, a in audits.items() if a.cells or a.checks}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
