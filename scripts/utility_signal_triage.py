"""Is there anything to predict? A go/no-go before building a utility estimator.

    python scripts/utility_signal_triage.py

The proposed extension is a prospective estimator of

    U(T, A, b) = E[ recall_MT(T, A, b) - recall_STL(T, b) | data known before
                    the campaign ],

validated leave-one-family-out. Before any of that is worth building, three
questions have to be answered from the campaigns already on disk, because each
can kill the project on its own and none needs a new run.

**1. Is the target-level benefit a stable quantity, or is it noise?**
No feature can predict noise. The ceiling on any estimator's R^2 is the
*reliability* of the thing being predicted: split the 32 repeats of a cell in
half, average each half, and ask how well one half predicts the other across
cells (Spearman-Brown corrected to full length). Near zero means the per-target
benefit does not exist as a stable quantity at this repeat count, and the only
honest estimator is a constant.

**2. Does family identity explain everything?**
Leave-one-family-out withholds the family. If essentially all the variance in
the benefit sits *between* families and none *within*, then an estimator that
has never seen the held-out family has nothing to lock onto, and the validation
the advisor proposes fails by construction rather than by modelling choice. The
decomposition below is an exact orthogonal ANOVA on the balanced 6 x 4 x 2 x 2
grid of (family, target, budget rank, history design).

**3. Is the benefit heterogeneous enough for prediction to be worth anything?**
If auxiliary data help in every cell, "use it" is a complete decision rule and a
calibrated estimator adds nothing a practitioner would act on. The cells where
the benefit is negative or near zero are what the estimator exists to find, and
there have to be some.

Two baselines are reported against the leave-one-family-out split, both of which
any proposed estimator must beat out of fold:

``grand``   predict the mean benefit over the five training families.
``known``   predict the mean for this budget rank and history design -- the two
            inputs a researcher genuinely has before screening.

and two that are *not* attainable at deployment, as reference points:

``family``  the held-out family's own mean. The gap from ``grand`` to ``family``
            is the whole budget any feature set is competing for.
``cell``    each cell's own mean -- the sampling floor.

Reads the 24 measured artefacts of the six-family experiment (`win32_*`
withheld, `winleak_*` uniform), 3,072 campaigns per arm. Writes
`results/utility_signal_triage.json`.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DESIGNS = {"withheld": "win32_{fam}_b{b}.json",
           "uniform": "winleak_{fam}_b{b}.json"}
# Budget rank, not absolute budget: budgets are family-specific (a fraction of
# the pool), so "low" and "high" are the labels that compare across families.
FAMILIES = {"cyp": (74, 111), "dpp": (66, 99), "hdac": (83, 125),
            "jak": (200, 300), "mmp": (55, 82), "pi3k": (130, 195)}
ARMS = ("mt", "pooled")


def load_cells(arm):
    """Per-campaign benefit of `arm` over `stl`, as [family, target, budget, design].

    The recall vector in an artefact is target-major -- the campaign loop runs
    ``for ti, t in enumerate(targets): for r in range(repeats)`` -- so campaign
    ``32*ti + r`` belongs to the ``ti``-th target of ``provenance["targets"]``.
    That is asserted against the artefact's own ``per_target`` means rather than
    trusted, because every number below depends on it.
    """
    fams = list(FAMILIES)
    out = np.full((len(fams), 4, 2, len(DESIGNS)), np.nan)       # cell means
    rep = np.full((len(fams), 4, 2, len(DESIGNS), 32), np.nan)   # per repeat
    names = {}
    for fi, fam in enumerate(fams):
        for bi, budget in enumerate(FAMILIES[fam]):
            for di, (design, pat) in enumerate(DESIGNS.items()):
                art = json.loads((RESULTS / pat.format(fam=fam, b=budget))
                                 .read_text(encoding="utf-8"))
                # `win32_*` predates the provenance block; `per_target` is
                # written in the campaign loop's own target order, so it is the
                # same list. The assertion below checks that either way.
                targets = ((art.get("provenance") or {}).get("targets")
                           or list(art["per_target"]))
                a = np.asarray(art["arms"][arm]["recall"], float)
                s = np.asarray(art["arms"]["stl"]["recall"], float)
                assert len(a) == 32 * len(targets) == 128, len(a)
                for ti, t in enumerate(targets):
                    sl = slice(32 * ti, 32 * (ti + 1))
                    pt = (art.get("per_target") or {}).get(t)
                    if pt is not None and f"{arm}_recall" in pt:
                        assert abs(a[sl].mean() - pt[f"{arm}_recall"]) < 1e-9, (
                            f"{fam} b{budget} {design}: campaign block {ti} does "
                            f"not match per_target[{t}] -- the target-major "
                            "assumption is wrong")
                        assert abs(s[sl].mean() - pt["stl_recall"]) < 1e-9
                    d = (a[sl] - s[sl]) * 100.0
                    rep[fi, ti, bi, di] = d
                    out[fi, ti, bi, di] = d.mean()
                    names[(fi, ti)] = t
    assert not np.isnan(out).any()
    return out, rep, names


def split_half(rep, rng, draws=200):
    """Reliability of a cell mean: how well half the repeats predict the other.

    Spearman-Brown corrects the half-length correlation up to the reliability of
    the full 32-repeat mean, which is the quantity an estimator would be fitted
    against. Averaged over random splits so the answer does not rest on one.
    """
    flat = rep.reshape(-1, rep.shape[-1])
    n = flat.shape[-1]
    rs = []
    for _ in range(draws):
        perm = rng.permutation(n)
        a = flat[:, perm[: n // 2]].mean(1)
        b = flat[:, perm[n // 2:]].mean(1)
        rs.append(np.corrcoef(a, b)[0, 1])
    r = float(np.mean(rs))
    return {"half_r": r, "spearman_brown": float(2 * r / (1 + r)),
            "draws": int(draws)}


def anova(A):
    """Exact orthogonal variance decomposition of a balanced factorial array.

    Axes are (family, target, budget, design). The effect of a factor set is its
    cell mean with every lower-order effect it contains subtracted off
    (inclusion-exclusion). The design is balanced, so the terms are orthogonal
    and their sums of squares add to the total exactly -- which is asserted
    rather than assumed.

    `target` is nested in `family`: a bare target index means nothing across
    families. The nested term is therefore the sum of the `target` and
    `family x target` terms, which `group_terms` takes together.
    """
    axes = {"family": 0, "target": 1, "budget": 2, "design": 3}
    eff, ss = {}, {}
    for size in range(len(axes) + 1):
        for S in itertools.combinations(axes, size):
            drop = tuple(axes[a] for a in axes if a not in S)
            m = A.mean(axis=drop, keepdims=True)
            for T, e in eff.items():
                if set(T) <= set(S):
                    m = m - e
            eff[S] = m
            ss[S] = float((np.broadcast_to(m, A.shape) ** 2).sum())
    total = float(((A - A.mean()) ** 2).sum())
    got = sum(v for k, v in ss.items() if k)
    assert abs(got - total) < 1e-6 * max(1.0, total), (got, total)
    return ss, total


def group_terms(ss, total):
    """The decomposition, grouped by what a deployed estimator could know.

    `budget` and `design` are inputs the researcher supplies, so those terms are
    free. Every term containing `family` is unavailable leave-one-family-out and
    has to be recovered from features. `target within family` is the term that
    separates a feature set doing real work from one that has only learned to
    recognise which family it is looking at.
    """
    keys = [k for k in ss if k]

    def pct(ks):
        return 100.0 * sum(ss[k] for k in ks) / total

    known = [k for k in keys if "family" not in k and "target" not in k]
    fam_only = [("family",)]
    # nested target effect, pooled over the conditions it does not interact with
    tgt_in_fam = [k for k in keys if "target" in k
                  and not ({"budget", "design"} & set(k))]
    fam_x_cond = [k for k in keys if "family" in k and "target" not in k
                  and ({"budget", "design"} & set(k))]
    named = set(known) | set(fam_only) | set(tgt_in_fam) | set(fam_x_cond)
    rest = [k for k in keys if k not in named]
    return {"known at deploy (budget, design)": pct(known),
            "family": pct(fam_only),
            "target within family": pct(tgt_in_fam),
            "family x (budget/design)": pct(fam_x_cond),
            "higher-order interactions": pct(rest)}


def lofo(A, rep):
    """Leave-one-family-out error of the baselines any estimator must beat."""
    nf = A.shape[0]
    noise = float(np.mean(rep.var(axis=-1, ddof=1) / rep.shape[-1]))
    denom = float(((A - A.mean()) ** 2).mean())
    out = {}
    for name in ("grand", "known", "family", "cell"):
        err, pr = [], []
        for f in range(nf):
            tr = np.delete(A, f, axis=0)
            if name == "grand":
                p = np.full(A[f].shape, tr.mean())
            elif name == "known":
                p = np.broadcast_to(tr.mean(axis=(0, 1)), A[f].shape)
            elif name == "family":
                p = np.full(A[f].shape, A[f].mean())   # oracle, not deployable
            else:
                p = A[f]                               # sampling floor
            err.append(((A[f] - p) ** 2).ravel())
            pr.append(np.asarray(p).ravel())
        e = np.concatenate(err)
        out[name] = {
            "rmse": float(np.sqrt(e.mean())),
            "r2_vs_grand_mean": float(1 - e.mean() / denom),
            "sign_accuracy": float(np.mean(np.sign(np.concatenate(pr))
                                           == np.sign(A.ravel()))),
        }
    out["always_positive"] = {"sign_accuracy": float(np.mean(A > 0))}
    out["sampling_noise_var_of_cell_mean"] = noise
    out["cell_mean_var"] = float(A.var())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=RESULTS / "utility_signal_triage.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    report = {"arms": {}}
    for arm in ARMS:
        A, rep, names = load_cells(arm)
        report["n_campaigns_per_arm"] = int(rep.size)
        ss, total = anova(A)
        rel = split_half(rep, rng)
        base = lofo(A, rep)
        groups = group_terms(ss, total)
        real = 100 * (1 - base["sampling_noise_var_of_cell_mean"]
                      / base["cell_mean_var"])

        print(f"\n{'=' * 74}\n{arm} - stl, recall points   "
              f"({A.size} cells x {rep.shape[-1]} repeats = {rep.size} campaigns)"
              f"\n{'=' * 74}")
        print(f"  overall {A.mean():+.2f} pts | cell range "
              f"[{A.min():+.2f}, {A.max():+.2f}] | sd across cells "
              f"{A.std(ddof=1):.2f}")
        neg = int((A < 0).sum())
        print(f"  cells where auxiliary data HURT: {neg}/{A.size} "
              f"({100 * neg / A.size:.0f}%) | "
              f"|benefit| < 2 pts: {int((np.abs(A) < 2).sum())}/{A.size}")

        print("\n  [1] reliability of a cell mean -- the ceiling on any R^2")
        print(f"      split-half r {rel['half_r']:.3f}  ->  Spearman-Brown "
              f"(32 repeats) {rel['spearman_brown']:.3f}")
        print(f"      sampling variance of a cell mean {base['sampling_noise_var_of_cell_mean']:.2f}"
              f"  vs between-cell variance {base['cell_mean_var']:.2f}"
              f"  ->  {real:.0f}% of the spread is real")

        print("\n  [2] where the variance lives")
        for k, v in groups.items():
            print(f"      {k:36s} {v:5.1f}%")

        print("\n  [3] leave-one-family-out baselines")
        print(f"      {'predictor':10s} {'RMSE':>7s} {'R2':>8s} {'sign acc':>9s}")
        for k in ("grand", "known", "family", "cell"):
            b = base[k]
            print(f"      {k:10s} {b['rmse']:7.2f} {b['r2_vs_grand_mean']:8.3f} "
                  f"{b['sign_accuracy']:8.0%}")
        print(f"      {'always +':10s} {'-':>7s} {'-':>8s} "
              f"{base['always_positive']['sign_accuracy']:8.0%}")

        report["arms"][arm] = {
            "cell_mean": float(A.mean()), "cell_sd": float(A.std(ddof=1)),
            "cell_min": float(A.min()), "cell_max": float(A.max()),
            "n_cells": int(A.size), "n_negative_cells": neg,
            "pct_of_spread_that_is_real": float(real),
            "reliability": rel, "variance_groups": groups, "baselines": base,
            "cells": {f"{fam}|{names[(fi, ti)]}|b{bi}|{d}":
                      float(A[fi, ti, bi, di])
                      for fi, fam in enumerate(FAMILIES)
                      for ti in range(4) for bi in range(2)
                      for di, d in enumerate(DESIGNS)},
        }

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
