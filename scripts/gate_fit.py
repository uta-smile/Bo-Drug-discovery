"""Can an abstention gate be LEARNED, and does deploying it survive a held-out test?

    python scripts/gate_fit.py --data results/selective_headroom_all.json

Sections 12-13 establish that abstention is worth something (ceiling +0.0117 over
a free bandit) and that no *hand-picked* signal reaches the quality needed. Both
halves share a weakness: **nothing was ever trained as a gate.** Every signal
tested was a by-product of a ranking model, read as if it were a confidence.

This fits the gate. One row per target, label ``y = 1[acting beat the bandit]``,
features computable before any oracle call. Two properties of the data make the
evaluation unusually clean and are the reason this instrument is cheap:

* **Full counterfactuals.** Every arm ran on every target with paired seeds, so
  both ``commit(pick)`` and ``bandit`` are observed everywhere. The value of any
  gate policy is computed exactly -- no propensities, no importance weighting,
  no off-policy estimator and none of its assumptions.
* **The unit of analysis is the target**, and targets are i.i.d. draws from the
  eligible pool. So a train/test split over targets is an honest holdout.

PRE-REGISTERED DECISION RULE
----------------------------
Fixed before looking at any output, because the in-sample threshold sweep in
section 12 reports a positive gain on a pure-noise signal **75.7% of the time**
(``--null`` reproduces that), and the temptation to read a win off a noisy table
is exactly what killed the n=24 ``pred_lift`` result.

The gate is called USEFUL only if **all four** hold:

1. cross-fitted out-of-fold AUROC > 0.60, permutation p < 0.05;
2. cross-fitted policy value > 0 against the free bandit, and outside the 95th
   percentile of the permuted-label null for the identical pipeline;
3. the certified one-sided 95% LCB on a single held-out 80% split is > 0;
4. it beats the matched-coverage random gate (the section-12 control).

Anything less is reported as a negative with the ceiling attached. The
``--n-curve`` output decides *which* negative: AUROC rising toward 0.60 with
target count means underpowered, flat at 0.50 means no signal in these features.

FEATURES
--------
Answer-key-free by construction. The five region-score confidence signals, plus
library/partition structure (size, region count, size concentration) and the
shape of the prior's own prediction distribution. Nothing derived from measured
values, from ``best_lift``, or from any campaign outcome -- asserted by
``_assert_no_leakage`` rather than by inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

# Anything carrying knowledge of the answer. Kept as a hard check because the
# whole result is worthless if one of these leaks in.
FORBIDDEN = {"best", "hit", "best_lift", "delta_vs_bandit", "bandit_minus_whole"}


def _assert_no_leakage(names: list[str]) -> None:
    bad = [n for n in names if any(f in n for f in FORBIDDEN)]
    if bad:
        raise SystemExit(f"leaked answer-key features into the gate: {bad}")


def featurise(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """-> (n_targets, n_features) and the feature names."""
    names, cols = [], []

    def add(name, fn):
        names.append(name)
        cols.append(np.array([float(fn(r)) for r in rows]))

    for f in sorted(rows[0]["conf"]):
        add(f"conf.{f}", lambda r, f=f: r["conf"][f])

    add("lib.n", lambda r: r["n"])
    add("lib.log_n", lambda r: np.log(r["n"]))
    add("part.n_nonempty", lambda r: sum(s > 0 for s in r["sizes"]))
    add("part.pick_size", lambda r: r["sizes"][r["pick"]])
    add("part.pick_frac", lambda r: r["sizes"][r["pick"]] / max(1.0, sum(r["sizes"])))
    add("part.size_gini", lambda r: _gini(np.asarray(r["sizes"], float)))
    add("part.size_max_frac", lambda r: max(r["sizes"]) / max(1.0, sum(r["sizes"])))

    # shape of the prior's own opinion over regions -- how peaked, how spread
    add("score.std", lambda r: np.std(_fin(r["region_scores"])))
    add("score.max", lambda r: np.max(_fin(r["region_scores"])))
    add("score.range", lambda r: np.ptp(_fin(r["region_scores"])))
    add("score.n_positive", lambda r: int((_fin(r["region_scores"]) > 0).sum()))
    add("score.top2_sum", lambda r: float(np.sort(_fin(r["region_scores"]))[-2:].sum()))

    _assert_no_leakage(names)
    X = np.column_stack(cols)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def _fin(v):
    a = np.asarray(v, float)
    return a[np.isfinite(a)] if np.isfinite(a).any() else np.zeros(1)


def _gini(x):
    x = np.sort(np.clip(x, 0, None))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def auroc(score, label) -> float:
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def fit_predict(Xtr, ytr, Xte):
    """Ridge-regularised logistic regression, standardised. Deliberately the
    smallest model that can use several features: at a few hundred targets a
    flexible learner would fit the noise the pre-registration is guarding
    against, and a linear gate is also the one a reviewer can read."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr.mean()))
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def pick_threshold(score, d, min_coverage=0.05):
    """Threshold maximising realised policy value ON THE TRAINING FOLD ONLY.

    This is section 12's sweep, which is badly biased in-sample -- that is the
    point. It is only ever applied to data the fold has not seen, so the bias
    stays inside the fold and the reported number is honest.
    """
    best, best_tau = -np.inf, np.inf
    for tau in np.unique(score):
        act = score >= tau
        if act.mean() < min_coverage:
            continue
        v = float(np.where(act, d, 0.0).mean())
        if v > best:
            best, best_tau = v, float(tau)
    return best_tau


def crossfit(X, y, d, folds=10, seed=0):
    """-> (oof gate score, oof policy value per target, oof coverage).

    The threshold is chosen inside each training fold, so both the model and
    its operating point are out-of-sample for every target scored.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    oof_score = np.zeros(n)
    oof_value = np.zeros(n)
    oof_act = np.zeros(n, bool)
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        s_tr = fit_predict(X[tr], y[tr], X[tr])
        tau = pick_threshold(s_tr, d[tr])
        s_te = fit_predict(X[tr], y[tr], X[te])
        oof_score[te] = s_te
        oof_act[te] = s_te >= tau
        oof_value[te] = np.where(s_te >= tau, d[te], 0.0)
    return oof_score, oof_value, oof_act


def certify(X, y, d, seed=0, tune_frac=0.2, delta=0.05):
    """HCPI-shaped deploy test: select the gate on a tuning split, then take a
    one-sided (1-delta) lower confidence bound for its value on the held-out
    split. Deploy only if the bound clears zero.

    The split fraction is small on the tuning side on purpose -- the testing
    half is what buys statistical power, and selecting and testing on the same
    targets is the failure this whole script exists to avoid.
    """
    from scipy.stats import norm

    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tune = max(10, int(round(tune_frac * n)))
    tune, test = idx[:n_tune], idx[n_tune:]
    s_tune = fit_predict(X[tune], y[tune], X[tune])
    tau = pick_threshold(s_tune, d[tune])
    s_test = fit_predict(X[tune], y[tune], X[test])
    act = s_test >= tau
    v = np.where(act, d[test], 0.0)
    lcb = v.mean() - norm.ppf(1 - delta) * v.std(ddof=1) / np.sqrt(len(v))
    return {"n_tune": int(n_tune), "n_test": int(len(test)), "tau": float(tau),
            "coverage": float(act.mean()), "value": float(v.mean()),
            "lcb": float(lcb), "deploy": bool(lcb > 0)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="results/selective_headroom_all.json")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--null-draws", type=int, default=500)
    p.add_argument("--target", default="pays", choices=["pays", "hit"],
                   help="what the gate is trained to predict. 'pays' = acting "
                        "beat the bandit on this target (the decision's actual "
                        "consequence); 'hit' = the pick was the best region.")
    p.add_argument("--n-curve", type=int, nargs="*",
                   default=[50, 100, 200, 400],
                   help="subsample sizes for the power curve. This is what "
                        "separates 'underpowered' from 'no signal'.")
    p.add_argument("--out", default="results/gate_fit.json")
    args = p.parse_args(argv)

    d_in = json.loads(Path(args.data).read_text(encoding="utf-8"))
    rows = d_in["rows"]
    for r in rows:
        r["delta_vs_bandit"] = {int(c): v for c, v in r["delta_vs_bandit"].items()}
    dv = np.array([r["delta_vs_bandit"][r["pick"]] for r in rows])
    y = (dv > 0) if args.target == "pays" else np.array([r["hit"] for r in rows])
    X, names = featurise(rows)
    n = len(rows)

    print(f"{'=' * 74}\nFITTING THE GATE  ({n} targets, {len(names)} features, "
          f"target='{args.target}')\n{'=' * 74}")
    print(f"  base rate P(y=1)            {y.mean():.3f}")
    print(f"  commit always               {dv.mean():+.4f}")
    print(f"  gate(outcome) ceiling       {np.maximum(dv, 0).mean():+.4f}\n")

    score, value, act = crossfit(X, y, dv, folds=args.folds)
    a = auroc(score, y)
    print(f"  cross-fitted OOF AUROC      {a:.3f}")
    print(f"  cross-fitted policy value   {value.mean():+.4f}  "
          f"(acts on {act.mean():.0%})")

    # ---- permuted-label null for the ENTIRE pipeline ---------------------
    # Shuffling y alone would leave the threshold sweep free to exploit dv, so
    # the pair (y, dv) is permuted together: the gate then has a label and an
    # outcome that are real but belong to a different target.
    rng = np.random.default_rng(1)
    null_a, null_v = [], []
    for _ in range(args.null_draws):
        pi = rng.permutation(n)
        s0, v0, _ = crossfit(X, y[pi], dv[pi], folds=args.folds)
        null_a.append(auroc(s0, y[pi]))
        null_v.append(v0.mean())
    null_a, null_v = np.array(null_a), np.array(null_v)
    p_a = float((null_a >= a).mean())
    p_v = float((null_v >= value.mean()).mean())
    print(f"\n  permuted-label null ({args.null_draws} draws)")
    print(f"    AUROC   null mean {null_a.mean():.3f}  95th {np.percentile(null_a, 95):.3f}"
          f"   ->  p = {p_a:.3f}")
    print(f"    value   null mean {null_v.mean():+.4f}  95th "
          f"{np.percentile(null_v, 95):+.4f}   ->  p = {p_v:.3f}")

    cert = certify(X, y, dv)
    print(f"\n  certified deploy test (tune {cert['n_tune']} / test {cert['n_test']})")
    print(f"    value on held-out         {cert['value']:+.4f} "
          f"(acts on {cert['coverage']:.0%})")
    print(f"    one-sided 95% LCB         {cert['lcb']:+.4f}   "
          f"->  {'DEPLOY' if cert['deploy'] else 'do not deploy'}")

    # ---- matched-coverage random gate, the section-12 control ------------
    cov = act.mean()
    rnd = np.array([np.where(rng.random(n) < cov, dv, 0.0).mean() for _ in range(5000)])
    print(f"\n  random gate at the same {cov:.0%} coverage  {rnd.mean():+.4f} "
          f"[{np.percentile(rnd, 2.5):+.4f}, {np.percentile(rnd, 97.5):+.4f}]")
    print(f"  knowing-which gap           {value.mean() - rnd.mean():+.4f}")

    # ---- the power curve -------------------------------------------------
    print(f"\n  POWER CURVE -- is this underpowered, or is there nothing here?")
    print(f"    {'n':>6s}{'OOF AUROC':>12s}{'policy value':>15s}{'coverage':>11s}")
    curve = []
    for m in [x for x in args.n_curve if x <= n] + ([n] if n not in args.n_curve else []):
        aa, vv = [], []
        for rep in range(20):
            sub = np.random.default_rng(100 + rep).choice(n, size=m, replace=False)
            s1, v1, _ = crossfit(X[sub], y[sub], dv[sub], folds=min(args.folds, 5),
                                 seed=rep)
            aa.append(auroc(s1, y[sub]))
            vv.append(v1.mean())
        curve.append({"n": int(m), "auroc": float(np.nanmean(aa)),
                      "value": float(np.mean(vv))})
        print(f"    {m:>6d}{np.nanmean(aa):>12.3f}{np.mean(vv):>+15.4f}")

    # ---- the pre-registered verdict --------------------------------------
    crit = {
        "auroc > 0.60 and p < 0.05": bool(a > 0.60 and p_a < 0.05),
        "policy value > 0 and beats null": bool(value.mean() > 0 and p_v < 0.05),
        "certified LCB > 0": bool(cert["deploy"]),
        "beats matched-coverage random gate": bool(value.mean() > np.percentile(rnd, 95)),
    }
    print(f"\n{'=' * 74}\nPRE-REGISTERED VERDICT\n{'=' * 74}")
    for k, v in crit.items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    ok = all(crit.values())
    print(f"\n  ==> the gate is {'USEFUL' if ok else 'NOT demonstrated'}. "
          f"{'' if ok else 'Report the ceiling and this negative together.'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "data": args.data, "n_targets": n, "target": args.target,
        "features": names, "base_rate": float(y.mean()),
        "commit_always": float(dv.mean()),
        "ceiling": float(np.maximum(dv, 0).mean()),
        "oof_auroc": a, "oof_value": float(value.mean()),
        "oof_coverage": float(act.mean()),
        "null": {"auroc_mean": float(null_a.mean()), "auroc_p": p_a,
                 "value_mean": float(null_v.mean()), "value_p": p_v},
        "certified": cert, "random_gate": float(rnd.mean()),
        "curve": curve, "criteria": crit, "useful": ok,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
