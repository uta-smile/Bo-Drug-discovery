"""Can a deployable feature predict the benefit, leave-one-family-out?

    python scripts/utility_feature_probe.py

`utility_signal_triage.py` establishes that the per-target benefit of auxiliary
data is a stable quantity (split-half reliability 0.90-0.96) and that roughly
half its variance sits *within* families, so leave-one-family-out prediction is
not ruled out by the design. It does not show that anything predicts it. This
does the next step, still without running a campaign.

C1 identified the auxiliary *response* information as the causal driver, so the
obvious feature is the rank correlation between the primary target's activity
column and its auxiliary columns. The difficulty is that at deployment **the
primary column is the thing being screened for** -- it is not available. That
turns the advisor's closing question into a measurement rather than a choice,
because the feature sets differ exactly in how they stand in for it:

``zero_shot``  the auxiliary columns' agreement *with each other*, plus library
               size. No primary activity data at all.
``warm10``     adds the primary-auxiliary rank correlation estimated from 10
               primary measurements -- the size of the initial design every
               campaign here already runs.
``warm20``     the same from 20.
``oracle_rho`` the same from the complete primary column. Not deployable; it is
               the ceiling that the warm-start estimates are chasing, and the
               gap between it and ``warm10`` is the price of not knowing the
               answer in advance.

Each is fitted on five families and scored on the sixth, with standardisation
and coefficients fitted inside the fold. Reported against the constant
predictor, which is what the triage showed to be the current state of the art.

Correlations are computed on the same censored block a campaign sees
(`drop_censored`, pile_frac=0.02), because the censoring removes exactly the
rows that inflate cross-target correlation.

Writes `results/utility_feature_probe.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dense_block_h1 import build_block, drop_censored, nanomolar_potency  # noqa: E402
import dense_block_campaign as dbc  # noqa: E402
import utility_signal_triage as tri  # noqa: E402

RESULTS = ROOT / "results"
WARM = (10, 20)


def block_values(fam):
    """The censored block a campaign sees, as (n_pool x n_targets) potencies."""
    mapping = dbc.FAMILIES[fam]
    block, _ = build_block(mapping, potency=nanomolar_potency)
    targets = [t for t in mapping.values() if t in block.columns]
    block, _ = drop_censored(block, targets, pile_frac=0.02, verbose=False)
    return block[targets].to_numpy(float), targets


def features(fam, rng, draws=200):
    """Per-primary-target features, split by what deployment actually affords."""
    vals, targets = block_values(fam)
    n = len(vals)
    out = {}
    for ti, t in enumerate(targets):
        aux = [a for a in range(len(targets)) if a != ti]
        rho = [float(spearmanr(vals[:, ti], vals[:, a]).statistic) for a in aux]
        # Auxiliary columns against each other: no primary activity involved,
        # so this is computable before a single assay is run.
        pair = [float(spearmanr(vals[:, a], vals[:, b]).statistic)
                for i, a in enumerate(aux) for b in aux[i + 1:]]
        f = {"rho_oracle": float(np.mean(rho)),
             "rho_oracle_max": float(np.max(rho)),
             "aux_rho": float(np.mean(pair)),
             "log_n_pool": float(np.log(n)),
             "n_pool": int(n)}
        for m in WARM:
            # The primary values a researcher would hold after an initial
            # design of m compounds, averaged over which m they happened to be.
            est = []
            for _ in range(draws):
                idx = rng.choice(n, size=m, replace=False)
                r = [spearmanr(vals[idx, ti], vals[idx, a]).statistic
                     for a in aux]
                r = [x for x in r if np.isfinite(x)]
                est.append(np.mean(r) if r else 0.0)
            f[f"rho_hat{m}"] = float(np.mean(est))
            f[f"rho_hat{m}_sd"] = float(np.std(est, ddof=1))
        out[t] = f
    return out


SETS = {
    "zero_shot": ["aux_rho", "log_n_pool"],
    "warm10": ["rho_hat10", "aux_rho", "log_n_pool"],
    "warm20": ["rho_hat20", "aux_rho", "log_n_pool"],
    "oracle_rho": ["rho_oracle", "aux_rho", "log_n_pool"],
}


def design_matrix(feat, names, cols, fams):
    """Rows are (family, target, budget rank, design); `design` is always in."""
    X, y_key = [], []
    for fi, fam in enumerate(fams):
        for ti, t in enumerate(names[fam]):
            for bi in range(2):
                for di, d in enumerate(tri.DESIGNS):
                    f = feat[fam][t]
                    X.append([f[c] for c in cols]
                             + [float(di), float(bi)])
                    y_key.append((fi, ti, bi, di))
    return np.asarray(X, float), y_key


def lofo_fit(X, y, fold, ridge=1e-6):
    """Out-of-fold predictions; standardisation and coefficients fitted inside."""
    pred = np.empty_like(y)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        A = np.c_[np.ones(tr.sum()), (X[tr] - mu) / sd]
        B = np.c_[np.ones(te.sum()), (X[te] - mu) / sd]
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]),
                            A.T @ y[tr])
        pred[te] = B @ w
    return pred


def score(y, pred):
    mse = float(np.mean((y - pred) ** 2))
    return {"rmse": float(np.sqrt(mse)),
            "r2": float(1 - mse / float(np.mean((y - y.mean()) ** 2))),
            "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(y))),
            "pearson_r": float(np.corrcoef(pred, y)[0, 1])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=RESULTS / "utility_feature_probe.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--draws", type=int, default=200)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    fams = list(tri.FAMILIES)
    feat, names = {}, {}
    for fam in fams:
        feat[fam] = features(fam, rng, draws=args.draws)
        names[fam] = list(feat[fam])
        print(f"  {fam}: " + "  ".join(
            f"{t} rho={feat[fam][t]['rho_oracle']:+.2f}"
            f"/hat10={feat[fam][t]['rho_hat10']:+.2f}"
            for t in names[fam]), flush=True)

    report = {"features": feat, "arms": {}}
    for arm in tri.ARMS:
        A, rep, tri_names = tri.load_cells(arm)
        # The triage's target order is the artefact's; align this probe to it.
        for fi, fam in enumerate(fams):
            assert names[fam] == [tri_names[(fi, ti)] for ti in range(4)], (
                fam, names[fam], [tri_names[(fi, ti)] for ti in range(4)])

        print(f"\n{'=' * 70}\n{arm} - stl   leave-one-family-out\n{'=' * 70}")
        print(f"  {'feature set':12s} {'RMSE':>7s} {'R2':>8s} {'r':>7s} {'sign':>7s}")
        res = {}
        for name, cols in SETS.items():
            X, keys = design_matrix(feat, names, cols, fams)
            y = np.asarray([A[k] for k in keys], float)
            fold = np.asarray([k[0] for k in keys])
            s = score(y, lofo_fit(X, y, fold))
            res[name] = s
            print(f"  {name:12s} {s['rmse']:7.2f} {s['r2']:8.3f} "
                  f"{s['pearson_r']:7.3f} {s['sign_accuracy']:6.0%}")
        # The constant predictor, repeated here so the table is self-contained.
        X, keys = design_matrix(feat, names, [], fams)
        y = np.asarray([A[k] for k in keys], float)
        fold = np.asarray([k[0] for k in keys])
        s = score(y, lofo_fit(np.zeros((len(y), 1)), y, fold))
        res["constant"] = s
        print(f"  {'constant':12s} {s['rmse']:7.2f} {s['r2']:8.3f} "
              f"{'-':>7s} {s['sign_accuracy']:6.0%}")
        report["arms"][arm] = res

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
