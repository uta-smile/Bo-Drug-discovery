"""The two explanations of the FINDINGS 40 gain, and the measurements that kill them.

    python scripts/mechanism_probes.py --out results/mechanism_probes.json

FINDINGS 40 established that single-task BO at ``sd x 0.80`` reproduces
everything a pure-noise auxiliary buys on JAK, and that no sd scale in a
fifty-sevenfold range reproduces any of it on CYP. Two explanations were
available from the real blocks. Both fail, and the measurements are here rather
than in a scratch file because FINDINGS 41 quotes them.

**calibration** -- "EI over-explores because the surrogate's error bars are
wider than its errors, and 0.80 is the factor that fixes it". That predicts
``std(z) ~ 0.80`` on JAK and ``~1.0`` on CYP, where ``z = (y - mu) / sd`` on
held-out candidates. It is wrong in direction (both blocks are *under*-dispersed,
std(z) = 1.4-4.8), in magnitude, and in discrimination (the number does not
separate the two blocks). The ``|z| > 2`` column says why a variance ratio is
the wrong summary at all: the residuals are cliff-driven, not Gaussian.

**isolation** -- "compressing sd stops EI buying chemically isolated compounds,
which are rarely actives". That predicts the picks ``sd x 0.80`` drops are more
isolated and lower valued than the ones it gains. On JAK they are
indistinguishable on both counts.

What survives is an accounting identity rather than a mechanism, and the probe
prints it: among the picks that change, the rate of landing in the top-k
doubles, and that alone reproduces the measured recall gain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from molbo.surrogate import (TanimotoGP, fingerprint_matrix,  # noqa: E402
                             tanimoto_gram)
from dense_block_campaign import FAMILIES, run_campaign  # noqa: E402
from dense_block_h1 import build_block, drop_censored  # noqa: E402

# The sd scale FINDINGS 40 measured on the posterior and then confirmed in the
# campaign. Not refitted here.
SCALE = 0.80


def block(family, pile_frac=0.02):
    mapping = FAMILIES[family]
    b, _ = build_block(mapping)
    targets = [t for t in mapping.values() if t in b.columns]
    b, _ = drop_censored(b, targets, pile_frac=pile_frac, verbose=False)
    X = fingerprint_matrix(b.smiles.tolist())
    return tanimoto_gram(X, X), b[targets].to_numpy(float), targets


def calibration(K, vals, *, n_obs_grid, draws):
    """std(z) and the tail mass, per observation count."""
    n_pool, T = vals.shape
    out = []
    for n_obs in n_obs_grid:
        if n_obs >= n_pool - 20:
            continue
        zs, sds, errs, rhos = [], [], [], []
        for d in range(draws):
            rng = np.random.default_rng(d)
            y = vals[:, d % T]
            obs = rng.choice(n_pool, size=n_obs, replace=False)
            cand = np.setdiff1d(np.arange(n_pool), obs)
            gp = TanimotoGP().fit_gram(K[np.ix_(obs, obs)], y[obs])
            mu, sd = gp.predict_gram(K[np.ix_(cand, obs)],
                                     k_diag=np.ones(len(cand)),
                                     return_std=True)
            zs.append((y[cand] - mu) / np.maximum(sd, 1e-12))
            sds.append(float(sd.mean()))
            errs.append(float(np.abs(y[cand] - mu).mean()))
            rhos.append(float(spearmanr(mu, y[cand]).statistic))
        z = np.concatenate(zs)
        out.append({"n_obs": int(n_obs), "std_z": float(z.std()),
                    "tail_frac": float((np.abs(z) > 2).mean()),
                    "mean_sd": float(np.mean(sds)),
                    "mean_abs_err": float(np.mean(errs)),
                    "rho_vs_truth": float(np.mean(rhos))})
    return out


def picks(K, vals, *, budget, batch, n_init, k, history, repeats, scale):
    """Diff the selected sets of two exploration levels on identical seeds."""
    n_pool, T = vals.shape
    S = K.copy()
    np.fill_diagonal(S, -np.inf)
    iso = S.max(axis=1)          # isolation: nearest neighbour in the library

    acc = {"dropped": {"iso": [], "val": [], "top": 0, "n": 0},
           "gained": {"iso": [], "val": [], "top": 0, "n": 0}}
    for ti in range(T):
        y = vals[:, ti]
        top = set(np.argsort(y)[-k:].tolist())
        others = [j for j in range(T) if j != ti]
        for r in range(repeats):
            seed = 1009 * r + 97 * ti
            sel = {}
            for s in (1.0, scale):
                hg = np.random.default_rng(seed)
                hist = hg.choice(n_pool, size=history * len(others),
                                 replace=False)
                aux_idx = [hist[a * history:(a + 1) * history]
                           for a in range(len(others))]
                aux_val = [vals[aux_idx[a], others[a]]
                           for a in range(len(others))]
                _, idx = run_campaign(K, y, aux_idx, aux_val, arm="stl",
                                      budget=budget, batch=batch,
                                      n_init=n_init, k=k, acquisition="ei",
                                      sd_scale=s,
                                      rng=np.random.default_rng(seed))
                sel[s] = set(idx.tolist())
            for name, ids in (("dropped", sel[1.0] - sel[scale]),
                              ("gained", sel[scale] - sel[1.0])):
                a = acc[name]
                a["n"] += len(ids)
                a["top"] += sum(i in top for i in ids)
                a["iso"].extend(iso[i] for i in ids)
                a["val"].extend(y[i] for i in ids)
    return acc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", nargs="+", default=["jak", "cyp"])
    p.add_argument("--budgets", nargs="+", type=int, default=[300, 111],
                   help="one per family, matching the stored window cells")
    p.add_argument("--draws", type=int, default=16,
                   help="draws per observation count in the calibration probe")
    p.add_argument("--repeats", type=int, default=16,
                   help="repeats per target in the picks probe")
    p.add_argument("--scale", type=float, default=SCALE)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--history", type=int, default=50)
    p.add_argument("--out", type=Path,
                   default=Path("results/mechanism_probes.json"))
    args = p.parse_args(argv)
    if len(args.budgets) != len(args.families):
        raise SystemExit("--budgets needs one value per family")

    report = {"scale": args.scale, "k": args.k, "history": args.history,
              "families": {}}
    for fam, budget in zip(args.families, args.budgets):
        K, vals, targets = block(fam)
        n_pool, T = vals.shape
        print(f"\n{fam}: {n_pool} compounds x {T} targets, budget {budget}")

        cal = calibration(K, vals, draws=args.draws,
                          n_obs_grid=(20, 50, 100, 200, 300))
        print("  calibration -- if the surrogate were over-dispersed, "
              "std(z) would be < 1")
        print(f"    {'n_obs':>6}{'std(z)':>9}{'|z|>2':>8}{'mean sd':>10}"
              f"{'mean |err|':>12}{'rho vs truth':>14}")
        for c in cal:
            print(f"    {c['n_obs']:>6}{c['std_z']:>9.3f}{c['tail_frac']:>8.3f}"
                  f"{c['mean_sd']:>10.3f}{c['mean_abs_err']:>12.3f}"
                  f"{c['rho_vs_truth']:>14.3f}")

        pk = picks(K, vals, budget=budget, batch=10, n_init=10, k=args.k,
                   history=args.history, repeats=args.repeats,
                   scale=args.scale)
        d, g = pk["dropped"], pk["gained"]
        di, gi = np.array(d["iso"]), np.array(g["iso"])
        dv, gv = np.array(d["val"]), np.array(g["val"])
        n_camp = args.repeats * T
        d_rate = d["top"] / max(d["n"], 1)
        g_rate = g["top"] / max(g["n"], 1)
        predicted = (g_rate - d_rate) * d["n"] / n_camp / args.k * 100
        print(f"  picks -- sd x 1.0 against sd x {args.scale}, "
              f"{n_camp} campaigns")
        print(f"    {'':<34}{'isolation':>11}{'value':>9}{'in top-k':>11}{'n':>7}")
        print(f"    {'bought only at FULL sd (dropped)':<34}{di.mean():>11.3f}"
              f"{dv.mean():>9.3f}{d_rate:>11.3f}{d['n']:>7}")
        print(f"    {'bought only at TIGHT sd (gained)':<34}{gi.mean():>11.3f}"
              f"{gv.mean():>9.3f}{g_rate:>11.3f}{g['n']:>7}")
        p_iso = float(mannwhitneyu(di, gi).pvalue)
        p_val = float(mannwhitneyu(dv, gv).pvalue)
        print(f"    isolation p = {p_iso:.2e}    value p = {p_val:.2e}")
        print(f"    {d['n'] / n_camp:.0f} picks differ per campaign; the top-k "
              f"rate goes {d_rate:.3f} -> {g_rate:.3f},")
        print(f"    which predicts {predicted:+.2f} recall points.")

        report["families"][fam] = {
            "budget": budget, "n_pool": int(n_pool), "targets": targets,
            "calibration": cal,
            "picks": {"n_campaigns": n_camp, "n_swapped": d["n"],
                      "dropped": {"isolation": float(di.mean()),
                                  "value": float(dv.mean()),
                                  "top_k_rate": d_rate},
                      "gained": {"isolation": float(gi.mean()),
                                 "value": float(gv.mean()),
                                 "top_k_rate": g_rate},
                      "p_isolation": p_iso, "p_value": p_val,
                      "predicted_recall_points": float(predicted)}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
