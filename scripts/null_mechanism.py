"""What actually causes the null-auxiliary gain? A ladder, one control at a time.

    python scripts/null_mechanism.py --family jak --out results/null_mechanism.json

FINDINGS 32 reported that an auxiliary column of pure noise buys recall under
expected improvement, and attributed it to "extra kernel rows compress the
model's uncertainty whether or not they are informative". That explanation
bundles three mechanisms that the experiment never separated:

1. **more rows** -- the joint kernel is larger;
2. **a spuriously positive between-target correlation** -- the fit selects
   ``r = 0.083`` on noise, not exactly 0, from a finite sample;
3. **refitted shared hyperparameters** -- the inner GP's signal scale and
   noise-to-signal ratio are chosen on the *joint* data, so the primary task's
   effective smoothing changes even when the tasks are uncoupled.

If ``r = 0`` exactly and the inner hyperparameters are frozen to the single-task
fit, the joint kernel is block diagonal and the primary block's posterior must
be *identical* to the single-task one. Any difference at that rung is a
computational surprise, not a modelling effect. Releasing one control at a time
says which mechanism carries the effect.

Rungs, in order:

* ``stl``          -- single-task GP on the primary rows only.
* ``r0_frozen``    -- joint model, ``r`` forced to 0, inner noise ratio pinned
                      to the value the single-task fit chose.
* ``r0_refit``     -- joint model, ``r`` forced to 0, noise ratio refitted on
                      the joint data.
* ``r_fitted``     -- joint model, ``r`` chosen from the grid. This is the arm
                      FINDINGS 32 calls the noise auxiliary.
* ``real``         -- the same, with the true auxiliary values.

Reported per rung: rank correlation of the primary prediction against truth,
mean predicted uncertainty on the candidates, and the rank agreement of the
expected-improvement ordering with the single-task one. The first says whether
the posterior got better or worse, the second whether uncertainty moved, and the
third whether any of it reached the decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.acquisition import EI_XI, acquisition_scores  # noqa: E402
from molbo.metrics import degrade_to_rho, spearman0  # noqa: E402
from molbo.surrogate import CoregionalisedGP, TanimotoGP  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_block_campaign import FAMILIES  # noqa: E402
from dense_block_h1 import build_block, drop_censored  # noqa: E402

RUNGS = ("stl", "r0_all_frozen", "r0_scale_pooled", "r0_refit",
         "r_fitted", "real")


def one_draw(K, vals, ti, *, n_obs, n_hist, rng):
    """Fit every rung on one draw and return their primary-task posteriors."""
    n_pool, T = vals.shape
    y = vals[:, ti]
    others = [j for j in range(T) if j != ti]

    obs = rng.choice(n_pool, size=n_obs, replace=False)
    rest = np.setdiff1d(np.arange(n_pool), obs)
    hist = rng.choice(rest, size=n_hist * len(others), replace=False)
    aux_idx = [hist[a * n_hist:(a + 1) * n_hist] for a in range(len(others))]
    cand = np.setdiff1d(np.arange(n_pool), np.concatenate([obs, hist]))

    noise_val = [degrade_to_rho(vals[:, others[a]], 0.0, rng)[aux_idx[a]]
                 for a in range(len(others))]
    real_val = [vals[aux_idx[a], others[a]] for a in range(len(others))]

    out = {}

    # -- rung 0: single task -------------------------------------------------
    stl = TanimotoGP().fit_gram(K[np.ix_(obs, obs)], y[obs])
    mu, sd = stl.predict_gram(K[np.ix_(cand, obs)],
                              k_diag=np.ones(len(cand)), return_std=True)
    out["stl"] = (mu, sd)
    eta_star = float(stl.eta_)

    scale_star = float(stl.scale_)

    # -- rungs 1-4: noise auxiliary, releasing one control at a time ----------
    # r0_all_frozen is the mathematical control: with a diagonal task covariance
    # and every inner hyperparameter pinned to the single-task fit, the
    # auxiliary rows must drop out of the primary posterior entirely -- mean and
    # uncertainty alike. r0_scale_pooled then releases only the profiled signal
    # variance, which is the one quantity computed across all tasks at once.
    for rung, kw in (
        ("r0_all_frozen", {"rho_grid": (0.0,), "eta_grid": (eta_star,),
                           "fixed_scale": scale_star}),
        ("r0_scale_pooled", {"rho_grid": (0.0,), "eta_grid": (eta_star,)}),
        ("r0_refit", {"rho_grid": (0.0,)}),
        ("r_fitted", {}),
    ):
        m = CoregionalisedGP(**kw)
        m.fit_gram(K, [obs] + aux_idx, [y[obs]] + noise_val)
        out[rung] = m.predict_gram(K, cand, return_std=True) + (m.rho_,)

    # -- rung 4: the real panel ---------------------------------------------
    m = CoregionalisedGP().fit_gram(K, [obs] + aux_idx, [y[obs]] + real_val)
    out["real"] = m.predict_gram(K, cand, return_std=True) + (m.rho_,)

    return out, y[cand]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="jak", choices=sorted(FAMILIES))
    p.add_argument("--n-obs", type=int, default=100)
    p.add_argument("--history", type=int, default=50)
    p.add_argument("--draws", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-censored", action="store_true")
    p.add_argument("--out", default="results/null_mechanism.json")
    args = p.parse_args(argv)

    from molbo.surrogate import fingerprint_matrix, tanimoto_gram

    mapping = FAMILIES[args.family]
    block, _ = build_block(mapping)
    targets = [t for t in mapping.values() if t in block.columns]
    if not args.keep_censored:
        block, _ = drop_censored(block, targets)
    X = fingerprint_matrix(block.smiles.tolist())
    K = tanimoto_gram(X, X)
    vals = block[targets].to_numpy(float)
    print(f"{args.family}: {len(block):,} compounds x {len(targets)} targets; "
          f"{args.n_obs} primary observations, {args.history} per auxiliary, "
          f"{args.draws} draws x {len(targets)} targets")

    acc = {r: {"rho": [], "sd": [], "agree": [], "r_sel": [],
           "dmu": [], "dsd": []} for r in RUNGS}
    for ti in range(len(targets)):
        for d in range(args.draws):
            rng = np.random.default_rng(args.seed + 1009 * d + 97 * ti)
            out, truth = one_draw(K, vals, ti, n_obs=args.n_obs,
                                  n_hist=args.history, rng=rng)
            base_mu, base_sd = out["stl"][0], out["stl"][1]
            g = np.random.default_rng(0)
            base_order = np.argsort(-acquisition_scores(
                "ei", base_mu, base_sd, best=float(base_mu.max()), rng=g))
            for rung in RUNGS:
                mu, sd = out[rung][0], out[rung][1]
                acc[rung]["rho"].append(spearman0(mu, truth))
                acc[rung]["sd"].append(float(sd.mean()))
                acc[rung]["dmu"].append(float(np.abs(mu - base_mu).max()))
                acc[rung]["dsd"].append(float(np.abs(sd - base_sd).max()))
                g = np.random.default_rng(0)
                order = np.argsort(-acquisition_scores(
                    "ei", mu, sd, best=float(mu.max()), rng=g))
                acc[rung]["agree"].append(spearman0(base_order, order))
                if len(out[rung]) > 2:
                    acc[rung]["r_sel"].append(float(out[rung][2]))
        print(f"    {targets[ti]} done", flush=True)

    print(f"\n    {'rung':<12}{'rho vs truth':>14}{'mean sd':>11}"
          f"{'EI order vs stl':>18}{'r selected':>13}")
    print("    " + "-" * 68)
    report = {"family": args.family, "n_obs": args.n_obs,
              "history": args.history, "draws": args.draws,
              # The EI-order column is an acquisition comparison, so the offset
              # it was computed under belongs with the numbers it produced.
              "xi": float(EI_XI), "rungs": {}}
    for rung in RUNGS:
        a = acc[rung]
        rho, sd = float(np.mean(a["rho"])), float(np.mean(a["sd"]))
        agree = float(np.mean(a["agree"]))
        rsel = float(np.mean(a["r_sel"])) if a["r_sel"] else float("nan")
        dmu, dsd = float(np.max(a["dmu"])), float(np.max(a["dsd"]))
        print(f"    {rung:<17}{dmu:>11.2e}{dsd:>11.2e}{rho:>14.4f}"
              f"{sd:>10.4f}{agree:>10.4f}"
              + (f"{rsel:>8.3f}" if a["r_sel"] else f"{'--':>8}"))
        report["rungs"][rung] = {"rho_vs_truth": rho, "mean_sd": sd,
                                 "ei_order_agreement_with_stl": agree,
                                 "r_selected": rsel,
                                 "max_abs_dmu_vs_stl": dmu,
                                 "max_abs_dsd_vs_stl": dsd}

    f = report["rungs"]["r0_all_frozen"]
    g = report["rungs"]["r0_scale_pooled"]
    print()
    print("    verdict:")
    if f["max_abs_dmu_vs_stl"] < 1e-9 and f["max_abs_dsd_vs_stl"] < 1e-9:
        print("      With a diagonal task covariance and every inner")
        print("      hyperparameter pinned, the auxiliary rows drop out of the")
        print("      primary posterior exactly -- mean and uncertainty alike.")
        print(f"      Releasing only the pooled signal variance moves the")
        print(f"      uncertainty by up to {g['max_abs_dsd_vs_stl']:.3e} while the")
        print(f"      mean still moves by {g['max_abs_dmu_vs_stl']:.3e}.")
    else:
        print("      UNEXPECTED: the fully frozen rung does not reproduce the")
        print(f"      single-task posterior (d_mu {f['max_abs_dmu_vs_stl']:.3e}, "
              f"d_sd {f['max_abs_dsd_vs_stl']:.3e}).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
