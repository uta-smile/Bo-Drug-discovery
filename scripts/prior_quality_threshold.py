"""How good must a learned prior be before it helps Bayesian optimisation?

    python scripts/prior_quality_threshold.py --targets 8 --seeds 200

Every learned component tried in this project has failed, and the usual
response is "train a better model". This measures whether that would help, by
sweeping prior *quality* directly instead of training anything.

For each target a synthetic prior is built with a controlled rank correlation
to the true objective, from rho = 0 (noise) to rho = 1 (the answer key), and
used as the GP's mean function. The output is the **threshold**: the rho at
which a prior starts improving campaign AUC.

That number is what tells you whether a better predictor is worth building.
Measured cold-target performance of the ridge on (ESM2, ECFP4) is rho ~ 0.22,
and published cold-target DTI models are not far above that -- so if the
threshold sits well beyond it, no amount of model work pays and the honest
conclusion is that the prior route is closed.

Acquisition matters and is swept too. Expected Improvement **degenerates**
with a confident prior: once the incumbent is the library maximum, every
remaining candidate has mu < best and sd ~ 0, so EI is zero everywhere and
the arm picks at random for the rest of its budget. Measured: a *perfect*
prior under EI scores -0.019 to -0.144 against plain BO while the same prior
under UCB scores +0.057 to +0.113. A prior and an acquisition function cannot
be chosen independently.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from molbo.metrics import degrade_to_rho as prior_at_rho  # noqa: E402


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="CSV libraries; defaults to a spread of BindingDB targets")
    p.add_argument("--targets", type=int, default=8)
    p.add_argument("--rhos", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    p.add_argument("--acquisitions", nargs="+", default=["ucb", "ei"])
    p.add_argument("--budgets", type=int, nargs="+", default=[30],
                   help="the threshold is expected to MOVE with budget: at a tiny "
                        "budget the GP has almost no data so a weak prior can still "
                        "beat it, while at a large budget the GP outlearns any prior. "
                        "Where the curve sits relative to the ~0.22 that cold-target "
                        "prediction achieves decides whether there is a usable regime "
                        "at all.")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/prior_quality_threshold.json")
    args = p.parse_args(argv)

    from scipy.stats import spearmanr, wilcoxon

    from molbo.campaign import ArmConfig
    from molbo.experiment import run_sweep
    from molbo.oracles import LookupOracle

    if args.datasets:
        paths = [Path(d) for d in args.datasets]
    else:
        pool = sorted(Path("data/cache/sweep_libs").glob("*.csv"))
        if not pool:
            raise SystemExit("no libraries found; run scripts/prior_sweep.py first "
                             "or pass --datasets")
        step = max(1, len(pool) // args.targets)
        paths = pool[::step][:args.targets]
    print(f"[thr] {len(paths)} libraries, rho grid {args.rhos}, "
          f"{args.seeds} seeds, budgets {args.budgets}\n", flush=True)

    rows = []
    for path in paths:
        try:
            oracle = LookupOracle.from_dataset(path, budget=max(args.budgets))
        except Exception:
            continue
        lib = list(oracle.library)
        if len(lib) < 80:
            continue
        truth = np.array([oracle._table[s] for s in lib])
        rng = np.random.default_rng(args.seed)
        priors = {r: prior_at_rho(truth, r, rng) for r in args.rhos}
        achieved = {r: float(spearmanr(priors[r], truth).statistic) for r in args.rhos}

        for budget in args.budgets:
            oracle.budget = budget
            n_init = min(args.n_init, max(3, budget // 3))
            for acq in args.acquisitions:
                arms = [ArmConfig("plain", strategy="bo", acquisition=acq, n_init=n_init)]
                arms += [ArmConfig(f"rho{r:g}", strategy="bo", acquisition=acq,
                                   n_init=n_init, prior_mean=priors[r]) for r in args.rhos]
                m, _ = run_sweep(oracle, arms, range(args.seeds), budget=budget,
                                 k=10, library=lib, verbose=False)
                base = np.array([x.auc for x in m if x.arm == "plain"])
                for r in args.rhos:
                    v = np.array([x.auc for x in m if x.arm == f"rho{r:g}"])
                    d = v - base
                    rows.append({"library": path.stem, "n": len(lib), "acq": acq,
                                 "budget": budget, "n_init": n_init,
                                 "rho_target": r, "rho_achieved": achieved[r],
                                 "delta": float(d.mean()),
                                 "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
        print(f"[thr] {path.stem} ({len(lib)} cpds) done", flush=True)

    print(f"\nMean dAUC vs plain BO, averaged over {len(paths)} libraries\n")
    for budget in args.budgets:
      for acq in args.acquisitions:
        print(f"  budget {budget}, acquisition = {acq}")
        print(f"    {'rho':>6s}{'achieved':>10s}{'dAUC':>10s}{'libs better':>13s}")
        prev = None
        cross = None
        for r in args.rhos:
            sel = [x for x in rows if x["acq"] == acq and x["rho_target"] == r
                   and x["budget"] == budget]
            d = float(np.mean([x["delta"] for x in sel]))
            ach = float(np.mean([x["rho_achieved"] for x in sel]))
            frac = float(np.mean([x["delta"] > 0 for x in sel]))
            print(f"    {r:>6.2f}{ach:>10.3f}{d:>+10.4f}{100 * frac:>12.0f}%")
            if prev is not None and prev[1] <= 0 < d and cross is None:
                # linear interpolation between the two bracketing grid points
                cross = prev[0] + (0 - prev[1]) * (r - prev[0]) / (d - prev[1])
            prev = (r, d)
        if cross is not None:
            print(f"    -> crosses zero at rho ~ {cross:.2f}")
        else:
            print(f"    -> never crosses zero on this grid")
        print()

    print("Cold-target predictors for comparison:")
    print("  ridge on (ESM2-35M, ECFP4), 150k pairs, held-out targets : rho ~ 0.22")
    print("  per-target learned region prior                          : rho n/a (29% top-1)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rhos": args.rhos, "budgets": args.budgets,
                               "seeds": args.seeds, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
