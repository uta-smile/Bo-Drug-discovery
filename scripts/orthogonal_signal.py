"""Does an ORTHOGONAL signal keep paying at realistic budgets?

    python scripts/orthogonal_signal.py --budgets 30 60 100 200 --seeds 200

The prior-quality threshold closes one door: a prior that predicts the same
thing the GP predicts must beat the GP's own accuracy, and the GP reaches
Spearman 0.46 by 30 observations and 0.54 by 80. Cold-target prediction
delivers ~0.22, so a competing prior is useless at any realistic budget, and
the gap widens as the budget grows.

But that argument only covers a prior that *competes*. The GP knows exactly
one thing -- 2D fingerprint similarity to what has been measured. It has no
binding mode, no 3D, no selectivity. A signal carrying information the GP
structurally cannot have does not have to beat it; it adds a dimension.

Measured motivation: AutoDock Vina scores correlate rho ~ 0.24 with measured
pIC50 but only rho ~ 0.03 with the GP's posterior mean. Weak, and almost
entirely new.

This separates the two axes that were confounded in the threshold experiment.
A signal is built with a controlled correlation to the truth (``rho_true``)
AND a controlled correlation to what the GP already believes
(``rho_gp``), and both are swept. The prediction, if orthogonality is what
matters:

    at high budget, a WEAK-but-ORTHOGONAL signal beats a
    STRONGER-but-REDUNDANT one of the same rho_true.

If instead only rho_true matters, the two collapse onto one curve and the
orthogonality story is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402


def gram_schmidt(v: np.ndarray, against: np.ndarray) -> np.ndarray:
    """Component of ``v`` orthogonal to ``against`` (both z-scored)."""
    a = (against - against.mean()) / (against.std() + 1e-12)
    r = v - (v @ a) / len(a) * a
    return (r - r.mean()) / (r.std() + 1e-12)


def build_signal(truth: np.ndarray, gp_mu: np.ndarray, rho_true: float,
                 redundant: bool, rng: np.random.Generator) -> np.ndarray:
    """A signal with ~``rho_true`` to truth, either aligned with the GP or not.

    ``redundant``: the signal's non-truth component copies the GP's opinion, so
    it tells the optimiser nothing new. Otherwise that component is made
    orthogonal to the GP. Both versions have the same correlation with the
    objective -- only their overlap with what the GP already knows differs.
    """
    z = (truth - truth.mean()) / (truth.std() + 1e-12)
    g = (gp_mu - gp_mu.mean()) / (gp_mu.std() + 1e-12)
    filler = g if redundant else gram_schmidt(rng.normal(size=len(z)), g)
    filler = gram_schmidt(filler, z) if not redundant else filler
    mixed = rho_true * z + np.sqrt(max(0.0, 1 - rho_true ** 2)) * filler
    return mixed * truth.std() + truth.mean()


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", type=int, default=6)
    p.add_argument("--budgets", type=int, nargs="+", default=[30, 60, 100, 200])
    p.add_argument("--rho-true", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--acq", default="ucb", help="EI degenerates with a confident prior")
    p.add_argument("--gp-obs", type=int, default=40,
                   help="observations used to build the reference GP opinion that "
                        "'redundant' copies; roughly mid-campaign")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/orthogonal_signal.json")
    args = p.parse_args(argv)

    from scipy.stats import spearmanr, wilcoxon

    from molbo.campaign import ArmConfig
    from molbo.experiment import run_sweep
    from molbo.oracles import LookupOracle
    from molbo.surrogate import TanimotoGP, fingerprint_matrix

    pool = sorted(Path("data/cache/sweep_libs").glob("*.csv"))
    if not pool:
        raise SystemExit("no libraries; run scripts/prior_sweep.py first")
    step = max(1, len(pool) // args.targets)
    paths = pool[::step][:args.targets]
    print(f"[orth] {len(paths)} libraries, budgets {args.budgets}, "
          f"rho_true {args.rho_true}, acq {args.acq}\n", flush=True)

    rows = []
    for path in paths:
        try:
            oracle = LookupOracle.from_dataset(path, budget=max(args.budgets))
        except Exception:
            continue
        lib = list(oracle.library)
        if len(lib) < max(args.budgets) + 60:
            continue
        truth = np.array([oracle._table[s] for s in lib])
        X = fingerprint_matrix(lib)
        rng = np.random.default_rng(args.seed)
        # A stand-in for "what the GP will believe": fit on a random sample the
        # size of a mid-campaign observation set.
        idx = rng.choice(len(lib), min(args.gp_obs, len(lib) // 3), replace=False)
        gp_mu, _ = TanimotoGP().fit(X[idx], truth[idx]).predict(X)

        for rt in args.rho_true:
            sig = {k: build_signal(truth, gp_mu, rt, k == "redundant", rng)
                   for k in ("redundant", "orthogonal")}
            info = {k: (float(spearmanr(v, truth).statistic),
                        float(spearmanr(v, gp_mu).statistic)) for k, v in sig.items()}
            for budget in args.budgets:
                oracle.budget = budget
                arms = [ArmConfig("plain", strategy="bo", acquisition=args.acq,
                                  n_init=args.n_init)]
                arms += [ArmConfig(k, strategy="bo", acquisition=args.acq,
                                   n_init=args.n_init, prior_mean=v)
                         for k, v in sig.items()]
                m, _ = run_sweep(oracle, arms, range(args.seeds), budget=budget,
                                 k=10, library=lib, verbose=False)
                base = np.array([x.auc for x in m if x.arm == "plain"])
                row = {"library": path.stem, "n": len(lib), "budget": budget,
                       "rho_true": rt}
                for k in sig:
                    v = np.array([x.auc for x in m if x.arm == k])
                    row[f"d_{k}"] = float((v - base).mean())
                    row[f"p_{k}"] = float(wilcoxon(v - base).pvalue)
                    row[f"rho_true_{k}"], row[f"rho_gp_{k}"] = info[k]
                rows.append(row)
        print(f"[orth] {path.stem} ({len(lib)} cpds) done", flush=True)

    print(f"\nMean dAUC vs plain BO ({args.acq}), averaged over libraries")
    print(f"{'rho_true':>9s}{'budget':>8s}{'redundant':>12s}{'orthogonal':>12s}"
          f"{'orth - red':>12s}")
    print("-" * 53)
    for rt in args.rho_true:
        for b in args.budgets:
            sel = [x for x in rows if x["rho_true"] == rt and x["budget"] == b]
            if not sel:
                continue
            red = float(np.mean([x["d_redundant"] for x in sel]))
            orth = float(np.mean([x["d_orthogonal"] for x in sel]))
            print(f"{rt:>9.2f}{b:>8d}{red:>+12.4f}{orth:>+12.4f}{orth - red:>+12.4f}")
        print()

    if rows:
        a = np.array([x["rho_gp_redundant"] for x in rows])
        b = np.array([x["rho_gp_orthogonal"] for x in rows])
        print(f"sanity: correlation with the GP's opinion -- "
              f"redundant {a.mean():+.2f}, orthogonal {b.mean():+.2f}")
        d = np.array([x["d_orthogonal"] - x["d_redundant"] for x in rows])
        print(f"\nOVERALL orthogonal minus redundant: {d.mean():+.4f} "
              f"(p={wilcoxon(d).pvalue:.4f}, better on {100 * (d > 0).mean():.0f}%)")
        hi = [x for x in rows if x["budget"] >= 100]
        if hi:
            dh = np.array([x["d_orthogonal"] - x["d_redundant"] for x in hi])
            print(f"  restricted to budget >= 100: {dh.mean():+.4f} "
                  f"(n={len(hi)})  <- the realistic regime")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"budgets": args.budgets, "rho_true": args.rho_true,
                               "seeds": args.seeds, "acq": args.acq, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
