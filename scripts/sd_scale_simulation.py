"""When does compressing the acquisition's uncertainty pay? Ground truth known.

    python scripts/sd_scale_simulation.py --out results/sd_scale_sim.json

FINDINGS 40 measured that a single-task arm at sd x 0.80 reproduces everything a
pure-noise auxiliary buys on JAK, and that the same knob is worth nothing on CYP.
FINDINGS 41 then refuted both explanations available from the real blocks: the
GP is *under*-dispersed on both (std(z) = 1.4-4.8), and the compounds the two
exploration levels buy are statistically indistinguishable. What survived was an
accounting identity, not a mechanism.

Two real blocks cannot settle this, because everything that distinguishes them
-- library size, budget, assay noise, how well a Tanimoto kernel describes the
target -- moves at once. Here they are separated, on synthetic data where the
answer is known.

**The hypothesis.** Expected improvement trades mu against sd at a fixed rate,
and that rate is right for some surrogates and wrong for others: one that ranks
well does not need to explore, and EI makes it explore anyway. So the
recall-optimal sd scale should *fall* as the surrogate's held-out rank
correlation rises. JAK (rho = 0.73 at n = 300, optimum 0.80) and CYP
(rho = 0.59, optimum 1.00) are two points consistent with that, which is not
evidence.

**The design.** A synthetic library of `n` points in a `d`-dimensional latent
space, with the campaign's gram matrix built from those coordinates. The target
is

    y = alpha * f + sqrt(1 - alpha^2) * g        (+ cliffs)

where `f` is a GP draw on that same kernel -- everything the surrogate can
possibly learn -- and `g` is an independent draw the kernel cannot see.
``alpha`` is therefore a direct, monotone dial on surrogate quality rather than
a property of a protein.

`--cliffs` adds a sparse set of large single-point spikes. FINDINGS 41 found the
real blocks' residuals are cliff-driven (|z| > 2 at 6-12% against 4.6% for a
Gaussian), and a pure GP draw has no cliffs at all, so without this the
simulation is in a regime the real data is not.

Everything downstream -- the fit, the acquisition, the batch selection, the
tie-breaking -- is `dense_block_campaign.run_campaign`, unchanged, so what is
simulated is this package's campaign rather than a reimplementation of one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from molbo.surrogate import TanimotoGP  # noqa: E402
from dense_block_campaign import run_campaign  # noqa: E402

# Declared before any run, as in FINDINGS 40. Both sides of 1.0, because a
# flattened posterior would make an arm more exploratory and the CYP sweep had
# to be run in both directions before it could be called negative.
SCALES = (1.60, 1.30, 1.15, 1.00, 0.90, 0.80, 0.70, 0.60, 0.45, 0.30)

# Learnable fraction of the target. Chosen to bracket the real blocks: the
# single-task Tanimoto GP reaches Spearman 0.59 (CYP) to 0.73 (JAK) against
# held-out truth at campaign scale.
ALPHAS = (0.30, 0.45, 0.60, 0.72, 0.82, 0.90, 0.96, 1.00)


def rbf(Z: np.ndarray, ell: float) -> np.ndarray:
    """Unit-diagonal gram. The campaign assumes ``k(x, x) = 1``, as Tanimoto is."""
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / ell ** 2)


def target(rng, K, *, alpha, cliffs, cliff_frac=0.02, cliff_size=3.0,
           jitter=1e-8):
    """``alpha`` of a GP draw on ``K``, the rest unlearnable, plus optional cliffs.

    The two components are standardised before mixing, so ``alpha`` is the
    correlation between the learnable part and the target and not an arbitrary
    weight.
    """
    n = K.shape[0]
    L = np.linalg.cholesky(K + jitter * np.eye(n))
    f = L @ rng.standard_normal(n)
    g = rng.standard_normal(n)
    f = (f - f.mean()) / f.std()
    g = (g - g.mean()) / g.std()
    y = alpha * f + np.sqrt(max(1.0 - alpha ** 2, 0.0)) * g
    if cliffs:
        # A handful of compounds that are far more potent than their
        # neighbourhood -- the single substitution worth three log units. They
        # are what makes a chemical landscape not a GP draw.
        hit = rng.choice(n, size=max(1, int(cliff_frac * n)), replace=False)
        y[hit] += cliff_size * rng.random(len(hit))
    return y


def ranking_quality(K, y, *, n_obs, rng, draws=8):
    """Held-out Spearman of the surrogate at campaign scale.

    The quantity a practitioner can measure before committing a budget, which
    is the point: if the optimum tracks it, it is a usable diagnostic.
    """
    n = len(y)
    out = []
    for _ in range(draws):
        obs = rng.choice(n, size=n_obs, replace=False)
        cand = np.setdiff1d(np.arange(n), obs)
        gp = TanimotoGP().fit_gram(K[np.ix_(obs, obs)], y[obs])
        # `predict_gram` returns the sd by default; only the mean is a ranking.
        mu = gp.predict_gram(K[np.ix_(cand, obs)], k_diag=np.ones(len(cand)),
                             return_std=False)
        out.append(float(spearmanr(mu, y[cand]).statistic))
    return float(np.mean(out))


def cell(seed, *, n, d, ell, alpha, cliffs, budget, batch, n_init, k, repeats):
    """One quality level: its ranking quality and its recall at every sd scale."""
    rng = np.random.default_rng(seed)
    Z = rng.random((n, d))
    K = rbf(Z, ell)
    y = target(rng, K, alpha=alpha, cliffs=cliffs)
    rho = ranking_quality(K, y, n_obs=budget // 2,
                          rng=np.random.default_rng(seed + 1))
    top = set(np.argsort(y)[-k:].tolist())
    out = {}
    for s in SCALES:
        hits = []
        for r in range(repeats):
            # Same stream per repeat across scales, so the initial design is
            # identical and the only difference is the exploration level.
            _, idx = run_campaign(
                K, y, [], [], arm="stl", budget=budget, batch=batch,
                n_init=n_init, k=k, acquisition="ei", sd_scale=s,
                rng=np.random.default_rng(seed + 1009 * r))
            hits.append(len(top & set(idx.tolist())) / k)
        out[s] = hits
    return rho, out


def run(args, cliffs):
    rows = []
    for alpha in ALPHAS:
        rhos, pooled = [], {s: [] for s in SCALES}
        for rep in range(args.replicates):
            seed = args.seed + 10_000 * rep + int(alpha * 977)
            rho, rec = cell(seed, n=args.n, d=args.d, ell=args.ell,
                            alpha=alpha, cliffs=cliffs, budget=args.budget,
                            batch=args.batch, n_init=args.n_init, k=args.k,
                            repeats=args.repeats)
            rhos.append(rho)
            for s in SCALES:
                pooled[s].extend(rec[s])
        m = {s: float(np.mean(v)) for s, v in pooled.items()}
        best = max(m, key=m.get)
        rows.append({"alpha": alpha, "rho": float(np.mean(rhos)), "recall": m,
                     "best_scale": best,
                     "gain_vs_1": (m[best] - m[1.0]) * 100,
                     "n_campaigns": len(pooled[1.0])})
    rows.sort(key=lambda r: r["rho"])
    return rows


def show(label, rows):
    print(f"\n  === {label} ===")
    print(f"  {'alpha':>6}{'rho':>8}{'best':>7}{'gain':>8}   " +
          "".join(f"{s:>6}" for s in SCALES))
    print("  " + "-" * (29 + 6 * len(SCALES)))
    for r in rows:
        print(f"  {r['alpha']:>6.2f}{r['rho']:>+8.3f}{r['best_scale']:>7.2f}"
              f"{r['gain_vs_1']:>+8.2f}   " +
              "".join(f"{r['recall'][s] * 100:>6.1f}" for s in SCALES))
    rho = np.array([r["rho"] for r in rows])
    best = np.array([r["best_scale"] for r in rows])
    st = spearmanr(rho, best)
    print(f"  rho(surrogate quality, optimal sd scale) = {st.statistic:+.3f}, "
          f"p = {st.pvalue:.4f}, n = {len(rows)}")
    return {"statistic": float(st.statistic), "p": float(st.pvalue)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=500, help="library size")
    p.add_argument("--d", type=int, default=5, help="latent dimension")
    p.add_argument("--ell", type=float, default=0.45, help="kernel length scale")
    p.add_argument("--budget", type=int, default=75,
                   help="kept well below saturation; at 30%% of this library "
                        "recall hits 100%% and every scale ties")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--repeats", type=int, default=24)
    p.add_argument("--replicates", type=int, default=4,
                   help="independent libraries per level; one draw is a sample "
                        "of a landscape, not of a regime")
    p.add_argument("--alphas", default=None,
                   help="comma-separated learnable fractions, overriding the "
                        "eight the pilot used. More levels is the only lever "
                        "on the Spearman's power: replicates tighten each "
                        "point, they do not add one")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/sd_scale_sim.json"))
    args = p.parse_args(argv)
    if args.alphas:
        global ALPHAS
        ALPHAS = tuple(float(x) for x in args.alphas.split(","))

    print(f"synthetic: n={args.n} d={args.d} ell={args.ell} "
          f"budget={args.budget} ({args.budget / args.n:.0%} of library) "
          f"k={args.k}")
    print(f"           {args.repeats} repeats x {args.replicates} libraries "
          f"= {args.repeats * args.replicates} campaigns per (level, scale)")

    smooth = run(args, cliffs=False)
    cliffy = run(args, cliffs=True)
    st_s = show("smooth landscape (a GP draw)", smooth)
    st_c = show("with activity cliffs (what the real blocks look like)", cliffy)

    print()
    for lab, st in (("smooth", st_s), ("cliffs", st_c)):
        if st["statistic"] < 0 and st["p"] < 0.05:
            print(f"  {lab}: PREDICTION HELD -- the better the surrogate ranks, "
                  f"the more compressing sd is worth.")
        elif st["statistic"] < 0:
            print(f"  {lab}: direction as predicted, not significant at "
                  f"{len(ALPHAS)} levels.")
        else:
            print(f"  {lab}: PREDICTION FAILED -- the optimum does not fall "
                  f"with surrogate quality.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"config": {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()},
         "scales": SCALES, "alphas": ALPHAS,
         "smooth": {"rows": smooth, "quality_vs_optimum": st_s},
         "cliffs": {"rows": cliffy, "quality_vs_optimum": st_c}},
        indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
