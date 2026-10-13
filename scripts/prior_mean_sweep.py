"""Use the cold-target predictor as the GP's MEAN, not as a filter.

    python scripts/prior_mean_sweep.py --n-targets 24 --budget 30 --seeds 200

Three ways of using the same learned prior have now been measured over the
same 24 held-out targets:

    hard mask over regions   -0.0295 vs plain BO  (loses even when correct)
    soft penalty over regions -0.0023 vs plain BO  (a wash; +0.027 over hard)
    GP prior mean             <- this script

The trend says the softer the application, the better, and the prior mean is
the softest available: the GP is fitted to ``y - prior`` and the prior is
added back to the posterior. It steers only where no data has arrived and is
overwritten everywhere it has, automatically, with no penalty to tune. It is
also the classical way to put knowledge into a GP, which the region-masking
approach was not.

It is also per-molecule rather than per-region, so it uses the predictor at
the resolution it was trained at instead of collapsing it to one of ten
choices.

Arms, and what each rules out:

``bo_plain``            zero-mean GP, the reference
``bo_priormean``        prior mean = cold-target ridge prediction
``bo_priormean_shuf``   the same predictions, shuffled across molecules. If
                        this helps too, the gain is from perturbing the
                        acquisition surface and not from the prior being right.
``bo_priormean_oracle`` prior mean = the true values. The ceiling for this
                        mechanism; reads the answer key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

ESM2 = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D",
        "150M": "facebook/esm2_t30_150M_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-targets", type=int, default=24)
    p.add_argument("--min-compounds", type=int, default=250)
    p.add_argument("--max-compounds", type=int, default=1200)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--esm", default="35M", choices=list(ESM2))
    p.add_argument("--n-train", type=int, default=150_000)
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--scale", type=float, nargs="+", default=[1.0],
                   help="multiplier on the prior before it becomes the GP mean. "
                        "<1 shrinks a prior you do not fully trust toward zero-mean.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/prior_mean_sweep.json")
    args = p.parse_args(argv)

    from prior_sweep import load_pairs

    from molbo.campaign import ArmConfig
    from molbo.clm import Encoder
    from molbo.experiment import run_sweep, summarize_sweep
    from molbo.oracles import LookupOracle
    from molbo.surrogate import fingerprint_matrix

    df = load_pairs()
    size = df.groupby("target_id").size()
    seqs = df.drop_duplicates("target_id").set_index("target_id").sequence
    pool = [t for t in size[(size >= args.min_compounds)
                            & (size <= args.max_compounds)].index
            if len(seqs[t]) <= 1022]
    rng = np.random.default_rng(args.seed)
    test = [pool[i] for i in rng.permutation(len(pool))[:args.n_targets]]
    print(f"[pm] {len(test)} targets (same draw as the other sweeps)", flush=True)

    train = df[~df.target_id.isin(set(test))]
    train = train[train.target_id.isin(size[(size >= 200) & (size <= 3000)].index)]
    train = train[[len(seqs[t]) <= 1022 for t in train.target_id]]
    enc = Encoder(model=ESM2[args.esm], revision=None, pooling="mean", max_len=1024,
                  cache_dir="data/cache/esm")
    uniq = sorted(set(train.target_id) | set(test))
    P = dict(zip(uniq, enc.embed([seqs[t] for t in uniq], progress=True)))
    tr = train.sample(n=min(args.n_train, len(train)), random_state=0)
    t0 = time.time()
    Xtr = np.hstack([np.stack([P[t] for t in tr.target_id]).astype(np.float32),
                     fingerprint_matrix(list(tr.smiles)).astype(np.float32)])
    ytr = tr.pic50.to_numpy()
    mu0, ym = Xtr.mean(0), ytr.mean()
    A = Xtr - mu0
    G = A.T @ A
    G[np.diag_indices_from(G)] += args.alpha
    w = np.linalg.solve(G, A.T @ (ytr - ym))
    print(f"[pm] ridge on {Xtr.shape} in {time.time() - t0:.0f}s\n", flush=True)

    tmp = Path("data/cache/sweep_libs")
    tmp.mkdir(parents=True, exist_ok=True)
    scale_names = [f"bo_priormean{s:g}" for s in args.scale]
    hdr = (f"{'target':<10s}{'n':>5s}{'rho':>7s}"
           + "".join(f"{n.replace('bo_', ''):>13s}" for n in scale_names)
           + f"{'shuffled':>11s}{'oracle':>9s}")
    print(hdr + "\n" + "-" * len(hdr), flush=True)

    rows = []
    for tid in test:
        sub = df[df.target_id == tid]
        csv = tmp / f"{tid}.csv"
        sub.rename(columns={"smiles": "canonical_smiles", "pic50": "pchembl_value"})[
            ["canonical_smiles", "pchembl_value"]].to_csv(csv, index=False)
        try:
            oracle = LookupOracle.from_dataset(csv, budget=args.budget)
        except Exception:
            continue
        lib = list(oracle.library)
        if len(lib) < 80:
            continue
        truth = np.array([oracle._table[s] for s in lib])
        Xl = np.hstack([np.repeat(P[tid][None, :], len(lib), 0).astype(np.float32),
                        fingerprint_matrix(lib).astype(np.float32)])
        pred = (Xl - mu0) @ w + ym
        from scipy.stats import spearmanr
        rho = float(spearmanr(pred, truth).statistic)
        shuf = np.random.default_rng(7).permutation(pred)

        arms = [ArmConfig("bo_plain", strategy="bo", acquisition="ei", n_init=args.n_init)]
        for sc, name in zip(args.scale, scale_names):
            arms.append(ArmConfig(name, strategy="bo", acquisition="ei",
                                  n_init=args.n_init, prior_mean=sc * pred))
        arms += [
            ArmConfig("bo_priormean_shuf", strategy="bo", acquisition="ei",
                      n_init=args.n_init, prior_mean=shuf),
            ArmConfig("bo_priormean_oracle", strategy="bo", acquisition="ei",
                      n_init=args.n_init, prior_mean=truth),
        ]
        m, _ = run_sweep(oracle, arms, range(args.seeds), budget=args.budget, k=10,
                         library=lib, verbose=False)
        s = summarize_sweep(m, reference="bo_plain")
        d = {a.name: (s["comparisons"].get(a.name, {}).get("mean_diff") or 0.0) for a in arms}
        rows.append({"target_id": tid, "n": len(lib), "rho": rho,
                     **{f"d_{k2}": v for k2, v in d.items()}})
        print(f"{tid:<10s}{len(lib):>5d}{rho:>+7.3f}"
              + "".join(f"{d[n]:>+13.4f}" for n in scale_names)
              + f"{d['bo_priormean_shuf']:>+11.4f}{d['bo_priormean_oracle']:>+9.4f}",
              flush=True)

    from scipy.stats import wilcoxon

    print(f"\n{len(rows)} targets | mean cold-target rho "
          f"{np.mean([r['rho'] for r in rows]):+.3f}")
    print(f"\n{'arm':<24s}{'mean dAUC':>11s}{'p vs plain':>12s}{'wins':>7s}")
    print("-" * 54)
    for name in scale_names + ["bo_priormean_shuf", "bo_priormean_oracle"]:
        v = np.array([r[f"d_{name}"] for r in rows])
        pv = wilcoxon(v).pvalue if np.any(v != 0) else 1.0
        print(f"{name:<24s}{v.mean():>+11.4f}{pv:>12.4f}{100 * (v > 0).mean():>6.0f}%")

    best = max(scale_names, key=lambda n: np.mean([r[f"d_{n}"] for r in rows]))
    bv = np.array([r[f"d_{best}"] for r in rows])
    sv = np.array([r["d_bo_priormean_shuf"] for r in rows])
    print("\nVERDICT")
    print(f"  best prior-mean arm: {best} at {bv.mean():+.4f} (p={wilcoxon(bv).pvalue:.4f})")
    print(f"  shuffled control   : {sv.mean():+.4f}  -- the gain is "
          f"{'REAL' if bv.mean() > sv.mean() else 'NOT'} about the prior being right")
    d = bv - sv
    print(f"  real minus shuffled: {d.mean():+.4f}  p={wilcoxon(d).pvalue:.4f}")
    for other, label in [("results/soft_prior_sweep.json", "soft region penalty"),
                         ("results/prior_sweep.json", "hard region mask")]:
        try:
            o = json.load(open(other))["rows"]
            k = "d_bo_prior_soft0.002" if "soft" in other else "d_bo_prior_hard"
            print(f"  vs {label:<20s}: {np.mean([x[k] for x in o]):+.4f}")
        except Exception:
            pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"budget": args.budget, "seeds": args.seeds,
                               "scale": args.scale, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
