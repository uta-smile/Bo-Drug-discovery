"""Is it the prior that fails, or committing to a region at all?

    python scripts/soft_prior_sweep.py --n-targets 24 --budget 30 --seeds 200

The 24-target sweep found the learned prior loses to a free bandit by -0.019
overall, and -- the surprising part -- **loses even on the targets where it
picked the right region** (-0.0016, n=7). Break-even accuracy comes out above
100%, so no chooser accuracy rescues it. That relocates the problem from the
model to the mechanism: a hard region mask cannot be walked away from cheaply,
while a bandit can.

This tests that directly. Same prior scores, same regions, same seeds; the only
change is how the preference is applied:

``hard``
    The region becomes a mask. Molecules outside it cannot be bought.
``soft``
    The region's score is a *penalty* on everything outside it, applied to the
    acquisition function. BO keeps the whole library and can leave whenever the
    evidence says so.

``molbo.constraints.apply_constraints`` already implements both -- a mask
degrades to soft when it admits fewer than ``min_candidates`` unseen molecules
-- so forcing soft mode is a parameter, not new machinery.

Prediction if the diagnosis is right: soft keeps most of the upside on the
targets where the prior is correct, and loses far less where it is wrong.
Prediction if the diagnosis is wrong: soft is simply a weaker version of hard
and both lose.
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

FORCE_SOFT = 10 ** 9
"""A min_candidates no mask can satisfy, so apply_constraints always demotes
to the soft path. Cleaner than adding a mode flag to ArmConfig for one study."""

ESM2 = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D",
        "150M": "facebook/esm2_t30_150M_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-targets", type=int, default=24)
    p.add_argument("--min-compounds", type=int, default=250)
    p.add_argument("--max-compounds", type=int, default=1200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--soft-penalty", type=float, nargs="+", default=[0.002, 0.01, 0.05],
                   help="acquisition penalty for molecules outside the chosen region, "
                        "subtracted from EI. MUST be sized against the EI scale, not "
                        "the objective's: measured EI spread here is 0.010 at 10 "
                        "observations rising to 0.138 at 30, so a penalty of 0.5 "
                        "(the ArmConfig default) is 4-100x the whole range and makes "
                        "'soft' behave exactly like a hard mask -- the first run of "
                        "this script scored hard and soft identically to four decimals.")
    p.add_argument("--esm", default="35M", choices=list(ESM2))
    p.add_argument("--n-train", type=int, default=150_000)
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/soft_prior_sweep.json")
    args = p.parse_args(argv)

    from prior_sweep import load_pairs  # same corpus, same filters

    from molbo.campaign import ArmConfig
    from molbo.clm import Encoder
    from molbo.clusters import UCBClusterSource, cluster_library, concentration
    from molbo.experiment import run_sweep, summarize_sweep
    from molbo.oracles import LookupOracle
    from molbo.prior import PriorClusterSource
    from molbo.surrogate import fingerprint_matrix

    df = load_pairs()
    size = df.groupby("target_id").size()
    seqs = df.drop_duplicates("target_id").set_index("target_id").sequence
    pool = [t for t in size[(size >= args.min_compounds)
                            & (size <= args.max_compounds)].index
            if len(seqs[t]) <= 1022]
    rng = np.random.default_rng(args.seed)
    # Same seed as prior_sweep.py, so these are the SAME targets -- the hard
    # and soft numbers are paired, not two independent draws.
    test = [pool[i] for i in rng.permutation(len(pool))[:args.n_targets]]
    print(f"[soft] {len(test)} targets (same draw as prior_sweep.py)", flush=True)

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
    mu, ym = Xtr.mean(0), ytr.mean()
    A = Xtr - mu
    G = A.T @ A
    G[np.diag_indices_from(G)] += args.alpha
    w = np.linalg.solve(G, A.T @ (ytr - ym))
    print(f"[soft] ridge in {time.time() - t0:.0f}s\n", flush=True)

    tmp = Path("data/cache/sweep_libs")
    tmp.mkdir(parents=True, exist_ok=True)
    soft_names = [f"bo_prior_soft{pen:g}" for pen in args.soft_penalty]
    hdr = f"{'target':<10s}{'n':>5s}{'':>6s}{'bandit':>9s}{'hard':>9s}" + \
          "".join(f"{n.replace('bo_prior_', ''):>10s}" for n in soft_names)
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
        if len(lib) < args.k * 8:
            continue
        values = np.array([oracle._table[s] for s in lib])
        labels = cluster_library(lib, args.k, method="kmeans")
        best = concentration(labels, values, 0.01)["best_cluster"]

        Xl = np.hstack([np.repeat(P[tid][None, :], len(lib), 0).astype(np.float32),
                        fingerprint_matrix(lib).astype(np.float32)])
        pred = (Xl - mu) @ w + ym
        thr = np.quantile(pred, 0.99)
        sizes = np.bincount(labels, minlength=args.k).astype(float)
        pscore = np.array([(pred[labels == c] >= thr).sum() / max(1.0, sizes[c])
                           if (labels == c).any() else -1.0 for c in range(args.k)])
        pick = int(np.argmax(pscore))

        arms = [
            ArmConfig("bo_whole", strategy="bo", acquisition="ei", n_init=args.n_init),
            ArmConfig("bo_ucb_cluster", strategy="bo", acquisition="ei", n_init=args.n_init,
                      min_candidates=10, constraint_every=1,
                      constraint_source=UCBClusterSource(labels=labels)),
            ArmConfig("bo_prior_hard", strategy="bo", acquisition="ei", n_init=args.n_init,
                      min_candidates=10, constraint_every=1,
                      constraint_source=PriorClusterSource(labels=labels, scores=pscore)),
        ]
        for pen, name in zip(args.soft_penalty, soft_names):
            arms.append(ArmConfig(
                name, strategy="bo", acquisition="ei", n_init=args.n_init,
                min_candidates=FORCE_SOFT, soft_penalty=pen, constraint_every=1,
                constraint_source=PriorClusterSource(labels=labels, scores=pscore)))

        m, _ = run_sweep(oracle, arms, range(args.seeds), budget=args.budget, k=10,
                         library=lib, verbose=False)
        s = summarize_sweep(m, reference="bo_whole")
        d = {a.name: (s["comparisons"].get(a.name, {}).get("mean_diff") or 0.0) for a in arms}
        row = {"target_id": tid, "n": len(lib), "pick": pick, "best": int(best),
               "hit": pick == int(best), **{f"d_{k2}": v for k2, v in d.items()}}
        rows.append(row)
        print(f"{tid:<10s}{len(lib):>5d}{'  HIT' if row['hit'] else ' miss':>6s}"
              f"{d['bo_ucb_cluster']:>+9.4f}{d['bo_prior_hard']:>+9.4f}"
              + "".join(f"{d[n]:>+10.4f}" for n in soft_names), flush=True)

    def mean(key, subset=None):
        sel = rows if subset is None else [r for r in rows if r["hit"] == subset]
        return float(np.mean([r[key] for r in sel])) if sel else float("nan")

    band = "d_bo_ucb_cluster"
    print(f"\n{len(rows)} targets | prior accuracy "
          f"{100 * np.mean([r['hit'] for r in rows]):.1f}%")
    print(f"\n{'arm':<20s}{'overall':>10s}{'when right':>13s}{'when wrong':>13s}"
          f"{'vs bandit':>12s}")
    print("-" * 68)
    for name in ["d_bo_ucb_cluster", "d_bo_prior_hard"] + [f"d_{n}" for n in soft_names]:
        print(f"{name[2:]:<20s}{mean(name):>+10.4f}{mean(name, True):>+13.4f}"
              f"{mean(name, False):>+13.4f}{mean(name) - mean(band):>+12.4f}")

    best_soft = max(soft_names, key=lambda n: mean(f"d_{n}"))
    delta = mean(f"d_{best_soft}") - mean(band)
    print("\nVERDICT")
    if delta > 0:
        print(f"  Soft application beats the free bandit by {delta:+.4f} where hard "
              f"application lost by {mean('d_bo_prior_hard') - mean(band):+.4f}.")
        print("  The diagnosis holds: the prior was never the problem -- committing was.")
    else:
        print(f"  Soft does not rescue it either ({delta:+.4f} vs the bandit), so the "
              f"failure\n  is not about commitment. Both ways of using a region prior "
              f"lose, and the\n  region hook should be reported as closed.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"budget": args.budget, "seeds": args.seeds, "k": args.k,
                               "soft_penalty": args.soft_penalty, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
