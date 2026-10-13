"""Does a learned region prior pay, across many held-out targets?

    python scripts/prior_sweep.py --n-targets 24 --budget 30 --seeds 200

The asymmetry result rests on two campaigns -- one where the prior picked the
right region (USP7, +0.008 over a free bandit) and one where it missed
(cruzipain, -0.094). The arithmetic that follows from those two numbers says a
prior needs ~92% accuracy to break even, but the magnitudes each come from a
single example. This runs the whole pipeline over many targets so the
distribution is measured rather than extrapolated.

Each BindingDB target with 200-3000 measured compounds is already a library
with a real pIC50 oracle, so no new data is needed -- 607 of them exist.

Design, and the two things it is careful about:

**One model, all test targets held out.** The ridge is fitted once on the
targets *not* in the test set. Fitting per target would be 20x the compute and
would also invite fitting on a target that later appears as a test case.

**Campaign outcome, not prediction accuracy.** Region top-1 accuracy was
measured at 25% for the best chooser and was actively misleading twice in this
project -- 'pick the largest region' wins the offline metric and is the worst
campaign arm tested. So every target here is run as a paired campaign against
plain BO, a cold bandit, and the hindsight ceiling.
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


def load_pairs():
    from tdc.utils.load import pd_load

    df = pd_load("bindingdb_ic50", "data/cache/tdc")
    df.columns = ["drug_id", "smiles", "target_id", "sequence", "y"]
    df = df.dropna(subset=["smiles", "sequence", "y"])
    df["pic50"] = 9.0 - np.log10(np.clip(df.y.astype(float), 1e-4, 1e11))
    return df.groupby(["target_id", "smiles"], as_index=False).agg(
        pic50=("pic50", "median"), sequence=("sequence", "first"))


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
    p.add_argument("--min-candidates", type=int, default=10)
    p.add_argument("--esm", default="35M", choices=list(ESM2))
    p.add_argument("--n-train", type=int, default=150_000)
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/prior_sweep.json")
    args = p.parse_args(argv)

    from molbo.campaign import ArmConfig
    from molbo.clm import Encoder
    from molbo.clusters import FixedClusterSource, UCBClusterSource, cluster_library, concentration
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
    test = [pool[i] for i in rng.permutation(len(pool))[:args.n_targets]]
    print(f"[sweep] {len(pool)} eligible targets; testing {len(test)}", flush=True)

    # One model, fitted on everything that is not a test target.
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
    print(f"[sweep] ridge on {Xtr.shape} from {tr.target_id.nunique()} targets "
          f"in {time.time() - t0:.0f}s\n", flush=True)

    tmp = Path("data/cache/sweep_libs")
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    hdr = (f"{'target':<10s}{'n':>5s}{'pick':>5s}{'best':>5s}{'':>4s}"
           f"{'ceiling':>9s}{'bandit':>9s}{'prior':>9s}{'prior-bandit':>14s}")
    print(hdr + "\n" + "-" * len(hdr), flush=True)
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
        conc = concentration(labels, values, 0.01)
        best = conc["best_cluster"]

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
                      min_candidates=args.min_candidates, constraint_every=1,
                      constraint_source=UCBClusterSource(labels=labels)),
            ArmConfig("bo_prior_cluster", strategy="bo", acquisition="ei", n_init=args.n_init,
                      min_candidates=args.min_candidates, constraint_every=1,
                      constraint_source=PriorClusterSource(labels=labels, scores=pscore)),
            ArmConfig("bo_best_cluster", strategy="bo", acquisition="ei", n_init=args.n_init,
                      min_candidates=args.min_candidates,
                      constraint_source=FixedClusterSource(labels, best)),
        ]
        m, _ = run_sweep(oracle, arms, range(args.seeds), budget=args.budget, k=10,
                         library=lib, verbose=False)
        s = summarize_sweep(m, reference="bo_whole")
        d = {a.name: (s["comparisons"].get(a.name, {}).get("mean_diff") or 0.0) for a in arms}
        row = {"target_id": tid, "n": len(lib), "pick": pick, "best": int(best),
               "hit": pick == int(best), "best_lift": conc["best_lift"],
               "sizes": sizes.tolist(), **{f"d_{k2}": v for k2, v in d.items()},
               "prior_minus_bandit": d["bo_prior_cluster"] - d["bo_ucb_cluster"]}
        rows.append(row)
        print(f"{tid:<10s}{len(lib):>5d}{pick:>5d}{best:>5d}"
              f"{'  HIT' if row['hit'] else ' miss':>4s}"
              f"{d['bo_best_cluster']:>+9.4f}{d['bo_ucb_cluster']:>+9.4f}"
              f"{d['bo_prior_cluster']:>+9.4f}{row['prior_minus_bandit']:>+14.4f}", flush=True)

    hits = [r for r in rows if r["hit"]]
    miss = [r for r in rows if not r["hit"]]
    g = float(np.mean([r["prior_minus_bandit"] for r in hits])) if hits else float("nan")
    l = float(np.mean([r["prior_minus_bandit"] for r in miss])) if miss else float("nan")
    acc = len(hits) / max(1, len(rows))
    print(f"\n{len(rows)} targets | prior top-1 accuracy {acc:.1%} "
          f"(random {1 / args.k:.0%})")
    print(f"  mean prior-minus-bandit when right : {g:+.4f}  (n={len(hits)})")
    print(f"  mean prior-minus-bandit when wrong : {l:+.4f}  (n={len(miss)})")
    overall = float(np.mean([r["prior_minus_bandit"] for r in rows])) if rows else float("nan")
    print(f"  OVERALL mean prior-minus-bandit    : {overall:+.4f}")
    if np.isfinite(g) and np.isfinite(l) and g > l:
        print(f"  break-even accuracy needed         : {-l / (g - l):.1%}  "
              f"(measured {acc:.1%})")
    print(f"\n  verdict: the prior {'PAYS' if overall > 0 else 'does NOT pay'} "
          f"against a free bandit, averaged over held-out targets.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"budget": args.budget, "seeds": args.seeds, "k": args.k,
                               "n_targets": len(rows), "accuracy": acc,
                               "gain_when_right": g, "loss_when_wrong": l,
                               "overall": overall, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
