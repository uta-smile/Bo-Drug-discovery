"""Build the training corpus for a learned cluster prior, from BindingDB.

    python scripts/build_prior_data.py --out data/prior/clusters.npz

What the prior has to do: look at a target and a set of candidate regions of
chemical space, and say which region holds the actives -- **before any
compound is measured**. That is the only thing it can offer that a bandit over
regions cannot, because a bandit must pay samples to find out.

Measured on cruzipain and USP7 at k=10: the hindsight-best region is worth
+0.061 to +0.084 AUC over searching the whole library, and at a budget of 30
calls a UCB bandit captures only 2-22% of that, because it spends the budget
on first visits. The gap is what this prior is for, and it widens as the
budget shrinks.

Training signal, per target:
    inputs   the protein sequence, and one centroid fingerprint per region
    label    which region actually contained the most top-1% compounds
That is a discrete choice over ~k options -- exactly the shape a contrastive
scoring model handles in one forward pass, and exactly the shape an
autoregressive model handles badly.

BindingDB via PyTDC gives 746 targets with 200-3000 measured compounds each
and their sequences, so the prior can be trained across targets and tested on
targets it has never seen. Held-out targets are the only honest test: on a
target in the training set it could simply have memorised the answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402


def load_bindingdb(path: str = "data/cache/tdc") -> "object":
    """BindingDB IC50 with sequences, bypassing ``tdc.multi_pred``.

    That module imports ``tiledbsoma`` (single-cell), which this project
    deliberately does not install -- see the ``pmo`` extra in pyproject.
    """
    from tdc.utils.load import pd_load

    df = pd_load("bindingdb_ic50", path)
    df.columns = ["drug_id", "smiles", "target_id", "sequence", "y"]
    return df


def to_pic50(nm) -> np.ndarray:
    """BindingDB reports IC50 in nM. pIC50 = -log10(M) = 9 - log10(nM)."""
    nm = np.asarray(nm, dtype=float)
    nm = np.clip(nm, 1e-4, 1e11)      # a reported 0 nM is a data error, not infinite potency
    return 9.0 - np.log10(nm)


def main(argv=None) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-compounds", type=int, default=200)
    p.add_argument("--max-compounds", type=int, default=3000)
    p.add_argument("--max-seq-len", type=int, default=1022, help="ESM2 caps at 1026 positions")
    p.add_argument("--k", type=int, default=10, help="regions per target")
    p.add_argument("--top-frac", type=float, default=0.01)
    p.add_argument("--max-targets", type=int, default=None, help="cap, for a trial run")
    p.add_argument("--out", default="data/prior/clusters.npz")
    args = p.parse_args(argv)

    from molbo.clusters import cluster_library, concentration
    from molbo.surrogate import fingerprint_matrix

    print("[prior] loading BindingDB...", flush=True)
    df = load_bindingdb()
    df = df.dropna(subset=["smiles", "sequence", "y"])
    df["pic50"] = to_pic50(df["y"])
    # One value per (target, molecule): BindingDB carries replicate rows from
    # different papers, and a duplicate would weight that compound twice.
    df = df.groupby(["target_id", "smiles"], as_index=False).agg(
        pic50=("pic50", "median"), sequence=("sequence", "first"))

    size = df.groupby("target_id").size()
    keep = size[(size >= args.min_compounds) & (size <= args.max_compounds)].index
    df = df[df.target_id.isin(keep)]
    seqs = df.drop_duplicates("target_id").set_index("target_id").sequence
    keep = [t for t in keep if len(seqs[t]) <= args.max_seq_len]
    df = df[df.target_id.isin(keep)]
    if args.max_targets:
        keep = list(keep)[:args.max_targets]
        df = df[df.target_id.isin(keep)]
    print(f"[prior] {len(keep)} targets, {len(df):,} (target, molecule) pairs", flush=True)

    centroids, labels_out, seq_out, tid_out, meta = [], [], [], [], []
    t0 = time.time()
    for i, tid in enumerate(keep, 1):
        sub = df[df.target_id == tid]
        smiles = list(sub.smiles)
        values = sub.pic50.to_numpy()
        if len(smiles) < args.k * 3:
            continue
        try:
            lab = cluster_library(smiles, args.k, method="kmeans")
            X = fingerprint_matrix(smiles)
        except Exception:
            continue
        conc = concentration(lab, values, top_frac=args.top_frac)
        # One centroid fingerprint per region: the mean ECFP4 of its members.
        # Fingerprints beat every neural molecular encoder we measured
        # (Spearman 0.64-0.74 vs 0.20-0.50), so the ligand side stays classical.
        cen = np.zeros((args.k, X.shape[1]), dtype=np.float32)
        occupied = np.zeros(args.k, dtype=bool)
        for c in range(args.k):
            m = lab == c
            if m.any():
                cen[c] = X[m].mean(0)
                occupied[c] = True
        per = {r["cluster"]: r for r in conc["per_cluster"]}
        best = conc["best_cluster"]
        centroids.append(cen)
        labels_out.append(best)
        seq_out.append(seqs[tid])
        tid_out.append(tid)
        meta.append({"target_id": tid, "n": len(smiles), "best": int(best),
                     "best_lift": conc["best_lift"], "coverage_best": conc["coverage_best"],
                     "occupied": int(occupied.sum()),
                     "sizes": [per.get(c, {}).get("size", 0) for c in range(args.k)],
                     "n_top_per_cluster": [per.get(c, {}).get("n_top", 0) for c in range(args.k)],
                     "pic50_max": float(values.max()), "pic50_std": float(values.std())})
        if i % 25 == 0 or i == len(keep):
            print(f"\r[prior] {i}/{len(keep)} targets  {time.time() - t0:.0f}s", end="", flush=True)
    print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, centroids=np.stack(centroids), best=np.array(labels_out),
                        target_ids=np.array(tid_out), sequences=np.array(seq_out))
    out.with_suffix(".meta.json").write_text(json.dumps(
        {"k": args.k, "top_frac": args.top_frac, "n_targets": len(meta),
         "min_compounds": args.min_compounds, "max_compounds": args.max_compounds,
         "targets": meta}, indent=2), encoding="utf-8")

    lifts = np.array([m["best_lift"] for m in meta])
    print(f"[prior] {len(meta)} targets written to {out}")
    print(f"[prior] best-region lift: median {np.median(lifts):.1f}, "
          f"{100 * (lifts > 2).mean():.0f}% of targets above 2.0")
    print(f"[prior] a chooser that picks at random would be right "
          f"{100 / args.k:.0f}% of the time -- that is the floor to beat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
