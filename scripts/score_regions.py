"""Score a library's regions with a cold-target affinity model, for a campaign arm.

    python scripts/score_regions.py --dataset data/chembl/cruzipain.csv \
        --sequence data/targets/cruzipain.txt --exclude P25779 \
        --out data/prior/regions_cruzipain.npy

Trains ridge on (ESM2 protein embedding, ECFP4 fingerprint) -> pIC50 across
BindingDB targets, then scores each region of the given library by the
predicted *density* of top compounds among its members.

Two things this is careful about, both learned the hard way in this project:

**Held-out target.** ``--exclude`` drops the campaign's own target from
training. BindingDB's 606 usable targets include cruzipain, so without this
the model would be scoring regions for a protein it was fitted on.

**Density, not count.** Scoring a region by how many predicted-good compounds
it contains just picks the biggest region -- and restricting a campaign to the
largest region was measured at -0.165 AUC on cruzipain, the worst arm tested.
The region that wins is small and concentrated, so the score is a rate.

The output is a plain ``.npy`` of one score per region, which
``scripts/cluster_headroom.py --prior-scores`` consumes. Keeping it a file
rather than a live model means the campaign run is reproducible without
re-fitting, and the scores can be inspected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

ESM2 = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D",
        "150M": "facebook/esm2_t30_150M_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}


def main(argv=None) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="the campaign's library CSV")
    p.add_argument("--sequence", required=True, help="target sequence, or a file holding it")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="UniProt accessions to drop from training (the campaign's target)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--esm", default="35M", choices=list(ESM2))
    p.add_argument("--n-train", type=int, default=150_000)
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--top-quantile", type=float, default=0.99,
                   help="a molecule counts as predicted-good above this quantile "
                        "of the model's predictions on this library")
    p.add_argument("--smiles-column", default="canonical_smiles")
    p.add_argument("--min-heavy-atoms", type=int, default=None)
    p.add_argument("--max-heavy-atoms", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import pandas as pd
    from tdc.utils.load import pd_load

    from molbo.clm import Encoder
    from molbo.clusters import cluster_library
    from molbo.surrogate import fingerprint_matrix

    seq = args.sequence
    if Path(seq).exists():
        text = Path(seq).read_text(encoding="utf-8")
        seq = "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))

    print("[regions] loading BindingDB...", flush=True)
    df = pd_load("bindingdb_ic50", "data/cache/tdc")
    df.columns = ["drug_id", "smiles", "target_id", "sequence", "y"]
    df = df.dropna(subset=["smiles", "sequence", "y"])
    df["pic50"] = 9.0 - np.log10(np.clip(df.y.astype(float), 1e-4, 1e11))
    df = df.groupby(["target_id", "smiles"], as_index=False).agg(
        pic50=("pic50", "median"), sequence=("sequence", "first"))
    size = df.groupby("target_id").size()
    keep = size[(size >= 200) & (size <= 3000)].index
    seqs = df.drop_duplicates("target_id").set_index("target_id").sequence
    drop = {a.upper() for a in args.exclude}
    keep = [t for t in keep if len(seqs[t]) <= 1022 and t.upper() not in drop]
    df = df[df.target_id.isin(keep)]
    print(f"[regions] training on {len(keep)} targets "
          f"(excluded {sorted(drop) or 'nothing'}), {len(df):,} pairs")

    enc = Encoder(model=ESM2[args.esm], revision=None, pooling="mean", max_len=1024,
                  cache_dir="data/cache/esm")
    uniq = sorted(set(df.target_id))
    P = dict(zip(uniq, enc.embed([seqs[t] for t in uniq], progress=True)))
    target_emb = enc.embed([seq])[0]

    tr = df.sample(n=min(args.n_train, len(df)), random_state=0)
    t0 = time.time()
    Xtr = np.hstack([np.stack([P[t] for t in tr.target_id]).astype(np.float32),
                     fingerprint_matrix(list(tr.smiles)).astype(np.float32)])
    ytr = tr.pic50.to_numpy()
    mu, ym = Xtr.mean(0), ytr.mean()
    A = Xtr - mu
    G = A.T @ A
    G[np.diag_indices_from(G)] += args.alpha
    w = np.linalg.solve(G, A.T @ (ytr - ym))
    print(f"[regions] ridge on {Xtr.shape} in {time.time() - t0:.0f}s")

    # Go through LookupOracle, not the raw CSV. It canonicalises, dedups and
    # applies the heavy-atom filter, so reading the CSV directly gives a
    # different library order -- and then region index 7 here would not be
    # region 7 in the campaign. The scores would be silently misaligned.
    from molbo.oracles import LookupOracle

    oracle = LookupOracle.from_dataset(args.dataset, budget=1,
                                       min_heavy_atoms=args.min_heavy_atoms,
                                       max_heavy_atoms=args.max_heavy_atoms)
    smiles = list(oracle.library)
    print(f"[regions] library {len(smiles)} compounds (via LookupOracle, "
          f"same path the campaign uses)")
    Xl = np.hstack([np.repeat(target_emb[None, :], len(smiles), 0).astype(np.float32),
                    fingerprint_matrix(smiles).astype(np.float32)])
    pred = (Xl - mu) @ w + ym

    labels = cluster_library(smiles, args.k, method="kmeans")
    thr = np.quantile(pred, args.top_quantile)
    sizes = np.bincount(labels, minlength=args.k).astype(float)
    scores = np.array([(pred[labels == c] >= thr).sum() / max(1.0, sizes[c])
                       if (labels == c).any() else -1.0 for c in range(args.k)])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    out.with_suffix(".json").write_text(json.dumps(
        {"dataset": args.dataset, "k": args.k, "esm": args.esm, "excluded": args.exclude,
         "n_train_pairs": len(tr), "n_train_targets": len(keep),
         "scores": scores.tolist(), "sizes": sizes.tolist(),
         "pick": int(np.argmax(scores)),
         "pred_range": [float(pred.min()), float(pred.max())]}, indent=2), encoding="utf-8")
    print(f"[regions] scores {np.round(scores, 4)}")
    print(f"[regions] picks region {int(np.argmax(scores))} (sizes {sizes.astype(int)})")
    print(f"[regions] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
