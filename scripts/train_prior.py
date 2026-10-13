"""Train the region prior across targets, and test it on targets it never saw.

    python scripts/train_prior.py --data data/prior/clusters.npz --out data/prior/prior_v1.pt

The claim being tested: shown only a protein sequence and the centroid
fingerprints of a library's regions, can a model say which region holds the
actives -- with no compound measured?

Evaluation is on **held-out targets**. On a target seen in training the model
could have memorised which region won, which is not the question. Two metrics,
because top-1 accuracy alone is misleading when the second-best region is
nearly as good:

``top1``
    Fraction of held-out targets where the chosen region is the one with the
    most top-1% compounds. Random is 1/k.
``coverage``
    Fraction of that target's top-1% compounds sitting in the chosen region,
    as a share of what the best region would have given. This is what actually
    matters for a campaign: choosing the second-best of two near-equal regions
    costs almost nothing.

Four free controls, and none of them needs a model:

``largest``  the biggest region -- strong, because a big region holds more good
             compounds by chance and starves BO less
``diverse``  the odd-one-out centroid
``dense``    the mainstream chemotype
``random``   the floor

And one control that is not free but is the one that decides the paper:
``--shuffle-proteins`` pairs each target with another target's embedding. If
accuracy survives that, the protein contributed nothing and the model is
ranking regions on shape alone -- in which case the ESM2 tower is decoration
and should be cut.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.prior import ClusterPrior, baseline_scores  # noqa: E402

ESM2 = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D",
        "150M": "facebook/esm2_t30_150M_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}


def embed_proteins(seqs, size: str, cache_dir: str, batch: int = 8) -> np.ndarray:
    from molbo.clm import Encoder

    enc = Encoder(model=ESM2[size], revision=None, pooling="mean", max_len=1024,
                  cache_dir=cache_dir, token_budget=4096)
    return enc.embed(list(seqs), progress=True)


def evaluate(scores_fn, centroids, sizes, n_top, best) -> dict:
    """``scores_fn(i) -> per-region score`` for target ``i``."""
    hits, cover = [], []
    for i in range(len(centroids)):
        s = scores_fn(i)
        pick = int(np.argmax(s))
        hits.append(pick == int(best[i]))
        tot = n_top[i].sum()
        cover.append(n_top[i][pick] / max(1, n_top[i][int(best[i])]))
    return {"top1": float(np.mean(hits)), "coverage": float(np.mean(cover)),
            "n": int(len(centroids))}


def main(argv=None) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/prior/clusters.npz")
    p.add_argument("--esm", default="35M", choices=list(ESM2))
    p.add_argument("--cache-dir", default="data/cache/esm")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="early stopping uses THIS, never the test split")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.5)
    p.add_argument("--width", type=int, default=32,
                   help="tiny on purpose: 454 training targets. At width 256 "
                        "(713k params) train top-1 hit 1.000 while test sat at 0.145")
    p.add_argument("--proj", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", nargs="*", default=[],
                   help="UniProt accessions to drop entirely, for targets a campaign "
                        "will later be run on. BindingDB's 607 targets include "
                        "cruzipain (P25779), so testing the prior on a cruzipain "
                        "campaign without this would be scoring it on its own "
                        "training data.")
    p.add_argument("--shuffle-proteins", action="store_true",
                   help="pair each target with another's embedding; accuracy that "
                        "survives this was never about the protein")
    p.add_argument("--out", default="data/prior/prior_v1.pt")
    args = p.parse_args(argv)

    z = np.load(args.data, allow_pickle=True)
    centroids = z["centroids"].astype(np.float32)
    best = z["best"].astype(int)
    seqs = [str(s) for s in z["sequences"]]
    tids = [str(t) for t in z["target_ids"]]
    meta = json.loads(Path(args.data).with_suffix(".meta.json").read_text(encoding="utf-8"))
    sizes = np.array([m["sizes"] for m in meta["targets"]], dtype=float)
    n_top = np.array([m["n_top_per_cluster"] for m in meta["targets"]], dtype=float)

    if args.exclude:
        drop = {a.upper() for a in args.exclude}
        keep = [i for i, t in enumerate(tids) if t.upper() not in drop]
        removed = sorted({t for t in tids if t.upper() in drop})
        centroids, best, sizes, n_top = (centroids[keep], best[keep],
                                         sizes[keep], n_top[keep])
        seqs = [seqs[i] for i in keep]
        tids = [tids[i] for i in keep]
        print(f"[prior] excluded {removed or 'nothing (none of those were present)'} "
              f"-> {len(tids)} targets remain")
    n_t, k, dr = centroids.shape
    print(f"[prior] {n_t} targets, {k} regions each, region dim {dr}")

    print(f"[prior] embedding {n_t} sequences with ESM2-{args.esm}...", flush=True)
    P = embed_proteins(seqs, args.esm, args.cache_dir)
    print(f"[prior] protein embeddings {P.shape}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_t)
    n_test = int(round(args.test_frac * n_t))
    n_val = int(round(args.val_frac * n_t))
    # Three-way, because early stopping on the test split is model selection on
    # the held-out data: the first version of this script did exactly that and
    # reported a test number it had already optimised against.
    test_idx, val_idx, train_idx = (order[:n_test], order[n_test:n_test + n_val],
                                    order[n_test + n_val:])
    if args.shuffle_proteins:
        # Break the target-region correspondence, keeping the marginals.
        P = P[rng.permutation(n_t)]
        print("[prior] PROTEIN EMBEDDINGS SHUFFLED -- this is the ablation")
    print(f"[prior] train {len(train_idx)} targets | test {len(test_idx)} (never seen)\n")

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    model = ClusterPrior(protein_dim=P.shape[1], region_dim=dr, width=args.width,
                         proj=args.proj, dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dev = model.device
    Pt = torch.tensor(P, device=dev)
    Ct = torch.tensor(centroids, device=dev)
    Bt = torch.tensor(best, device=dev, dtype=torch.long)
    tr = torch.tensor(train_idx, device=dev)
    te = torch.tensor(test_idx, device=dev)
    va = torch.tensor(val_idx, device=dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[prior] {n_params / 1e3:.0f}k trainable parameters")

    def acc(idx):
        model.train(False)
        with torch.no_grad():
            pred = model.logits(Pt[idx], Ct[idx]).argmax(1)
        model.train(True)
        return float((pred == Bt[idx]).float().mean())

    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(True)
        perm = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            loss = F.cross_entropy(model.logits(Pt[b], Ct[b]), Bt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        a = acc(va)          # validation, never test
        if a > best_acc:
            best_acc, bad = a, 0
            best_state = {n: {kk: vv.detach().clone() for kk, vv in m.state_dict().items()}
                          for n, m in (("p", model.protein_head), ("r", model.region_head))}
            best_scale = model.logit_scale.detach().clone()
        else:
            bad += 1
            if bad >= args.patience:
                break
        if ep % 20 == 0:
            print(f"[prior] epoch {ep:3d}  loss {loss.item():.4f}  "
                  f"train top1 {acc(tr):.3f}  val top1 {a:.3f}  (test held back)")
    model.protein_head.load_state_dict(best_state["p"])
    model.region_head.load_state_dict(best_state["r"])
    with torch.no_grad():
        model.logit_scale.copy_(best_scale)

    ct, st, nt, bt = centroids[test_idx], sizes[test_idx], n_top[test_idx], best[test_idx]
    rows = {"prior (learned)": evaluate(lambda i: model.score(P[test_idx][i], ct[i]),
                                        ct, st, nt, bt)}
    for kind in ("largest", "diverse", "dense", "random"):
        rows[kind] = evaluate(
            lambda i, kk=kind: baseline_scores(ct[i], st[i], kk, np.random.default_rng(i)),
            ct, st, nt, bt)

    print(f"\nheld-out targets (n={len(test_idx)}), {k} regions each")
    print(f"{'chooser':<18s}{'top-1':>9s}{'coverage':>11s}")
    print("-" * 38)
    for name, m in rows.items():
        print(f"{name:<18s}{m['top1']:>9.3f}{m['coverage']:>11.3f}")
    print(f"{'(random floor)':<18s}{1 / k:>9.3f}")

    learned, bestfree = rows["prior (learned)"], max(
        (v for kk, v in rows.items() if kk != "prior (learned)"), key=lambda v: v["top1"])
    print("\nVERDICT")
    if learned["top1"] <= max(1 / k, bestfree["top1"]):
        print("  The learned prior does not beat the free controls. Either the protein")
        print("  signal is not there, or centroid fingerprints are the wrong region")
        print("  representation. Do not wire it into a campaign.")
    else:
        print(f"  Learned {learned['top1']:.3f} vs best free control {bestfree['top1']:.3f} "
              f"and random {1 / k:.3f}.")
        print("  Next: --shuffle-proteins. If accuracy holds up there, the protein tower")
        print("  is decoration and the win came from region shape alone.")

    model.save(args.out, esm=args.esm, test_targets=[tids[i] for i in test_idx],
               results=rows, shuffled=args.shuffle_proteins, excluded=list(args.exclude),
               args=vars(args))
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"results": rows, "k": k, "n_train": len(train_idx), "n_test": len(test_idx),
         "esm": args.esm, "shuffled": args.shuffle_proteins}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
