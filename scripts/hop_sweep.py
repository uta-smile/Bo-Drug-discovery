"""Can a cross-target model pick which unsampled scaffold BO should enter?

    python scripts/hop_sweep.py --targets 8 --budgets 30 60 120 --seeds 100

Directed scaffold hopping is worth +0.013 to +0.069 AUC against plain BO, and
random hopping *loses* -- so the value is in knowing which chemotype, not in
exploring more. This asks whether a model trained across other targets can
supply that, which is the only part of the pipeline still using an answer key.

Arms:

``plain``        ordinary BO
``hop_random``   a random unsampled scaffold every 4th round -- the control
                 that separates "explore" from "explore *there*"
``hop_model``    the unsampled scaffold whose members score highest under a
                 ridge on (ESM2 protein embedding, ECFP4), fitted on other
                 targets and never on this one
``hop_oracle``   the unsampled scaffold containing the best compound. Reads
                 the answer key; a ceiling, not a method.

The bar here is much lower than for the priors in sections 8-10 of FINDINGS.
Those had to beat the GP's own accuracy, which reaches rho 0.46 by 30
observations. In a scaffold the GP has never sampled its accuracy is zero, so
``hop_model`` only has to beat ``hop_random``.
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
    p.add_argument("--targets", type=int, default=8)
    p.add_argument("--min-compounds", type=int, default=400)
    p.add_argument("--max-compounds", type=int, default=1200)
    p.add_argument("--budgets", type=int, nargs="+", default=[30, 60, 120])
    p.add_argument("--every", type=int, default=4, help="rounds between hops")
    p.add_argument("--agg", default="max", choices=["max", "mean"])
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--esm", default="35M", choices=list(ESM2))
    p.add_argument("--n-train", type=int, default=150_000)
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/hop_sweep.json")
    args = p.parse_args(argv)

    from scipy.stats import spearmanr, wilcoxon

    from prior_sweep import load_pairs

    from molbo.campaign import ArmConfig
    from molbo.clm import Encoder
    from molbo.experiment import run_sweep, summarize_sweep
    from molbo.hopping import ScaffoldHopSource, murcko_labels, oracle_scores, predicted_scores
    from molbo.oracles import LookupOracle
    from molbo.surrogate import fingerprint_matrix

    df = load_pairs()
    size = df.groupby("target_id").size()
    seqs = df.drop_duplicates("target_id").set_index("target_id").sequence
    pool = [t for t in size[(size >= args.min_compounds)
                            & (size <= args.max_compounds)].index
            if len(seqs[t]) <= 1022]
    rng = np.random.default_rng(args.seed)
    test = [pool[i] for i in rng.permutation(len(pool))[:args.targets]]
    print(f"[hop] {len(test)} held-out targets, budgets {args.budgets}, "
          f"hop every {args.every} rounds\n", flush=True)

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
    print(f"[hop] ridge on {Xtr.shape} from {tr.target_id.nunique()} targets "
          f"in {time.time() - t0:.0f}s\n", flush=True)

    tmp = Path("data/cache/sweep_libs")
    tmp.mkdir(parents=True, exist_ok=True)
    ARMS = ["plain", "hop_random", "hop_model", "hop_oracle"]
    hdr = (f"{'target':<9s}{'n':>5s}{'scaf':>6s}{'bud':>5s}{'rho':>7s}"
           + "".join(f"{a.replace('hop_', ''):>11s}" for a in ARMS[1:]))
    print(hdr + "\n" + "-" * len(hdr), flush=True)

    rows = []
    for tid in test:
        sub = df[df.target_id == tid]
        csv = tmp / f"{tid}.csv"
        sub.rename(columns={"smiles": "canonical_smiles", "pic50": "pchembl_value"})[
            ["canonical_smiles", "pchembl_value"]].to_csv(csv, index=False)
        try:
            oracle = LookupOracle.from_dataset(csv, budget=max(args.budgets))
        except Exception:
            continue
        lib = list(oracle.library)
        if len(lib) < max(args.budgets) + 60:
            continue
        truth = np.array([oracle._table[s] for s in lib])
        labels = murcko_labels(lib)
        Xl = np.hstack([np.repeat(P[tid][None, :], len(lib), 0).astype(np.float32),
                        fingerprint_matrix(lib).astype(np.float32)])
        pred = (Xl - mu0) @ w + ym
        rho = float(spearmanr(pred, truth).statistic)
        s_model = predicted_scores(labels, pred, agg=args.agg)
        s_oracle = oracle_scores(labels, truth)
        # How well does the model rank scaffolds? This is the quantity that
        # decides the arm, and it is not the same as per-molecule accuracy.
        ok = np.isfinite(s_model) & np.isfinite(s_oracle)
        rho_scaf = float(spearmanr(s_model[ok], s_oracle[ok]).statistic)

        for budget in args.budgets:
            oracle.budget = budget
            arms = [
                ArmConfig("plain", strategy="bo", acquisition="ei", n_init=args.n_init),
                ArmConfig("hop_random", strategy="bo", acquisition="ei", n_init=args.n_init,
                          min_candidates=1, constraint_every=1,
                          constraint_source=ScaffoldHopSource(labels, None, args.every)),
                ArmConfig("hop_model", strategy="bo", acquisition="ei", n_init=args.n_init,
                          min_candidates=1, constraint_every=1,
                          constraint_source=ScaffoldHopSource(
                              labels, lambda _l, s=s_model: s, args.every)),
                ArmConfig("hop_oracle", strategy="bo", acquisition="ei", n_init=args.n_init,
                          min_candidates=1, constraint_every=1,
                          constraint_source=ScaffoldHopSource(
                              labels, lambda _l, s=s_oracle: s, args.every)),
            ]
            m, _ = run_sweep(oracle, arms, range(args.seeds), budget=budget, k=10,
                             library=lib, verbose=False)
            s = summarize_sweep(m, reference="plain")
            row = {"target_id": tid, "n": len(lib), "n_scaffolds": int(labels.max()) + 1,
                   "budget": budget, "rho_molecule": rho, "rho_scaffold": rho_scaf,
                   "max": float(truth.max())}
            for a in ARMS:
                row[f"auc_{a}"] = float(np.mean([x.auc for x in m if x.arm == a]))
                row[f"best_{a}"] = float(np.mean([x.best for x in m if x.arm == a]))
                c = s["comparisons"].get(a, {})
                row[f"d_{a}"] = c.get("mean_diff")
                row[f"p_{a}"] = c.get("p_value")
            rows.append(row)
            print(f"{tid:<9s}{len(lib):>5d}{row['n_scaffolds']:>6d}{budget:>5d}"
                  f"{rho_scaf:>+7.2f}"
                  + "".join(f"{row[f'd_{a}']:>+11.4f}" for a in ARMS[1:]), flush=True)
        print(flush=True)

    print(f"\nMean dAUC vs plain BO over {len({r['target_id'] for r in rows})} targets")
    print(f"{'budget':>7s}{'random':>11s}{'model':>11s}{'oracle':>11s}"
          f"{'model-random':>14s}{'p':>9s}{'% of ceiling':>14s}")
    print("-" * 78)
    for b in args.budgets:
        sel = [r for r in rows if r["budget"] == b]
        if not sel:
            continue
        g = {a: np.array([r[f"d_{a}"] for r in sel]) for a in ARMS[1:]}
        d = g["hop_model"] - g["hop_random"]
        pv = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        share = (g["hop_model"].mean() / g["hop_oracle"].mean()
                 if g["hop_oracle"].mean() else float("nan"))
        print(f"{b:>7d}{g['hop_random'].mean():>+11.4f}{g['hop_model'].mean():>+11.4f}"
              f"{g['hop_oracle'].mean():>+11.4f}{d.mean():>+14.4f}{pv:>9.4f}"
              f"{100 * share:>13.0f}%")

    allr = [r for r in rows]
    dm = np.array([r["d_hop_model"] for r in allr])
    dr = np.array([r["d_hop_random"] for r in allr])
    print(f"\nmean scaffold-ranking rho (model vs truth): "
          f"{np.mean([r['rho_scaffold'] for r in allr]):+.3f}")
    print(f"mean per-molecule rho                     : "
          f"{np.mean([r['rho_molecule'] for r in allr]):+.3f}")
    print("\nVERDICT")
    if dm.mean() > 0 and wilcoxon(dm).pvalue < 0.05:
        print(f"  hop_model beats plain BO by {dm.mean():+.4f} "
              f"(p={wilcoxon(dm).pvalue:.4f}) and random hopping by "
              f"{(dm - dr).mean():+.4f}.")
        print("  A cross-target model CAN say where to hop. This is the one mechanism")
        print("  here where the model is not competing with the surrogate.")
    else:
        print(f"  hop_model is {dm.mean():+.4f} vs plain BO "
              f"(p={wilcoxon(dm).pvalue:.4f}). The ceiling is real but the model "
              f"cannot\n  find the right scaffold; report the ceiling and the "
              f"negative together.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"budgets": args.budgets, "every": args.every,
                               "seeds": args.seeds, "agg": args.agg, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
