"""Is choosing a region of chemical space worth anything, and who should choose?

    python scripts/cluster_headroom.py --dataset data/chembl/cruzipain.csv --k 10

The gate on region-based search. The proposal is: cluster the library, let
something pick a region, let BO search inside it, and switch regions when
progress stalls. Before training any chooser, three questions decide whether
the structure can pay:

1. **Is the good chemistry concentrated?** ``molbo.clusters.concentration``,
   no optimiser needed. Measured on cruzipain at k=10: one cluster holding 7.7%
   of the library contains 36 of the top 50 compounds, lift ~9. On EGFR's
   whole-target library the same measurement gives lift ~1.5 -- nothing to
   choose between. This is target-dependent, so run it per library.

2. **How much does the choice buy?** ``bo_best_cluster`` restricts BO to the
   hindsight-best region. It reads the answer key, so it is a ceiling, not a
   method -- but it bounds every possible chooser, learned or not.

3. **Does a chooser need to be learned?** ``bo_ucb_cluster`` is a bandit over
   regions: value is the best seen inside a region, plus the usual optimism
   bonus. Ten lines, regret guarantees, no training. **This is the baseline a
   learned chooser must beat** -- not plain BO.

The arm to be suspicious of is ``bo_random_cluster``. Restricting the candidate
pool changes BO's behaviour on its own, regardless of which region is chosen,
so a win by any restricted arm must be measured against a randomly restricted
one of the same tightness.

Seed count defaults to 200. Measured on this project: an identical selector arm
gave -0.0195 (p=0.011) and +0.0087 (p=0.19) at 20 seeds on library definitions
differing by 17 of 2,103 compounds. Twenty seeds does not resolve effects of
this size.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.campaign import ArmConfig  # noqa: E402
from molbo.clusters import (  # noqa: E402
    FixedClusterSource,
    HeuristicClusterSource,
    UCBClusterSource,
    cluster_library,
    coherence,
    concentration,
)
from molbo.experiment import run_sweep, summarize_sweep  # noqa: E402
from molbo.oracles import LookupOracle  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--k", type=int, nargs="+", default=[10],
                   help="cluster counts to test. A bandit needs arms << budget, "
                        "so k far above budget/10 will simply reduce to random")
    p.add_argument("--method", default="kmeans", choices=["kmeans", "scaffold", "sphere"])
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 100])
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--k-top", type=int, default=10, help="top-k for the AUC metric")
    p.add_argument("--min-candidates", type=int, default=20,
                   help="below this many unseen survivors a restriction is applied "
                        "softly instead of hard; see molbo.constraints.apply_constraints")
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--ucb-c", type=float, default=1.0)
    p.add_argument("--prior-scores", default=None,
                   help="a .npy of one score per region (scripts/score_regions.py). "
                        "Adds bo_prior_cluster: the bandit warm-started with those "
                        "scores. It must beat bo_ucb_cluster, not bo_whole.")
    p.add_argument("--prior", default=None,
                   help="a trained region prior (scripts/train_prior.py). Adds "
                        "bo_prior_cluster: the same bandit, warm-started with the "
                        "model's opinion. This is the arm under test -- it must beat "
                        "bo_ucb_cluster, not bo_whole.")
    p.add_argument("--esm", default="35M", help="protein encoder the prior was trained with")
    p.add_argument("--sequence", default=None,
                   help="the target's amino-acid sequence, or a path to a FASTA/txt file. "
                        "Required with --prior.")
    p.add_argument("--shuffle-prior", action="store_true",
                   help="permute the prior's region scores. If bo_prior_cluster keeps "
                        "its advantage under this, the advantage was never the model's "
                        "opinion about which region -- it was the mere presence of a prior "
                        "damping the bandit's exploration.")
    p.add_argument("--out", default="results/cluster_headroom.json")
    return p.parse_args(argv)


def prior_scores(args, library, labels, k: int) -> np.ndarray:
    """-> one prior score per region, from a trained model and the target sequence."""
    from molbo.clm import Encoder
    from molbo.prior import ClusterPrior
    from molbo.surrogate import fingerprint_matrix

    seq = args.sequence
    if seq and Path(seq).exists():
        text = Path(seq).read_text(encoding="utf-8")
        seq = "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))
    if not seq:
        raise SystemExit("--prior needs --sequence (the target's amino-acid sequence)")

    model = ClusterPrior.load(args.prior)
    esm = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D",
           "150M": "facebook/esm2_t30_150M_UR50D",
           "650M": "facebook/esm2_t33_650M_UR50D"}[args.esm]
    emb = Encoder(model=esm, revision=None, pooling="mean", max_len=1024,
                  cache_dir="data/cache/esm").embed([seq])[0]

    X = fingerprint_matrix(list(library))
    cen = np.zeros((k, X.shape[1]), dtype=np.float32)
    for c in range(k):
        m = labels == c
        if m.any():
            cen[c] = X[m].mean(0)
    return model.score(emb, cen)


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    args = parse_args(argv)
    oracle = LookupOracle.from_dataset(args.dataset, value_column=args.value_column,
                                       budget=max(args.budgets))
    library = list(oracle.library)
    values = np.array([oracle._table[s] for s in library], dtype=float)
    print(f"dataset   {args.dataset}\nlibrary   {len(library)} compounds  "
          f"seeds {args.seeds}  method {args.method}\n")

    rows, diag = [], []
    for k in args.k:
        labels = cluster_library(library, k, method=args.method)
        conc = concentration(labels, values, top_frac=0.01)
        coh = coherence(labels, library)
        best_cluster = conc["best_cluster"]
        best_size = next(r["size"] for r in conc["per_cluster"] if r["cluster"] == best_cluster)
        diag.append({"k": k, "concentration": {kk: vv for kk, vv in conc.items()
                                               if kk != "per_cluster"}, "coherence": coh})
        print(f"[k={k}] {conc['n_clusters']} clusters | best holds {best_size}/{len(library)} "
              f"compounds and {100 * conc['coverage_best']:.0f}% of the top 1% "
              f"(lift {conc['best_lift']:.1f})")
        print(f"       coherence: within {coh['within']:.3f} vs between {coh['between']:.3f} "
              f"(separation {coh['separation']:+.3f}) | sizes "
              f"{coh['size_min']}/{coh['size_median']}/{coh['size_max']}")

        if args.prior_scores:
            pscores = np.load(args.prior_scores)
            if len(pscores) != conc["n_clusters"]:
                raise SystemExit(f"--prior-scores has {len(pscores)} entries but there "
                                 f"are {conc['n_clusters']} regions at k={k}")
        elif args.prior:
            pscores = prior_scores(args, library, labels, conc["n_clusters"])
        else:
            pscores = None
        if pscores is not None:
            pick = int(np.argmax(pscores))
            print(f"       prior picks region {pick} (best is {best_cluster}) -- "
                  f"{'HIT' if pick == best_cluster else 'miss'}"
                  f"{'  [SCORES SHUFFLED]' if args.shuffle_prior else ''}")

        # A random cluster is drawn per seed inside the arm, so it is not one
        # unlucky region -- but it must be the same tightness as the best one,
        # else the comparison is about restriction size, not about choosing.
        sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
        nearest = min(sizes, key=lambda c: abs(sizes[c] - best_size) + 1e6 * (c == best_cluster))

        for budget in args.budgets:
            oracle.budget = budget
            if budget / max(1, conc["n_clusters"]) < 2:
                print(f"       [!] budget {budget} over {conc['n_clusters']} clusters is "
                      f"{budget / conc['n_clusters']:.1f} pulls/arm -- the bandit cannot "
                      f"leave its first round")
            ucb = UCBClusterSource(labels=labels, c=args.ucb_c)
            arms = [
                ArmConfig("bo_whole", strategy="bo", acquisition="ei", n_init=args.n_init),
                ArmConfig("bo_random_cluster", strategy="bo", acquisition="ei",
                          n_init=args.n_init, min_candidates=args.min_candidates,
                          constraint_source=FixedClusterSource(labels, nearest)),
                ArmConfig("bo_largest_cluster", strategy="bo", acquisition="ei",
                          n_init=args.n_init, min_candidates=args.min_candidates,
                          constraint_source=HeuristicClusterSource(labels, rule="largest")),
                ArmConfig("bo_densest_cluster", strategy="bo", acquisition="ei",
                          n_init=args.n_init, min_candidates=args.min_candidates,
                          constraint_source=HeuristicClusterSource(
                              labels, rule="densest", values=values)),
                ArmConfig("bo_ucb_cluster", strategy="bo", acquisition="ei",
                          n_init=args.n_init, min_candidates=args.min_candidates,
                          constraint_source=ucb, constraint_every=1),
            ]
            if pscores is not None:
                from molbo.prior import PriorClusterSource

                s = pscores if not args.shuffle_prior else \
                    np.random.default_rng(0).permutation(pscores)
                arms.append(ArmConfig(
                    "bo_prior_cluster", strategy="bo", acquisition="ei",
                    n_init=args.n_init, min_candidates=args.min_candidates,
                    constraint_source=PriorClusterSource(labels=labels, scores=s, c=args.ucb_c),
                    constraint_every=1))
            arms.append(ArmConfig("bo_best_cluster", strategy="bo", acquisition="ei",
                                  n_init=args.n_init, min_candidates=args.min_candidates,
                                  constraint_source=FixedClusterSource(labels, best_cluster)))
            metrics, _ = run_sweep(oracle, arms, range(args.seeds), budget=budget,
                                   k=args.k_top, library=library, verbose=False)
            s = summarize_sweep(metrics, reference="bo_whole")
            head = (f"{'arm':<20s}{'AUC':>9s}{'sem':>8s}{'dAUC':>9s}{'p':>9s}"
                    f"{'dz':>7s}{'top1%':>8s}")
            print(f"\n  budget {budget}\n  {head}\n  {'-' * len(head)}")
            for arm in arms:
                a = s["summaries"][arm.name]
                c = s["comparisons"].get(arm.name, {})
                recall = float(np.mean([m.top1pct_recall for m in metrics
                                        if m.arm == arm.name and m.top1pct_recall is not None]))
                row = {"k": k, "budget": budget, "arm": arm.name, "auc": a.auc_mean,
                       "sem": a.auc_sem, "top1pct_recall": recall,
                       "delta_vs_whole": c.get("mean_diff"), "p_value": c.get("p_value"),
                       "cohens_dz": c.get("cohens_dz")}
                rows.append(row)
                d = "" if row["delta_vs_whole"] is None else f"{row['delta_vs_whole']:+9.4f}"
                pv = "" if row["p_value"] is None else f"{row['p_value']:9.4f}"
                dz = "" if row["cohens_dz"] is None else f"{row['cohens_dz']:+7.2f}"
                print(f"  {arm.name:<20s}{a.auc_mean:9.4f}{a.auc_sem:8.4f}{d:>9s}"
                      f"{pv:>9s}{dz:>7s}{recall:8.3f}")

            get = lambda n: next(r for r in rows if r["k"] == k and r["budget"] == budget
                                 and r["arm"] == n)
            ceil, ucb_row, rand = get("bo_best_cluster"), get("bo_ucb_cluster"), get("bo_random_cluster")
            print(f"\n  VERDICT (k={k}, budget {budget})")
            if (ceil["delta_vs_whole"] or 0) <= 2 * ceil["sem"]:
                print("    The hindsight-best region does not beat the whole library. No chooser")
                print("    can win here -- region selection is the wrong hook for this target.")
            else:
                took = (ucb_row["delta_vs_whole"] or 0) / ceil["delta_vs_whole"]
                print(f"    Ceiling {ceil['delta_vs_whole']:+.4f} AUC. A free bandit already takes "
                      f"{100 * took:.0f}% of it\n    ({ucb_row['delta_vs_whole']:+.4f}, "
                      f"p={ucb_row['p_value']:.3f}); random restriction gives "
                      f"{rand['delta_vs_whole']:+.4f}.")
                left = ceil["delta_vs_whole"] - (ucb_row["delta_vs_whole"] or 0)
                print(f"    Room left for a LEARNED chooser: {left:+.4f} AUC. That is what a "
                      f"CLM prior\n    would have to capture, and it must beat the bandit, "
                      f"not plain BO.")
            print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": args.dataset, "library_size": len(library),
                               "seeds": args.seeds, "method": args.method,
                               "diagnostics": diag, "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
