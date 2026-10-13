"""How much room is there for *any* constraint to help? Run this before an arm.

    python scripts/constraint_headroom.py --dataset data/chembl/egfr_anilinoquinazoline.csv

Measures the gap between plain BO and a ceiling arm that is restricted to the
library's true best ``fraction`` -- the constraint you would write if you
already knew every answer. Nothing any real constraint source does can beat
that line, so the gap is an upper bound on what the LLM arm could possibly
demonstrate.

This is a design instrument, not a result. Running it first is what stops you
spending a week on an arm in a regime where the effect it is meant to show
cannot be resolved even in principle. The measured pattern on EGFR is that
headroom roughly triples between a quarter of the library and a twentieth, and
grows as the budget shrinks: a tight budget against a loose constraint is the
worst place to look for a constraint effect, and is also the configuration
people reach for first.

Report the headroom next to the arm's effect size. A +0.01 AUC improvement
means something very different against +0.09 of headroom than against +0.02.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.campaign import ArmConfig, run_campaign  # noqa: E402
from molbo.constraints import LibraryContext, oracle_constraint  # noqa: E402
from molbo.metrics import auc_top_k, compare_arms, normalize_values  # noqa: E402
from molbo.oracles import LookupOracle  # noqa: E402
from molbo.surrogate import LibraryKernel, fingerprint_matrix  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.10, 0.05])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--out", default=None, help="write the table as JSON here")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    oracle = LookupOracle.from_dataset(args.dataset, value_column=args.value_column)
    library = oracle.library
    values = np.asarray(oracle.values)
    lo, hi = float(values.min()), float(values.max())

    t0 = time.perf_counter()
    ctx = LibraryContext.build(library)
    X = fingerprint_matrix(library)
    kernel = LibraryKernel.maybe_build(X)
    seeds = list(range(args.seeds))

    print(
        f"dataset   {args.dataset}\n"
        f"library   {len(library)} compounds, {args.value_column} in [{lo:.2f}, {hi:.2f}]\n"
        f"seeds     {args.seeds}   (setup {time.perf_counter() - t0:.1f}s)\n"
    )

    def campaign_auc(arm, budget, seed):
        oracle.budget = budget
        res = run_campaign(
            oracle, arm, seed, budget=budget, library=library, ctx=ctx, X=X, kernel=kernel
        )
        return auc_top_k(normalize_values(res.scored, lo, hi), k=args.k, budget=budget)

    n_top = max(1, int(round(0.01 * len(library))))
    top_set = {library[i] for i in np.argsort(-values)[:n_top]}

    rows = []
    print(
        f"{'budget':>7} {'frac':>6} {'n_allowed':>10} {'plain BO':>10} "
        f"{'ceiling':>10} {'headroom':>10} {'p':>8}"
    )
    print("-" * 68)

    for budget in args.budgets:
        base_arm = ArmConfig("bo_ei", strategy="bo", acquisition="ei", n_init=args.n_init)
        base = [campaign_auc(base_arm, budget, s) for s in seeds]

        # Recall of the true top 1% by plain BO, for context on how much of the
        # problem is already solved without any constraint at all.
        oracle.budget = budget
        recall = np.mean(
            [
                len(top_set & set(
                    run_campaign(
                        oracle, base_arm, s, budget=budget, library=library,
                        ctx=ctx, X=X, kernel=kernel,
                    ).smiles
                )) / len(top_set)
                for s in seeds
            ]
        )

        for frac in args.fractions:
            ceil_arm = ArmConfig(
                f"ceiling_{frac:g}",
                strategy="bo",
                acquisition="ei",
                n_init=args.n_init,
                constraint_source=(
                    lambda f: (lambda **kw: oracle_constraint(values, fraction=f))
                )(frac),
            )
            ceil = [campaign_auc(ceil_arm, budget, s) for s in seeds]
            cmp = compare_arms(ceil, base, paired=True)
            n_allowed = max(1, int(round(frac * len(library))))
            rows.append(
                {
                    "budget": budget,
                    "fraction": frac,
                    "n_allowed": n_allowed,
                    "bo_auc": float(np.mean(base)),
                    "ceiling_auc": float(np.mean(ceil)),
                    "headroom": float(np.mean(ceil) - np.mean(base)),
                    "p_value": cmp["p_value"],
                    "cohens_dz": cmp["cohens_dz"],
                    "bo_top1pct_recall": float(recall),
                }
            )
            print(
                f"{budget:>7} {frac:>6.2f} {n_allowed:>10} {np.mean(base):>10.4f} "
                f"{np.mean(ceil):>10.4f} {np.mean(ceil) - np.mean(base):>+10.4f} "
                f"{cmp['p_value']:>8.4f}"
            )
        print(f"{'':>7} {'':>6} plain BO recalls {recall:.1%} of the true top 1% at this budget")

    best = max(rows, key=lambda r: r["headroom"])
    print(
        f"\nmost headroom: budget {best['budget']}, fraction {best['fraction']:g} "
        f"-> {best['headroom']:+.4f} AUC ({best['n_allowed']} compounds allowed)\n"
        "Run the constraint arms there; an effect smaller than this is not\n"
        "resolvable no matter how good the constraint source is."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "library_size": len(library),
                    "seeds": seeds,
                    "n_init": args.n_init,
                    "k": args.k,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
