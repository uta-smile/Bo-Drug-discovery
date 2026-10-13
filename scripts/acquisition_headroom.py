"""How much room is there for *any* controller to help? Run this before SARA.

    python scripts/acquisition_headroom.py --dataset data/chembl/egfr_anilinoquinazoline.csv

Agentic BO (arXiv 2608.00316) bets that an LLM steering the optimiser beats a
fixed policy. Before testing that bet on this library, measure whether the bet
is winnable here at all.

The instrument mirrors ``constraint_headroom.py``. It compares:

* **every fixed acquisition**, run alone -- ``ei``, ``ucb``, ``pi``, ``greedy``,
  ``ts`` (Thompson sampling). This is what "standard BO" can do with the policy decided up
  front.
* **a hindsight ceiling** that switches acquisition every round to whichever one
  reaches the best *true* value. No controller can beat it, because it already
  knows the answers.

**Headroom is the ceiling minus the best fixed acquisition, not minus ``ei``.**
If ``ucb`` alone beats ``ei`` alone, that is a better fixed policy, not evidence
for adaptivity -- and crediting it to an agent would be the same mistake as
crediting the LLM warm start for gains a free MaxMin heuristic also produced.

Read the verdict at the bottom. If the headroom is inside the seed-to-seed
noise, the action space is inert on this problem and no agent can demonstrate
anything: the honest conclusion is that this library is the wrong testbed, not
that the agent failed. That is a far cheaper thing to learn here -- in seconds,
with no endpoint, no API key and no GPU -- than after wiring up a model.

The controls (``scripted``, ``random``, ``round-robin``) are run alongside so
that a later LLM number has something to be read against from the first table
it appears in.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.agentic.diagnostics import ACQUISITIONS  # noqa: E402
from molbo.agentic.policies import (  # noqa: E402
    OraclePolicy,
    RandomPolicy,
    RoundRobinPolicy,
    ScriptedPolicy,
)
from molbo.campaign import ArmConfig, run_campaign  # noqa: E402
from molbo.constraints import LibraryContext  # noqa: E402
from molbo.metrics import auc_top_k, compare_arms, normalize_values  # noqa: E402
from molbo.oracles import LookupOracle  # noqa: E402
from molbo.surrogate import LibraryKernel, fingerprint_matrix  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 100])
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument(
        "--acquisitions",
        nargs="+",
        default=list(ACQUISITIONS),
        help="fixed acquisitions to run alone",
    )
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

    rows = []
    for budget in args.budgets:
        print(f"=== budget {budget} ===")
        print(f"{'policy':<22} {'AUC':>9} {'sem':>8} {'vs best fixed':>14} {'p':>8}")
        print("-" * 64)

        # ---- fixed policies: what standard BO can do -----------------------
        fixed: dict[str, list[float]] = {}
        for kind in args.acquisitions:
            arm = ArmConfig(f"bo_{kind}", strategy="bo", acquisition=kind, n_init=args.n_init)
            fixed[kind] = [campaign_auc(arm, budget, s) for s in seeds]

        best_kind = max(fixed, key=lambda k: float(np.mean(fixed[k])))
        reference = fixed[best_kind]
        ref_mean = float(np.mean(reference))

        for kind in args.acquisitions:
            vals = fixed[kind]
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            tag = f"fixed {kind}" + (" *" if kind == best_kind else "")
            if kind == best_kind:
                print(f"{tag:<22} {np.mean(vals):>9.4f} {sem:>8.4f} {'--':>14} {'--':>8}")
            else:
                cmp = compare_arms(vals, reference, paired=True)
                print(
                    f"{tag:<22} {np.mean(vals):>9.4f} {sem:>8.4f} "
                    f"{np.mean(vals) - ref_mean:>+14.4f} {cmp['p_value']:>8.4f}"
                )
            rows.append(
                {
                    "budget": budget,
                    "policy": f"fixed_{kind}",
                    "auc": float(np.mean(vals)),
                    "sem": sem,
                    "is_best_fixed": kind == best_kind,
                }
            )

        # ---- adaptive policies ---------------------------------------------
        # `OraclePolicy` reads `values`, which no method may do. It is a
        # reference line, labelled as such, exactly as `oracle_constraint` is in
        # the constraint study.
        adaptive = {
            "ceiling (hindsight)": OraclePolicy(values=values),
            "scripted": ScriptedPolicy(),
            "random switch": RandomPolicy(),
            "round-robin": RoundRobinPolicy(),
        }
        for label, policy in adaptive.items():
            arm = ArmConfig(
                label.replace(" ", "_"),
                strategy="bo",
                acquisition="ei",
                n_init=args.n_init,
                policy_source=policy,
            )
            vals = [campaign_auc(arm, budget, s) for s in seeds]
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            cmp = compare_arms(vals, reference, paired=True)
            print(
                f"{label:<22} {np.mean(vals):>9.4f} {sem:>8.4f} "
                f"{np.mean(vals) - ref_mean:>+14.4f} {cmp['p_value']:>8.4f}"
            )
            rows.append(
                {
                    "budget": budget,
                    "policy": label,
                    "auc": float(np.mean(vals)),
                    "sem": sem,
                    "delta_vs_best_fixed": float(np.mean(vals) - ref_mean),
                    "p_value": cmp["p_value"],
                    "cohens_dz": cmp["cohens_dz"],
                }
            )

        ceiling = next(
            r for r in rows if r["budget"] == budget and r["policy"] == "ceiling (hindsight)"
        )
        headroom = ceiling["delta_vs_best_fixed"]
        ref_sem = float(np.std(reference, ddof=1) / np.sqrt(len(reference)))
        print(
            f"\nbest fixed policy : {best_kind}  ({ref_mean:.4f} +/- {ref_sem:.4f})\n"
            f"headroom for any controller : {headroom:+.4f} AUC\n"
        )
        if headroom <= 2 * ref_sem:
            print(
                "VERDICT: headroom is inside twice the standard error of the best\n"
                "fixed policy. Switching acquisition cannot be shown to help on this\n"
                "library at this budget, even with hindsight. Do not spend model calls\n"
                "here -- change the budget, the library, or the action space first.\n"
            )
        else:
            print(
                f"VERDICT: there is room. A controller has up to {headroom:+.4f} AUC to\n"
                f"find. Read any LLM effect against that number, and against the\n"
                "scripted control above -- which is free.\n"
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
                    "acquisitions": args.acquisitions,
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
