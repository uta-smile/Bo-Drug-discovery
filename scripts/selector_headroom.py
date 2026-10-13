"""How much room is there for *any* selector on BO's shortlist hook?

    python scripts/selector_headroom.py --dataset data/chembl/egfr_anilinoquinazoline.csv

Run this before training a selector, for the same reason
``scripts/acquisition_headroom.py`` runs a hindsight policy before any
controller: a ceiling that cannot beat plain BO retires the hook, and that is
much cheaper to learn from one script than from a trained model.

The instrument also settles a second question the offline metric cannot. A
selector can be scored two ways:

``shortlist_metrics`` (offline, ``molbo.clm``)
    gain = true value of the pick minus the shortlist mean. Single-round and
    greedy: it rewards buying the most potent candidate *now*.
campaign AUC (what the project reports)
    rewards finding the best compound within the whole budget, which needs the
    optimiser to keep learning.

These disagree, and the disagreement is the point. ``max_mu`` has 3.4x
``bo_rank0``'s offline gain and *loses* AUC, because buying the highest
posterior mean every round never spends a call resolving uncertainty. So
offline gain alone is not a valid gate on a selector; this script measures the
mapping between the two so a gate can be set on evidence.

Every arm here is a reference line, not a method. ``oracle`` reads the answer
key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.campaign import ArmConfig  # noqa: E402
from molbo.clm import MaxMeanSelector, OracleShortlistSelector, shortlist_metrics  # noqa: E402
from molbo.experiment import run_sweep, summarize_sweep  # noqa: E402
from molbo.llm import RandomShortlistSelector  # noqa: E402
from molbo.oracles import LookupOracle  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[50, 100])
    p.add_argument("--seeds", type=int, default=200,
                   help="200, not the project's usual 20. Measured: the identical "
                        "bo_maxmu_select arm gave -0.0195 (p=0.011) and +0.0087 "
                        "(p=0.19) on two library definitions differing by 17 of "
                        "2103 compounds. At n=200 both collapse to ~0.004 |dAUC|. "
                        "Selector effects here are dz~0.1-0.5 and 20 seeds "
                        "resolves neither their size nor their sign.")
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--shortlist-k", type=int, default=15)
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--min-heavy-atoms", type=int, default=10,
                   help="match run_lookup_experiment.py's default, so this "
                        "instrument and the sweeps describe the same library")
    p.add_argument("--max-heavy-atoms", type=int, default=55)
    p.add_argument("--shortlists", default=None,
                   help="a clm_collect.py .jsonl; adds the offline gain column "
                        "so the offline/campaign mapping can be read off directly")
    p.add_argument("--out", default="results/selector_headroom.json")
    return p.parse_args(argv)


def offline_gains(path: str) -> dict[str, float]:
    """Offline ``gain`` for each reference selector on recorded shortlists."""
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    true = [np.asarray(r["true"]) for r in recs]
    return {
        "bo_ei": shortlist_metrics([-np.arange(len(r["true"])) for r in recs], true)["gain"],
        "bo_random_select": 0.0,
        "bo_maxmu_select": shortlist_metrics([np.asarray(r["mu"]) for r in recs], true)["gain"],
        "bo_oracle_select": shortlist_metrics(true, true)["gain"],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    oracle = LookupOracle.from_dataset(args.dataset, value_column=args.value_column,
                                       budget=max(args.budgets),
                                       min_heavy_atoms=args.min_heavy_atoms,
                                       max_heavy_atoms=args.max_heavy_atoms)
    library = oracle.library
    table = dict(oracle._table)
    print(f"dataset      {args.dataset}\nlibrary      {len(library)} compounds  "
          f"seeds {args.seeds}  shortlist_k {args.shortlist_k}\n")

    off = offline_gains(args.shortlists) if args.shortlists else {}
    rows = []
    for budget in args.budgets:
        oracle.budget = budget
        arms = [
            ArmConfig("bo_ei", strategy="bo", acquisition="ei", n_init=args.n_init),
            ArmConfig("bo_random_select", strategy="bo", acquisition="ei", n_init=args.n_init,
                      selector=RandomShortlistSelector(), shortlist_k=args.shortlist_k),
            ArmConfig("bo_maxmu_select", strategy="bo", acquisition="ei", n_init=args.n_init,
                      selector=MaxMeanSelector(), shortlist_k=args.shortlist_k),
            ArmConfig("bo_oracle_select", strategy="bo", acquisition="ei", n_init=args.n_init,
                      selector=OracleShortlistSelector(values=table),
                      shortlist_k=args.shortlist_k),
        ]
        metrics, _ = run_sweep(oracle, arms, range(args.seeds), budget=budget,
                               k=args.k, library=library, verbose=False)
        summary = summarize_sweep(metrics, reference="bo_ei")
        # summarize_sweep returns dataclasses under "summaries"/"comparisons";
        # save_sweep is what renames them to "arms"/"comparisons_vs_reference".
        cmp = summary["comparisons"]

        head = f"{'selector':<20s}{'AUC':>9s}{'sem':>8s}{'dAUC':>9s}{'p':>9s}{'top1%':>8s}"
        if off:
            head += f"{'offline':>9s}"
        print(f"budget {budget}\n{head}\n{'-' * len(head)}")
        for arm in arms:
            a = summary["summaries"][arm.name]
            recall = float(np.mean([m.top1pct_recall for m in metrics
                                    if m.arm == arm.name and m.top1pct_recall is not None]))
            c = cmp.get(arm.name, {})
            row = {"budget": budget, "selector": arm.name, "auc": a.auc_mean,
                   "sem": a.auc_sem, "top1pct_recall": recall,
                   "delta_vs_bo_ei": c.get("mean_diff"), "p_value": c.get("p_value"),
                   "cohens_dz": c.get("cohens_dz"), "offline_gain": off.get(arm.name)}
            rows.append(row)
            d = "" if row["delta_vs_bo_ei"] is None else f"{row['delta_vs_bo_ei']:+9.4f}"
            pv = "" if row["p_value"] is None else f"{row['p_value']:9.4f}"
            line = f"{arm.name:<20s}{a.auc_mean:9.4f}{a.auc_sem:8.4f}{d:>9s}{pv:>9s}{recall:8.3f}"
            if off:
                g = row["offline_gain"]
                line += f"{g:9.3f}" if g is not None else f"{'-':>9s}"
            print(line)

        ceiling = next(r for r in rows if r["budget"] == budget and r["selector"] == "bo_oracle_select")
        best_real = max((r for r in rows if r["budget"] == budget
                         and r["selector"] != "bo_oracle_select"), key=lambda r: r["auc"])
        print(f"\nVERDICT (budget {budget}): the hook's ceiling is "
              f"{ceiling['delta_vs_bo_ei']:+.4f} AUC over plain BO "
              f"(p={ceiling['p_value']:.4f}).")
        if ceiling["delta_vs_bo_ei"] is None or ceiling["delta_vs_bo_ei"] <= 2 * ceiling["sem"]:
            print("  That is within 2 sem of zero -- the shortlist hook has no room, and no\n"
                  "  selector can win here. Intervene somewhere else.")
        else:
            free = [r for r in rows if r["budget"] == budget
                    and r["selector"] not in ("bo_oracle_select", "bo_ei")]
            best_free = max(free, key=lambda r: r["auc"])
            captured = (best_free["delta_vs_bo_ei"] or 0.0) / ceiling["delta_vs_bo_ei"]
            print(f"  Real room, and nothing free takes it. Best selector that does not read\n"
                  f"  the answer key is {best_free['selector']} at "
                  f"{best_free['delta_vs_bo_ei']:+.4f} AUC (p={best_free['p_value']:.3f}), i.e. "
                  f"{100 * captured:.0f}% of the ceiling.")
            if off:
                mm = next(r for r in rows if r["budget"] == budget
                          and r["selector"] == "bo_maxmu_select")
                print(f"  Offline->campaign mapping is steep: offline gain "
                      f"{off['bo_maxmu_select']:.2f} (max_mu) buys {mm['delta_vs_bo_ei']:+.4f} AUC, "
                      f"offline {off['bo_oracle_select']:.2f} (oracle)\n"
                      f"  buys {ceiling['delta_vs_bo_ei']:+.4f}. A selector scoring at or below "
                      f"{off['bo_maxmu_select']:.2f} offline is not worth a campaign run.")
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"dataset": args.dataset, "library_size": len(library), "seeds": args.seeds,
         "n_init": args.n_init, "k": args.k, "shortlist_k": args.shortlist_k,
         "shortlists": args.shortlists, "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
