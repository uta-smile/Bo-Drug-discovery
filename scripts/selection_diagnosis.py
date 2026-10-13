"""Where does the history-design advantage enter: eligibility, or acquisition?

    python scripts/selection_diagnosis.py --family jak --budget 300
    python scripts/selection_diagnosis.py --family cyp --budget 111 \
        --json results/selection_diagnosis_cyp.json

Two different mechanisms produce the same outward behaviour. A model may avoid
the auxiliary history because its acquisition value there is suppressed, or the
history may simply have been removed from the candidate set. Only the second is
a bug, and the manuscript's account of the pooled arm rests on it being the
first.

The campaign code answers the question directly: ``cand`` is every compound not
yet assayed *on the primary target*, so a compound measured only on an
auxiliary target stays eligible. This script shows it rather than asserting it,
by instrumenting the selection loop and reporting, per round:

* how many compounds are eligible, and how many of them are in the auxiliary
  history;
* the mean posterior mean, posterior standard deviation and acquisition value
  **split by history membership**, which is where a suppression effect would
  appear;
* what fraction of the eligible set belongs to the target's true top-k (truth
  used only for retrospective evaluation, never by a method);
* what fraction of the compounds actually selected were in the history.

Three selection conditions are crossed with the two history designs:

* ``eligible_all``   -- every primary-unmeasured compound is eligible. This is
                        what the campaigns in the paper do.
* ``exclude_hist``   -- auxiliary-history compounds are explicitly removed from
                        the candidate set. Not a method; it is the upper bound
                        on what an exclusion bug could buy.
* ``random``         -- uniform selection from the eligible set, no model. If
                        this gains from withholding the top-k, the design
                        advantage needs no chemistry modelling at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import dense_block_campaign as dbc  # noqa: E402
from dense_block_h1 import build_block, drop_censored, nanomolar_potency  # noqa: E402
from molbo.metrics import auc_top_k  # noqa: E402
from molbo.surrogate import fingerprint_matrix, tanimoto_gram  # noqa: E402

CONDITIONS = ("eligible_all", "exclude_hist")


def load_block(args):
    """The same block the campaigns use, cut the same way.

    Kept to the two BindingDB families this diagnosis runs on rather than
    reimplementing the driver's whole family/panel resolution: a second,
    subtly different loader is how two scripts end up disagreeing about what
    "the JAK block" means.
    """
    families = {**dbc.FAMILIES, **dbc.PANEL_FAMILIES}
    if args.family not in families:
        raise SystemExit(f"unknown family {args.family!r}")
    mapping = families[args.family]
    block, _ = build_block(mapping, source=Path(args.source),
                           potency=nanomolar_potency)
    targets = [t for t in mapping.values() if t in block.columns]
    if not args.keep_censored:
        block, _ = drop_censored(block, targets, pile_frac=0.02, verbose=False)
    K = tanimoto_gram(*(2 * (fingerprint_matrix(block.smiles.tolist()),)))
    return K, block[targets].to_numpy(float), targets


def trace_collector(hist_set: set[int], top_set: set[int]) -> tuple[list, callable]:
    rows: list[dict] = []

    def cb(ev):
        elig = np.asarray(ev["eligible"], dtype=np.intp)
        in_hist = np.array([int(i) in hist_set for i in elig])
        row = {
            "round": int(ev["round"]), "n_observed": int(ev["n_observed"]),
            "n_eligible": int(len(elig)),
            "n_eligible_in_history": int(in_hist.sum()),
            "frac_eligible_in_topk": float(
                np.mean([int(i) in top_set for i in elig])),
            "n_selected": int(len(ev["selected"])),
            "n_selected_in_history": int(
                sum(int(i) in hist_set for i in np.asarray(ev["selected"]))),
            "n_selected_in_topk": int(
                sum(int(i) in top_set for i in np.asarray(ev["selected"]))),
        }
        for name in ("mu", "sd", "acq"):
            v = ev[name]
            if v is None:
                continue
            v = np.asarray(v, dtype=float)
            row[f"{name}_hist"] = (float(v[in_hist].mean())
                                   if in_hist.any() else None)
            row[f"{name}_rest"] = (float(v[~in_hist].mean())
                                   if (~in_hist).any() else None)
        rows.append(row)

    return rows, cb


def one(block, *, arm, condition, exclude_top, budget, args, repeat, target_i):
    """One campaign, traced. Returns (recall, auc, trace rows, aux stats)."""
    K, vals, targets = block
    g = np.random.default_rng(args.seed + 9001 * repeat + 17 * target_i)
    others = [j for j in range(vals.shape[1]) if j != target_i]
    y = vals[:, target_i]

    hist = dbc.draw_history(K, y, args.history * len(others), g,
                            cluster_frac=0.0, exclude_top=exclude_top,
                            k=args.k)
    aux_idx = [hist[a * args.history:(a + 1) * args.history]
               for a in range(len(others))]
    aux_val = [vals[aux_idx[a], others[a]] for a in range(len(others))]

    hist_set = {int(i) for i in hist}
    top_set = {int(i) for i in np.argsort(-y)[:args.k]}

    eligible = None
    if condition == "exclude_hist":
        eligible = np.setdiff1d(np.arange(len(y)), hist)

    rows, cb = trace_collector(hist_set, top_set)
    seq, idx = dbc.run_campaign(
        K, y, aux_idx, aux_val,
        arm=arm,
        budget=args.budget, batch=args.batch, n_init=args.n_init, k=args.k,
        acquisition="ei", rng=g, eligible=eligible, trace=cb)
    found = len(set(int(i) for i in idx) & top_set)
    return {
        "recall": found / args.k,
        "auc": float(auc_top_k(y[idx], args.k)),
        "trace": rows,
        "frac_history_in_topk": float(
            len(hist_set & top_set) / max(len(top_set), 1)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="jak")
    p.add_argument("--source", default="data/cache/tdc/bindingdb_ic50.csv")
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--history", type=int, default=50)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="*", default=["stl", "mt", "pooled"])
    p.add_argument("--keep-censored", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    block = load_block(args)
    K, vals, targets = block
    print(f"{args.family}: {len(vals)} compounds, {len(targets)} targets, "
          f"budget {args.budget}, {args.repeats} repeats x {len(targets)} "
          f"held-out targets\n")

    out = {}
    for exclude_top in (False, True):
        design = "withheld" if exclude_top else "uniform"
        # The selection rule and the eligibility rule are separate axes. In
        # particular `random` must be run under BOTH eligibility rules: random
        # selection from outside the history is the condition that says whether
        # the design advantage needs any chemistry modelling at all.
        for condition in ("eligible_all", "exclude_hist"):
            for arm in [*args.arms, "random"]:
                key = f"{design}/{condition}/{arm}"
                recs, aucs, traces, inhist = [], [], [], []
                for ti in range(len(targets)):
                    for r in range(args.repeats):
                        res = one(block, arm=arm, condition=condition,
                                  exclude_top=exclude_top, budget=args.budget,
                                  args=args, repeat=r, target_i=ti)
                        recs.append(res["recall"])
                        aucs.append(res["auc"])
                        traces.append(res["trace"])
                        inhist.append(res["frac_history_in_topk"])
                out[key] = summarise(traces, recs, aucs, inhist)
                print(f"  {key:<34}recall {np.mean(recs):.1%}   "
                      f"eligible in top-k {out[key]['frac_eligible_in_topk_r0']:.3%}"
                      f" -> {out[key]['frac_eligible_in_topk_last']:.3%}   "
                      f"picks from history {out[key]['frac_picks_from_history']:.1%}")

    report(out)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def summarise(traces, recs, aucs, inhist) -> dict:
    first = [t[0] for t in traces if t]
    last = [t[-1] for t in traces if t]
    allr = [r for t in traces for r in t]

    def m(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    picks = sum(r["n_selected"] for r in allr)
    # The mean acquisition value over the history is the wrong statistic: a
    # method selects the *top* of the ranking, not the mean, and on CYP the
    # pooled arm's mean acquisition over the history is slightly higher than
    # over the rest while it picks from the history at a third of the base
    # rate. What matters is the selection ratio -- how often the history is
    # chosen, against how often indifference would choose it. 1.0 is
    # indifferent, below 1 is avoidance, and no arm is told to avoid anything.
    base = [r["n_eligible_in_history"] / max(r["n_eligible"], 1) for r in allr]
    base_rate = float(np.mean(base)) if base else float("nan")
    frac_picks = sum(r["n_selected_in_history"] for r in allr) / max(picks, 1)
    return {
        "history_base_rate": base_rate,
        "history_selection_ratio": (float(frac_picks / base_rate)
                                    if base_rate else None),
        "recall_mean": float(np.mean(recs)), "auc_mean": float(np.mean(aucs)),
        "frac_history_in_topk": float(np.mean(inhist)),
        "n_eligible_r0": m(first, "n_eligible"),
        "n_eligible_in_history_r0": m(first, "n_eligible_in_history"),
        "frac_eligible_in_topk_r0": m(first, "frac_eligible_in_topk"),
        "frac_eligible_in_topk_last": m(last, "frac_eligible_in_topk"),
        "frac_picks_from_history": float(
            sum(r["n_selected_in_history"] for r in allr) / max(picks, 1)),
        "acq_hist": m(allr, "acq_hist"), "acq_rest": m(allr, "acq_rest"),
        "mu_hist": m(allr, "mu_hist"), "mu_rest": m(allr, "mu_rest"),
        "sd_hist": m(allr, "sd_hist"), "sd_rest": m(allr, "sd_rest"),
        "rounds": len(allr) // max(len(traces), 1),
    }


def report(out: dict) -> None:
    print("\n  Is the auxiliary history eligible for selection?")
    for key, v in out.items():
        if "/eligible_all/" not in key:
            continue
        n_hist = v["n_eligible_in_history_r0"]
        print(f"    {key:<34}{n_hist:.0f} of {v['n_eligible_r0']:.0f} eligible "
              f"compounds are in the history at round 0")
    print("    A nonzero count means the history was never removed from the "
          "candidate set.\n")

    print("  How often each arm selects from the history, against "
          "indifference (1.00):")
    print(f"    {'condition':<34}{'base rate':>10}{'picked':>9}"
          f"{'ratio':>8}{'mean acq h/rest':>18}")
    for key, v in out.items():
        if "/eligible_all/" not in key:
            continue
        acq = ("--" if v["acq_hist"] is None
               else f"{v['acq_hist'] / v['acq_rest']:.3f}")
        print(f"    {key:<34}{v['history_base_rate']:>9.1%}"
              f"{v['frac_picks_from_history']:>9.1%}"
              f"{v['history_selection_ratio']:>8.2f}{acq:>18}")
    print("    No arm is told to avoid the history; a ratio below 1 is the "
          "model doing it.\n    The mean acquisition value does not capture "
          "this -- selection reads the top of\n    the ranking, not the mean.")

    print("\n  What withholding the top-k does to the eligible set:")
    for cond in ("eligible_all", "exclude_hist"):
        for arm in ("stl", "mt", "pooled", "random"):
            u, w = f"uniform/{cond}/{arm}", f"withheld/{cond}/{arm}"
            if u not in out or w not in out:
                continue
            du = out[u]["recall_mean"]
            dw = out[w]["recall_mean"]
            print(f"    {cond + '/' + arm:<28}recall uniform {du:.1%} -> "
                  f"withheld {dw:.1%}  ({dw - du:+.1%})")


if __name__ == "__main__":
    raise SystemExit(main())
