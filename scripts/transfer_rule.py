"""A rule for what to do with free history, validated out of sample.

    python scripts/transfer_rule.py --out results/transfer_rule.json

Every threshold in this package so far has been *fitted*: section 38 fits a zero
crossing to 18 blocks and reports where it lands. That says the relationship
exists. It does not say a practitioner holding a new block and the number
``rho`` would do better by following it, because the crossing was fitted on the
same blocks it is quoted against.

This runs the honest version. For each block in turn: fit on the other
``n - 1``, apply to the held-out one, record what was actually earned. Three
things to do with a free history, measured under one protocol so they are
comparable:

* **abstain**    -- single-task BO, random initial design. Ignore the history.
* **seed**       -- single-task BO, initial design drawn from the history's
                    best-scoring compounds (``--seed-pool aux_top``).
* **transfer**   -- coregionalised GP on the history, random initial design.

and four references the rule has to be read against:

* always abstain (zero by construction), always seed, always transfer,
* and the **oracle**, which picks the best option per block knowing the answer.
  It is a ceiling, not a method, and is labelled as one everywhere below.

The rule itself is deliberately the simplest thing that can use ``rho``: a
linear fit of each option's gain on the block's coupling, fitted on the training
blocks, with the held-out block assigned the option whose predicted gain is
highest. Two parameters per option on 17 blocks. Anything richer would be fitting
the noise, and the point of the exercise is to find out whether even this much
generalises.

``--x`` selects the feature. ``rho`` is the block's median off-diagonal Spearman
on the raw block; ``rho_eff`` is the correlation the rule actually sees on the
rows it is handed, which is what section 38 fits and is measurable before any
budget is spent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

OPTIONS = ("abstain", "seed", "transfer")


def couplings(blocks: Path) -> dict[str, float]:
    """``{label: rho}``, from the selection artefact or the scan beside it.

    ``--blocks-from`` labels blocks ``k01``, ``k02``, ... in file order, so the
    label is positional. The selection artefact does not always carry ``rho``;
    the scan it was cut from does. When both are present the two are matched on
    the target set rather than on position, because a positional match that
    silently went wrong would attach every gain to the wrong coupling and still
    produce a plausible-looking fit.
    """
    d = json.loads(blocks.read_text(encoding="utf-8"))
    rows = d["blocks"]
    if all("rho" in b for b in rows):
        return {f"k{i + 1:02d}": b["rho"] for i, b in enumerate(rows)}

    scan = blocks.with_name(blocks.name.replace("_blocks", "_scan"))
    if not scan.exists():
        raise SystemExit(
            f"{blocks.name} carries no `rho` and {scan.name} is not beside it. "
            "The coupling is the rule's only feature; it cannot be guessed.")
    by_targets = {tuple(b["uniprot"]): b["rho"]
                  for b in json.loads(scan.read_text(encoding="utf-8"))["blocks"]
                  if "rho" in b}
    out = {}
    for i, b in enumerate(rows):
        key = tuple(b["uniprot"])
        if key not in by_targets:
            raise SystemExit(
                f"block k{i + 1:02d} ({', '.join(key)}) is not in "
                f"{scan.name}; the two artefacts are not the same selection.")
        out[f"k{i + 1:02d}"] = by_targets[key]
    return out


def load(results: Path, blocks: Path, seed_dir: Path | None):
    """Per block: its coupling, and the AUC gain of each option over abstain."""
    rho = couplings(blocks)

    rows = []
    for label in sorted(rho):
        f = results / f"{label}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        arms = d["arms"]
        if "stl" not in arms or "mt" not in arms:
            continue
        base = np.asarray(arms["stl"]["auc"], float)
        row = {"block": label, "rho": rho[label], "n_pool": d.get("n_pool"),
               "abstain": np.zeros_like(base),
               "transfer": np.asarray(arms["mt"]["auc"], float) - base}
        if seed_dir is not None:
            g = seed_dir / f"{label}.json"
            if g.exists():
                sd = json.loads(g.read_text(encoding="utf-8"))
                if "stl" in sd.get("arms", {}):
                    row["seed"] = (np.asarray(sd["arms"]["stl"]["auc"], float)
                                   - base)
        rows.append(row)
    return rows


def options_present(rows):
    return tuple(o for o in OPTIONS if all(o in r for r in rows))


def fit_predict(train, test, opts, x):
    """Least squares of each option's mean gain on ``x``, then argmax."""
    xs = np.array([r[x] for r in train], float)
    pred = {}
    for o in opts:
        ys = np.array([r[o].mean() for r in train], float)
        if o == "abstain":
            pred[o] = 0.0
            continue
        # Degree 1 on 17 points. `polyfit` is the whole model.
        c = np.polyfit(xs, ys, 1)
        pred[o] = float(np.polyval(c, test[x]))
    return max(pred, key=pred.get), pred


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path,
                   default=Path("results/xfam_transfer"))
    p.add_argument("--seed-results", type=Path,
                   default=Path("results/xfam_seed"),
                   help="campaigns with --seed-pool aux_top, same protocol; "
                        "omitted if absent, and the rule then chooses between "
                        "abstain and transfer only")
    p.add_argument("--blocks", type=Path,
                   default=Path("results/cross_family_blocks.json"))
    p.add_argument("--x", default="rho", help="the feature the rule reads")
    p.add_argument("--out", type=Path,
                   default=Path("results/transfer_rule.json"))
    args = p.parse_args(argv)

    seed_dir = args.seed_results if args.seed_results.exists() else None
    rows = load(args.results, args.blocks, seed_dir)
    if len(rows) < 4:
        raise SystemExit(
            f"only {len(rows)} block(s) under {args.results}; "
            "leave-one-out needs the sweep to have finished.")
    opts = options_present(rows)
    print(f"{len(rows)} blocks, options: {', '.join(opts)}, feature {args.x}")
    if "seed" not in opts:
        print("  (no --seed-pool aux_top runs found, so the rule chooses "
              "between abstain and transfer only)")

    # -- what each fixed policy earns ----------------------------------------
    fixed = {o: float(np.mean([r[o].mean() for r in rows])) for o in opts}
    oracle = float(np.mean([max(r[o].mean() for o in opts) for r in rows]))

    # -- the rule, leave one block out ---------------------------------------
    chosen, earned, preds = [], [], []
    for i, test in enumerate(rows):
        train = rows[:i] + rows[i + 1:]
        pick, pred = fit_predict(train, test, opts, args.x)
        chosen.append(pick)
        earned.append(float(test[pick].mean()))
        preds.append(pred)
    rule = float(np.mean(earned))

    print(f"\n  {'policy':<28}{'mean AUC gain':>15}{'regret vs ceiling':>20}")
    print("  " + "-" * 63)
    for o in opts:
        print(f"  always {o:<21}{fixed[o]:>+15.4f}{oracle - fixed[o]:>20.4f}")
    print(f"  {'RULE (leave-one-block-out)':<28}{rule:>+15.4f}"
          f"{oracle - rule:>20.4f}")
    print(f"  {'per-block ceiling':<28}{oracle:>+15.4f}{0.0:>20.4f}"
          f"   <- reads the answer")

    best_fixed = max(fixed, key=fixed.get)
    print(f"\n  {'block':<8}{args.x:>9}" +
          "".join(f"{o:>11}" for o in opts) + f"{'rule picks':>13}")
    print("  " + "-" * (30 + 11 * len(opts)))
    for r, pick in zip(rows, chosen):
        print(f"  {r['block']:<8}{r[args.x]:>+9.3f}" +
              "".join(f"{r[o].mean():>+11.4f}" for o in opts) +
              f"{pick:>13}")

    # -- the rule against that fixed policy, tested honestly -----------------
    # On most blocks the rule *picks* the fixed policy, so the paired
    # difference is exactly zero. A signed-rank test discards those pairs and
    # then leaves scipy's `method="auto"` no exact path, because zeros are
    # present; the probability it returns is a normal approximation over the
    # handful of discordant blocks, not an exact test over eighteen. Quoting
    # it as the latter overstates the sample by a factor of six. We report the
    # discordant blocks and test only those, exactly.
    diff = np.array([e - r[best_fixed].mean() for e, r in zip(earned, rows)])
    disc = diff[diff != 0.0]
    test = {"best_fixed": best_fixed,
            "mean_difference": float(diff.mean()),
            "sem_difference": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "n_blocks": int(len(diff)),
            "n_zero": int((diff == 0.0).sum()),
            "n_discordant": int(len(disc)),
            "better": int((diff > 0).sum()), "worse": int((diff < 0).sum())}
    if len(disc):
        test["signed_rank_p_exact"] = float(
            wilcoxon(disc, method="exact").pvalue)
        test["sign_test_p"] = float(
            binomtest((disc > 0).sum(), len(disc), 0.5).pvalue)
        test["scipy_default_p"] = float(wilcoxon(diff).pvalue)
    print(f"\n  rule minus always-{best_fixed}: "
          f"{diff.mean():+.4f} +- {test['sem_difference']:.4f} AUC over "
          f"{len(diff)} blocks,\n  but the rule picks that policy on "
          f"{test['n_zero']} of them, so only {len(disc)} blocks are "
          f"discordant\n  ({test['better']} better, {test['worse']} worse). "
          "Tested on those:")
    if len(disc):
        print(f"    exact signed-rank p = {test['signed_rank_p_exact']:.4f}, "
              f"sign test p = {test['sign_test_p']:.4f}")
        print(f"    (scipy's default over all {len(diff)} returns "
              f"{test['scipy_default_p']:.4f}, a normal approximation over "
              f"{len(disc)} pairs -- do not quote it)")

    print()
    if rule > fixed[best_fixed] + 1e-9:
        print(f"  The rule beats the best fixed policy (always {best_fixed}) by "
              f"{rule - fixed[best_fixed]:+.4f} AUC,\n  closing "
              f"{(rule - fixed[best_fixed]) / max(oracle - fixed[best_fixed], 1e-12):.0%} "
              f"of the gap to the per-block ceiling.")
    else:
        print(f"  The rule does NOT beat always-{best_fixed} "
              f"({rule:+.4f} against {fixed[best_fixed]:+.4f}).\n"
              f"  On these blocks, choosing per block with {args.x} is not worth "
              f"doing: the\n  honest recommendation is the fixed policy.")
    agree = sum(1 for r, pick in zip(rows, chosen)
                if pick == max(opts, key=lambda o: r[o].mean()))
    print(f"  It picks the per-block best on {agree}/{len(rows)} blocks.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "feature": args.x, "options": list(opts), "n_blocks": len(rows),
        "fixed": fixed, "oracle": oracle, "rule": rule,
        "picks": {r["block"]: pick for r, pick in zip(rows, chosen)},
        "per_block": [{"block": r["block"], args.x: r[args.x],
                       **{o: float(r[o].mean()) for o in opts}} for r in rows],
        "rule_vs_best_fixed": test,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
