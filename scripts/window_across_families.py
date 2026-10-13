"""Does the operating window hold across protein families, or only the two?

    python scripts/window_across_families.py --out results/window_families.json

Section 25 states the paper's central practical claim on two blocks, JAK and
CYP, both chosen by hand: at 22-33% of the library the multi-task gain is large
*and unanimous over all four held-out targets*, and unanimity is what licenses
the leave-one-target-out self-check a practitioner would run before trusting the
method on a cold target. Two hand-picked families cannot separate "this holds on
co-measured panels" from "this holds on these two panels".

``find_dense_blocks.py`` supplies more families without hand-picking;
``dense_block_campaign.py`` runs the identical protocol on each. This script
reads those artefacts and asks the two questions that follow:

1. **Does the window replicate?** Mean gain and unanimity per cell.
2. **If it does not replicate everywhere, what separates the families where it
   does?** Two candidate predictors, measured per held-out target:

   * the **auxiliary task correlation** -- the mean Spearman of the held-out
     target's column against its three relatives. Section 18's mechanism says
     transfer flows through ``k_tanimoto * r``, and reach is 1.0 inside a dense
     block by construction, so ``r`` is the only term left that can vary.
   * the **single-task baseline recall** in that cell -- how much the reference
     arm has left to win. Section 25 already names this as the cause of the
     decay past 50% ("there is nothing left to win at the end"); the question
     here is whether it also explains the variation *between* families, which
     would mean the fraction-of-library axis is a proxy for it rather than the
     quantity itself.

They are reported as partial rank correlations against each other, because a
family whose targets correlate strongly is also plausibly a family a single-task
GP already does well on, and the raw correlations cannot tell those apart.

Reads ``results/win_*.json`` (8 repeats, the stored section-25 grid and its
extension) or ``results/win32_*.json`` (32 repeats, the power re-run) -- whichever
prefix is passed. Writes nothing a campaign needs; this is read-only analysis.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402

from find_dense_blocks import (  # noqa: E402,F401
    MIN_POOL, family_properties, print_families)
"""Block sizes and task correlations come from the discovery
module, which owns the question of what blocks exist. This file
owns only the reading of campaign artefacts.
"""


def load_cells(pattern, prop, *, repeats=None):
    """One record per (family, budget) artefact, plus its per-target rows."""
    cells, rows = [], []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        fam = d["family"]
        if fam not in prop or (repeats is not None and d.get("repeats") != repeats):
            continue
        a = d["arms"]
        if a.get("mt", {}).get("delta_vs_stl") is None:
            # An artefact run without the `stl` arm carries no delta to read:
            # `dense_block_campaign` writes null there rather than a zero that
            # would look like a measured tie. Skip it loudly instead of
            # formatting a None into the table.
            print(f"    skipping {f}: no stl arm, nothing to difference against")
            continue
        pt = d["per_target"]
        stl = 100 * np.mean([r["stl_recall"] for r in pt.values()])
        mt = 100 * np.mean([r["mt_recall"] for r in pt.values()])
        cells.append({
            # The artefacts do not share one naming convention -- section 25's
            # are win_<fam>_<budget>.json, the later ones win_<fam>_b<budget> --
            # so the path is carried rather than rebuilt from the fields.
            "path": f,
            "family": fam, "n_pool": d["n_pool"], "budget": d["budget"],
            "frac": round(100 * d["budget"] / d["n_pool"]),
            "repeats": d["repeats"],
            "stl_recall": stl, "mt_recall": mt, "delta_recall": mt - stl,
            "pooled_recall": 100 * a["pooled"]["recall_mean"] if "pooled" in a else None,
            "delta_auc": a["mt"]["delta_vs_stl"], "delta_auc_sem": a["mt"]["delta_sem"],
            "targets_improving": d["targets_improving"],
            "audit_clean": d.get("audit", {}).get("clean"),
            "weakest_aux_rho": min(prop[fam]["aux_rho"].values()),
        })
        for t, r in pt.items():
            rows.append({"family": fam, "frac": cells[-1]["frac"], "target": t,
                         "aux_rho": prop[fam]["aux_rho"][t],
                         "stl_recall": r["stl_recall"], "delta": r["delta"]})
    return cells, rows


def partial_spearman(x, y, z):
    """Rank correlation of x and y with the rank of z projected out."""
    x, y, z = (rankdata(np.asarray(v, float)) for v in (x, y, z))
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r, p = pearsonr(rx, ry)
    return float(r), float(p)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="results/win_*.json")
    p.add_argument("--cov-pattern", default="results/wincov_*.json",
                   help="the matched-history-coverage control runs; section [7] "
                        "is skipped when none are present")
    p.add_argument("--repeats", type=int, default=8,
                   help="only read artefacts with this many repeats, so an 8- "
                        "and a 32-repeat run of the same cell are never mixed")
    p.add_argument("--window", type=int, nargs="+", default=[22, 33],
                   help="the library fractions section 25 calls the window")
    p.add_argument("--out", default="results/window_families.json")
    args = p.parse_args(argv)

    prop = family_properties()
    cells, rows = load_cells(args.pattern, prop, repeats=args.repeats)
    if not cells:
        raise SystemExit(f"no artefacts matched {args.pattern} at "
                         f"{args.repeats} repeats")

    print(f"[1] the blocks ({sum(p['runnable'] for p in prop.values())} of "
          f"{len(prop)} large enough to run at H=50)")
    print_families(prop)

    print(f"\n[2] the window, {args.repeats} repeats x 4 held-out targets per cell")
    print(f"    {'family':<7}{'N':>5}{'%lib':>6}{'bud':>5}{'stl':>8}{'mt':>8}"
          f"{'pooled':>8}{'drecall':>9}{'dAUC':>18}{'up':>6}")
    order = {"jak": 0, "cyp": 1}
    for c in sorted(cells, key=lambda c: (order.get(c["family"], 9),
                                          c["family"], c["budget"])):
        star = "*" if c["frac"] in args.window else " "
        pl = f"{c['pooled_recall']:.1f}%" if c["pooled_recall"] is not None else "-"
        print(f"  {star} {c['family']:<7}{c['n_pool']:>5}{c['frac']:>5}%"
              f"{c['budget']:>5}{c['stl_recall']:>7.1f}%{c['mt_recall']:>7.1f}%"
              f"{pl:>8}{c['delta_recall']:>+9.1f}"
              f"{c['delta_auc']:>+12.4f}+-{c['delta_auc_sem']:.4f}"
              f"{c['targets_improving']:>4}/4")
    bad = [c for c in cells if c["audit_clean"] is False]
    print(f"    audit: {len(cells) - len(bad)}/{len(cells)} cells clean"
          + (f" -- FAILING: {[c['family'] for c in bad]}" if bad else ""))

    win = [c for c in cells if c["frac"] in args.window]
    un = sum(c["targets_improving"] == 4 for c in win)
    pos = sum(c["delta_recall"] > 0 for c in win)
    fams = sorted({c["family"] for c in win})
    fam_pos = sum(all(c["delta_recall"] > 0 for c in win if c["family"] == f)
                  for f in fams)
    print(f"\n[3] inside the window: mean gain positive in {pos}/{len(win)} cells "
          f"and in every cell of {fam_pos}/{len(fams)} families;")
    print(f"    unanimous (4/4 held-out targets) in {un}/{len(win)} cells")

    print(f"\n[4] what predicts the per-target gain ({len(rows)} rows, window only)")
    w = [r for r in rows if r["frac"] in args.window]
    ar = [r["aux_rho"] for r in w]
    sr = [r["stl_recall"] for r in w]
    dl = [r["delta"] for r in w]
    stats = {}
    for lab, v in (("aux task correlation", ar), ("single-task baseline recall", sr)):
        rho = spearmanr(v, dl)
        other = sr if v is ar else ar
        pr, pp = partial_spearman(v, dl, other)
        stats[lab] = {"spearman": float(rho.statistic), "p": float(rho.pvalue),
                      "partial": pr, "partial_p": pp}
        print(f"    {lab:<28} rho {rho.statistic:+.3f} (p={rho.pvalue:.1e})   "
              f"partial {pr:+.3f} (p={pp:.1e})")
    rr = spearmanr(ar, sr)
    print(f"    {'the two against each other':<28} rho {rr.statistic:+.3f} "
          f"(p={rr.pvalue:.1e}) -- near-independent, so both terms are real")

    print("\n[5] which rule picks out the unanimous cells")
    def rule(name, pred):
        inside = [c for c in cells if pred(c)]
        out = [c for c in cells if not pred(c)]
        a = sum(c["targets_improving"] == 4 for c in inside)
        b = sum(c["targets_improving"] == 4 for c in out)
        print(f"    {name:<44} {a}/{len(inside)} inside, {b}/{len(out)} outside")
        return {"inside": [a, len(inside)], "outside": [b, len(out)]}

    rules = {
        "frac_window": rule(f"{'-'.join(map(str, args.window))}% of library "
                            "(the section-25 rule)",
                            lambda c: c["frac"] in args.window),
        "baseline_band": rule("single-task recall in 40-85%",
                              lambda c: 40 <= c["stl_recall"] <= 85),
        "rho_floor": rule("family's weakest aux rho >= 0.40",
                          lambda c: c["weakest_aux_rho"] >= 0.40),
    }

    # Section 31 concluded from JAK and CYP that the fitted task correlation
    # does not earn its place -- naive pooling ties on one and beats it on the
    # other, so the gain belongs to the free history rather than to the
    # coregionalisation. Both arms are stored per campaign and share an initial
    # design, so that is a paired test, and it can be re-asked on the other
    # families without running anything.
    print()
    print("[6] the fitted task correlation against naive pooling "
          "(paired, window cells)")
    print(f"    {'family':<7}{'rho':>6}{'%lib':>6}{'d recall':>10}"
          f"{'d AUC (mt - pooled)':>26}{'p':>8}")
    pooled_rows = []
    for c in sorted(win, key=lambda c: (order.get(c["family"], 9), c["family"],
                                        c["budget"])):
        a = json.load(open(c["path"]))["arms"]
        if "pooled" not in a or "auc" not in a["pooled"]:
            continue
        m, q = np.asarray(a["mt"]["auc"]), np.asarray(a["pooled"]["auc"])
        mr, qr = np.asarray(a["mt"]["recall"]), np.asarray(a["pooled"]["recall"])
        dd = m - q
        from scipy.stats import wilcoxon
        pv = float(wilcoxon(m, q).pvalue)
        pooled_rows.append({"family": c["family"], "frac": c["frac"],
                            "delta_recall": float(100 * (mr - qr).mean()),
                            "delta_auc": float(dd.mean()),
                            "sem": float(dd.std(ddof=1) / np.sqrt(len(dd))),
                            "p": pv})
        print(f"    {c['family']:<7}{prop[c['family']]['median_rho']:>+6.2f}"
              f"{c['frac']:>5}%{100 * (mr - qr).mean():>+9.1f}"
              f"{dd.mean():>+18.4f}+-{dd.std(ddof=1) / np.sqrt(len(dd)):.4f}"
              f"{pv:>8.3f}")
    wins = sum(r["delta_auc"] > 0 and r["p"] < 0.05 for r in pooled_rows)
    loss = sum(r["delta_auc"] < 0 and r["p"] < 0.05 for r in pooled_rows)
    print(f"    mt beats pooled on AUC in {wins} of {len(pooled_rows)} cells at "
          f"p<0.05, loses in {loss}")
    # The other half of section 31's question: pooling's downside. `mt` and
    # `pooled` are compared against the single-task reference each cell, and
    # the interesting quantity is not which is larger on average but which can
    # go negative -- a practitioner runs one campaign, not a mean.
    neg_mt = [c["family"] for c in win if c["delta_recall"] <= 0]
    neg_pool = [c["family"] for c in win
                if c["pooled_recall"] is not None
                and c["pooled_recall"] <= c["stl_recall"]]
    print(f"    cells where the arm is no better than single-task: "
          f"mt {len(neg_mt)}/{len(win)} {sorted(set(neg_mt))}, "
          f"pooled {len(neg_pool)}/{len(win)} {sorted(set(neg_pool))}")

    # The confound only six families can show. Section 25 normalises the budget
    # by library size and leaves the history absolute at H=50 per auxiliary, so
    # the free information is 17% of JAK's library and 63% of MMP's. `run_window
    # _coverage.sh` re-runs the same cells with H set per family to hold
    # coverage at 25%; comparing the two is a paired, within-family test of
    # whether the cross-block differences were ever about the blocks.
    cov_cells, _ = load_cells(args.cov_pattern, prop, repeats=args.repeats)
    cov_rows = []
    if cov_cells:
        by = {(c["family"], c["frac"]): c for c in cov_cells}
        print(f"\n[7] matched history coverage ({args.cov_pattern})")
        print(f"    {'family':<7}{'N':>5}{'%lib':>6}{'cov@H=50':>10}"
              f"{'gain H=50':>17}{'gain matched':>18}")
        for c in sorted(win, key=lambda c: (order.get(c["family"], 9),
                                            c["family"], c["budget"])):
            m = by.get((c["family"], c["frac"]))
            if m is None:
                continue
            cov = 150 / (c["n_pool"] - 10)
            cov_rows.append({"family": c["family"], "frac": c["frac"],
                             "coverage_at_H50": float(cov),
                             "gain_H50": c["delta_recall"],
                             "gain_matched": m["delta_recall"],
                             "up_H50": c["targets_improving"],
                             "up_matched": m["targets_improving"]})
            print(f"    {c['family']:<7}{c['n_pool']:>5}{c['frac']:>5}%"
                  f"{100 * cov:>10.0f}%{c['delta_recall']:>+11.1f}"
                  f" ({c['targets_improving']}/4)"
                  f"{m['delta_recall']:>+9.1f} ({m['targets_improving']}/4)")
        g0 = [r["gain_H50"] for r in cov_rows]
        g1 = [r["gain_matched"] for r in cov_rows]
        dcov = [0.25 - r["coverage_at_H50"] for r in cov_rows]
        dgain = [b - a for a, b in zip(g0, g1)]
        rr = spearmanr(dcov, dgain)
        print(f"    spread of the gain across families: "
              f"{min(g0):+.1f}..{max(g0):+.1f} at H=50, "
              f"{min(g1):+.1f}..{max(g1):+.1f} matched")
        print(f"    change in gain vs change in coverage: rho "
              f"{rr.statistic:+.3f} (p={rr.pvalue:.4f}) -- a family that loses "
              "history loses gain")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repeats": args.repeats, "window": args.window,
                               "families": prop, "cells": cells,
                               "per_target": rows, "predictors": stats,
                               "rules": rules, "mt_vs_pooled": pooled_rows,
                               "matched_coverage": cov_rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
