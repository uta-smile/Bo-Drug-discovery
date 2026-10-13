"""Does the acquisition function matter once the surrogate is multi-task?

    python scripts/acquisition_across_families.py

section 1 retired the acquisition hook: a per-round hindsight ceiling over the
portfolio beat the best fixed acquisition by +0.0018 (p = 0.90), so no
controller -- bandit, LLM or otherwise -- could demonstrate anything. That
measurement has two holes, and this script closes both on the dense blocks.

**Hole A: the portfolio is not five distinct actions.** Measured inside real
multi-task campaigns, against EI's own batch, UCB and PI agree at Spearman
0.97/0.996 and share 67-79% of the top-10 picks. `greedy` sits at 0.74-0.80
(28-38%) and `ts` at 0.39-0.45 (8-15%). Three of the five arms are one arm, so
a near-zero ceiling over them is a fact about the *action space*, not evidence
that adaptivity cannot pay.

**Hole B: section 1 was measured on a single-task ChEMBL library and never on a dense
block.** section 9 is the reason that gap is not free: a prior and an acquisition
function cannot be chosen separately, and a *perfect* prior scores -0.019...-0.144
under EI against +0.057...+0.113 under UCB. The coregionalised GP changes the
posterior sd structure, which is the only thing an acquisition reads.

Every cell of section 34 is EI, and in section 1's own artefact EI is never the best fixed
acquisition (PI wins at budget 50, UCB at 100), so this also asks what that
choice cost.

Three ceilings are reported, because they bound different choosers:

* **best fixed** -- the largest mean over the fixed arms. What you get by
  choosing one acquisition up front, correctly.
* **static oracle (split-half)** -- choose the best acquisition per cell on half
  the repeats, score it on the other half. This is what a *perfect static*
  chooser achieves, with the winner's curse removed; taking a plain max over
  arms would be biased upward by selecting on noise.
* **per-round oracle** -- `--acquisition oracle`, which reads the true values and
  switches every round. Loose in both directions: myopic, so it can land below
  the best fixed arm (section 1 at budget 100: -0.0158).

The headroom that matters for an LLM switcher is **per-round oracle minus best
fixed**. Anything less than that is unreachable by any controller.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

FIXED = ("ei", "ucb", "pi", "greedy", "ts")


def load(arm="mt"):
    """Per-cell, per-acquisition campaign traces, keyed (family, budget).

    EI comes from the section 34 artefacts rather than being re-run: same seeds, same
    settings, so it is already paired with everything here.
    """
    cells: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for f in glob.glob("results/acq_*.json"):
        m = re.match(r".*acq_(\w+?)_b(\d+)_(\w+)\.json$", f.replace("\\", "/"))
        if not m:
            continue
        fam, b, acq = m.group(1), int(m.group(2)), m.group(3)
        d = json.load(open(f))
        if arm not in d["arms"] or "auc" not in d["arms"][arm]:
            continue
        cells.setdefault((fam, b), {})[acq] = np.asarray(d["arms"][arm]["auc"])
    for f in glob.glob("results/win_*.json"):
        d = json.load(open(f))
        if d.get("repeats") != 8 or d.get("batch") != 10:
            continue
        key = (d["family"], d["budget"])
        if key in cells and arm in d["arms"] and "auc" in d["arms"][arm]:
            cells[key].setdefault("ei", np.asarray(d["arms"][arm]["auc"]))
    return cells


def split_half(by_acq, rng):
    """Unbiased estimate of a perfect *static* chooser.

    Choosing the best arm and reporting its own mean selects on noise and is
    biased upward, which matters here because the arms are close and the
    repeats are few. Picking on one half and scoring on the other removes it.
    Averaged over random splits so the answer does not depend on one.
    """
    acqs = [a for a in FIXED if a in by_acq]
    if len(acqs) < 2:
        return np.nan
    n = len(by_acq[acqs[0]])
    out = []
    for _ in range(200):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        pick = max(acqs, key=lambda k: by_acq[k][a].mean())
        out.append(by_acq[pick][b].mean())
    return float(np.mean(out))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="mt", choices=["mt", "stl"])
    p.add_argument("--out", default="results/acquisition_families.json")
    args = p.parse_args(argv)

    cells = load(args.arm)
    if not cells:
        raise SystemExit("no results/acq_*.json found; run "
                         "`run_window_experiments.py acquisition oracle "
                         "controls` first")
    rng = np.random.default_rng(0)

    order = {"jak": 0, "cyp": 1}
    hdr = "".join(f"{a:>9}" for a in FIXED)
    print(f"[1] fixed acquisitions, arm={args.arm}, AUC of the top-10 trace")
    print(f"    {'cell':<12}{hdr}{'best':>8}{'oracle':>9}{'head':>9}{'p':>8}")
    rows, heads = [], []
    for (fam, b) in sorted(cells, key=lambda k: (order.get(k[0], 9), k[0], k[1])):
        by = cells[(fam, b)]
        if not all(a in by for a in FIXED):
            missing = [a for a in FIXED if a not in by]
            print(f"    {fam + ' b' + str(b):<12}  incomplete, missing {missing}")
            continue
        means = {a: by[a].mean() for a in FIXED}
        best_a = max(means, key=means.get)
        line = "".join(f"{means[a]:>9.3f}" for a in FIXED)
        orc = by.get("oracle")
        if orc is None:
            print(f"    {fam + ' b' + str(b):<12}{line}{means[best_a]:>8.3f}"
                  f"{'-':>9}{'-':>9}{'-':>8}   best={best_a}")
            continue
        head = orc.mean() - means[best_a]
        pv = float(wilcoxon(orc, by[best_a]).pvalue)
        heads.append(head)
        rows.append({"family": fam, "budget": b, "arm": args.arm,
                     "means": {a: float(means[a]) for a in FIXED},
                     "best_fixed": best_a, "best_fixed_auc": float(means[best_a]),
                     "oracle_auc": float(orc.mean()),
                     "headroom": float(head), "p": pv,
                     "static_oracle_splithalf": split_half(by, rng),
                     "ei_gap": float(means[best_a] - means["ei"])})
        print(f"    {fam + ' b' + str(b):<12}{line}{means[best_a]:>8.3f}"
              f"{orc.mean():>9.3f}{head:>+9.4f}{pv:>8.3f}   best={best_a}")

    if not rows:
        print("\n    no cell has both the fixed sweep and the oracle yet")
        return 0

    print("\n[2] what a chooser could win")
    pos = sum(r["headroom"] > 0 for r in rows)
    sig = sum(r["headroom"] > 0 and r["p"] < 0.05 for r in rows)
    print(f"    per-round oracle beats best fixed in {pos}/{len(rows)} cells, "
          f"{sig} at p<0.05")
    print(f"    mean headroom {np.mean(heads):+.4f} AUC "
          f"(sd {np.std(heads, ddof=1):.4f}); section 1 single-task was +0.0018 (p=0.90)")
    sh = [r["static_oracle_splithalf"] - r["best_fixed_auc"] for r in rows]
    print(f"    a perfect STATIC chooser (split-half) gains {np.mean(sh):+.4f} "
          "over best fixed -- the part of the headroom that needs no switching")

    # The part that decides whether a controller is worth building. A ceiling a
    # free schedule already reaches is not headroom for anything.
    print("\n[2b] how much of it costs nothing (free switching controls)")
    taken = {}
    for ctl in ("switch_rr", "switch_random"):
        vals = []
        for r in rows:
            by = cells[(r["family"], r["budget"])]
            if ctl in by:
                vals.append(by[ctl].mean() - r["best_fixed_auc"])
        if not vals:
            print(f"    {ctl:<15} not run yet")
            continue
        # "share of the ceiling taken" is only meaningful when the control is
        # between the best fixed arm and the ceiling. A control that loses to
        # best fixed has taken none of it, and dividing a negative delta by a
        # positive headroom prints things like "-170% of the ceiling", which is
        # not a quantity -- it is a ratio of two numbers on opposite sides of
        # the reference. Report the share only where it is defined.
        share = (np.mean(vals) / np.mean(heads)
                 if np.mean(vals) > 0 and np.mean(heads) > 0 else None)
        taken[ctl] = {"delta_vs_best_fixed": float(np.mean(vals)),
                      "share_of_headroom": None if share is None else float(share),
                      "n_cells": len(vals),
                      "beats_best_fixed": int(sum(v > 0 for v in vals))}
        tail = (f"takes {share:.0%} of the ceiling" if share is not None
                else "below best fixed, so it takes none of the ceiling")
        print(f"    {ctl:<15} {np.mean(vals):+.4f} vs best fixed over "
              f"{len(vals)} cells, beats it in {sum(v > 0 for v in vals)}; "
              f"{tail}")
    if taken:
        best_ctl = max(taken.values(), key=lambda t: t["delta_vs_best_fixed"])
        if best_ctl["delta_vs_best_fixed"] >= np.mean(heads):
            print("    -> a free schedule already reaches the ceiling; there is "
                  "nothing for an LLM to earn here")
        else:
            print("    -> the ceiling is above both free controls: the gap "
                  "between them is what a controller would have to capture")

    print("\n[3] what section 34's EI-only choice cost")
    gaps = [r["ei_gap"] for r in rows]
    ei_best = sum(r["best_fixed"] == "ei" for r in rows)
    print(f"    EI is the best fixed acquisition in {ei_best}/{len(rows)} cells")
    print(f"    best fixed - EI: mean {np.mean(gaps):+.4f} AUC, "
          f"max {max(gaps):+.4f}")
    from collections import Counter
    print("    best fixed by cell: "
          + ", ".join(f"{k}x{v}" for k, v in
                      Counter(r["best_fixed"] for r in rows).most_common()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm": args.arm, "cells": rows,
                               "controls": taken}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
