"""Re-state a sweep's result at the aggregation level its claim is about.

    python scripts/block_level.py "results/win32_*.json" --arm mt
    python scripts/block_level.py "results/win32_*.json" --arm mt --baseline pooled

Every campaign artefact in this package reports a paired Wilcoxon over
campaigns. Within one block that is correct: 32 repeats x 4 held-out targets are
32 x 4 paired draws of the same experiment. For a claim about *transfer in
general* it is not, because those 128 campaigns share a block, a library, an
assay history and a censoring filter. Reporting p < 1e-4 from 1,536 campaigns
when the claim ranges over six families overstates what was measured by roughly
the ratio of the two sample sizes.

This script reads stored artefacts only -- nothing is re-run -- and prints the
same contrast at three levels:

* **campaign** -- what the artefacts report. Correct within a block.
* **cell** -- one number per (block, budget). Removes the within-block pairing
  but still double-counts a block that was run at two budgets.
* **block** -- one number per block, averaged over its budgets. This is the unit
  a cross-block claim is about.

It also prints the **floor**: the smallest p a signed-rank test can return at
that sample size. With six blocks that floor is 0.031, so a six-block sweep
cannot report anything stronger however large the effect, and a p near the floor
is a statement about the number of blocks rather than about the effect.
"""

from __future__ import annotations

import argparse
import collections
import glob as globmod
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def signed_rank_floor(n: int) -> float:
    """Smallest two-sided p a signed-rank test can return with ``n`` pairs.

    Reached when every difference has the same sign, which is `2 / 2**n`.
    Returns 1.0 below n = 6, where no configuration clears 0.05.
    """
    return min(1.0, 2.0 ** (1 - n))


def read(path: Path, arm: str, baseline: str):
    d = json.loads(path.read_text(encoding="utf-8"))
    arms = d.get("arms") or {}
    if arm not in arms or baseline not in arms:
        return None
    a, b = arms[arm], arms[baseline]
    if "recall" not in a or "recall" not in b:
        return None
    per = (np.asarray(a["recall"], float) - np.asarray(b["recall"], float)) * 100
    return {
        "block": d.get("family"),
        "budget": d.get("budget"),
        "per_campaign": per,
        "cell_mean": float(per.mean()),
    }


def report(label: str, x: np.ndarray) -> dict:
    """The contrast at one level, with the facts that decide how to read `p`.

    `scipy.stats.wilcoxon` uses its exact null only when no pair is zero and no
    two absolute differences tie; otherwise `method="auto"` silently falls back
    to the normal approximation. Sample size alone does not establish that a
    probability was computed exactly, so we record the two conditions and the
    method, rather than inferring exactness from `n`.
    """
    x = np.asarray(x, float)
    zeros = int((x == 0.0).sum())
    absx = np.abs(x[x != 0.0])
    ties = int(len(absx) - len(np.unique(absx)))
    exact = bool(zeros == 0 and ties == 0 and 0 < len(x) <= 50)
    p = float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    floor = signed_rank_floor(len(x))
    # A floor only means anything when the test is exact and sees every pair.
    at_floor = bool(exact and p <= floor * 1.0001)
    return {"level": label, "n": int(len(x)), "mean": float(x.mean()),
            "sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
            "positive": int((x > 0).sum()), "p": p, "p_floor": floor,
            "zeros": zeros, "ties": ties, "exact": exact,
            "method": "exact" if exact else "normal approximation",
            "at_floor": at_floor}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pattern", help="glob over results/*.json, quoted")
    p.add_argument("--arm", default="mt")
    p.add_argument("--baseline", default="stl")
    p.add_argument("--against", default=None,
                   help="a second sweep's glob. Reports the *difference* "
                        "between the two sweeps' contrasts, matched cell by "
                        "cell on (block, budget), at all three levels. Use it "
                        "for a claim about what changing the protocol did.")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    files = sorted(Path(f) for f in globmod.glob(args.pattern))
    rows = [r for r in (read(f, args.arm, args.baseline) for f in files) if r]
    if not rows:
        raise SystemExit(
            f"no artefact matching {args.pattern!r} carries both "
            f"{args.arm!r} and {args.baseline!r} with per-campaign recall.")

    if args.against:
        # Two sweeps that differ in one protocol setting. The campaigns are
        # paired across them -- same block, same budget, same seed, same
        # history draw -- so the subtraction is valid at every level, and the
        # aggregation below then applies unchanged.
        rhs = {(o["block"], o["budget"]): o
               for o in (read(f, args.arm, args.baseline)
                         for f in sorted(Path(f)
                                         for f in globmod.glob(args.against)))
               if o}
        paired = []
        for r in rows:
            o = rhs.get((r["block"], r["budget"]))
            if o is None or len(o["per_campaign"]) != len(r["per_campaign"]):
                raise SystemExit(
                    f"{r['block']} at budget {r['budget']} has no matching "
                    f"cell in {args.against!r}. The two sweeps must cover the "
                    f"same grid with the same number of campaigns, or the "
                    f"difference is not paired.")
            d = r["per_campaign"] - o["per_campaign"]
            paired.append({**r, "per_campaign": d, "cell_mean": float(d.mean())})
        rows = paired

    by_block = collections.defaultdict(list)
    for r in rows:
        by_block[r["block"]].append(r["cell_mean"])

    campaign = np.concatenate([r["per_campaign"] for r in rows])
    cell = np.array([r["cell_mean"] for r in rows])
    block = np.array([np.mean(v) for v in by_block.values()])

    levels = [report("campaign", campaign), report("cell", cell),
              report("block", block)]

    print(f"{args.arm} vs {args.baseline}"
          + (f", minus the same contrast in {args.against}" if args.against
             else "")
          + f", {len(rows)} cells over "
          f"{len(by_block)} blocks, delta top-k recall in points\n")
    print(f"  {'level':<12}{'n':>6}{'mean':>9}{'sem':>7}{'positive':>11}"
          f"{'p':>10}{'floor':>9}{'method':>22}")
    print("  " + "-" * 86)
    for L in levels:
        mark = "  <- at the floor" if L["at_floor"] and L["n"] < 12 else ""
        note = L["method"]
        if L["zeros"] or L["ties"]:
            note += f" ({L['zeros']} zero, {L['ties']} tied)"
        print(f"  {L['level']:<12}{L['n']:>6}{L['mean']:>+9.2f}{L['sem']:>7.2f}"
              f"{str(L['positive']) + '/' + str(L['n']):>11}{L['p']:>10.4f}"
              f"{L['p_floor']:>9.4f}  {note:<40}{mark}")

    print(f"\n  per block (mean over its budgets):")
    for b, v in sorted(by_block.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"    {str(b):<10}{np.mean(v):>+7.2f}   "
              + ", ".join(f"{x:+.1f}" for x in v))

    blk = levels[-1]
    print()
    if blk["at_floor"]:
        print(f"  The block-level p of {blk['p']:.4f} IS the floor for "
              f"{blk['n']} blocks: every block\n  agrees in sign, and no effect "
              f"size could push it lower. Quote it as\n  "
              f"\"{blk['positive']}/{blk['n']} blocks, p = {blk['p']:.3f} "
              f"(the minimum at n = {blk['n']})\".")
    else:
        print(f"  Block-level: {blk['positive']}/{blk['n']} positive, "
              f"p = {blk['p']:.4f}, floor {blk['p_floor']:.4f}.")
    print(f"  The campaign-level p of {levels[0]['p']:.2g} is over "
          f"{levels[0]['n']} non-independent units\n  and should not be quoted "
          f"for a claim that ranges over blocks.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"arm": args.arm, "baseline": args.baseline,
             "pattern": args.pattern, "against": args.against, "levels": levels,
             "per_block": {str(k): v for k, v in by_block.items()}},
            indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
