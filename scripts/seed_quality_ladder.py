"""The coupling below which a relative's data stops being worth seeding from.

    python scripts/seed_quality_ladder.py

`seed_rho_curve.py` ran the gate on all 15 KIBA blocks and could not answer this.
Every block in the panel sits at task correlation >= 0.277, so the fitted zero
crossing of the seeding gain (0.153, CI [0.09, 0.32]) lies *below every point
that produced it* -- an extrapolation, not a measurement. The honest statement
after that sweep was "we cannot locate the threshold", and this script is the
way to stop that being the final word.

## The instrument

`--aux-quality q` degrades every auxiliary column to Spearman `q` with its own
true values before anything reads it, leaving the primary's column and the
scoring untouched. Walking `q` down walks the coupling a seeding rule has to the
target through the region the real panels do not cover: on `kiba_cdk5`
(block rho 0.505) `q = 0.3` puts the rule at an effective rho of 0.175.

**The x axis is measured, not requested.** For every (target, repeat, auxiliary)
the gate records Spearman between the values the rule actually sees and the
primary's true values on those same rows, and the run reports the median. A
requested `q` is an input; the effective rho is what the rule had. They differ,
and the second is the one a threshold can be stated in.

`q = 0` is §32's null-information control, and it is the anchor that makes the
rest credible: a pure-noise auxiliary must buy nothing. If the ladder shows a
gain there, the gain is not transfer and no crossing read off it means anything.

## What it does not settle

A synthetic ladder degrades an auxiliary *uniformly at random*. Real weak
coupling is structured -- a relative that ranks one chemotype well and another
not at all. So this brackets the crossing under the most favourable kind of
weakness, and a cross-family scan is still owed. Run both and check they agree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import init_headroom  # noqa: E402
from dense_block_campaign import PANELS  # noqa: E402
from seed_rho_curve import crossing  # noqa: E402

RULES = ("aux_top", "aux_scaffold_nn", "maxmin")
BLOCKS = ("kiba_prkc", "kiba_cdk5", "kiba_lck")
QUALITY = (None, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.0)
"""``None`` is the undegraded block -- the ladder's top rung, and the point that
ties it to `seed_rho_curve.py`'s measurements."""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocks", nargs="+", default=list(BLOCKS))
    p.add_argument("--quality", nargs="+", type=float, default=None,
                   help="requested aux-quality rungs; the undegraded block is "
                        "always run as well")
    p.add_argument("--arms", nargs="+", default=["stl"])
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--history", type=int, default=50)
    p.add_argument("--out-dir", default="results/seed_quality")
    p.add_argument("--out", default="results/seed_quality_ladder.json")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    rungs = [None] + list(args.quality) if args.quality else list(QUALITY)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = []
    for block in args.blocks:
        for q in rungs:
            tag = "raw" if q is None else f"q{q:g}".replace(".", "")
            for arm in args.arms:
                dest = out_dir / f"{block}_{tag}_{arm}.json"
                if args.force or not dest.exists():
                    argv_ = ["--family", block, "--keep-censored",
                             "--history-excludes-top", "--arm", arm,
                             "--candidates", str(args.candidates),
                             "--repeats", str(args.repeats),
                             "--budget", str(args.budget),
                             "--history", str(args.history),
                             "--out", str(dest)]
                    if q is not None:
                        argv_ += ["--aux-quality", str(q)]
                    if init_headroom.main(argv_) != 0:
                        raise SystemExit(f"{block}/{tag}/{arm} failed")
                rep = json.loads(dest.read_text())
                by = {r["label"]: r for r in rep["rows"]}
                rows.append({
                    "block": block, "arm": arm, "requested": q,
                    "rho_eff": rep["rho_eff"],
                    "ceiling": by["hindsight ceiling"]["delta_mean"],
                    **{r: by[r]["delta_mean"] for r in RULES if r in by},
                    **{f"{r}_sem": by[r]["delta_sem"] for r in RULES if r in by},
                    "top_k_seeded": rep.get("top_k_seeded", {}),
                    "audit_clean": rep["audit"]["clean"],
                    "artefact": dest.as_posix()})
                print(f"  {block:<11s} {tag:<5s} {arm:<4s} "
                      f"rho_eff={rep['rho_eff']:+.3f} "
                      + "  ".join(f"{r}={rows[-1].get(r, float('nan')):+.3f}"
                                  for r in ("aux_top", "aux_scaffold_nn"))
                      + f"   ({time.time() - t0:.0f}s)", flush=True)

    fits, null, within = {}, {}, {}
    for arm in args.arms:
        sel = [r for r in rows if r["arm"] == arm and r["rho_eff"] is not None]
        for rule in RULES:
            x = [r["rho_eff"] for r in sel if rule in r]
            y = [r[rule] for r in sel if rule in r]
            f = crossing(x, y)
            if f:
                f["x_min"] = float(min(x))
                f["x_max"] = float(max(x))
                # The whole point of the ladder: is the crossing inside the
                # measured range this time, or still an extrapolation?
                f["interpolated"] = bool(f["crossing"] is not None
                                         and f["x_min"] <= f["crossing"]
                                         <= f["x_max"])
                f["spearman_p"] = float(spearmanr(x, y).pvalue)
            fits[f"{arm}/{rule}"] = f
            # The pooled fit mixes two different things: the ladder's *within*
            # block intervention, which is causal because the block is held
            # fixed, and the differences *between* blocks, which are not. A
            # pooled slope can be carried entirely by the second -- `kiba_prkc`
            # is high at every rung and `kiba_lck` is flat at zero, and a line
            # through both would show a trend even if neither block moved. So
            # each block gets its own fit, and the pooled one is read only
            # beside them.
            for b in args.blocks:
                bx = [r["rho_eff"] for r in sel if r["block"] == b and rule in r]
                by_ = [r[rule] for r in sel if r["block"] == b and rule in r]
                g = crossing(bx, by_, n_boot=800)
                if g:
                    g["x_min"], g["x_max"] = float(min(bx)), float(max(bx))
                    g["interpolated"] = bool(
                        g["crossing"] is not None
                        and g["x_min"] <= g["crossing"] <= g["x_max"])
                    g["spearman_p"] = float(spearmanr(bx, by_).pvalue)
                within[f"{arm}/{rule}/{b}"] = g
        # Section 32's control, read straight off the bottom rung.
        zero = [r for r in sel if r["requested"] == 0.0]
        if zero:
            null[arm] = {
                "rho_eff": float(np.median([r["rho_eff"] for r in zero])),
                **{rule: float(np.mean([r[rule] for r in zero if rule in r]))
                   for rule in RULES},
                "n": len(zero)}

    print(f"\n    {'arm/rule':<22s}{'rho(x,gain)':>12s}{'p':>8s}"
          f"{'crossing':>11s}{'95% CI':>18s}{'in range':>10s}")
    print("    " + "-" * 81)
    for key, f in fits.items():
        if not f:
            continue
        ci = (f"[{f['ci_lo']:.2f}, {f['ci_hi']:.2f}]"
              if f.get("ci_lo") is not None else "n/a")
        cr = f"{f['crossing']:+.3f}" if f["crossing"] is not None else "n/a"
        print(f"    {key:<22s}{f['spearman_rho_vs_gain']:>+12.3f}"
              f"{f['spearman_p']:>8.4f}{cr:>11s}{ci:>18s}"
              f"{('YES' if f['interpolated'] else 'no'):>10s}")
    print("\n    within one block, where the ladder is the only thing "
          "that moved:")
    print(f"    {'arm/rule/block':<34s}{'rho(x,gain)':>12s}{'p':>8s}"
          f"{'crossing':>11s}{'in range':>10s}")
    print("    " + "-" * 75)
    for key, g in within.items():
        if not g:
            continue
        cr = f"{g['crossing']:+.3f}" if g["crossing"] is not None else "n/a"
        print(f"    {key:<34s}{g['spearman_rho_vs_gain']:>+12.3f}"
              f"{g['spearman_p']:>8.4f}{cr:>11s}"
              f"{('YES' if g['interpolated'] else 'no'):>10s}")

    for arm, n in null.items():
        print(f"\n    null control ({arm}, requested q=0, effective rho "
              f"{n['rho_eff']:+.3f}): "
              + ", ".join(f"{r} {n[r]:+.3f}" for r in RULES if r in n))

    clean = all(r["audit_clean"] for r in rows)
    print(f"\n    {len(rows)} runs, audit clean on every one: {clean}")
    report = {"blocks": args.blocks, "rungs": rungs, "arms": args.arms,
              "candidates": args.candidates, "repeats": args.repeats,
              "budget": args.budget, "history": args.history,
              "rows": rows, "fits": fits, "within_block": within,
              "null_control": null,
              "audit_clean": bool(clean)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
