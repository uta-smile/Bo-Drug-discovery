"""Where does history-aware seeding stop paying? The curve, and the crossing.

    python scripts/seed_rho_curve.py --blocks results/dense_blocks_kiba.json

`init_headroom.py` answers the question for one block. Four blocks bracketed the
crossing between a task correlation that pays and one that does not, which is a
bracket and not a number. This runs the gate over **every** block in a scan and
fits the crossing, because that threshold -- *below this correlation, do not
spend effort on your relatives' data* -- is the one quantity a practitioner can
act on before committing an assay.

## The axis, and why the stored one is the wrong one

`find_dense_blocks --labelled` reports each block's task correlation on the
**censored-filtered** block. On the KIBA panel that filter leaves 25-99 of ~900
compounds, so the stored `median_rho` is a correlation over a few dozen rows --
and the campaigns run on the raw block, with `--keep-censored`. The two are not
the same number:

    kiba_prkc   raw 0.866   stored (n=263) 0.991
    kiba_cdk5   raw 0.505   stored (n= 59) 0.781
    kiba_lck    raw 0.277   stored (n= 32) 0.482

The ordering survives but the values do not, so this script recomputes the
correlation **on the block the campaign actually sees** and carries the stored
one beside it. Plotting a gain measured on 900 compounds against a correlation
measured on 32 would put the crossing in the wrong place.

## What is fitted

Per block: median off-diagonal Spearman among the four targets (raw), against
the free rules' dAUC over a random initial design, from `init_headroom`. The
crossing is read off a least-squares line with a paired bootstrap over blocks,
and reported as an interval -- a threshold stated without one invites exactly
the over-reading that section 34 had to retract.

Every per-block artefact is kept under ``results/seed_rho/``. The aggregate is
a join over them, so re-running one block does not mean re-running the sweep.
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
from dense_block_campaign import (  # noqa: E402
    PANELS,
    build_block,
    drop_censored,
)
from find_dense_blocks import blocks_as_families  # noqa: E402

RULES = ("aux_top", "aux_scaffold_nn", "maxmin")


def block_rho(mapping, source, potency, pile_frac):
    """Task correlation on the raw block and on the filtered one.

    Spearman, matching `family_properties` -- the campaign metric is a top-k
    recall, so the ranking is what matters and the scale is not.
    """
    block, _ = build_block(mapping, source=source, potency=potency)
    tg = [t for t in mapping.values() if t in block.columns]
    out = {"targets": tg, "n_raw": len(block)}
    for label, df in (("raw", block),
                      ("clean", drop_censored(block, tg, pile_frac=pile_frac,
                                              verbose=False)[0])):
        v = df[tg].to_numpy(float)
        if len(v) < 5:
            out[f"rho_{label}"] = float("nan")
            out[f"n_{label}"] = len(v)
            continue
        R = np.atleast_2d(spearmanr(v).statistic)
        off = (R[np.triu_indices(len(tg), 1)] if R.shape[0] == len(tg)
               else np.array([np.nan]))
        out[f"rho_{label}"] = float(np.median(off))
        out[f"rho_{label}_min"] = float(np.min(off))
        out[f"rho_{label}_max"] = float(np.max(off))
        out[f"n_{label}"] = len(v)
    return out


def crossing(x, y, *, n_boot=2000, seed=0):
    """Where a least-squares line through (rho, dAUC) crosses zero.

    Bootstrapped over blocks, not over repeats: the question is whether the
    threshold would move if a different set of panels had been scanned, and
    resampling repeats would answer a narrower question with a tighter and
    misleading interval.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    r = float(spearmanr(x, y).statistic)
    g = np.random.default_rng(seed)
    roots = []
    for _ in range(n_boot):
        j = g.integers(0, len(x), size=len(x))
        if len(np.unique(x[j])) < 2:
            continue
        s, b = np.polyfit(x[j], y[j], 1)
        if abs(s) > 1e-9:
            roots.append(-b / s)
    roots = np.asarray(roots)
    inside = roots[(roots >= x.min() - 0.2) & (roots <= x.max() + 0.2)]
    return {"slope": float(slope), "intercept": float(intercept),
            "spearman_rho_vs_gain": r, "n_blocks": int(len(x)),
            "crossing": float(-intercept / slope) if abs(slope) > 1e-9 else None,
            "ci_lo": float(np.percentile(inside, 2.5)) if len(inside) else None,
            "ci_hi": float(np.percentile(inside, 97.5)) if len(inside) else None,
            "boot_in_range": int(len(inside)), "boot_total": int(len(roots))}


def robustness_checks(rows, arm, rule):
    """The two things that decide how much of the fit to believe.

    **Level vs slope.** A gain that is positive on nearly every block is a
    different claim from a gain that *grows with* task correlation, and the
    second is far easier to manufacture out of one extreme block. Both are
    reported: a signed-rank test against zero for the level, and a
    leave-the-leverage-point-out refit for the slope.

    **The leverage point.** On the KIBA panel 14 of 15 blocks sit in
    rho 0.28-0.51 and one sits at 0.87, so an ordinary least-squares slope is
    largely that one block's doing. Dropping the highest-rho block and refitting
    says whether the trend is a trend or a lever. Section 34 is the reason this
    is computed rather than eyeballed: an effect that looked unanimous over two
    hand-picked families was not, and the check that would have caught it was a
    refit on an unbiased draw.
    """
    from scipy.stats import wilcoxon

    y = np.array([r["arms"][arm][rule] for r in rows], float)
    x = np.array([r["rho_raw"] for r in rows], float)
    out = {"n_blocks": int(len(y)), "mean": float(y.mean()),
           "sem": float(y.std(ddof=1) / np.sqrt(len(y))),
           "median": float(np.median(y)),
           "n_positive": int((y > 0).sum()),
           "wilcoxon_p": float(wilcoxon(y).pvalue) if len(y) >= 6 else None}
    s = spearmanr(x, y)
    out["spearman"] = float(s.statistic)
    out["spearman_p"] = float(s.pvalue)
    hi = int(np.argmax(x))
    keep = np.ones(len(x), bool)
    keep[hi] = False
    if keep.sum() >= 4:
        s2 = spearmanr(x[keep], y[keep])
        out["leverage_block"] = rows[hi]["block"]
        out["leverage_rho"] = float(x[hi])
        out["spearman_drop_leverage"] = float(s2.statistic)
        out["spearman_p_drop_leverage"] = float(s2.pvalue)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocks", default="results/dense_blocks_kiba.json",
                   help="a find_dense_blocks scan artefact")
    p.add_argument("--panel", default="kiba", choices=sorted(PANELS))
    p.add_argument("--top", type=int, default=15,
                   help="how many of the scan's blocks to run")
    p.add_argument("--arms", nargs="+", default=["stl", "mt"])
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--history", type=int, default=50)
    p.add_argument("--pile-frac", type=float, default=0.02)
    p.add_argument("--out-dir", default="results/seed_rho")
    p.add_argument("--out", default="results/seed_rho_curve.json")
    p.add_argument("--force", action="store_true",
                   help="re-run blocks whose artefact already exists")
    args = p.parse_args(argv)

    t0 = time.time()
    panel = PANELS[args.panel]
    families = blocks_as_families(args.blocks, top=args.top)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fam, mapping in families.items():
        rho = block_rho(mapping, panel["source"], lambda y: y, args.pile_frac)
        row = {"block": fam, **rho, "arms": {}}
        for arm in args.arms:
            dest = out_dir / f"{args.panel}_{fam}_{arm}.json"
            if args.force or not dest.exists():
                rc = init_headroom.main([
                    "--family", fam, "--blocks-from", args.blocks,
                    "--source", str(panel["source"]),
                    "--potency", panel["potency"],
                    "--keep-censored", "--history-excludes-top",
                    "--arm", arm, "--candidates", str(args.candidates),
                    "--repeats", str(args.repeats), "--budget", str(args.budget),
                    "--history", str(args.history), "--out", str(dest)])
                if rc != 0:
                    raise SystemExit(f"{fam}/{arm} failed")
            rep = json.loads(dest.read_text())
            by_label = {r["label"]: r for r in rep["rows"]}
            row["arms"][arm] = {
                "ceiling": by_label["hindsight ceiling"]["delta_mean"],
                **{r: by_label[r]["delta_mean"] for r in RULES if r in by_label},
                **{f"{r}_sem": by_label[r]["delta_sem"]
                   for r in RULES if r in by_label},
                "gate_inert": rep["gate"]["inert"] if rep["gate"] else None,
                "audit_clean": rep["audit"]["clean"],
                "artefact": dest.as_posix()}
        rows.append(row)
        print(f"  {fam}: rho_raw={rho['rho_raw']:+.3f} "
              + "  ".join(f"{a} aux_top={row['arms'][a]['aux_top']:+.3f}"
                          for a in args.arms)
              + f"   ({time.time() - t0:.0f}s)", flush=True)

    fits, robustness = {}, {}
    for arm in args.arms:
        for rule in RULES:
            sel = [r for r in rows if rule in r["arms"][arm]]
            x = [r["rho_raw"] for r in sel]
            y = [r["arms"][arm][rule] for r in sel]
            fits[f"{arm}/{rule}"] = crossing(x, y)
            robustness[f"{arm}/{rule}"] = robustness_checks(sel, arm, rule)

    print(f"\n    {'arm/rule':<22s}{'rho(x,gain)':>13s}{'crossing':>12s}"
          f"{'95% CI':>20s}")
    print("    " + "-" * 67)
    for key, f in fits.items():
        if not f:
            continue
        ci = (f"[{f['ci_lo']:.2f}, {f['ci_hi']:.2f}]"
              if f["ci_lo"] is not None else "n/a")
        cross = f"{f['crossing']:+.3f}" if f["crossing"] is not None else "n/a"
        print(f"    {key:<22s}{f['spearman_rho_vs_gain']:>+13.3f}"
              f"{cross:>12s}{ci:>20s}")

    clean = all(r["arms"][a]["audit_clean"] for r in rows for a in args.arms)
    print(f"\n    {len(rows)} blocks x {len(args.arms)} arms, "
          f"audit clean on every cell: {clean}")

    for key, rb in robustness.items():
        if rb.get("wilcoxon_p") is None:
            continue
        print(f"    {key:<22s} level {rb['mean']:+.3f} +-{rb['sem']:.3f}, "
              f"{rb['n_positive']}/{rb['n_blocks']} positive, "
              f"signed-rank p={rb['wilcoxon_p']:.4f} | slope rho={rb['spearman']:+.3f} "
              f"(p={rb['spearman_p']:.3f}) -> drop {rb.get('leverage_block')}: "
              f"{rb.get('spearman_drop_leverage', float('nan')):+.3f} "
              f"(p={rb.get('spearman_p_drop_leverage', float('nan')):.3f})")

    report = {"blocks_scan": args.blocks, "panel": args.panel,
              "top": args.top, "arms": args.arms,
              "candidates": args.candidates, "repeats": args.repeats,
              "budget": args.budget, "history": args.history,
              "rows": rows, "fits": fits, "robustness": robustness,
              "audit_clean": bool(clean)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
