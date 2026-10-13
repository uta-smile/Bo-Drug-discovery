"""The money figure: two ways of weakening a relative, one threshold.

    python scripts/figure_threshold.py   # after seed_quality_ladder + cross_family_scan

Left: the synthetic ladder. One block, degraded step by step, three times over.
Right: 18 blocks chosen for being genuinely weakly coupled, nothing degraded.
The two were produced by different mechanisms and the question the figure exists
to answer is whether they put the zero crossing in the same place. Each panel
carries its own bootstrap CI as a band and the *other* panel's crossing as a
dashed line, so the agreement is read directly rather than from two tables.

Design notes:

* Left panel: three blocks, three series, categorical slots 1-3 with distinct
  marker shapes so the panel survives grayscale print. Three is the cap for a
  scatter and we are at it.
* Right panel is **one** series. A single series takes no categorical hue --
  there is nothing to tell it apart from -- so it is drawn in ink. Reusing slot
  1 there would make blue mean `kiba_prkc` on the left and "any block" on the
  right, which is the one thing colour is not allowed to do.
* No dual axes; the panels share the y scale, and both x axes are the same
  measured quantity so they share limits too. That shared x is the whole point:
  the crossings are only comparable if the axes are.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLOCK_COLOUR = {"kiba_prkc": "#2a78d6", "kiba_cdk5": "#eb6834",
                "kiba_lck": "#1baf7a"}
BLOCK_MARKER = {"kiba_prkc": "o", "kiba_cdk5": "s", "kiba_lck": "^"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a84", "#e4e4df"
PRIOR_BAR = 0.37


def frame(ax, *, title, xlabel, ylabel=None):
    ax.axhline(0.0, color=MUTED, linewidth=1.0, zorder=1)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=7.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def band(ax, fit, *, label):
    """The bootstrap CI on this panel's own crossing."""
    if not fit or fit.get("ci_lo") is None:
        return
    ax.axvspan(fit["ci_lo"], fit["ci_hi"], color=MUTED, alpha=0.16, zorder=0)
    ax.annotate(f"{label} {fit['crossing']:+.3f}\n95% CI "
                f"[{fit['ci_lo']:.2f}, {fit['ci_hi']:.2f}]",
                (fit["crossing"], ax.get_ylim()[1]), xytext=(6, -6),
                textcoords="offset points", fontsize=7, color=INK2, va="top")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ladder", default="results/seed_quality_ladder.json")
    p.add_argument("--cross", default="results/cross_family_scan.json")
    p.add_argument("--arm", default="stl")
    p.add_argument("--rule", default="aux_top")
    p.add_argument("--out", default="report/fig_threshold")
    args = p.parse_args(argv)

    lad = json.loads(Path(args.ladder).read_text(encoding="utf-8"))
    xf = json.loads(Path(args.cross).read_text(encoding="utf-8"))
    f_lad = (lad.get("fits") or {}).get(f"{args.arm}/{args.rule}")
    f_xf = (xf.get("fits") or {}).get(f"{args.arm}/{args.rule}/rho_eff")
    if f_xf is None:
        raise SystemExit(f"no cross-family fit for arm {args.arm!r}; "
                         f"available: {sorted(xf.get('fits') or {})}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)

    # ---- left: the synthetic ladder ---------------------------------------
    rows = [r for r in lad["rows"]
            if r["arm"] == args.arm and r["rho_eff"] is not None
            and args.rule in r]
    for block, colour in BLOCK_COLOUR.items():
        sel = sorted((r for r in rows if r["block"] == block),
                     key=lambda r: r["rho_eff"])
        if not sel:
            continue
        x = np.array([r["rho_eff"] for r in sel])
        y = np.array([r[args.rule] for r in sel])
        null = np.array([r["requested"] == 0.0 for r in sel])
        m = BLOCK_MARKER[block]
        axL.plot(x, y, color=colour, linewidth=1.6, alpha=0.9, zorder=2)
        axL.scatter(x[~null], y[~null], s=38, marker=m, facecolor=colour,
                    edgecolor="white", linewidth=0.9, zorder=3)
        axL.scatter(x[null], y[null], s=48, marker=m, facecolor="white",
                    edgecolor=colour, linewidth=1.6, zorder=4)
        axL.annotate(block.replace("kiba_", ""), (x[-1], y[-1]),
                     xytext=(5, 6 if abs(y[-1]) < 0.05 else 0),
                     textcoords="offset points", fontsize=7.5, color=INK2,
                     va="center")
    frame(axL, title="Synthetic: one block, degraded step by step",
          xlabel="effective coupling ρ the rule sees",
          ylabel=f"ΔAUC of {args.rule} vs a random initial design")

    # ---- right: the cross-family scan -------------------------------------
    sel = [r for r in xf["rows"]
           if r["arm"] == args.arm and r.get("rho_eff") is not None
           and args.rule in r]
    x = np.array([r["rho_eff"] for r in sel])
    y = np.array([r[args.rule] for r in sel])
    e = np.array([r.get(f"{args.rule}_sem", 0.0) for r in sel])
    axR.errorbar(x, y, yerr=e, fmt="none", ecolor=MUTED, elinewidth=1.0,
                 capsize=0, alpha=0.6, zorder=2)
    axR.scatter(x, y, s=38, marker="o", facecolor=INK2, edgecolor="white",
                linewidth=0.9, zorder=3)
    s, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    axR.plot(xs, s * xs + b, color=INK2, linewidth=1.6, alpha=0.85, zorder=2)
    frame(axR, title=f"Real: {len(sel)} blocks chosen for being weakly coupled",
          xlabel="effective coupling ρ the rule sees")

    for ax in (axL, axR):
        ax.set_xlim(-0.08, 1.0)
        ax.axvline(PRIOR_BAR, color=MUTED, linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(f"§8 prior bar  ρ = {PRIOR_BAR}",
                    (PRIOR_BAR, ax.get_ylim()[0]), xytext=(4, 5),
                    textcoords="offset points", fontsize=7, color=INK2,
                    va="bottom")
    band(axL, f_lad, label="crossing")
    band(axR, f_xf, label="crossing")
    # Each panel carries the other's crossing, which is what makes the
    # agreement a thing you see rather than a thing you are told.
    for ax, other, name in ((axL, f_xf, "cross-family"),
                            (axR, f_lad, "ladder")):
        if other and other.get("crossing") is not None:
            ax.axvline(other["crossing"], color=INK, linewidth=1.1,
                       linestyle=(0, (1, 2)), zorder=5)
            ax.annotate(f"{name}\n{other['crossing']:+.3f}",
                        (other["crossing"], ax.get_ylim()[0]), xytext=(4, 4),
                        textcoords="offset points", fontsize=6.5, color=INK,
                        va="bottom")

    handles = [plt.Line2D([], [], color=c, marker=BLOCK_MARKER[b],
                          markeredgecolor="white", linewidth=1.6,
                          label=b.replace("kiba_", ""))
               for b, c in BLOCK_COLOUR.items()]
    handles += [plt.Line2D([], [], color=MUTED, marker="o", linestyle="none",
                           markerfacecolor="white",
                           label="null control (q = 0)"),
                plt.Line2D([], [], color=INK, linestyle=(0, (1, 2)),
                           label="the other panel's crossing")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.5, -0.04))

    inside = (f_lad and f_xf and f_xf.get("ci_lo") is not None
              and f_xf["ci_lo"] <= f_lad["crossing"] <= f_xf["ci_hi"])
    fig.suptitle("Two different ways of weakening a relative put the threshold "
                 "in the same place" if inside else
                 "The two ways of weakening a relative disagree about the "
                 "threshold",
                 fontsize=10.5, color=INK, x=0.012, ha="left", y=1.05)
    fig.text(0.012, 0.975,
             f"Seeding stops paying near ρ ≈ 0.07 either way — far below the "
             f"ρ = {PRIOR_BAR} the same data needs to be worth a prior mean.",
             fontsize=8, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight",
                    facecolor="#ffffff")
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")
    print(f"\n    ladder crossing       {f_lad['crossing']:+.3f}")
    print(f"    cross-family crossing {f_xf['crossing']:+.3f}  "
          f"95% CI [{f_xf['ci_lo']:.3f}, {f_xf['ci_hi']:.3f}]")
    print(f"    ladder inside the cross-family CI: {bool(inside)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
