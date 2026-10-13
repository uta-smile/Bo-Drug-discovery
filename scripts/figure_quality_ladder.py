"""The ladder figure: the seeding gain against the coupling a rule actually has.

    python scripts/figure_quality_ladder.py      # after seed_quality_ladder.py

One line per block, walking down `--aux-quality`. This is the figure that can
carry a threshold, and `fig_seed_rho.pdf` is not: there the x axis is a property
of the panel and every block sits above 0.28, so the zero crossing is outside
the data. Here the coupling is the thing being varied, the block is held fixed,
and the crossing is interpolated.

Design notes:

* One series per **block**, not per rule, because the block is what is held
  fixed within a line -- connecting the rungs is only meaningful inside one.
  Three blocks, so slots 1-3, the three that clear the all-pairs floors.
* The null control (requested quality 0) is drawn as an open marker rather than
  a fourth colour. It is the same measurement at a different setting, not
  another series, and status colours are reserved.
* `aux_top` only. `aux_scaffold_nn` tells the same story one panel over; putting
  both on one plot would need six lines and the series cap is three.
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


def panel(ax, data, arm, rule, *, show_ylabel):
    rows = [r for r in data["rows"]
            if r["arm"] == arm and r["rho_eff"] is not None and rule in r]
    for block, colour in BLOCK_COLOUR.items():
        sel = sorted((r for r in rows if r["block"] == block),
                     key=lambda r: r["rho_eff"])
        if not sel:
            continue
        x = np.array([r["rho_eff"] for r in sel])
        y = np.array([r[rule] for r in sel])
        e = np.array([r.get(f"{rule}_sem", 0.0) for r in sel])
        null = np.array([r["requested"] == 0.0 for r in sel])
        ax.plot(x, y, color=colour, linewidth=1.6, alpha=0.9, zorder=2)
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=colour, elinewidth=1.0,
                    capsize=0, alpha=0.5, zorder=2)
        m = BLOCK_MARKER[block]
        ax.scatter(x[~null], y[~null], s=42, marker=m, facecolor=colour,
                   edgecolor="white", linewidth=0.9, zorder=3)
        # The null rung: same series, hollow, so it reads as a control rather
        # than a fourth entity.
        ax.scatter(x[null], y[null], s=52, marker=m, facecolor="white",
                   edgecolor=colour, linewidth=1.6, zorder=4)
        # Nudge vertically as well: a series that ends flat puts its label
        # straight on the zero rule, where it is unreadable.
        dy = 6.0 if abs(y[-1]) < 0.05 else 0.0
        ax.annotate(block.replace("kiba_", ""), (x[-1], y[-1]),
                    xytext=(5, dy), textcoords="offset points", fontsize=7.5,
                    color=INK2, va="center")

    ax.axhline(0.0, color=MUTED, linewidth=1.0, zorder=1)
    ax.axvline(PRIOR_BAR, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=1)
    f = (data.get("fits") or {}).get(f"{arm}/{rule}")
    if f and f.get("ci_lo") is not None and f.get("interpolated"):
        ax.axvspan(f["ci_lo"], f["ci_hi"], color=MUTED, alpha=0.16, zorder=0)
        ax.annotate(f"crossing {f['crossing']:+.2f}\n95% CI "
                    f"[{f['ci_lo']:.2f}, {f['ci_hi']:.2f}]",
                    (f["crossing"], ax.get_ylim()[1]), xytext=(6, -6),
                    textcoords="offset points", fontsize=7, color=INK2,
                    va="top")
    ax.annotate(f"§8 prior bar  ρ = {PRIOR_BAR}", (PRIOR_BAR, ax.get_ylim()[0]),
                xytext=(4, 5), textcoords="offset points", fontsize=7,
                color=INK2, va="bottom")
    ax.set_title(f"{arm} surrogate", fontsize=9.5, color=INK, loc="left", pad=8)
    # No per-axis x label: both panels measure the same quantity, and two
    # copies of a label this long run into each other at the panel seam. One
    # figure-level label below, set in `main`.
    if show_ylabel:
        ax.set_ylabel(f"ΔAUC of {rule} vs a random initial design",
                      fontsize=8, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.16)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ladder", default="results/seed_quality_ladder.json")
    p.add_argument("--rule", default="aux_top")
    p.add_argument("--out", default="report/fig_quality_ladder")
    args = p.parse_args(argv)

    data = json.loads(Path(args.ladder).read_text())
    arms = data["arms"]
    fig, axes = plt.subplots(1, len(arms), figsize=(4.3 * len(arms), 3.4),
                             sharey=len(arms) > 1)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        panel(ax, data, arm, args.rule, show_ylabel=(ax is axes[0]))

    handles = [plt.Line2D([], [], color=c, marker=BLOCK_MARKER[b],
                          markeredgecolor="white", linewidth=1.6,
                          label=b.replace("kiba_", ""))
               for b, c in BLOCK_COLOUR.items()]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", linestyle="none",
                              markerfacecolor="white",
                              label="null control (q = 0)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Degrading a relative's data walks the gain to zero — and the "
                 "crossing is inside the data",
                 fontsize=10.5, color=INK, x=0.012, ha="left", y=1.04)
    fig.text(0.012, 0.975, "A pure-noise auxiliary (hollow marker) buys "
             "nothing, which is what makes the rest of the curve readable.",
             fontsize=8, color=INK2, ha="left")
    fig.supxlabel("effective coupling ρ the rule sees  (Spearman, degraded "
                  "auxiliary vs primary)", fontsize=8, color=INK2, y=0.05)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight",
                    facecolor="#ffffff")
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")

    print(f"\n    {'block':<11s}{'requested':>11s}{'rho_eff':>10s}"
          + "".join(f"{a + '/' + args.rule:>22s}" for a in arms))
    for block in BLOCK_COLOUR:
        for r in sorted((r for r in data["rows"]
                         if r["block"] == block and r["arm"] == arms[0]),
                        key=lambda r: -(r["rho_eff"] or 0)):
            cells = ""
            for a in arms:
                match = [q for q in data["rows"] if q["block"] == block
                         and q["arm"] == a and q["requested"] == r["requested"]]
                cells += (f"{match[0].get(args.rule, float('nan')):>+22.3f}"
                          if match else f"{'-':>22s}")
            req = "raw" if r["requested"] is None else f"{r['requested']:g}"
            print(f"    {block:<11s}{req:>11s}{r['rho_eff']:>+10.3f}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
