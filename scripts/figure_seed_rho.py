"""The paper's figure: what a relative's data is worth, against how related it is.

    python scripts/figure_seed_rho.py          # after seed_rho_curve.py

One point per dense block: its task correlation on the x axis, and what seeding
the initial design from the free history is worth on the y. Two panels, one per
surrogate, because the answer differs between them and that difference is a
finding -- the coregionalised GP already spends the same history, so the room
left for seeding is smaller under `mt` than under `stl`.

Design notes, so a later edit does not undo them:

* Three series, drawn in categorical slots 1-3, which are the three that clear
  the all-pairs colour-distance floors a scatter needs. A fourth rule would have
  to fold into "other" or get its own panel rather than take slot 4.
* Marker *shape* carries identity as well as hue, so the figure survives being
  printed in grayscale -- which is the realistic failure mode for a paper, more
  than colour-vision deficiency is.
* Two panels share one y scale and each has its own x. Never two y scales.
* `maxmin` is on the plot although it is negative everywhere. It is the control
  that beat the LLM warm start on a single congeneric library, and showing it
  underwater on every co-measured block is the point, not an embarrassment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1-3 of the reference palette; validated all-pairs.
SERIES = [
    ("aux_top", "#2a78d6", "o", "aux_top"),
    ("aux_scaffold_nn", "#eb6834", "s", "aux_scaffold_nn"),
    ("maxmin", "#1baf7a", "^", "maxmin (no history)"),
]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a84"
GRID = "#e4e4df"
PRIOR_BAR = 0.37  # FINDINGS section 8's prior-mean threshold, derived elsewhere


def panel(ax, rows, arm, *, show_ylabel):
    x_all = [r["rho_raw"] for r in rows]
    for key, colour, marker, label in SERIES:
        x = np.array([r["rho_raw"] for r in rows if key in r["arms"][arm]])
        y = np.array([r["arms"][arm][key] for r in rows if key in r["arms"][arm]])
        e = np.array([r["arms"][arm].get(f"{key}_sem", 0.0)
                      for r in rows if key in r["arms"][arm]])
        if not len(x):
            continue
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=colour, elinewidth=1.0,
                    capsize=0, alpha=0.55, zorder=2)
        ax.scatter(x, y, s=46, marker=marker, facecolor=colour,
                   edgecolor="white", linewidth=0.9, zorder=3, label=label)
        if len(x) >= 3:
            s, b = np.polyfit(x, y, 1)
            xs = np.linspace(min(x_all), max(x_all), 50)
            ax.plot(xs, s * xs + b, color=colour, linewidth=1.6, alpha=0.85,
                    zorder=2)
            # Direct label at the line's right end: identity is never colour
            # alone, and with three series a legend plus labels is the rule.
            ax.annotate(label, (xs[-1], s * xs[-1] + b),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7.5, color=INK2, va="center")

    ax.axhline(0.0, color=MUTED, linewidth=1.0, zorder=1)
    ax.axvline(PRIOR_BAR, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=1)
    # Bottom of the frame, not the top: the top is where the strongest points
    # sit and the label landed on them.
    ax.annotate(f"§8 prior bar  ρ = {PRIOR_BAR}", (PRIOR_BAR, ax.get_ylim()[0]),
                xytext=(4, 5), textcoords="offset points", fontsize=7,
                color=INK2, va="bottom")
    ax.set_title(f"{arm} surrogate", fontsize=9.5, color=INK, loc="left",
                 pad=8)
    ax.set_xlabel("median task correlation ρ  (Spearman, block the campaign saw)",
                  fontsize=8, color=INK2)
    if show_ylabel:
        ax.set_ylabel("ΔAUC vs a random initial design", fontsize=8, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--curve", default="results/seed_rho_curve.json")
    p.add_argument("--out", default="report/fig_seed_rho")
    args = p.parse_args(argv)

    data = json.loads(Path(args.curve).read_text())
    rows, arms = data["rows"], data["arms"]

    fig, axes = plt.subplots(1, len(arms), figsize=(4.0 * len(arms), 3.3),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        panel(ax, rows, arm, show_ylabel=(ax is axes[0]))
        # Right margin on *every* panel: the direct labels hang off the last
        # fitted point, and without this they overrun the left panel's frame
        # and sit on top of the next one.
        ax.margins(x=0.30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(SERIES),
               frameon=False, fontsize=8, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.02))
    # The title states the claim that survives the robustness checks -- the
    # *level*. An earlier version read "pays in proportion to how related it
    # is", which the slope cannot carry: drop the single high-rho block and the
    # trend stops being significant for aux_top. Section 34 is the standing
    # reminder of what that kind of title costs.
    n_pos = sum(r["arms"][arms[0]]["aux_top"] > 0 for r in rows)
    fig.suptitle("Seeding from a relative's measured data beats a random "
                 f"start on {n_pos} of {len(rows)} panels",
                 fontsize=11, color=INK, x=0.012, ha="left", y=1.04)
    fig.text(0.012, 0.975, "How much it is worth is only weakly predicted by "
             "task correlation — the trend is carried by one high-ρ "
             "block.", fontsize=8, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="#ffffff")
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")

    # The table view the relief rule asks for, and what a reader checks numbers
    # against. Aqua sits under 3:1 on a light surface, so this is not optional.
    print(f"\n    {'block':<7s}{'rho':>8s}" +
          "".join(f"{a + '/' + k:>22s}" for a in arms for k, *_ in SERIES))
    for r in sorted(rows, key=lambda r: -r["rho_raw"]):
        cells = "".join(
            f"{r['arms'][a].get(k, float('nan')):>+22.3f}"
            for a in arms for k, *_ in SERIES)
        print(f"    {r['block']:<7s}{r['rho_raw']:>+8.3f}{cells}")
    for key, f in (data.get("fits") or {}).items():
        if f and f.get("crossing") is not None:
            ci = (f"[{f['ci_lo']:.2f}, {f['ci_hi']:.2f}]"
                  if f.get("ci_lo") is not None else "n/a")
            print(f"    fit {key:<22s} crossing {f['crossing']:+.3f}  95% CI {ci}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
