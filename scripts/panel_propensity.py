"""Where does a *real* assay panel sit on the history-design axis?

    python scripts/panel_propensity.py --out results/panel_propensity.json

``dense_block_campaign.py --history-prefer-aux ALPHA`` walks the historical
panel from "assembled without reference to activity" (``ALPHA=0``) towards
"assembled by people who expected these compounds to be interesting". The
campaign results say the pooled multitask GP degrades along that axis. They do
not say where a real panel falls on it, so the headline claim rests on a
direction rather than a measurement.

This script measures it. The axis is calibrated in two scale-free statistics,
each computed the same way on both sides:

**SMD** -- standardised mean difference in primary-target activity between the
compounds the panel covers and the compounds it does not.

**Top-k capture lift** -- how many of the primary's true top-k the panel holds,
divided by what a panel of the same size drawn without reference to activity
would hold. 1.0 means no preference.

*Real side.* For a family's four targets, take every compound BindingDB holds
an IC50 for against the primary. The "panel" is the subset also measured
against the family's other targets -- that subset is a real decision made by
real people, and the question is whether they picked actives. Each of the four
targets takes a turn as primary, matching the campaign's rotation.

*Simulated side.* The same statistics on the family's dense block, over a grid
of ``ALPHA``, using ``draw_history`` itself rather than a reimplementation, at
the history size the sweep used.

Inverting the simulated curve at the real value gives the equivalent ALPHA.

Two things this does NOT establish. The real population (everything BindingDB
measured against the target) and the simulated library (the censoring-filtered
dense block) are different sets of compounds, so the match is on the strength
of the selection, not a claim that the two libraries are interchangeable. And
co-assay is observational: a compound may be in the panel because it was
potent, or because both assays ran in one paper on one series. This measures
the association, not the mechanism.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dense_block_campaign import FAMILIES, draw_history  # noqa: E402
from dense_block_h1 import drop_censored  # noqa: E402

SOURCE = Path("data/cache/tdc/bindingdb_ic50.csv")
BLOCK_DIR = Path("data/bindingdb")


def to_pic50(nm) -> np.ndarray:
    """BindingDB reports IC50 in nM. pIC50 = -log10(M) = 9 - log10(nM)."""
    nm = np.asarray(nm, dtype=float)
    nm = np.clip(nm, 1e-4, 1e11)
    return 9.0 - np.log10(nm)


def load_pairs(accessions, source=SOURCE, chunksize=500_000):
    """(compound, target, pIC50) for the given accessions only.

    Chunked for the reason ``find_dense_blocks.incidence`` gives: the dump is
    839 MB and reading it whole costs several GB. Replicate rows from
    different papers are collapsed to a median, as ``build_prior_data`` does,
    so a compound reported twice does not count twice.
    """
    if not source.exists():
        raise SystemExit(f"{source} not found; it is the TDC BindingDB IC50 dump.")
    want = set(accessions)
    keep = []
    for chunk in pd.read_csv(source, usecols=["X1", "ID2", "Y"], chunksize=chunksize):
        chunk = chunk.dropna()
        chunk = chunk[chunk.ID2.isin(want) & (chunk.Y > 0) & (chunk.Y < 1e9)]
        if len(chunk):
            keep.append(chunk)
    if not keep:
        raise SystemExit("no rows matched the requested accessions")
    df = pd.concat(keep, ignore_index=True)
    df["pic50"] = to_pic50(df["Y"])
    return df.groupby(["ID2", "X1"], as_index=False).agg(pic50=("pic50", "median"))


def strip_pileups(values, pile_frac=0.02, mode="low"):
    """Mask of values that are NOT on a reporting ceiling.

    Censoring matters here beyond the usual reason: censored readouts pile at
    *low* activity, so if panel compounds are censored at a different rate
    than non-panel ones, the SMD picks up the censoring rather than the
    selection.

    ``dense_block_h1.drop_censored`` drops every value held by more than
    ``pile_frac`` of a column. That rule is calibrated for a ~1,500-compound
    dense block. These populations run to 6,600 compounds, where 2% is 130
    compounds and round potencies like 8.0 clear it innocently -- on PIK3CD
    the unmodified rule discards spikes at 8.0, 8.4 and 8.52, which are potent
    compounds, not ">10 uM" readouts. Discarding the top of the distribution
    is fatal to a top-k statistic.

    So ``mode`` selects the rule, and the caller reports the answer under all
    three rather than picking one:

    ``all``   the block rule, unmodified.
    ``low``   spikes at or below the median only -- a censored "> X" readout
              is by construction a weak measurement, so a spike above the
              median is a real cluster of potencies.
    ``none``  no filter.
    """
    v = np.round(np.asarray(values, dtype=float), 4)
    if mode == "none":
        return np.ones(len(v), dtype=bool)
    uniq, counts = np.unique(v, return_counts=True)
    spike = uniq[counts > pile_frac * len(v)]
    if mode == "low":
        spike = spike[spike <= np.median(v)]
    return ~np.isin(v, spike)


def smd(z, selected):
    """Standardised mean difference between the selected and the rest.

    ``z`` is already standardised over the whole population, so this is the
    raw difference of group means -- in population standard deviations.
    """
    selected = np.asarray(selected, dtype=bool)
    if selected.all() or not selected.any():
        return float("nan")
    return float(z[selected].mean() - z[~selected].mean())


def capture_lift(y, selected, k):
    """Top-k capture by the selected set, over what its size would give anyway.

    A panel covering a fraction f of the library holds f*k of the top k on
    average if it was assembled without reference to activity. The ratio to
    that expectation is comparable across panels of different sizes, and
    across the real and simulated sides.
    """
    selected = np.asarray(selected, dtype=bool)
    frac = selected.mean()
    if frac <= 0 or frac >= 1:
        return float("nan"), float("nan"), float("nan")
    top = np.argsort(y)[-k:]
    observed = float(selected[top].sum())
    expected = frac * k
    return observed, expected, observed / expected


def real_side(df, family, accessions, k, pile_frac, require_all,
              mode="low", max_panel_frac=0.85):
    """Selection statistics for each target of one family, from BindingDB.

    A rotation whose panel covers more than ``max_panel_frac`` of the
    population is marked unreliable rather than dropped: when 97% of the
    compounds measured against JAK1 were also measured against another JAK,
    the "not in the panel" group is 98 compounds of leftovers, and a contrast
    against it is not a measurement of how panels get assembled.
    """
    by_target = {a: df[df.ID2 == a].set_index("X1").pic50 for a in accessions}
    out = []
    for primary in accessions:
        prim = by_target[primary]
        if len(prim) < 50:
            out.append({"primary": primary, "label": FAMILIES[family][primary],
                        "skipped": f"only {len(prim)} compounds"})
            continue
        others = [a for a in accessions if a != primary]
        # Co-assay is a property of which cells BindingDB has filled, so it is
        # read from the index alone, never from the potency.
        hits = np.zeros(len(prim), dtype=int)
        for a in others:
            hits += prim.index.isin(by_target[a].index).astype(int)
        panel = (hits == len(others)) if require_all else (hits >= 1)

        y = prim.to_numpy()
        ok = strip_pileups(y, pile_frac, mode)
        y, panel = y[ok], panel[ok]
        if panel.sum() < 10 or (~panel).sum() < 10 or len(y) < 50:
            out.append({"primary": primary, "label": FAMILIES[family][primary],
                        "skipped": f"panel {int(panel.sum())} of {len(y)} "
                                   f"after the {mode} censoring filter"})
            continue
        sd = y.std()
        z = (y - y.mean()) / (sd if sd > 1e-12 else 1.0)
        obs, exp, lift = capture_lift(y, panel, k)
        out.append({
            "primary": primary, "label": FAMILIES[family][primary],
            "n_measured": int(len(y)), "n_panel": int(panel.sum()),
            "panel_frac": float(panel.mean()), "smd": smd(z, panel),
            "capture_observed": obs, "capture_expected": exp,
            "capture_lift": lift, "filter": mode,
            "reliable": bool(panel.mean() <= max_panel_frac),
        })
    return out


def sim_curve(vals, ti, size, k, alphas, repeats, seed):
    """Selection statistics on the dense block for one primary, over ALPHA.

    Uses ``draw_history`` itself, so the curve is calibrated against the axis
    the campaign actually walks rather than against a copy of it.

    ``size`` matters. Capture lift is bounded by 1/coverage, so a curve drawn
    at the sweep's history size cannot be compared with a real panel that
    covers a different fraction of its library. ALPHA, by contrast, is a
    softmax temperature on the score and means the same thing at any size, so
    the comparison is made by drawing the simulated panel at the real panel's
    coverage and reading ALPHA off that curve.
    """
    n = len(vals)
    y = vals[:, ti]
    others = [j for j in range(vals.shape[1]) if j != ti]
    sd_y = y.std()
    z = (y - y.mean()) / (sd_y if sd_y > 1e-12 else 1.0)
    # The score a panel assembler could have known: mean standardised activity
    # on the AUXILIARY targets. The primary column is never read, as in the
    # campaign.
    a = vals[:, others]
    sd = a.std(axis=0)
    prefer = ((a - a.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)).mean(axis=1)
    size = int(np.clip(size, k + 1, n - 1))

    curve = {}
    for alpha in alphas:
        smds, lifts = [], []
        for r in range(repeats):
            # The grid runs negative, and a seed tuple must be non-negative.
            rng = np.random.default_rng(
                (seed, ti, r, int(alpha * 1000) + 1_000_000, size))
            # cluster_frac=0 and exclude_top=False are the sweep's settings;
            # the Gram matrix is only read when clustering, so the identity
            # stands in for it here.
            hist = draw_history(np.eye(n), y, size, rng,
                                cluster_frac=0.0, exclude_top=False, k=k,
                                prefer=prefer, prefer_alpha=alpha)
            sel = np.zeros(n, dtype=bool)
            sel[hist] = True
            smds.append(smd(z, sel))
            lifts.append(capture_lift(y, sel, k)[2])
        curve[alpha] = {"smd": float(np.nanmean(smds)),
                        "smd_sd": float(np.nanstd(smds)),
                        "capture_lift": float(np.nanmean(lifts)),
                        "lift_ceiling": float(n / size),
                        "panel_frac": size / n}
    return curve


def invert(curve, value, key):
    """ALPHA whose simulated statistic matches ``value``, linearly interpolated.

    Returns None outside the grid rather than extrapolating: past the end of
    the grid the honest answer is "beyond what was simulated".
    """
    if value is None or not np.isfinite(value):
        return None
    xs = sorted(curve)
    ys = [curve[a][key] for a in xs]
    if not all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1)):
        return None                      # not monotone; inversion undefined
    if value <= ys[0] or value >= ys[-1]:
        return None
    j = int(np.searchsorted(ys, value))
    lo, hi = ys[j - 1], ys[j]
    w = 0.0 if hi == lo else (value - lo) / (hi - lo)
    return float(xs[j - 1] + w * (xs[j] - xs[j - 1]))


def block_path(family, accessions):
    labels = sorted(FAMILIES[family][a].lower() for a in accessions)
    return BLOCK_DIR / f"block_{'-'.join(labels)}.csv"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", nargs="+",
                   default=["mmp", "dpp", "cyp", "hdac", "pi3k", "jak"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--history", type=int, default=50,
                   help="per auxiliary target, as the sweep used")
    p.add_argument("--repeats", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pile-frac", type=float, default=0.02)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5,
                            -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                            4.0, 6.0],
                   help="negative values are panels that avoid actives, which "
                        "the first run found some families actually do")
    p.add_argument("--filters", nargs="+", default=["low", "none", "all"],
                   help="censoring rules to report; the answer is only usable "
                        "if it survives all of them")
    p.add_argument("--max-panel-frac", type=float, default=0.85)
    p.add_argument("--require-all", action="store_true",
                   help="panel = measured on ALL other family targets "
                        "(default: at least one)")
    p.add_argument("--out", default="results/panel_propensity.json")
    args = p.parse_args(argv)

    accs = sorted({a for f in args.families for a in FAMILIES[f]})
    print(f"[0] reading BindingDB for {len(accs)} accessions "
          f"across {len(args.families)} families...", flush=True)
    df = load_pairs(accs)
    print(f"    {len(df):,} unique (target, compound) pairs", flush=True)

    report = {"families": {}, "settings": vars(args)}
    for family in args.families:
        accessions = list(FAMILIES[family])
        print(f"\n[{family}]", flush=True)

        # Sensitivity to the censoring rule first: if the sign of the top-k
        # lift depends on which rule is used, nothing below is reportable.
        by_filter = {m: real_side(df, family, accessions, args.k,
                                  args.pile_frac, args.require_all, m,
                                  args.max_panel_frac)
                     for m in args.filters}
        print(f"    {'target':<9s} " + "".join(
            f"{'lift/' + m:>12s}{'SMD':>8s}" for m in args.filters))
        for i, primary in enumerate(accessions):
            cells = ""
            for m in args.filters:
                r = by_filter[m][i]
                cells += ("     skipped     --" if "skipped" in r else
                          f"{r['capture_lift']:11.2f}x{r['smd']:+8.2f}")
            flag = "" if all("skipped" in by_filter[m][i]
                             or by_filter[m][i]["reliable"]
                             for m in args.filters) else "  [coverage>%.0f%%]" % (
                                 args.max_panel_frac * 100)
            print(f"    {FAMILIES[family][primary]:<9s} {cells}{flag}")

        real = by_filter[args.filters[0]]
        for r in real:
            if "skipped" not in r:
                print(f"    {r['label']:<9s} panel {r['n_panel']:>5d}/"
                      f"{r['n_measured']:<6d} ({r['panel_frac']:5.1%})"
                      f"{'' if r['reliable'] else '  UNRELIABLE'}")

        path = block_path(family, accessions)
        if not path.exists():
            print(f"    no dense block at {path}; simulation skipped")
            report["families"][family] = {"real": real, "block": None}
            continue
        block = pd.read_csv(path)
        targets = [FAMILIES[family][a] for a in accessions]
        block, _ = drop_censored(block, targets, pile_frac=args.pile_frac,
                                 verbose=False)
        vals = block[targets].to_numpy(dtype=float)
        n_block = len(block)

        # Reference: where the sweep's own designs sit, at its history size.
        ref = sim_curve(vals, 0, args.history * 3, args.k, args.alphas,
                        args.repeats, args.seed)
        print(f"    block {n_block} compounds; the sweep's history "
              f"({args.history}x3) covers {ref[args.alphas[0]]['panel_frac']:.0%}"
              f" -- its uniform design is alpha=0, 'pref1' is alpha=1")

        # Each real rotation is matched on coverage before ALPHA is read off.
        # Only rotations with an informative non-panel group are used.
        scored = [r for r in real if "skipped" not in r and r["reliable"]]
        for r in scored:
            ti = targets.index(r["label"])
            size = int(round(r["panel_frac"] * n_block))
            c = sim_curve(vals, ti, size, args.k, args.alphas, args.repeats,
                          args.seed)
            r["alpha_from_smd"] = invert(c, r["smd"], "smd")
            r["alpha_from_lift"] = invert(c, r["capture_lift"], "capture_lift")
            r["sim_matched_size"] = size
            r["lift_ceiling"] = c[args.alphas[0]]["lift_ceiling"]
            r["curve_matched"] = {str(a): v for a, v in c.items()}
            a_s, a_l = r["alpha_from_smd"], r["alpha_from_lift"]
            print(f"      {r['label']:<9s} matched sim size {size:>4d}  "
                  f"alpha(SMD) {'  n/a' if a_s is None else f'{a_s:5.2f}'}  "
                  f"alpha(lift) {'  n/a' if a_l is None else f'{a_l:5.2f}'}"
                  + ("" if r["capture_lift"] <= r["lift_ceiling"] * 0.95
                     else "   [lift at ceiling]"))

        if not scored:
            print("    no rotation has an informative non-panel group; "
                  "this family cannot place a real panel on the axis")
        for key, lab in (("alpha_from_smd", "SMD"),
                         ("alpha_from_lift", "capture lift")):
            got = [r[key] for r in scored if r[key] is not None]
            off = len(scored) - len(got)
            txt = (f"{np.mean(got):+.2f} (range {min(got):+.2f}..{max(got):+.2f})"
                   if got else "all outside the grid")
            print(f"    equivalent alpha by {lab:<13s}: {txt}"
                  + (f"; {off} of {len(scored)} outside grid" if off else ""))
        report["families"][family] = {
            "real_by_filter": by_filter, "block_n": n_block,
            "curve_sweep_size": {str(a): v for a, v in ref.items()}}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
