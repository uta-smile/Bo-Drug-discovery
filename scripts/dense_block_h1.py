"""H1 on measured data, with the geometry confound removed by construction.

    python scripts/dense_block_h1.py --out results/dense_block_h1.json

Section 17 found no transfer on ChEMBL and could not say why, because two
explanations were confounded:

1. measured kinases are coupled at r ~ 0.4, not at DOCKSTRING's 1.0;
2. ChEMBL libraries are disjoint chemotypes -- auxiliary compounds reach only
   0.53 of the primary's own nearest-neighbour similarity -- so the Tanimoto
   kernel may have nothing to carry a transfer through whatever r is.

Transfer flows through ``k_tanimoto(m, m') * r``. On ChEMBL those were about
0.35 and 0.40, so the effective coupling was ~0.14 and the two failures
multiplied. Fixing one alone cannot work.

A **dense block** fixes both. BindingDB contains compound sets measured against
every member of a protein family -- selectivity panels, run on purpose. Inside
one, the auxiliary observations are on *the same molecules* as the primary's, so
reach is 1.0 by construction and cannot be the explanation for anything. What
is left is the task correlation, measured rather than assumed, and it is
strictly inside the range H1 needs:

    JAK1-JAK2 +0.79   JAK2-TYK2 +0.65   JAK2-JAK3 +0.54
    JAK1-JAK3 +0.54   JAK1-TYK2 +0.38   JAK3-TYK2 +0.25

This is the cleanest test of H1 the available data supports, and it runs both
forms of the hypothesis against the same block:

* **H1-split** (section 14's question): spend a fixed number of *cells* either
  all on the primary, or spread across the family. Blocked and spread designs.
* **H1-cold** (section 16's question): the primary is cold at ``n`` charged
  observations while the family carries ``H`` free historical ones.

pIC50, median over replicate measurements. Higher is better, so nothing needs
negating -- unlike docking scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.audit import Audit  # noqa: E402
from molbo.metrics import spearman0 as _rho  # noqa: E402

from molbo.surrogate import (  # noqa: E402
    CoregionalisedGP,
    TanimotoGP,
    fingerprint_matrix,
    tanimoto_gram,
)

SOURCE = Path("data/cache/tdc/bindingdb_ic50.csv")
CACHE_DIR = Path("data/bindingdb")

FAMILIES = {
    "jak": {"P23458": "JAK1", "O60674": "JAK2", "P52333": "JAK3", "P29597": "TYK2"},
}
"""UniProt -> label. The JAK family is the largest dense kinase block in
BindingDB at 1,563 compounds x 4 targets; ``--uniprot`` takes any other set."""

FREE_GRID: tuple[float, ...] = (0.0, 0.3, 0.6, 0.8, 0.95, 0.999)
"""Coarser grid for the free-B coordinate scan. The full grid at two passes
costs ~290 GP fits per model fit, which is minutes per sweep cell; six values
at one pass costs ~70 and resolves the only thing this arm is asked to show,
which is whether pairs separate at all."""

N_GRID = (10, 20, 40, 80)
H_GRID = (0, 5, 10, 25, 50, 100, 200)




def nanomolar_potency(y):
    """pIC50 from IC50 in nM -- the convention of the BindingDB dump."""
    return 9.0 - np.log10(y)


def build_block(mapping, *, source=SOURCE, cache_dir=CACHE_DIR, force=False,
                potency=nanomolar_potency):
    """The fully-observed sub-matrix for ``mapping``, cached as a CSV.

    The source is 839 MB and scanning it is 3 s in chunks but 20+ s and several
    GB if read whole, so it is read in chunks and the result memoised -- the
    block is the unit of work here, not the file.

    ``potency`` maps the dump's ``Y`` column to "higher is better", which is what
    every downstream caller assumes. It is *not* a formality: the kinome panels
    report a KIBA score where higher already means stronger (checked against this
    dump's own pIC50 on the compounds JAK2 shares with it, Spearman +0.855), so
    passing the nanomolar conversion there would silently invert the objective
    and have a campaign optimise for the weakest binders.

    The cache key carries the source for the same reason. Blocks are named after
    their genes, so a JAK block cut from a kinome panel and one cut from
    BindingDB would otherwise be the same file.
    """
    import pandas as pd

    source = Path(source)
    key = "-".join(sorted(mapping.values())).lower()
    stem = "" if source == SOURCE else f"{source.stem}_"
    cache = cache_dir / f"block_{stem}{key}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache), cache

    if not source.exists():
        raise SystemExit(
            f"{source} not found. It is the TDC BindingDB IC50 dump; this script "
            "needs it once to cut the block, after which the cached CSV suffices.")
    keep = []
    for chunk in pd.read_csv(source, usecols=["X1", "ID2", "Y"], chunksize=500_000):
        keep.append(chunk[chunk.ID2.isin(mapping)])
    df = pd.concat(keep, ignore_index=True).dropna()
    # IC50 in nM; drop non-positive and absurd values before the log.
    df = df[(df.Y > 0) & (df.Y < 1e9)]
    df["p"] = potency(df.Y)
    # Median over replicates: BindingDB carries the same pair from many papers,
    # and a mean would let one mis-keyed order of magnitude move a compound.
    g = df.groupby(["X1", "ID2"]).p.median().reset_index()
    wide = g.pivot(index="X1", columns="ID2", values="p")
    wide.columns = [mapping[c] for c in wide.columns]
    wide = wide.dropna().reset_index().rename(columns={"X1": "smiles"})
    cache_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(cache, index=False)
    return wide, cache


def drop_censored(block, targets, *, pile_frac=0.02, verbose=True):
    """Remove compounds sitting on an assay's reporting ceiling.

    A screening panel that reports "IC50 > 10 uM" stores it as a single
    number, and BindingDB carries it through. In the CYP block that is pIC50
    exactly 5.00 for 21-27% of each column, and **40.3% of compounds hold the
    identical value on all four targets** -- against 3.5% in the JAK block.

    Those rows wreck every quantity this script measures, and all in the same
    direction. Cross-target correlation is inflated, because a censored 5.00 is
    perfectly correlated with another censored 5.00. The fitted task
    correlation rails to 1. Naive pooling looks optimal, because for those
    compounds the four targets genuinely are one number. And the transfer gain
    itself is inflated, because predicting a censored constant on one target
    from the same censored constant on another is not transfer, it is copying.

    The rule: any value shared by more than ``pile_frac`` of a column is a
    pile-up rather than a measurement, and compounds touching one are dropped.
    A real potency distribution has no such spike.
    """
    vals = block[targets]
    bad = np.zeros(len(block), dtype=bool)
    piles = {}
    for t in targets:
        counts = vals[t].round(4).value_counts()
        spike = counts[counts > pile_frac * len(block)]
        if len(spike):
            piles[t] = {float(k): int(v) for k, v in spike.items()}
            bad |= vals[t].round(4).isin(spike.index).to_numpy()
    kept = block.loc[~bad].reset_index(drop=True)
    if verbose:
        print(f"\n[0] censoring filter (values held by >{pile_frac:.0%} of a column)")
        for t, spike in piles.items():
            shown = ", ".join(f"{k:.2f} x{v}" for k, v in spike.items())
            print(f"    {t:<8s} pile-ups: {shown}")
        if not piles:
            print("    none found -- no column has a value spike")
        print(f"    kept {len(kept):,} of {len(block):,} compounds "
              f"({len(kept) / len(block):.1%})")
    return kept, piles


def compute_qed(block):
    """QED per compound, NaN where RDKit cannot parse the SMILES."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

    RDLogger.DisableLog("rdApp.*")
    out = np.full(len(block), np.nan)
    for i, smi in enumerate(block.smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[i] = QED.qed(mol)
    return out


def pareto_mask(F):
    """Non-dominated rows of ``F`` (n, d), all objectives maximised.

    Used rather than a scalarisation because scalarising is precisely what
    breaks these benchmarks: section 14 measured the published DOCKSTRING
    objectives as 93% determined by their QED term once it was summed into the
    score. A Pareto front cannot be dominated by a free objective in that way --
    a compound with high QED and no potency sits at one end of the front and
    does not displace the potent end.
    """
    F = np.asarray(F, dtype=float)
    n = len(F)
    keep = np.ones(n, dtype=bool)
    order = np.argsort(-F[:, 0])
    for i in order:
        if not keep[i]:
            continue
        # Anything weakly dominated by i, and not equal to it, is removed.
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        keep &= ~dom
    return keep


def drop_non_druglike(block, *, qed_min, verbose=True):
    """Keep only drug-like compounds, as a library filter rather than a score.

    The whole of sections 18-25 optimises raw potency, so nothing stops the
    method selecting a large greasy compound that happens to bind. The obvious
    repair -- adding a drug-likeness term to the objective -- is exactly what
    ruins the published DOCKSTRING objectives, where ``-10(1 - QED)`` ends up
    carrying 93% of the ranking and a zero-assay QED ordering recovers half the
    top-50 (section 14).

    Applying it as an admission filter instead keeps the score entirely assayed:
    rejected compounds leave the library rather than scoring badly, so a free
    descriptor cannot earn any recall and the floor stays at chance.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

    RDLogger.DisableLog("rdApp.*")
    vals = np.full(len(block), np.nan)
    for i, smi in enumerate(block.smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            vals[i] = QED.qed(mol)
    keep = np.isfinite(vals) & (vals >= qed_min)
    if verbose:
        print(f"\n[0b] drug-likeness filter (QED >= {qed_min:g})")
        finite = vals[np.isfinite(vals)]
        print(f"    QED {finite.min():.2f}..{finite.max():.2f}  "
              f"median {np.median(finite):.2f}")
        print(f"    kept {int(keep.sum()):,} of {len(block):,} compounds "
              f"({keep.mean():.1%})")
    return block.loc[keep].reset_index(drop=True)


def describe(block, K, targets, report):
    print(f"\n[1] the block: {len(block):,} compounds x {len(targets)} targets "
          f"= {len(block) * len(targets):,} cells, fully observed")
    vals = block[targets].to_numpy(float)
    for i, t in enumerate(targets):
        print(f"    {t:<6s} pIC50 {vals[:, i].min():.2f}..{vals[:, i].max():.2f}  "
              f"median {np.median(vals[:, i]):.2f}  sd {vals[:, i].std():.2f}")

    C = np.corrcoef(vals.T)
    print("\n    empirical task correlation (Pearson on the same compounds):")
    print("           " + "".join(f"{t:>8s}" for t in targets))
    for i, t in enumerate(targets):
        print(f"    {t:<7s}" + "".join(
            f"{C[i, j]:>8.2f}" if i != j else f"{'--':>8s}" for j in range(len(targets))))
    off = C[np.triu_indices(len(targets), 1)]
    print(f"    off-diagonal: min {off.min():+.2f}  median {np.median(off):+.2f}  "
          f"max {off.max():+.2f}")

    # Reach is 1.0 by construction -- every task is measured on these same
    # molecules -- but the pool's own tightness still matters for the GP, so
    # report it against section 17's ChEMBL numbers (own NN 0.64-0.71).
    g = np.random.default_rng(0)
    pick = g.choice(len(block), size=min(300, len(block)), replace=False)
    own = K[np.ix_(pick, np.arange(len(block)))].copy()
    for r, j in enumerate(pick):
        own[r, j] = -1.0
    nn = float(own.max(axis=1).mean())
    print(f"\n    pool nearest-neighbour similarity {nn:.3f}  "
          "(ChEMBL libraries 0.64-0.71, DOCKSTRING 0.38)")
    print("    auxiliary reach ratio 1.00 by construction: the tasks share the "
          "molecules.")
    report["block"] = {
        "n_compounds": int(len(block)), "targets": list(targets),
        "corr": {f"{targets[i]}|{targets[j]}": float(C[i, j])
                 for i in range(len(targets)) for j in range(i + 1, len(targets))},
        "corr_median": float(np.median(off)), "nn_similarity": nn,
    }
    return C


def h1_split(block, K, targets, report, *, n_grid, n_repeats, seed):
    """Section 14's question on measured data: spend cells on one target or many."""
    print(f"\n[2] H1-split -- equal cells, concentrated vs spread "
          f"({n_repeats} repeats)")
    print("    a cell is one (compound, target) measurement; every arm spends n.")
    vals = block[targets].to_numpy(float)
    n_pool = len(block)
    T = len(targets)
    out = {}
    for n in n_grid:
        m = n // T
        acc = {"stl": [], "blocked": [], "spread": []}
        for ti, t in enumerate(targets):
            y = vals[:, ti]
            others = [j for j in range(T) if j != ti]
            for r in range(n_repeats):
                g = np.random.default_rng(seed + 977 * r + 31 * n + 7919 * ti)
                draw = g.choice(n_pool, size=max(n, T * m), replace=False)
                tr = draw[:n]
                te = np.setdiff1d(np.arange(n_pool), draw)
                gp = TanimotoGP().fit_gram(K[np.ix_(tr, tr)], y[tr])
                acc["stl"].append(
                    _rho(y[te], gp.predict_gram(K[np.ix_(te, tr)], return_std=False)))
                if m < 3:
                    acc["blocked"].append(np.nan)
                    acc["spread"].append(np.nan)
                    continue
                trm = draw[:m]
                idx_b = [trm] * T
                val_b = [y[trm]] + [vals[trm, j] for j in others]
                acc["blocked"].append(_rho(
                    y[te], CoregionalisedGP().fit_gram(K, idx_b, val_b)
                    .predict_gram(K, te)))
                idx_s = [draw[a * m:(a + 1) * m] for a in range(T)]
                val_s = ([y[idx_s[0]]]
                         + [vals[idx_s[a + 1], others[a]] for a in range(T - 1)])
                acc["spread"].append(_rho(
                    y[te], CoregionalisedGP().fit_gram(K, idx_s, val_s)
                    .predict_gram(K, te)))
        row = {}
        for k, v in acc.items():
            v = np.asarray(v, float)
            ok = np.isfinite(v)
            row[k] = float(v[ok].mean()) if ok.any() else float("nan")
        for k in ("blocked", "spread"):
            d = np.asarray(acc[k], float) - np.asarray(acc["stl"], float)
            ok = np.isfinite(d)
            row[f"{k}_gain"] = float(d[ok].mean()) if ok.any() else float("nan")
            row[f"{k}_sem"] = (float(d[ok].std() / np.sqrt(ok.sum())) if ok.any()
                               else float("nan"))
        row["m_per_target"] = m
        out[n] = row

    print(f"\n    {'cells':>6s}{'each':>6s}{'stl':>9s}{'blocked':>9s}{'spread':>9s}"
          f"{'spread gain':>19s}")
    print("    " + "-" * 58)
    for n in n_grid:
        r = out[n]
        g = (f"{'n/a':>19s}" if not np.isfinite(r["spread_gain"])
             else f"{r['spread_gain']:>+13.3f}+-{r['spread_sem']:<4.3f}")
        print(f"    {n:>6d}{r['m_per_target']:>6d}{r['stl']:>9.3f}"
              f"{r['blocked']:>9.3f}{r['spread']:>9.3f}{g}")
    best = max((r for r in out.values() if np.isfinite(r["spread_gain"])),
               key=lambda r: r["spread_gain"], default=None)
    print("\n    verdict:")
    if best is None:
        print("      no budget affords the multi-task arm.")
    elif best["spread_gain"] > 3 * best["spread_sem"]:
        print(f"      H1-split HOLDS on measured data: {best['spread_gain']:+.3f}.")
    else:
        print(f"      H1-split fails on measured data too: best "
              f"{best['spread_gain']:+.3f}+-{best['spread_sem']:.3f}.")
        print("      Same answer as DOCKSTRING (section 14), now without the")
        print("      docking-score and library-geometry caveats.")
    report["h1_split"] = {str(k): v for k, v in out.items()}
    return out


def h1_cold(block, K, targets, report, *, n_grid, h_grid, n_repeats, seed, free_b=False,
            holdout=0.0, audit=False):
    """Section 16's question on measured data: free history on the family."""
    print(f"\n[3] H1-cold -- primary charged n, family carries H free "
          f"({n_repeats} repeats)")
    vals = block[targets].to_numpy(float)
    n_pool = len(block)
    T = len(targets)
    out: dict = {}
    rho_picked: list[float] = []
    audit = Audit(enabled=bool(audit))
    for n in n_grid:
        for h in h_grid:
            acc, acc_pool, acc_free = [], [], []
            b_spread = []
            for ti, t in enumerate(targets):
                y = vals[:, ti]
                others = [j for j in range(T) if j != ti]
                for r in range(n_repeats):
                    g = np.random.default_rng(seed + 977 * r + 31 * n + h + 7919 * ti)
                    need = n + h * (T - 1)
                    # Without a reserved holdout, `te` is whatever the draw did
                    # not take -- so the H=200 arm is scored on 600 fewer
                    # compounds than the H=0 baseline it is differenced against.
                    # Those compounds are random, so the mean should not move,
                    # but "should not" is a claim: --holdout reserves one test
                    # set that every H is scored on, and settles it.
                    if holdout > 0:
                        # Includes the target, so each held-out target gets its
                        # own evaluation rows. Without `ti` every target in a
                        # repeat is scored on the same compounds, which
                        # correlates the per-target results that rule 4 exists
                        # to keep separate -- and it disagreed with the same
                        # stream in transfer_headroom.py and
                        # cold_target_chembl.py, making `--holdout` runs
                        # incomparable across the three instruments.
                        gh = np.random.default_rng(seed + 977 * r + 7919 * ti)
                        perm = gh.permutation(n_pool)
                        n_te = int(holdout * n_pool)
                        te, avail = perm[:n_te], perm[n_te:]
                        room = len(avail)
                    else:
                        te, avail, room = None, np.arange(n_pool), n_pool // 2
                    if need > room:
                        acc.append(np.nan)
                        acc_pool.append(np.nan)
                        acc_free.append(np.nan)
                        continue
                    draw = avail[g.choice(len(avail), size=need, replace=False)]
                    tr = draw[:n]
                    if te is None:
                        te = np.setdiff1d(np.arange(n_pool), draw)
                    # A leak here would inflate every number in this check
                    # and would not crash. The evaluation rows must touch
                    # neither the primary's training rows nor the free history.
                    audit.cell()
                    audit.disjoint(te, draw, f"{t} n={n} h={h} evaluation",
                                   "training/history")
                    audit.exactly(draw, need, f"{t} n={n} h={h} draw")
                    # The auxiliary columns must be other targets, not the
                    # primary's own values relabelled.
                    audit.excludes(ti, others, f"{t} n={n} h={h} auxiliary set")

                    if h == 0:
                        gp = TanimotoGP().fit_gram(K[np.ix_(tr, tr)], y[tr])
                        mu = gp.predict_gram(K[np.ix_(te, tr)], return_std=False)
                        acc.append(_rho(y[te], mu))
                        acc_pool.append(_rho(y[te], mu))
                        acc_free.append(_rho(y[te], mu))
                        continue
                    idx = [tr] + [draw[n + a * h: n + (a + 1) * h]
                                  for a in range(T - 1)]
                    val = ([y[tr]]
                           + [vals[idx[a + 1], others[a]] for a in range(T - 1)])
                    gp = CoregionalisedGP().fit_gram(K, idx, val)
                    acc.append(_rho(y[te], gp.predict_gram(K, te)))
                    rho_picked.append(gp.rho_)
                    if free_b:
                        # A full correlation matrix instead of one shared r.
                        # Predicted to pay where the family is unevenly coupled
                        # (JAK, 0.24-0.71) and not where it is uniform (CYP,
                        # 0.52-0.74) -- section 19.
                        fb = CoregionalisedGP(structure="free", passes=1,
                                              rho_grid=FREE_GRID).fit_gram(
                            K, idx, val)
                        acc_free.append(_rho(y[te], fb.predict_gram(K, te)))
                        off = fb._B[np.triu_indices(T, 1)]
                        b_spread.append(float(off.max() - off.min()))
                    # Control: the same observations with the task structure
                    # thrown away. r = 1 is B = 11', one function wearing four
                    # labels -- z-scored per task, that is exactly "pool
                    # everything and fit one GP". If it matches the fitted arm,
                    # the coregionalisation is decoration and the gain is just
                    # more data; if the fitted arm wins, estimating how much the
                    # targets share is doing work.
                    pooled = CoregionalisedGP(rho_grid=(1.0,)).fit_gram(K, idx, val)
                    acc_pool.append(_rho(y[te], pooled.predict_gram(K, te)))
            v = np.asarray(acc, float)
            vp = np.asarray(acc_pool, float)
            vf = np.asarray(acc_free, float)
            ok = np.isfinite(v)
            okp = np.isfinite(vp)
            okf = np.isfinite(vf)
            out[(n, h)] = {
                "free": float(vf[okf].mean()) if okf.any() else float("nan"),
                "raw_free": vf[okf].tolist(),
                "b_spread": float(np.mean(b_spread)) if b_spread else float("nan"),
                "mean": float(v[ok].mean()) if ok.any() else float("nan"),
                "sem": (float(v[ok].std() / np.sqrt(ok.sum())) if ok.sum() > 1
                        else float("nan")),
                "pooled": float(vp[okp].mean()) if okp.any() else float("nan"),
                "raw": v[ok].tolist(),
                "raw_pooled": vp[okp].tolist(),
            }
        print(f"    n={n} done", flush=True)

    print(f"\n    Spearman on held-out compounds")
    print(f"    {'n':>4s}{'H=0 (stl)':>12s}"
          + "".join(f"{'H=' + str(h):>11s}" for h in h_grid if h))
    print("    " + "-" * (16 + 11 * (len(h_grid) - 1)))
    summary = {}
    for n in n_grid:
        base = out[(n, 0)]["mean"]
        line, gains = f"    {n:>4d}{base:>12.3f}", {}
        for h in h_grid:
            if not h:
                continue
            v = out[(n, h)]["mean"]
            a = np.asarray(out[(n, h)]["raw"], float)
            b = np.asarray(out[(n, 0)]["raw"], float)
            k = min(len(a), len(b))
            d = a[:k] - b[:k]
            gains[h] = {"value": v, "gain": v - base,
                        "sem": (float(d.std() / np.sqrt(len(d))) if len(d) > 1
                                else float("nan")),
                        "frac_positive": float((d > 0).mean()) if len(d) else 0.0}
            line += f"{v:>11.3f}" if np.isfinite(v) else f"{'n/a':>11s}"
        summary[n] = {"stl": base, "by_h": gains}
        print(line)

    print("\n    gain over the single-task GP")
    print(f"    {'n':>4s}" + "".join(f"{'H=' + str(h):>18s}" for h in h_grid if h))
    print("    " + "-" * (4 + 18 * (len(h_grid) - 1)))
    for n in n_grid:
        cells = ""
        for h, g in summary[n]["by_h"].items():
            cells += (f"{'n/a':>18s}" if not np.isfinite(g["gain"])
                      else f"{g['gain']:>+12.3f}+-{g['sem']:<5.3f}")
        print(f"    {n:>4d}{cells}")

    if free_b:
        print("\n    free B -- a full correlation matrix instead of one shared r")
        print(f"    {'n':>4s}" + "".join(f"{'H=' + str(h):>18s}" for h in h_grid if h))
        print("    " + "-" * (4 + 18 * (len(h_grid) - 1)))
        for n in n_grid:
            cells = ""
            for h in h_grid:
                if not h:
                    continue
                a = np.asarray(out[(n, h)]["raw_free"], float)
                b = np.asarray(out[(n, h)]["raw"], float)
                k = min(len(a), len(b))
                d = a[:k] - b[:k]
                summary[n]["by_h"][h]["free"] = out[(n, h)]["free"]
                summary[n]["by_h"][h]["vs_shared"] = (
                    float(d.mean()) if k else float("nan"))
                summary[n]["by_h"][h]["vs_shared_sem"] = (
                    float(d.std() / np.sqrt(k)) if k > 1 else float("nan"))
                cells += (f"{out[(n, h)]['free']:>11.3f}"
                          f"{summary[n]['by_h'][h]['vs_shared']:>+7.3f}")
            print(f"    {n:>4d}{cells}")
        print("    each cell: free-B rho, then (free - shared r).")
        sp = [out[(n, h)]["b_spread"] for n in n_grid for h in h_grid if h
              and np.isfinite(out[(n, h)]["b_spread"])]
        if sp:
            print(f"    fitted B off-diagonal spread: {np.mean(sp):.2f} "
                  "(how unevenly the family came out coupled)")
            report["b_spread"] = float(np.mean(sp))

    print("\n    control: the same observations with the task structure discarded")
    print(f"    (r fixed at 1, i.e. pool all four targets into one function)")
    print(f"    {'n':>4s}" + "".join(f"{'H=' + str(h):>18s}" for h in h_grid if h))
    print("    " + "-" * (4 + 18 * (len(h_grid) - 1)))
    for n in n_grid:
        cells = ""
        for h in h_grid:
            if not h:
                continue
            a = np.asarray(out[(n, h)]["raw"], float)
            b = np.asarray(out[(n, h)]["raw_pooled"], float)
            k = min(len(a), len(b))
            d = a[:k] - b[:k]
            summary[n]["by_h"][h]["pooled"] = out[(n, h)]["pooled"]
            summary[n]["by_h"][h]["vs_pooled"] = float(d.mean()) if k else float("nan")
            summary[n]["by_h"][h]["vs_pooled_sem"] = (
                float(d.std() / np.sqrt(k)) if k > 1 else float("nan"))
            cells += (f"{out[(n, h)]['pooled']:>11.3f}"
                      f"{summary[n]['by_h'][h]['vs_pooled']:>+7.3f}")
        print(f"    {n:>4d}{cells}")
    print("    each cell: pooled rho, then (fitted ICM - pooled).")

    if rho_picked:
        q = np.percentile(rho_picked, [10, 50, 90])
        print(f"\n    task correlation selected: median {q[1]:.2f} "
              f"(10th {q[0]:.2f}, 90th {q[2]:.2f})")
        report["rho_selected"] = {"median": float(q[1]), "p10": float(q[0]),
                                  "p90": float(q[2])}

    audit.report("", clean_lines=(
        "evaluation rows disjoint from training and history;",
        "no duplicate draws; no target used as its own auxiliary",
    ))
    report["audit"] = audit.to_json()

    print("\n    verdict:")
    # Every arm can come back empty -- a holdout that leaves less room than
    # n + H(T-1) needs, or a block too small for the grid. The gains are then
    # all NaN, and the "fails" branch below would report a scientific
    # conclusion drawn from no data at all. Refuse instead.
    if not any(np.isfinite(g["gain"])
               for n in n_grid for g in summary[n]["by_h"].values()):
        biggest = max(n_grid) + max(h_grid) * (len(targets) - 1)
        print("      no configuration in the grid could be fitted: the largest")
        print(f"      needs {biggest} compounds and the holdout/pool left fewer.")
        print("      This is not a result about H1 -- lower --holdout, shrink")
        print("      --history, or use a larger block.")
        report["h1_cold"] = {"summary": {}, "status": "no_fits"}
        report["h1_cold_budgets_live"] = 0
        return summary

    live = 0
    for n in n_grid:
        hit = next((h for h, g in summary[n]["by_h"].items()
                    if np.isfinite(g["gain"]) and np.isfinite(g["sem"])
                    and g["gain"] > 3 * g["sem"]), None)
        best_h = max(summary[n]["by_h"],
                     key=lambda k: (summary[n]["by_h"][k]["gain"]
                                    if np.isfinite(summary[n]["by_h"][k]["gain"])
                                    else -np.inf))
        bg = summary[n]["by_h"][best_h]
        if hit is None:
            print(f"      n={n:<3d} nothing clears 3 SEM; best {bg['gain']:+.3f} "
                  f"at H={best_h}.")
        else:
            live += 1
            print(f"      n={n:<3d} breaks even at H={hit}; best {bg['gain']:+.3f} "
                  f"at H={best_h} ({bg['frac_positive']:.0%} of runs positive).")
    if live == len(n_grid):
        print("      H1-cold HOLDS on measured data, at every budget, on a block")
        print("      where chemical reach cannot be the explanation. This is the")
        print("      first support for any form of H1 outside docking scores.")
    elif live:
        print("      H1-cold holds at some budgets. Read the break-even column.")
    else:
        print("      H1-cold fails here too -- and here it cannot be blamed on")
        print("      library geometry, because the tasks share their compounds.")
        print("      That leaves the surrogate, not the hypothesis, as the suspect.")
    report["h1_cold"] = {"summary": {str(k): v for k, v in summary.items()}}
    report["h1_cold_budgets_live"] = live
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="jak", choices=sorted(FAMILIES))
    p.add_argument("--uniprot", nargs="+", default=None,
                   help="UNIPROT=LABEL pairs, overriding --family")
    p.add_argument("--budgets", type=int, nargs="+", default=list(N_GRID))
    p.add_argument("--history", type=int, nargs="+", default=list(H_GRID))
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild", action="store_true", help="re-cut the cached block")
    p.add_argument("--drop-censored", action="store_true",
                   help="remove compounds sitting on an assay reporting ceiling "
                        "(a value held by >2%% of a column). Essential for the "
                        "CYP block, where 40%% of compounds carry one identical "
                        "censored number across all four targets")
    p.add_argument("--free-b", action="store_true",
                   help="also fit a full task correlation matrix by coordinate "
                        "ascent, instead of one shared r. Predicted to pay only "
                        "where the family is unevenly coupled (FINDINGS 19)")
    p.add_argument("--holdout", type=float, default=0.0,
                   help="reserve this fraction as a single test set shared by "
                        "every H, so the arms being differenced are scored on "
                        "the same compounds (0 = score on whatever the draw "
                        "left, which differs by H)")
    p.add_argument("--audit", action="store_true",
                   help="check at runtime that evaluation rows never overlap "
                        "training or history rows, and that no target is used "
                        "as its own auxiliary")
    p.add_argument("--skip-split", action="store_true")
    p.add_argument("--skip-cold", action="store_true")
    p.add_argument("--out", default="results/dense_block_h1.json")
    args = p.parse_args(argv)

    if args.uniprot:
        try:
            mapping = dict(kv.split("=", 1) for kv in args.uniprot)
        except ValueError:
            raise SystemExit("--uniprot takes UNIPROT=LABEL pairs, e.g. O60674=JAK2")
    else:
        mapping = FAMILIES[args.family]
    if len(mapping) < 2:
        raise SystemExit("a dense block needs at least two targets")
    if 0 not in args.history:
        raise SystemExit("--history must include 0: the single-task control")

    t0 = time.time()
    block, cache = build_block(mapping, force=args.rebuild)
    targets = [t for t in mapping.values() if t in block.columns]
    print(f"block from {cache}: {len(block):,} compounds x {len(targets)} targets "
          f"({time.time() - t0:.0f}s)")
    if args.drop_censored:
        block, _piles = drop_censored(block, targets)
        if len(block) < 200:
            raise SystemExit(f"only {len(block)} compounds survive the censoring "
                             "filter; this block is mostly ceiling values")
    X = fingerprint_matrix(block.smiles.tolist())
    K = tanimoto_gram(X, X)

    report: dict = {"source": str(cache), "mapping": mapping, "seed": args.seed}
    describe(block, K, targets, report)
    if not args.skip_split:
        h1_split(block, K, targets, report, n_grid=tuple(args.budgets),
                 n_repeats=args.repeats, seed=args.seed)
    if not args.skip_cold:
        h1_cold(block, K, targets, report, n_grid=tuple(args.budgets),
                h_grid=tuple(args.history), n_repeats=args.repeats,
                seed=args.seed, holdout=args.holdout, free_b=args.free_b,
                audit=args.audit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
