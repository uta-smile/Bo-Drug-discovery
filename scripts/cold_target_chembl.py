"""Does H1-cold survive a realistic task correlation? DOCKSTRING cannot say.

    python scripts/cold_target_chembl.py --out results/cold_target_chembl.json

FINDINGS section 16 found that a cold target borrowing from *already-collected*
data on its relatives beats a single-task GP, break-even at 5-10 historical
observations per auxiliary, +0.45 Spearman at H = 200.  It also flagged the
reason not to believe the effect size: the coregionalised GP selected a task
correlation of 1.0, and that is a property of AutoDock Vina.  Docking scores are
driven heavily by generic ligand size and lipophilicity, which every pocket
shares, so DOCKSTRING's targets are nearly one function.  Measured pIC50 is not
like that.

This runs the identical experiment on measured ChEMBL data, where the task
correlation is whatever medicinal chemistry actually delivers.  The design is
check C from ``transfer_headroom.py``:

* the **primary** target is cold -- ``n`` observations, charged to the campaign;
* each **auxiliary** target carries ``H`` historical observations, free, because
  someone already paid for them;
* ``H = 0`` is exactly the single-task Tanimoto GP;
* the score is Spearman on the primary's held-out compounds.

Two things differ from DOCKSTRING, and both are the point:

**The libraries are separate.**  Each ChEMBL target has its own compounds, which
is why section 14's dense-matrix question was unaskable here.  H1-cold does not
need a dense matrix -- the auxiliaries are measured on *different* molecules by
construction -- so this is the one form of the hypothesis these libraries can
test, and the reason the DOCKSTRING detour was worth taking is that it is the
form that survived.

**Compounds that appear in two libraries are kept.**  A compound with both a
JAK2 and a VEGFR2 number is not leakage: the VEGFR2 value is a real auxiliary
measurement that a deployed model would also have, and withholding it would
model a world where assays are never run twice.  The primary's *own* held-out
values are never shown to any arm.  Overlap is reported per pair so a reader can
see how much of any gain could come from it.
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

LIBRARIES = ("jak2_all", "cdk2_all", "vegfr2_all", "egfr_all")
"""Four kinases with enough compounds each to be both primary and auxiliary.
Kinases rather than an arbitrary four targets for the same reason DOCKSTRING's
subset is: a null over proteases, GPCRs and nuclear receptors would be
uninformative, because nobody expects transfer between them."""

H_GRID = (0, 5, 10, 25, 50, 100, 200)
N_GRID = (10, 20, 80)




def load_libraries(paths, *, cap, value_column, seed):
    """Union the libraries into one compound pool with a per-target index.

    Returns ``(smiles, per_target)`` where ``per_target[name]`` is
    ``(row_indices, values)`` into the shared pool. Capping each library keeps
    the shared Tanimoto gram to something that fits in memory; the cap is
    applied by random subsample, not by taking the head, because ChEMBL files
    arrive sorted by potency and the head is not a library.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    pool: dict[str, int] = {}
    smiles: list[str] = []
    per_target: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for path in paths:
        p = Path(path)
        name = p.stem
        df = pd.read_csv(p)
        col = next((c for c in ("canonical_smiles", "smiles", "SMILES")
                    if c in df.columns), None)
        if col is None or value_column not in df.columns:
            raise SystemExit(f"{p}: need a smiles column and {value_column}")
        df = df[[col, value_column]].dropna().drop_duplicates(subset=col)
        if len(df) > cap:
            df = df.iloc[rng.choice(len(df), size=cap, replace=False)]
        idx, vals = [], []
        for smi, v in zip(df[col], df[value_column]):
            j = pool.get(smi)
            if j is None:
                j = len(smiles)
                pool[smi] = j
                smiles.append(smi)
            idx.append(j)
            vals.append(float(v))
        per_target[name] = (np.asarray(idx, dtype=np.intp),
                            np.asarray(vals, dtype=float))
        print(f"    {name:<14s} {len(idx):>5d} compounds  "
              f"{value_column} {np.min(vals):.2f}..{np.max(vals):.2f}")
    print(f"    union: {len(smiles):,} unique compounds")
    return smiles, per_target


def report_overlap(per_target, report):
    """How many compounds each pair of libraries shares, and how they agree."""
    print("\n[1] library overlap")
    names = list(per_target)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia, va = per_target[a]
            ib, vb = per_target[b]
            common = np.intersect1d(ia, ib)
            rec = {"n_shared": int(len(common)),
                   "frac_of_smaller": float(len(common) / min(len(ia), len(ib)))}
            if len(common) >= 20:
                # Correlation of the measured values where both exist: the
                # empirical task correlation, and the honest comparison to
                # DOCKSTRING's +0.75 median.
                ma = {j: v for j, v in zip(ia, va)}
                mb = {j: v for j, v in zip(ib, vb)}
                rec["rho_on_shared"] = _rho([ma[j] for j in common],
                                            [mb[j] for j in common])
            out[f"{a}|{b}"] = rec
            shown = (f"rho {rec['rho_on_shared']:+.2f}" if "rho_on_shared" in rec
                     else "too few to correlate")
            print(f"    {a:<14s} {b:<14s} shared {len(common):>5d} "
                  f"({rec['frac_of_smaller']:>5.1%})  {shown}")
    report["overlap"] = out
    return out


def report_reach(per_target, K, report, *, seed, n_sample=300):
    """Can the auxiliary compounds even reach the primary's test set?

    A non-replication here has two candidate mechanisms and they are not the
    same finding. Either the task correlation is genuinely lower on measured
    data, or the auxiliary observations sit in a different region of chemical
    space and the Tanimoto kernel has nothing to carry the transfer through.
    DOCKSTRING cannot distinguish them, because every task there is measured on
    one uniform ZINC pool: its auxiliary molecules are drawn from exactly the
    distribution the test molecules come from. ChEMBL libraries are chemotype
    clusters that someone assembled per target, so they need not overlap at all.

    This measures reach directly: for each primary compound, its nearest
    neighbour inside its own library, against its nearest neighbour in the
    auxiliary pool. If the second is far below the first, the kernel cannot
    transfer regardless of how correlated the tasks are.
    """
    print("\n[1b] can the auxiliaries reach the primary's chemistry?")
    rng = np.random.default_rng(seed)
    names = list(per_target)
    out = {}
    print(f"    {'primary':<14s}{'own NN':>9s}{'aux NN':>9s}{'ratio':>8s}")
    for t in names:
        idx_t, _ = per_target[t]
        aux = np.unique(np.concatenate([per_target[a][0] for a in names if a != t]))
        aux = np.setdiff1d(aux, idx_t)
        pick = idx_t[rng.choice(len(idx_t), size=min(n_sample, len(idx_t)),
                                replace=False)]
        own = K[np.ix_(pick, idx_t)].copy()
        # Exclude self-similarity: a compound's nearest neighbour is itself.
        for r, j in enumerate(pick):
            own[r, np.flatnonzero(idx_t == j)] = -1.0
        own_nn = float(own.max(axis=1).mean())
        aux_nn = float(K[np.ix_(pick, aux)].max(axis=1).mean())
        out[t] = {"own_nn": own_nn, "aux_nn": aux_nn,
                  "ratio": aux_nn / own_nn if own_nn > 0 else float("nan")}
        print(f"    {t:<14s}{own_nn:>9.3f}{aux_nn:>9.3f}{out[t]['ratio']:>8.2f}")
    mean_ratio = float(np.mean([v["ratio"] for v in out.values()]))
    print(f"    mean ratio {mean_ratio:.2f} -- below ~0.7 means the auxiliary "
          "libraries occupy")
    print("    different chemistry, and a null here is about reach, not about "
          "task correlation.")
    report["reach"] = {"per_target": out, "mean_ratio": mean_ratio}
    return out


def run(smiles, per_target, K, report, *, n_grid, h_grid, n_repeats, seed,
        holdout=0.0, paired=False, audit=None):
    """The section 17 table: a cold primary, free history on its relatives.

    ``holdout`` and ``paired`` bind the ``H = 0`` control to the arms it is
    differenced against, and both default off so the stored table stays
    reproducible. Without them the primary's train/test split is redrawn for
    every ``H`` -- the gain is still unbiased, but it carries the split-to-split
    variance of two independent draws, and this table's headline is a *null*.
    A null measured with avoidable variance is a weaker claim than the same null
    measured without it, so the paired run is the one to quote.
    """
    audit = audit or Audit()
    names = list(per_target)
    print(f"\n[2] cold target with historical auxiliaries  "
          f"({len(names)} targets, {n_repeats} repeats)")
    print("    H is free -- already-collected assays, not campaign budget.")
    print("    pairing: "
          + (f"one evaluation set per repeat shared by every H (holdout {holdout:g})"
             if holdout > 0 else "evaluation set varies with H")
          + (", shared primary training rows" if paired
             else ", primary training rows drawn independently per H"))

    out: dict = {}
    rho_picked: list[float] = []
    # Compounds appear in more than one library, and the module docstring argues
    # that keeping them is right -- a VEGFR2 number on a molecule the JAK2
    # campaign is scored on is prior information a deployed model would also
    # have. That argument is only usable if the size of the exposure is stated,
    # so it is counted here rather than left as a sentence.
    shared_exposure: list[float] = []
    for ti, t in enumerate(names):
        idx_t, y_t = per_target[t]
        aux = [a for a in names if a != t]
        audit.excludes(t, aux, f"{t} auxiliary set")
        # Checked once per target rather than per draw: a library that holds the
        # same compound twice would let one "observation" be counted as two, and
        # an auxiliary whose values coincide with the primary's on every shared
        # compound is the primary relabelled, not a relative.
        audit.exactly(idx_t, len(idx_t), f"{t} library rows")
        prim_map = dict(zip(idx_t.tolist(), y_t.tolist()))
        for a in aux:
            ia, va = per_target[a]
            audit.exactly(ia, len(ia), f"{a} library rows")
            common = np.intersect1d(idx_t, ia)
            if common.size >= 5:
                a_map = dict(zip(ia.tolist(), va.tolist()))
                audit.not_relabelled(
                    np.asarray([prim_map[int(j)] for j in common]),
                    [np.asarray([a_map[int(j)] for j in common])],
                    f"{t} vs {a} on {common.size} shared compounds")
        rows: dict = {}
        for n in n_grid:
            for h in h_grid:
                acc = []
                for r in range(n_repeats):
                    g = np.random.default_rng(seed + 977 * r + 31 * n + h + 7919 * ti)
                    # `h` is in `g`'s seed, so without `paired` the control and
                    # every arm get a different split of the primary library.
                    g_prim = (np.random.default_rng(seed + 977 * r + 31 * n
                                                    + 7919 * ti) if paired else g)
                    if n + 1 > len(idx_t):
                        acc.append(np.nan)
                        continue
                    if holdout > 0:
                        # Reserved from a stream that depends on neither `n` nor
                        # `h`, so every cell of the row is scored on the same
                        # compounds within a repeat.
                        gh = np.random.default_rng(seed + 977 * r + 7919 * ti)
                        base = gh.permutation(len(idx_t))
                        n_te = int(holdout * len(idx_t))
                        te_local, avail = base[:n_te], base[n_te:]
                        if n > len(avail):
                            acc.append(np.nan)
                            continue
                        tr_local = avail[g_prim.choice(len(avail), size=n,
                                                       replace=False)]
                    else:
                        perm = g_prim.permutation(len(idx_t))
                        tr_local, te_local = perm[:n], perm[n:]
                    tr, te = idx_t[tr_local], idx_t[te_local]
                    truth = y_t[te_local]
                    audit.cell()
                    audit.exactly(tr, n, f"{t} n={n} H={h} r={r} primary rows")
                    audit.disjoint(tr, te, "primary training", "evaluation")
                    # The primary's own held-out values are the thing no arm may
                    # see. Checked as an identity on the value arrays, not on
                    # indices, because the failure that matters is a column
                    # picked by the wrong name.
                    audit.same_across((t, n, r), h, "evaluation rows", te)
                    audit.same_across((t, n, r), h, "primary training rows", tr)

                    if h == 0:
                        gp = TanimotoGP().fit_gram(K[np.ix_(tr, tr)], y_t[tr_local])
                        mu = gp.predict_gram(K[np.ix_(te, tr)], return_std=False)
                    else:
                        task_idx, task_val = [tr], [y_t[tr_local]]
                        short = False
                        for a in aux:
                            ia, va = per_target[a]
                            if h > len(ia):
                                short = True
                                break
                            pick = g.choice(len(ia), size=h, replace=False)
                            task_idx.append(ia[pick])
                            task_val.append(va[pick])
                            audit.exactly(ia[pick], h,
                                          f"{t} n={n} H={h} r={r} {a} history")
                        if short:
                            acc.append(np.nan)
                            continue
                        hist_all = np.concatenate(task_idx[1:])
                        # Not a violation -- see `shared_exposure` above -- but
                        # the number that bounds how much of any gain could be
                        # a relabelled look at the evaluation compounds.
                        shared_exposure.append(
                            float(np.intersect1d(te, hist_all).size) / max(len(te), 1))
                        gp = CoregionalisedGP().fit_gram(K, task_idx, task_val)
                        mu = gp.predict_gram(K, te)
                        rho_picked.append(gp.rho_)
                    acc.append(_rho(truth, mu))
                v = np.asarray(acc, dtype=float)
                ok = np.isfinite(v)
                rows[(n, h)] = {
                    "mean": float(v[ok].mean()) if ok.any() else float("nan"),
                    "sem": (float(v[ok].std() / np.sqrt(ok.sum())) if ok.sum() > 1
                            else float("nan")),
                }
        out[t] = rows
        print(f"    {t:<14s} done", flush=True)

    print(f"\n    Spearman on the primary's held-out compounds, "
          f"mean over {len(names)} targets")
    print(f"    {'n':>4s}{'H=0 (stl)':>12s}"
          + "".join(f"{'H=' + str(h):>11s}" for h in h_grid if h))
    print("    " + "-" * (16 + 11 * (len(h_grid) - 1)))
    def _mean(vals):
        """nanmean without the RuntimeWarning when every entry is NaN.

        An all-NaN slice is a configuration that could not be fitted, not a
        number near zero, and it has to survive as NaN so the verdict below can
        refuse rather than average nothing into a conclusion.
        """
        v = np.asarray(vals, dtype=float)
        return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float("nan")

    summary = {}
    for n in n_grid:
        base = _mean([out[t][(n, 0)]["mean"] for t in names])
        line, gains = f"    {n:>4d}{base:>12.3f}", {}
        for h in h_grid:
            if not h:
                continue
            per = np.asarray([out[t][(n, h)]["mean"] - out[t][(n, 0)]["mean"]
                              for t in names], dtype=float)
            ok = np.isfinite(per)
            v = _mean([out[t][(n, h)]["mean"] for t in names])
            gains[h] = {
                "value": v, "gain": v - base,
                "sem": (float(per[ok].std() / np.sqrt(ok.sum())) if ok.sum() > 1
                        else float("nan")),
                "n_targets_positive": int((per[ok] > 0).sum()),
                "n_targets": int(ok.sum()),
            }
            line += f"{v:>11.3f}" if np.isfinite(v) else f"{'n/a':>11s}"
        summary[n] = {"stl": base, "by_h": gains}
        print(line)

    print("\n    gain over the single-task GP (+ = joint modelling helps)")
    print(f"    {'n':>4s}" + "".join(f"{'H=' + str(h):>18s}" for h in h_grid if h))
    print("    " + "-" * (4 + 18 * (len(h_grid) - 1)))
    for n in n_grid:
        cells = ""
        for h, g in summary[n]["by_h"].items():
            cells += (f"{'n/a':>18s}" if not np.isfinite(g["gain"])
                      else f"{g['gain']:>+12.3f}+-{g['sem']:<5.3f}")
        print(f"    {n:>4d}{cells}")

    if rho_picked:
        q = np.percentile(rho_picked, [10, 50, 90])
        print(f"\n    task correlation selected by marginal likelihood: "
              f"median {q[1]:.2f}  (10th {q[0]:.2f}, 90th {q[2]:.2f})")
        report["rho_selected"] = {"median": float(q[1]), "p10": float(q[0]),
                                  "p90": float(q[2]), "n_fits": len(rho_picked)}

    if shared_exposure:
        q = float(np.mean(shared_exposure))
        print(f"\n    cross-library exposure: {q:.2%} of the primary's evaluation "
              "compounds")
        print("    also carry an auxiliary measurement in the drawn history. Kept on "
              "purpose")
        print("    (a real campaign would have those numbers); this is the ceiling on "
              "how much")
        print("    of any gain could come from the overlap rather than from the task "
              "structure.")
        report["shared_exposure"] = q
    audit.report(clean_lines=(
        "primary training and evaluation rows disjoint; exact draw sizes;",
        "no target as its own auxiliary; every auxiliary value matches its "
        "own library;",
        "control and arms share the evaluation set and the primary training rows",
    ))

    print("\n    verdict:")
    # An all-NaN table means no configuration fitted, and the branch below would
    # otherwise announce that H1-cold does not replicate -- a conclusion from no
    # data. Refuse before the prose.
    if not any(np.isfinite(g["gain"])
               for n in n_grid for g in summary[n]["by_h"].values()):
        smallest = min(len(v[0]) for v in per_target.values())
        print("      nothing in the grid could be fitted: the largest cell needs")
        print(f"      n + H x {len(names) - 1} observations and the smallest library "
              f"holds {smallest}.")
        print("      Not a result about H1 -- shrink --history or raise --cap.")
        report["summary"] = {}
        report["verdict"] = "no_fits"
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
            print(f"      n={n:<3d} no H in the grid clears 3 SEM; "
                  f"best {bg['gain']:+.3f} at H={best_h} "
                  f"({bg['n_targets_positive']}/{bg['n_targets']} targets).")
        else:
            live += 1
            print(f"      n={n:<3d} breaks even at H={hit}; best {bg['gain']:+.3f} "
                  f"at H={best_h} ({bg['n_targets_positive']}/{bg['n_targets']} "
                  "targets).")
    if live == len(n_grid):
        print("      H1-cold replicates on measured data at every budget.")
    elif live:
        print("      H1-cold replicates at some budgets and not others -- read the")
        print("      break-even column, not the headline.")
    else:
        print("      H1-cold does NOT replicate on measured pIC50. The DOCKSTRING")
        print("      result was a property of docking scores, whose targets are")
        print("      nearly one function; real kinases are not. Section 16's effect")
        print("      size should be withdrawn, not merely caveated.")
    report["summary"] = {str(k): v for k, v in summary.items()}
    report["verdict_budgets_live"] = live
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libraries", nargs="+",
                   default=[f"data/chembl/{n}.csv" for n in LIBRARIES])
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--cap", type=int, default=1000,
                   help="compounds kept per library; bounds the shared gram")
    p.add_argument("--budgets", type=int, nargs="+", default=list(N_GRID))
    p.add_argument("--history", type=int, nargs="+", default=list(H_GRID))
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--audit", action="store_true",
                   help="check at runtime the invariants this table rests on: "
                        "evaluation rows never trained on, exact draw sizes, no "
                        "target used as its own auxiliary, every auxiliary value "
                        "matching its own library, and whether the H=0 control "
                        "and the H>0 arms actually shared a split")
    p.add_argument("--holdout", type=float, default=0.0,
                   help="reserve this fraction of each primary library as one "
                        "evaluation set that every H is scored on. 0 -- the "
                        "default, and what section 17 used -- redraws the split "
                        "for every H")
    p.add_argument("--paired", action="store_true",
                   help="fit the H=0 control and every H>0 arm on the SAME "
                        "primary training rows, so the reported gain is a paired "
                        "difference. Off by default to keep the stored table "
                        "reproducible")
    p.add_argument("--out", default="results/cold_target_chembl.json")
    args = p.parse_args(argv)

    if 0 not in args.history:
        raise SystemExit("--history must include 0: it is the single-task control "
                         "every gain in the table is measured against")
    if not any(h > 0 for h in args.history):
        raise SystemExit("--history needs at least one non-zero value; 0 alone is "
                         "the control with nothing to compare against")

    t0 = time.time()
    print("loading libraries ...")
    smiles, per_target = load_libraries(args.libraries, cap=args.cap,
                                        value_column=args.value_column,
                                        seed=args.seed)
    X = fingerprint_matrix(smiles)
    K = tanimoto_gram(X, X)
    print(f"    gram {K.shape[0]}x{K.shape[1]} in {time.time() - t0:.0f}s")

    report: dict = {"libraries": list(args.libraries), "cap": args.cap,
                    "value_column": args.value_column, "seed": args.seed,
                    "n_compounds": len(smiles)}
    audit = Audit(enabled=args.audit)
    report_overlap(per_target, report)
    report_reach(per_target, K, report, seed=args.seed)
    run(smiles, per_target, K, report, n_grid=tuple(args.budgets),
        h_grid=tuple(args.history), n_repeats=args.repeats, seed=args.seed,
        holdout=args.holdout, paired=args.paired, audit=audit)
    # Travels in the JSON with the numbers it vouches for: a table whose
    # provenance says `clean: false` is not quotable, and that has to be
    # readable from the artefact rather than from a lost terminal.
    report["audit"] = audit.to_json()
    report["holdout"] = args.holdout
    report["paired"] = bool(args.paired)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
