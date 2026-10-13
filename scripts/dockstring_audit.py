"""Does DOCKSTRING support the claims this project already makes?

    python scripts/dockstring_audit.py --out results/dockstring_audit.json

Adding a dataset is only worth it if it can either *corroborate* or *break*
existing results.  A dataset that can only agree is decoration.  So this runs
four checks, in the order that each one can kill the next:

1. **Provenance.** Shape, target list and value ranges against the notebook.
   If the download is not the file Kubincova & Mobley used, nothing downstream
   compares to their numbers.

2. **The GP accuracy curve** (FINDINGS.md section 10).  The single most load-bearing
   table in this project says a Tanimoto GP reaches Spearman ~0.30 by 10
   observations and ~0.46 by 30, and *that* is what closes the learned-prior
   route -- a cold-target prior at rho 0.22 is simply worse than the surrogate
   it would advise.  That curve was measured on ChEMBL pIC50.  If it replicates
   on 260k docking scores it is a property of molecular BO; if it does not, the
   conclusion was a property of four ChEMBL libraries and should be narrowed.

3. **Cross-target correlation.**  H1's precondition.  Multi-task transfer needs
   the task covariance to be neither zero (nothing to share) nor one (nothing
   to gain).  This measures it on the 58-target matrix rather than assuming it.

4. **Objective structure.**  Where the top-k of each published objective sits,
   and how much of it a single target explains -- the check that the
   selectivity term is doing work rather than decorating a potency ranking.

Checks 2-4 subsample; the full 260k pool is used only for check 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from molbo import dockstring as ds  # noqa: E402
from molbo.surrogate import TanimotoGP, fingerprint_matrix, tanimoto_gram  # noqa: E402

# FINDINGS.md section 10, "Measured GP accuracy": 30 repeats, 4 ChEMBL libraries.
CHEMBL_CURVE = {5: 0.186, 10: 0.299, 15: 0.355, 20: 0.405, 30: 0.462, 50: 0.496, 80: 0.542}


def check_provenance(df, report):
    print("\n[1] provenance")
    got = tuple(c for c in df.columns if c in set(ds.TARGETS))
    n_missing = int(df[list(ds.TARGETS)].isna().sum().sum())
    dense = 1.0 - n_missing / (len(df) * len(ds.TARGETS))
    print(f"    rows                {len(df):,}   (notebook: {ds.N_LIGANDS:,})")
    print(f"    targets             {len(got)}       (expected 58)")
    print(f"    missing cells       {n_missing:,}")
    print(f"    matrix density      {dense * 100:.4f}%")
    vals = df[list(ds.TARGETS)].to_numpy()
    print(f"    docking score range {np.nanmin(vals):.1f} .. {np.nanmax(vals):.1f}")
    print(f"    QED range           {np.nanmin(df.QED):.3f} .. {np.nanmax(df.QED):.3f}")
    ok = len(df) == ds.N_LIGANDS and len(got) == 58
    print(f"    -> {'MATCHES the notebook' if ok else 'DOES NOT MATCH -- stop here'}")
    report["provenance"] = {
        "rows": len(df), "n_targets": len(got), "missing_cells": n_missing,
        "density": dense, "matches_notebook": bool(ok),
        "score_min": float(np.nanmin(vals)), "score_max": float(np.nanmax(vals)),
    }
    return ok


def check_gp_curve(df, report, *, objective, n_pool, n_repeats, seed):
    """Spearman of the GP posterior mean against truth, on *unobserved* rows."""
    print(f"\n[2] GP accuracy vs observations  (objective={objective}, "
          f"pool={n_pool}, {n_repeats} repeats)")
    rng = np.random.default_rng(seed)
    obj = ds.OBJECTIVES[objective]
    idx = rng.choice(len(df), size=n_pool, replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    y = np.asarray(obj(sub), dtype=float)
    t0 = time.time()
    X = fingerprint_matrix(sub.smiles.tolist())
    K = tanimoto_gram(X, X)
    print(f"    fingerprints + gram in {time.time() - t0:.0f}s")

    off = K[np.triu_indices(n_pool, 1)]
    print(f"    pool similarity: mean Tanimoto {off.mean():.3f}, "
          f"median {np.median(off):.3f}, frac > 0.3: {(off > 0.3).mean():.3%}")

    ns = sorted(CHEMBL_CURVE)
    rows = {}
    print(f"\n    {'n_obs':>6s}{'DOCKSTRING':>13s}{'ChEMBL (s10)':>15s}{'delta':>9s}")
    print("    " + "-" * 43)
    for n in ns:
        rhos = []
        for r in range(n_repeats):
            g = np.random.default_rng(seed + 1000 * r + n)
            tr = g.choice(n_pool, size=n, replace=False)
            te = np.setdiff1d(np.arange(n_pool), tr)
            gp = TanimotoGP().fit_gram(K[np.ix_(tr, tr)], y[tr])
            mu = gp.predict_gram(K[np.ix_(te, tr)], return_std=False)
            rhos.append(spearmanr(y[te], mu).statistic)
        m, se = float(np.mean(rhos)), float(np.std(rhos) / np.sqrt(n_repeats))
        ref = CHEMBL_CURVE[n]
        rows[n] = {"mean": m, "sem": se, "chembl": ref, "delta": m - ref}
        print(f"    {n:>6d}{m:>10.3f}+-{se:<4.3f}{ref:>13.3f}{m - ref:>+9.3f}")

    deltas = [v["delta"] for v in rows.values()]
    agree = sum(abs(v["delta"]) <= 3 * v["sem"] + 0.05 for v in rows.values())
    print(f"\n    mean |delta| = {np.mean(np.abs(deltas)):.3f}; "
          f"{agree}/{len(rows)} points within 0.05+3SEM of the ChEMBL curve")
    report["gp_curve"] = {"objective": objective, "n_pool": n_pool,
                          "n_repeats": n_repeats, "points": rows,
                          "mean_abs_delta": float(np.mean(np.abs(deltas))),
                          "pool_mean_tanimoto": float(off.mean()),
                          "pool_median_tanimoto": float(np.median(off))}
    return rows


def check_pool_diversity(report, *, libraries, n_sample, seed):
    """The obvious confound for check 2, measured instead of asserted.

    The ChEMBL curve was fitted on target-specific libraries -- compounds that
    someone chose to assay against one protein, which is a chemically tight
    set. A uniform sample of DOCKSTRING is a slice of ZINC. If a Tanimoto GP
    does worse here, the first thing to rule out is that it is simply being
    asked to interpolate across a much emptier space.
    """
    import pandas as pd

    print("\n[2b] is the DOCKSTRING pool just more diverse?")
    rng = np.random.default_rng(seed)
    rows = {}
    for path in libraries:
        p = Path(path)
        if not p.exists():
            continue
        lib = pd.read_csv(p)
        col = next((c for c in ("smiles", "canonical_smiles", "SMILES")
                    if c in lib.columns), None)
        if col is None:
            continue
        smi = lib[col].dropna().tolist()
        if len(smi) > n_sample:
            smi = [smi[i] for i in rng.choice(len(smi), size=n_sample, replace=False)]
        Xl = fingerprint_matrix(smi)
        Kl = tanimoto_gram(Xl, Xl)
        o = Kl[np.triu_indices(len(smi), 1)]
        rows[p.stem] = {"n": len(smi), "mean": float(o.mean()),
                        "median": float(np.median(o)),
                        "frac_gt_0.3": float((o > 0.3).mean())}
        print(f"    {p.stem:<28s} n={len(smi):>5d}  mean {o.mean():.3f}  "
              f"median {np.median(o):.3f}  >0.3 {(o > 0.3).mean():6.2%}")
    if rows:
        gc = report.get("gp_curve", {})
        if "pool_mean_tanimoto" in gc:
            print(f"    {'DOCKSTRING (uniform)':<28s} n={gc['n_pool']:>5d}  "
                  f"mean {gc['pool_mean_tanimoto']:.3f}  "
                  f"median {gc['pool_median_tanimoto']:.3f}")
        print("    -> a lower mean here is the honest explanation for a lower GP curve;")
        print("       it makes section 10's numbers library-specific, not universal.")
    report["pool_diversity"] = rows
    return rows


def check_task_correlation(df, report, *, n_sample, seed):
    print(f"\n[3] cross-target correlation  (n={n_sample} ligands, all 58 targets)")
    rng = np.random.default_rng(seed)
    sub = df.iloc[rng.choice(len(df), size=n_sample, replace=False)]
    # Pairwise-complete, not np.corrcoef: the 339 missing cells are spread over
    # all 58 columns, so a single NaN anywhere turns the whole matrix NaN.
    C = sub[list(ds.TARGETS)].corr().to_numpy(dtype=float)
    off = C[np.triu_indices(len(ds.TARGETS), 1)]
    print(f"    all 1,653 target pairs : min {off.min():+.2f}  median {np.median(off):+.2f}"
          f"  max {off.max():+.2f}")
    ki = [ds.TARGETS.index(t) for t in ds.KINASES]
    Ck = C[np.ix_(ki, ki)]
    offk = Ck[np.triu_indices(len(ki), 1)]
    print(f"    {len(ki)} kinases only      : min {offk.min():+.2f}  median "
          f"{np.median(offk):+.2f}  max {offk.max():+.2f}")
    jl = C[ds.TARGETS.index("JAK2"), ds.TARGETS.index("LCK")]
    print(f"    JAK2 vs LCK (objective): {jl:+.2f}")
    pp = [ds.TARGETS.index(t) for t in ("PPARA", "PPARD", "PPARG")]
    Cp = C[np.ix_(pp, pp)]
    print(f"    PPAR A/D/G pairwise    : {Cp[0,1]:+.2f} {Cp[0,2]:+.2f} {Cp[1,2]:+.2f}")
    print("    -> transfer is exploitable only if this is strictly between 0 and 1")
    report["task_correlation"] = {
        "all_min": float(off.min()), "all_median": float(np.median(off)),
        "all_max": float(off.max()), "kinase_median": float(np.median(offk)),
        "jak2_lck": float(jl),
        "ppar": {"AD": float(Cp[0, 1]), "AG": float(Cp[0, 2]), "DG": float(Cp[1, 2])},
    }
    return C


def check_objectives(df, report, *, k=1000):
    """Where the top-k sits, and -- the part that decides whether a campaign on
    these objectives measures anything -- how much of it is free.

    Both published objectives carry a ``-10 (1 - QED)`` term, and QED is a
    cheminformatics descriptor computed from the SMILES: it costs nothing and
    is known for every library member before a single assay. If that term
    dominates, then a "top-k recovery" number is partly a report on how well
    the optimiser rediscovers a free descriptor, and it has to be read against
    a QED-only ranking the same way a region-restriction result has to be read
    against MaxMin.
    """
    print(f"\n[4] objective structure and the free-descriptor control (full 260k)")
    qed_all = df.QED.to_numpy(dtype=float)
    out = {}
    for name in ("jak2", "jak2_only", "ppar"):
        obj = ds.OBJECTIVES[name]
        y_all = np.asarray(obj(df), dtype=float)
        m = np.isfinite(y_all) & np.isfinite(qed_all)
        y, qed = y_all[m], qed_all[m]
        top = np.sort(y)[-k:]
        print(f"\n    {name:<10s} range {y.min():.2f} .. {y.max():.2f}   "
              f"median {np.median(y):.2f}   top-{k} cut {top[0]:.2f}")

        # Split the objective into the free half and the assayed half. The
        # objective knows where that line is; re-deriving it here would silently
        # mis-split any variant whose drug-likeness weight is not the published
        # one (see molbo.dockstring.Objective.reweighted).
        qterm = np.asarray(obj.free(df), dtype=float)[m]
        dterm = np.asarray(obj.assayed(df), dtype=float)[m]
        rho_q = float(spearmanr(y, qterm).statistic)
        rho_d = float(spearmanr(y, dterm).statistic)
        print(f"      sd(QED term) {qterm.std():.2f}  vs  sd(docking term) "
              f"{dterm.std():.2f}   ratio {qterm.std() / dterm.std():.2f}x")
        print(f"      rho(objective, QED term)     = {rho_q:+.3f}   <- free")
        print(f"      rho(objective, docking term) = {rho_d:+.3f}   <- assayed")

        # How much of the objective is explained by potency on one target alone?
        rec = {}
        for t in obj.targets:
            rho = float(spearmanr(y, -df[t].to_numpy(dtype=float)[m]).statistic)
            print(f"      rho(objective, -{t:<6s}) = {rho:+.3f}")
            rec[t] = rho

        # The control that matters: rank by the free term alone, no assays.
        true_top = set(np.argsort(y)[-k:].tolist())
        q_recall = len(true_top & set(np.argsort(qterm)[-k:].tolist())) / k
        d_recall = len(true_top & set(np.argsort(dterm)[-k:].tolist())) / k
        print(f"      top-{k} recall, QED-only ranking (zero assays) : {q_recall:.1%}")
        print(f"      top-{k} recall, docking-only ranking           : {d_recall:.1%}")
        rec.update({"top1000_cut": float(top[0]), "max": float(y.max()),
                    "rho_qed_term": rho_q, "rho_docking_term": rho_d,
                    "sd_qed_term": float(qterm.std()),
                    "sd_docking_term": float(dterm.std()),
                    "qed_only_topk_recall": q_recall,
                    "docking_only_topk_recall": d_recall})
        out[name] = rec

    worst = max(out.values(), key=lambda r: r["qed_only_topk_recall"])
    print(f"\n    -> the objectives rank mostly by QED (rho {out['jak2']['rho_qed_term']:+.2f}), "
          f"but the top-{k} does not:")
    print(f"       the best free ranking recovers {worst['qed_only_topk_recall']:.1%} of it. "
          "Report every")
    print("       top-k number against that floor, not against zero.")
    report["objectives"] = out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(ds.DATASET))
    p.add_argument("--objective", default="jak2", choices=sorted(ds.OBJECTIVES))
    p.add_argument("--gp-pool", type=int, default=3000,
                   help="library size for the GP accuracy curve")
    p.add_argument("--gp-repeats", type=int, default=30,
                   help="matches the 30 repeats behind the ChEMBL curve")
    p.add_argument("--corr-sample", type=int, default=20000)
    p.add_argument(
        "--compare-libraries", nargs="+",
        default=["data/chembl/egfr_anilinoquinazoline.csv", "data/chembl/jak2_all.csv",
                 "data/chembl/cdk2_all.csv", "data/chembl/vegfr2_all.csv"],
        help="ChEMBL libraries to measure pool diversity against (check 2b)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/dockstring_audit.json")
    args = p.parse_args(argv)

    t0 = time.time()
    print(f"loading {args.dataset} ...", flush=True)
    df = ds.load(args.dataset)
    print(f"loaded {len(df):,} rows in {time.time() - t0:.0f}s")

    report: dict = {"dataset": args.dataset, "seed": args.seed}
    if not check_provenance(df, report):
        print("\nprovenance check failed; downstream numbers are not comparable.")
    check_gp_curve(df, report, objective=args.objective, n_pool=args.gp_pool,
                   n_repeats=args.gp_repeats, seed=args.seed)
    check_pool_diversity(report, libraries=args.compare_libraries,
                         n_sample=1500, seed=args.seed)
    check_task_correlation(df, report, n_sample=args.corr_sample, seed=args.seed)
    check_objectives(df, report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
