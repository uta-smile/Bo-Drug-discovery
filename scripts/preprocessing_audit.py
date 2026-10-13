"""Every preprocessing claim in the paper, recomputed from the block CSVs.

    python scripts/preprocessing_audit.py
    python scripts/preprocessing_audit.py --json results/preprocessing_audit.json

The manuscript makes a dozen quantitative statements about how the data was
prepared --- the potency transform, which values the censoring filter removes,
how many compounds survive it, how far the task correlation moves when it is
applied. None of them had been checked against the files since they were
written, and the three review rounds so far have found that numbers written
from memory are wrong about a third of the time.

This recomputes each one from ``data/bindingdb/*.csv`` and the stored scan
artefacts, prints it beside the claim, and exits nonzero on a mismatch. It is
the preprocessing half of the evidence package; ``experiment_inventory.py`` and
``regenerate_tables.py`` are the campaign half.

Claims are listed in ``CLAIMS`` with the tolerance each is quoted to, so that
changing a number in the paper without rerunning this is a test failure rather
than a silent drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dense_block_h1 import drop_censored, nanomolar_potency  # noqa: E402

BD = ROOT / "data" / "bindingdb"

# The three KIBA blocks the paper pins by UniProt accession, by their block CSV.
PINNED = {
    "kiba_prkc": "block_kiba_score_prkcd-prkcg-prkcz-prkd3.csv",
    "kiba_cdk5": "block_kiba_score_cdk5-dyrk1a-pim1-prkaca.csv",
    "kiba_lck": "block_kiba_score_gsk3b-lck-mapk8-mapk9.csv",
}

BINDINGDB = {
    "jak": "block_jak1-jak2-jak3-tyk2.csv",
    "cyp": "block_cyp2c19-cyp2c9-cyp2d6-cyp3a4.csv",
}

# Fixed screening doses, in micromolar, that a censored readout lands on.
DOSES_UM = (5.0, 10.0, 30.0, 50.0, 100.0)


def load(name: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(BD / name)
    targets = [c for c in df.columns if c != "smiles"]
    return df, targets


def median_rho(df: pd.DataFrame, targets: list[str]) -> float:
    """Median off-diagonal Spearman -- the block's task correlation."""
    vals = df[targets].to_numpy(float)
    off = [spearmanr(vals[:, i], vals[:, j]).statistic
           for i in range(len(targets)) for j in range(i + 1, len(targets))]
    return float(np.median(off))


def identical_across_targets(df: pd.DataFrame, targets: list[str]) -> float:
    v = df[targets].to_numpy(float).round(4)
    return float((v == v[:, [0]]).all(axis=1).mean())


def censoring(name: str) -> dict:
    df, targets = load(name)
    kept, piles = drop_censored(df, targets, pile_frac=0.02, verbose=False)
    spikes = {t: {float(k): v for k, v in p.items()} for t, p in piles.items()}
    top = {}
    for t in targets:
        c = df[t].round(4).value_counts()
        top[t] = float(c.iloc[0] / len(df)) if len(c) else 0.0
    return {
        "file": name, "targets": targets,
        "n_raw": int(len(df)), "n_kept": int(len(kept)),
        "survival": float(len(kept) / len(df)),
        "identical_all_targets": identical_across_targets(df, targets),
        "modal_share_per_column": top,
        "spikes": spikes,
        "rho_raw": median_rho(df, targets),
        "rho_filtered": median_rho(kept, targets) if len(kept) > 5 else None,
    }


def check(label, got, expected, tol, unit=""):
    ok = got is not None and abs(got - expected) <= tol
    mark = "ok " if ok else "MISMATCH"
    g = "n/a" if got is None else f"{got:.4g}{unit}"
    print(f"  [{mark}] {label:<52} paper {expected:g}{unit:<3}  "
          f"computed {g}")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    out, failures = {}, []

    # -- the potency transform -------------------------------------------
    print("POTENCY TRANSFORM\n")
    print("  pIC50 = 9 - log10(IC50 in nM); KIBA scores are used unchanged.")
    ok = True
    for um in DOSES_UM:
        got = float(nanomolar_potency(um * 1000.0))
        want = 9.0 - np.log10(um * 1000.0)
        ok &= abs(got - want) < 1e-12
    print(f"  [{'ok ' if ok else 'MISMATCH'}] transform agrees with "
          f"9 - log10(nM) at the five screening doses")
    print("\n  The censored values the paper names, as the doses that produce "
          "them:")
    for um in DOSES_UM:
        print(f"    {um:>5.0f} uM  ->  pIC50 {nanomolar_potency(um * 1000):.2f}")
    if not ok:
        failures.append("potency transform")

    # -- BindingDB censoring ---------------------------------------------
    print("\nCENSORING, BindingDB blocks\n")
    for key, name in BINDINGDB.items():
        if not (BD / name).exists():
            print(f"  {key}: {name} not present, skipped")
            continue
        c = censoring(name)
        out[key] = c
        lo = min(c["modal_share_per_column"].values())
        hi = max(c["modal_share_per_column"].values())
        print(f"  {key.upper()}  {c['n_raw']} compounds -> {c['n_kept']} kept "
              f"({c['survival']:.1%}); modal value holds "
              f"{lo:.1%}-{hi:.1%} of a column")
        print(f"        identical on all four targets: "
              f"{c['identical_all_targets']:.1%}")
        print(f"        median off-diagonal rho: raw {c['rho_raw']:.3f}"
              + (f", filtered {c['rho_filtered']:.3f}"
                 if c["rho_filtered"] is not None else ""))

    print("\n  against the manuscript:")
    if "cyp" in out and "jak" in out:
        cyp, jak = out["cyp"], out["jak"]
        checks = [
            ("CYP compounds before the filter", float(cyp["n_raw"]), 1472, 0),
            ("CYP compounds after the filter", float(cyp["n_kept"]), 335, 0),
            ("CYP survival", cyp["survival"] * 100, 22.8, 0.1, "%"),
            ("CYP identical on all four targets",
             cyp["identical_all_targets"] * 100, 40.3, 0.1, "%"),
            ("JAK compounds before the filter", float(jak["n_raw"]), 1563, 0),
            ("JAK compounds after the filter", float(jak["n_kept"]), 904, 0),
            ("JAK survival", jak["survival"] * 100, 57.8, 0.1, "%"),
            ("JAK identical on all four targets",
             jak["identical_all_targets"] * 100, 3.5, 0.1, "%"),
        ]
        for row in checks:
            label, got, exp, tol = row[:4]
            unit = row[4] if len(row) > 4 else ""
            if not check(label, got, exp, tol, unit):
                failures.append(label)
        lo = min(cyp["modal_share_per_column"].values()) * 100
        hi = max(cyp["modal_share_per_column"].values()) * 100
        okrange = 21 - 0.5 <= lo and hi <= 27 + 0.5
        print(f"  [{'ok ' if okrange else 'MISMATCH'}] "
              f"{'CYP modal share per column':<52} paper 21-27%   "
              f"computed {lo:.1f}-{hi:.1f}%")
        if not okrange:
            failures.append("CYP modal share")

    # -- KIBA: what the filter would do, and the rho it moves -------------
    print("\nCENSORING, KIBA pinned blocks (campaigns run with "
          "--keep-censored)\n")
    kiba = {}
    for label, name in PINNED.items():
        if not (BD / name).exists():
            print(f"  {label}: {name} not present, skipped")
            continue
        c = censoring(name)
        kiba[label] = c
        print(f"  {label:<12}{c['n_raw']:>5} raw -> {c['n_kept']:>4} if "
              f"filtered    rho raw {c['rho_raw']:.3f}"
              + (f"  filtered {c['rho_filtered']:.3f}"
                 if c["rho_filtered"] is not None else "  filtered n/a"))
    out["kiba"] = kiba

    if kiba:
        print("\n  against the manuscript:")
        want = {"kiba_prkc": (904, 0.866, 0.991),
                "kiba_cdk5": (899, 0.505, 0.781),
                "kiba_lck": (843, 0.277, 0.482)}
        for label, (n, rraw, rfilt) in want.items():
            if label not in kiba:
                continue
            c = kiba[label]
            for nm, got, exp, tol in (
                    (f"{label} compounds", float(c["n_raw"]), n, 0),
                    (f"{label} rho (raw block)", c["rho_raw"], rraw, 0.002),
                    (f"{label} rho (filtered block)", c["rho_filtered"],
                     rfilt, 0.002)):
                if not check(nm, got, exp, tol):
                    failures.append(nm)
        kept = [c["n_kept"] for c in kiba.values()]
        print(f"\n  [{'ok ' if min(kept) >= 32 and max(kept) <= 263 else 'MISMATCH'}]"
              f" {'KIBA rows surviving the filter':<52} paper 32-263   "
              f"computed {min(kept)}-{max(kept)}")
        if not (min(kept) >= 32 and max(kept) <= 263):
            failures.append("KIBA surviving-row range")

    print()
    if failures:
        print(f"  {len(failures)} claim(s) do not match the data:")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  Every preprocessing claim in the manuscript matches the "
              "block files.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=float),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
