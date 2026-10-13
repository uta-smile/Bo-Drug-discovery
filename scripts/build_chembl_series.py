"""Build a congeneric-series lookup table from ChEMBL measured activities.

Pulls every pChEMBL-valued activity for a target, filters it down to something
defensible as a single comparable assay readout, then carves out the largest
Bemis-Murcko scaffold cluster -- that cluster is the congeneric series and
becomes the enumerated search space for the primary oracle.

    python scripts/build_chembl_series.py --target CHEMBL203 --out data/chembl/egfr

Add ``--report`` to see the candidate scaffold clusters without writing
anything, which is how you pick a target worth using.

Filtering choices worth knowing about, because they decide what the oracle
means:

* ``standard_relation = '='`` only.  A ``>10000 nM`` record is a censored
  bound, not a measurement; treating it as a pIC50 of 5 invents data and puts
  a cliff of fake inactives into the series.
* ``data_validity_comment`` must be empty -- ChEMBL flags transcription errors
  and out-of-range values here.
* Replicates are collapsed to the median, and a compound whose replicates
  disagree by more than ``--max-spread`` log units is dropped rather than
  averaged, since the oracle is supposed to be exact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molbo.chem import DEFAULT_OPTS, canonical_smiles, murcko_scaffold  # noqa: E402

API = "https://www.ebi.ac.uk/chembl/api/data"
PAGE = 1000


def fetch_activities(
    target: str,
    standard_type: str = "IC50",
    assay_type: str | None = "B",
    cache_dir: Path | None = None,
    timeout: int = 120,
) -> pd.DataFrame:
    """Download every pChEMBL-valued activity for a target, with a disk cache."""
    tag = f"{target}_{standard_type}_{assay_type or 'any'}"
    cache = (cache_dir / f"activities_{tag}.parquet") if cache_dir else None
    if cache and cache.exists():
        print(f"[cache] {cache}")
        return pd.read_parquet(cache)

    fields = [
        "molecule_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "standard_value",
        "standard_units",
        "standard_relation",
        "standard_type",
        "assay_chembl_id",
        "assay_type",
        "document_chembl_id",
        "data_validity_comment",
        "activity_comment",
        "potential_duplicate",
        "target_chembl_id",
    ]
    params = {
        "target_chembl_id": target,
        "standard_type": standard_type,
        "pchembl_value__isnull": "false",
        "limit": PAGE,
        "offset": 0,
        "only": ",".join(fields),
        "format": "json",
    }
    if assay_type:
        params["assay_type"] = assay_type

    rows: list[dict] = []
    session = requests.Session()
    session.headers["User-Agent"] = "molbo/0.1 (research; congeneric series build)"
    total = None
    while True:
        for attempt in range(4):
            try:
                r = session.get(f"{API}/activity.json", params=params, timeout=timeout)
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 3:
                    raise
                wait = 2**attempt
                print(f"  retry in {wait}s ({exc})")
                time.sleep(wait)
        payload = r.json()
        chunk = payload.get("activities", [])
        rows.extend(chunk)
        meta = payload.get("page_meta", {})
        if total is None:
            total = meta.get("total_count")
            print(f"[chembl] {target} {standard_type}: {total} activities")
        print(f"  fetched {len(rows)}/{total}", end="\r", flush=True)
        if not meta.get("next") or not chunk:
            break
        params["offset"] += PAGE
    print()

    df = pd.DataFrame(rows)
    if cache is not None and not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def clean(
    df: pd.DataFrame,
    max_sem: float | None = 0.5,
    max_spread: float | None = None,
    assay: str | None = None,
    document: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter to comparable measurements and collapse replicates to a median.

    Replicate disagreement is judged by the **standard error of the median**,
    not by ``max - min``.  The distinction is not academic: gefitinib has 224
    ChEMBL records spanning 5.4 log units (enzymatic vs cellular, WT vs mutant,
    different ATP concentrations), so a range filter throws it out -- and with
    it every other well-characterised compound, which are exactly the potent
    ones a lead-optimisation benchmark should contain.  Measured across this
    target, a ``range <= 1.0`` filter discards *100%* of compounds with ten or
    more measurements.

    The median of 224 measurements is a better estimate than a singleton, and
    the standard error says so: ``1.253 * sigma / sqrt(n)`` with sigma from the
    MAD.  Compounds measured once have no evidence against them and are kept.

    ``max_spread`` remains available as an additional, stricter filter, but it
    is off by default for the reason above.
    """
    def note(msg: str, frame: pd.DataFrame) -> None:
        if verbose:
            print(f"  {msg:<42s} {len(frame):>7d} rows  {frame.molecule_chembl_id.nunique():>6d} cpds")

    note("raw", df)
    if assay:
        df = df[df["assay_chembl_id"] == assay]
        note(f"assay == {assay}", df)
    if document:
        df = df[df["document_chembl_id"] == document]
        note(f"document == {document}", df)
    df = df[df["canonical_smiles"].notna()]
    df = df[df["pchembl_value"].notna()]
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df[df["pchembl_value"].notna()]
    note("has smiles + pchembl", df)

    df = df[df["standard_relation"] == "="]
    note("exact relation (drops censored)", df)

    if "data_validity_comment" in df:
        df = df[df["data_validity_comment"].isna()]
        note("no data-validity flag", df)
    if "potential_duplicate" in df:
        df = df[df["potential_duplicate"].fillna(0).astype(int) == 0]
        note("not flagged duplicate", df)

    df["smiles"] = [canonical_smiles(s, DEFAULT_OPTS) for s in df["canonical_smiles"]]
    df = df[df["smiles"].notna()]
    note("rdkit-parseable", df)

    mad = lambda s: float(np.median(np.abs(np.asarray(s) - np.median(s))))
    agg = (
        df.groupby("smiles")
        .agg(
            pchembl_value=("pchembl_value", "median"),
            n_measurements=("pchembl_value", "size"),
            spread=("pchembl_value", lambda s: float(s.max() - s.min())),
            mad=("pchembl_value", mad),
            molecule_chembl_id=("molecule_chembl_id", "first"),
            n_assays=("assay_chembl_id", "nunique"),
            n_docs=("document_chembl_id", "nunique"),
        )
        .reset_index()
    )
    # Robust sigma from the MAD, then the standard error of the median.
    # 1.4826 makes the MAD a consistent sigma estimator for a normal; 1.2533
    # is the median's asymptotic efficiency penalty relative to the mean.
    sigma = 1.4826 * agg["mad"]
    agg["sem_median"] = 1.2533 * sigma / np.sqrt(agg["n_measurements"])

    for col, thr, label in (("sem_median", max_sem, "std error of median"),
                            ("spread", max_spread, "replicate spread")):
        if thr is None:
            continue
        before = len(agg)
        agg = agg[agg[col] <= thr]
        if verbose:
            print(f"  {label + ' <= ' + str(thr):<42s} {len(agg):>7d} cpds "
                  f"({before - len(agg)} dropped as imprecise)")
    return agg.sort_values("pchembl_value", ascending=False).reset_index(drop=True)


def core_matches(df: pd.DataFrame, core: str) -> pd.DataFrame:
    """Select every compound containing ``core`` as a substructure.

    This is what a medicinal chemist means by a congeneric series -- one shared
    core, decorated differently -- and it is strictly more inclusive than exact
    Murcko clustering, which splits gefitinib away from erlotinib because their
    pendant rings differ.  Murcko clustering is still the right tool for
    *finding* the core; this is the right tool for collecting the series.
    """
    from rdkit import Chem

    patt = Chem.MolFromSmarts(core) or Chem.MolFromSmiles(core)
    if patt is None:
        raise ValueError(f"could not parse --core as SMARTS or SMILES: {core!r}")
    keep = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        keep.append(mol is not None and mol.HasSubstructMatch(patt))
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def scaffold_clusters(df: pd.DataFrame, generic: bool = False) -> list[tuple[str, pd.DataFrame]]:
    """Group compounds by exact Murcko scaffold, largest cluster first."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(df["smiles"]):
        scaf = murcko_scaffold(smi, generic=generic)
        if scaf:
            buckets[scaf].append(i)
    clusters = [(s, df.iloc[idx].reset_index(drop=True)) for s, idx in buckets.items()]
    clusters.sort(key=lambda kv: len(kv[1]), reverse=True)
    return clusters


def report(clusters: list[tuple[str, pd.DataFrame]], top: int = 15) -> None:
    print(f"\n{'n':>6}  {'range':>12}  {'sd':>5}  scaffold")
    print("-" * 92)
    for scaf, sub in clusters[:top]:
        v = sub["pchembl_value"]
        rng = f"{v.min():.2f}-{v.max():.2f}"
        print(f"{len(sub):>6}  {rng:>12}  {v.std():>5.2f}  {scaf[:60]}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="CHEMBL203", help="ChEMBL target id (default EGFR)")
    p.add_argument("--standard-type", default="IC50")
    p.add_argument("--assay-type", default="B", help="B=binding, F=functional, '' for any")
    p.add_argument("--out", type=Path, help="output stem, e.g. data/chembl/egfr")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--max-sem", type=float, default=0.5,
                   help="max standard error of the median, log units (n-aware; see clean.__doc__)")
    p.add_argument("--max-spread", type=float, default=None,
                   help="optional stricter filter on raw replicate range; off by default")
    p.add_argument("--assay", help="restrict to a single ChEMBL assay id (one protocol, one lab)")
    p.add_argument("--document", help="restrict to a single ChEMBL document id (one SAR paper)")
    p.add_argument("--min-size", type=int, default=200, help="minimum compounds in the series")
    p.add_argument("--generic-scaffold", action="store_true", help="ignore atom/bond types when clustering")
    p.add_argument(
        "--n-clusters",
        type=int,
        default=1,
        help="merge the N largest scaffold clusters; 0 keeps every compound for the target",
    )
    p.add_argument(
        "--core",
        help="SMARTS/SMILES core; keep every compound containing it (overrides clustering)",
    )
    p.add_argument("--report", action="store_true", help="list candidate clusters and exit")
    args = p.parse_args()

    raw = fetch_activities(
        args.target,
        standard_type=args.standard_type,
        assay_type=args.assay_type or None,
        cache_dir=args.cache_dir,
    )
    if raw.empty:
        print("no activities returned -- check the target id", file=sys.stderr)
        return 1

    print("\n[filter]")
    df = clean(raw, max_sem=args.max_sem, max_spread=args.max_spread,
               assay=args.assay, document=args.document)

    clusters = scaffold_clusters(df, generic=args.generic_scaffold)
    if args.report:
        report(clusters)
        print(f"\n{len(df)} compounds total across {len(clusters)} scaffolds")
        return 0

    if args.core:
        series = core_matches(df, args.core)
        chosen = [(args.core, series)]
        print(f"\n[core] {args.core} matches {len(series)} of {len(df)} compounds")
    elif args.n_clusters == 0:
        # Whole-target library: not congeneric, but the same measured readout and
        # ~20x larger, which is the regime where BO has room to beat random.
        chosen = []
        series = df.copy()
    else:
        chosen = clusters[: args.n_clusters]
        series = pd.concat([sub for _, sub in chosen], ignore_index=True)
    if len(series) < args.min_size:
        print(
            f"largest series has only {len(series)} compounds (< --min-size {args.min_size}).\n"
            "Re-run with --report to inspect, or try --generic-scaffold / --n-clusters.",
            file=sys.stderr,
        )
        return 1

    if not args.out:
        print("nothing written (pass --out). Series preview:")
        report(chosen, top=args.n_clusters)
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_suffix(".csv")
    series = series.rename(columns={"smiles": "canonical_smiles"})
    series.to_csv(csv, index=False)

    v = series["pchembl_value"]
    meta = {
        "target_chembl_id": args.target,
        "standard_type": args.standard_type,
        "assay_type": args.assay_type,
        "selection": "core-substructure" if args.core else ("whole-target" if args.n_clusters == 0 else "murcko-cluster"),
        "core": args.core,
        "scaffolds": [s for s, _ in chosen],
        "generic_scaffold": args.generic_scaffold,
        "n_compounds": int(len(series)),
        "value_column": "pchembl_value",
        "value_units": "pChEMBL (-log10 M); higher is more potent",
        "max_sem": args.max_sem,
        "max_replicate_spread": args.max_spread,
        "assay_chembl_id": args.assay,
        "document_chembl_id": args.document,
        "aggregation": "median of ChEMBL pchembl_value across retained measurements",
        "value_min": float(v.min()),
        "value_max": float(v.max()),
        "value_median": float(v.median()),
        "standardizer": DEFAULT_OPTS.key(),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "ChEMBL REST API",
    }
    meta_path = out.parent / (out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[write] {csv}  ({len(series)} compounds)")
    print(f"[write] {meta_path}")
    print(f"        pIC50 {v.min():.2f} - {v.max():.2f}  median {v.median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
