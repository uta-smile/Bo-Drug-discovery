"""Find a target worth optimising against: undrugged, low-data, gradeable.

    python scripts/find_target.py --out results/target_search.json

The benchmark's weakest point is its target. EGFR has six approved drugs and
9,153 measured compounds, 74% of them active -- so "find an EGFR binder" is a
question nobody is asking, on a library decades of medicinal chemistry already
enriched. A reviewer asks that first.

What is wanted instead is a target that is genuinely open but still gradeable:

``no approved drug``
    ChEMBL's ``/mechanism`` records link molecules to the targets they act on.
    A target with no max_phase-4 mechanism has no approved drug against it.
    Imperfect -- approval is per indication, and a phase-4 molecule may appear
    as an off-target hit -- so treat it as a filter, not a verdict.
``a few hundred to a couple of thousand measured compounds``
    Enough to hold most out as an answer key, few enough that the low-data
    regime is real rather than simulated. Below ~200 there is nothing to grade
    against; above a few thousand the target is no longer under-explored.
``chemically diverse``
    Measured as mean pairwise Tanimoto distance. A congeneric series sits near
    0.70 and is the regime where cheap extra information sources are published
    to *lose*; the whole-target libraries here sit at 0.84-0.87.

Candidates are a hand-picked list rather than a sweep over all 5,869 human
single-protein targets, which would cost one API call each. They are drawn
from areas where the undrugged-ness is the point: hard oncology targets,
tuberculosis, and antiparasitics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

API = "https://www.ebi.ac.uk/chembl/api/data"

# (search term, why it is interesting) -- the note is for the reader, not the filter.
CANDIDATES = [
    ("WRN", "synthetic lethal in MSI cancers; no approved drug"),
    ("Werner syndrome ATP-dependent helicase", "same, alternate name"),
    ("MAT2A", "synthetic lethal with MTAP deletion"),
    ("PRMT5", "MTAP-deleted cancers"),
    ("USP7", "deubiquitinase, long-undrugged"),
    ("HPK1", "immuno-oncology kinase"),
    ("Cbl-b", "E3 ligase, immuno-oncology"),
    ("Diacylglycerol kinase alpha", "T-cell exhaustion"),
    ("Tyrosine-protein phosphatase non-receptor type 11", "SHP2"),
    ("KRAS", "G12D and beyond remain open"),
    ("STAT3", "classic undruggable transcription factor"),
    ("Catenin beta-1", "Wnt pathway, undrugged"),
    ("Myc proto-oncogene protein", "the canonical undruggable"),
    ("Enoyl-[acyl-carrier-protein] reductase", "InhA, tuberculosis"),
    ("Decaprenylphosphoryl-beta-D-ribose oxidase", "DprE1, tuberculosis"),
    ("Trehalose monomycolate exporter MmpL3", "tuberculosis"),
    ("UDP-3-O-acyl-N-acetylglucosamine deacetylase", "LpxC, Gram-negative"),
    ("DNA gyrase subunit B", "antibacterial"),
    ("Dihydroorotate dehydrogenase", "PfDHODH, malaria"),
    ("Plasmepsin", "malaria"),
    ("Cysteine protease falcipain", "malaria"),
    ("Trypanothione reductase", "trypanosomiasis"),
    ("Cruzipain", "Chagas disease"),
    ("Leucine-rich repeat serine/threonine-protein kinase 2", "LRRK2, Parkinson's"),
    ("Soluble epoxide hydrolase", "inflammation"),
    ("Indoleamine 2,3-dioxygenase", "IDO1, immuno-oncology"),
    ("Menin", "recently approved -- a positive control for the filter"),
]


def get(path: str, **params) -> dict:
    params.setdefault("limit", 1)
    for attempt in range(4):
        try:
            r = requests.get(f"{API}/{path}.json", params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return {}


def n_activities(target_id: str) -> int:
    j = get("activity", target_chembl_id=target_id, standard_type="IC50",
            pchembl_value__isnull="false")
    return int((j.get("page_meta") or {}).get("total_count", 0))


def approved_drugs(target_id: str) -> list[str]:
    """Molecules with a ChEMBL mechanism against this target that reached phase 4."""
    j = get("mechanism", target_chembl_id=target_id, limit=1000)
    ids = [m["molecule_chembl_id"] for m in j.get("mechanisms", [])
           if m.get("molecule_chembl_id")]
    out = []
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        mj = get("molecule", molecule_chembl_id__in=",".join(chunk), limit=1000)
        out += [m["molecule_chembl_id"] for m in mj.get("molecules", [])
                if str(m.get("max_phase") or "0").startswith("4")]
    return sorted(set(out))


def resolve(term: str) -> tuple[str, str] | None:
    j = get("target/search", q=term, limit=10)
    best = None
    for t in j.get("targets", []):
        if t.get("target_type") != "SINGLE PROTEIN":
            continue
        if t.get("organism") not in (None, "Homo sapiens") and "Homo" not in str(t.get("organism")):
            # keep pathogen targets too, but prefer an exact-ish name match
            pass
        best = best or (t["target_chembl_id"], f"{t.get('pref_name')} [{t.get('organism')}]")
    return best


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-compounds", type=int, default=200)
    p.add_argument("--max-compounds", type=int, default=4000)
    p.add_argument("--out", default="results/target_search.json")
    args = p.parse_args(argv)

    rows = []
    print(f"{'search term':<48s}{'chembl':<14s}{'IC50 rows':>10s}{'phase-4 drugs':>14s}")
    print("-" * 88)
    for term, note in CANDIDATES:
        hit = resolve(term)
        if not hit:
            print(f"{term[:46]:<48s}{'(not found)':<14s}")
            continue
        cid, name = hit
        n = n_activities(cid)
        drugs = approved_drugs(cid) if n >= args.min_compounds else []
        rows.append({"term": term, "note": note, "target_chembl_id": cid, "name": name,
                     "n_ic50": n, "approved": drugs})
        print(f"{term[:46]:<48s}{cid:<14s}{n:>10d}{len(drugs):>14d}")

    keep = [r for r in rows
            if args.min_compounds <= r["n_ic50"] <= args.max_compounds and not r["approved"]]
    keep.sort(key=lambda r: -r["n_ic50"])
    print(f"\n{'='*88}\nUNDRUGGED, {args.min_compounds}-{args.max_compounds} measured compounds\n{'='*88}")
    if not keep:
        print("  nothing passed -- widen the bounds or extend CANDIDATES")
    for r in keep:
        print(f"  {r['target_chembl_id']:<14s}{r['n_ic50']:>6d} rows   {r['name'][:44]:<46s}{r['note']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"min": args.min_compounds, "max": args.max_compounds,
                               "all": rows, "shortlist": keep}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("Next: scripts/build_chembl_series.py --target <id> --n-clusters 0 --out data/chembl/<name>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
